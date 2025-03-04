import gc
import pdb
import yaml
import time
import torch
import random
import argparse
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from safetensors import safe_open
from sklearn.metrics import f1_score
from eval_model import eval_iteris_model
from get_midfeatures import T5WithHooks, BartWithHooks, BlipWithHook
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BartForConditionalGeneration, AutoTokenizer, AutoProcessor
from get_midfeatures import get_all_midfeatures, get_samples, get_pretrain_matrix, get_lora_matrix

GLUE_task_name = [
    "mnli", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc",
]
EMOTION_task_name = [
    "emoint", "emotion-cause",
    "tec", "isear",
]
SENTICAP_task_name = ['positive', 'negative']
FlickrStyle10k_task_name = ["roman", "humor"]
TASKS_blip_base = ['positive', 'negative', "roman", "humor"]

def get_loras_path(task_type, model_name):
    lora_path_dict = {}
    if 't5' in model_name and task_type == "GLUE_t5":
        for item in GLUE_task_name:
            lora_path_dict[item] = f"loras/GLUE-lora-t5/{item}"
    elif 'bart' in model_name and task_type == "GLUE_bart":
        for item in GLUE_task_name:
            lora_path_dict[item] = f"loras/GLUE-lora-bart/{item}"
    elif 't5-large' in model_name and task_type == "EMOTION_t5_large":
        for item in EMOTION_task_name:
            lora_path_dict[item] = f"loras/EMOTION-lora-t5/{item}"
    elif 'blip' in model_name and task_type == "TASKS_blip_base":
        for item in FlickrStyle10k_task_name:
            lora_path_dict[item] = f"loras/FlickrStyle10k-lora-blip/{item}"
        for item in SENTICAP_task_name:
            lora_path_dict[item] = f"loras/SENTICAP-lora-blip/{item}"
    return lora_path_dict

# Set all the seeds the same
def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def reg_math(term, alpha):
    term_list = [term[i] + alpha[i] * torch.eye(term[i].size(0), dtype=term.dtype, device=term.device) for i in range(term.shape[0])]
    return torch.stack(term_list)

def solution_matrix(
    W_list, 
    X_list, 
    X_tilde_list, 
    ceof_list, 
    manual_ceof,
    alpha_1=1e-7, 
    alpha_2=1e-7,
    reg_ceof=5e-4,
):  
    with torch.no_grad():
        N = W_list.shape[0]
        manual_ceof = torch.tensor(manual_ceof).to('cuda')
        X_list, X_tilde_list = X_list.transpose(0,1).flatten(start_dim=1,end_dim=2), X_tilde_list.transpose(0,1).flatten(start_dim=1,end_dim=2)

        X_tilde_list = (1 - reg_ceof) * X_tilde_list + reg_ceof * X_list
        X_X_tilde = torch.matmul(X_list.transpose(-1,-2), X_tilde_list)
        X_X_tilde_norm = torch.norm(X_X_tilde, p='fro', dim=[-2,-1]) * alpha_1
        X_X_tilde = reg_math(X_X_tilde, X_X_tilde_norm)

        X_tilde_X_tilde = torch.matmul(X_tilde_list.transpose(-1,-2), X_tilde_list)
        X_tilde_X_tilde_norm = torch.norm(X_tilde_X_tilde, p='fro', dim=[-2,-1]) * alpha_2
        X_tilde_X_tilde = reg_math(X_tilde_X_tilde, X_tilde_X_tilde_norm)

        term1 = torch.sum(torch.matmul(W_list, X_X_tilde) * (ceof_list*manual_ceof).view(N,1,1), dim=0).double()
        term2 = torch.sum(X_tilde_X_tilde * (ceof_list*manual_ceof).view(N,1,1), dim=0).double()
        results = torch.linalg.solve(term2.t(), term1.t()).double().t()
        
        norm_value = ceof_list * torch.norm(torch.matmul(W_list, X_list.transpose(-1,-2)) - torch.matmul(torch.stack([results, results]).float(), X_list.transpose(-1,-2)), dim=[-2, -1])**2
        # print(norm_value)
        return results.to('cpu')

def update_param(
    seed,
    max_iter,
    lora_path,
    model_name,
    task_targets,
    manual_ceof,
    shuffle,
    with_pretrain_matrix=0,
    max_length=512,
    lora_alpha=[32,32],
    alpha_1=1e-7,
    alpha_2=1e-7,
    reg_ceof=5e-4,
    rank=8,
    select_long=40,
    inner_num=2,
    outer_num=10,
    samples_num=20,
    if_divide=True,
    if_balance=True,
    **generation_kwargs,
):
    input_ids_list, X_dict = get_all_midfeatures(
        rank=rank,
        seed=seed,
        select_long=select_long,
        lora_path=lora_path,
        model_name=model_name,
        max_length=max_length,
        task_targets=task_targets,
        if_divide=if_divide,
        if_balance=if_balance,
        shuffle=shuffle,
        inner_num=inner_num,
        outer_num=outer_num,
        samples_num=samples_num,
        **generation_kwargs,
    )
    
    pretrain_matrix_dict = get_pretrain_matrix(X_dict.keys(), model_name=model_name)
    # Via the names of LoRAs, get the pretrain model matrix

    lora_adapter_path_list = [
        lora_adapter_path + "/adapter_model.safetensors" for lora_adapter_path in lora_path
    ]
    tensors_lora = [safe_open(tensor_lora, framework='pt') for tensor_lora in lora_adapter_path_list]
    torch.cuda.empty_cache()
    X_tilde_dict = {}
    for time in range(max_iter):
        torch.cuda.empty_cache() 
        gc.collect()
        tar_lora_list = {}
        print(f"-----------iter: {time}---------------")
        print("Calculate the opt solution...")
        with torch.no_grad():
            for idx in X_dict.keys():
                # print(idx)
                W_list, X_list = torch.stack(
                    [get_lora_matrix(model_name, tensors_lora[i], idx, lora_alpha[i], rank=rank, no_weight=True) for i in range(len(tensors_lora))]
                ).to('cuda'), X_dict[idx].to('cuda') # Get lora matrix and mid-features
                N = W_list.shape[0]  
                merge_W = W_list + pretrain_matrix_dict[idx].unsqueeze(0).repeat(N, 1, 1).to('cuda')
                ceof_list = torch.norm(merge_W, p='fro', dim=[-2,-1])**2 / \
                            torch.sum(torch.norm(torch.matmul(X_list, merge_W.transpose(1,2)), p='fro', dim=[-2,-1])**2, dim=0)
                # ceof_list = torch.tensor([1.0, 1.0]).to('cuda')
                if with_pretrain_matrix == 0:
                    tar_lora_list[idx] = solution_matrix(W_list, X_list, X_list, ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu') if time == 0 else \
                                        solution_matrix(W_list, X_list, X_tilde_dict[idx].to('cuda'), ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu')
                elif with_pretrain_matrix == 1:
                    tar_lora_list[idx] = solution_matrix(merge_W, X_list, X_list, ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu') if time == 0 else \
                                        solution_matrix(merge_W, X_list, X_tilde_dict[idx].to('cuda'), ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu')
                torch.cuda.empty_cache()
                gc.collect() 
        print("Calculation Done!")
        print("Loading and updating the original model...")
        model = None
        if 't5' in model_name:
            model = T5WithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'bart' in model_name:
            model = BartWithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'blip' in model_name:
            model = BlipWithHook.from_pretrained(model_name).to('cuda')
        # Updating the model
        number_update = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                # name like this: 'decoder.block.1.layer.0.SelfAttention.q.weight'
                if name[:-7] in tar_lora_list.keys(): #delete 'weight'
                    lora_matrix = tar_lora_list[name[:-7]].to('cuda')
                    if with_pretrain_matrix == 0:
                        param.copy_(lora_matrix + param)
                    elif with_pretrain_matrix == 1:
                        param.copy_(lora_matrix)
                    number_update += 1
        if number_update == len(tar_lora_list.keys()):
            print("All the targets which correspond to LoRAs are updated successfully!")
        else:
            print("Something got wrong...")
        torch.cuda.empty_cache() 
        max_memory = torch.cuda.max_memory_allocated()
        print(f"Max memory usage so far: {max_memory / 1024 ** 2:.2f} MB", flush=True)
        if time == max_iter-1:
            return model
        # Record the mid-features of updated model
        records_list = []
        if if_divide == True:
            assert inner_num * outer_num == len(input_ids_list[0])
            for input_ids in input_ids_list:
                print("Generating lora midfeatures...")
                dict_record_item = {}
                for i in range(outer_num):
                    with torch.no_grad():
                        outputs = model.generate(input_ids[i*inner_num:(i+1)*inner_num, :].to('cuda'))
                    temp_dict = dict(model.inputs_to_track.items())
                    dict_record_item = temp_dict if i == 0 else {key: torch.cat([value, temp_dict[key]], dim=0) for key, value in dict_record_item.items()}
                    model.inputs_to_track.clear()
                    torch.cuda.empty_cache()
                records_list.append(dict_record_item) 
        else:
            for input_ids in input_ids_list:
                model.inputs_to_track.clear()
                torch.cuda.empty_cache()
                print("Generating lora midfeatures...")
                dict_record_item = {}
                with torch.no_grad():
                    if 'blip' in model_name:
                        outputs = model.generate(**input_ids, max_length=max_length)
                    else:
                        outputs = model.generate(input_ids.to('cuda'))
                records_list.append(dict(model.inputs_to_track.items())) 

        for item in records_list[0].keys():
            X_tilde_dict[item] = torch.cat(
                [records[item].unsqueeze(dim=1) for records in records_list], 
                dim=1,
            ).to('cpu')


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, default="config/methods-config/iteris-config.yaml", help="Path to the config file")
    parser.add_argument('--task_type', type=str, choices=['GLUE_t5', 'EMOTION_t5_large', 'TASKS_blip_base'], 
                        default='GLUE_t5', help="Choose a task type from the list of options.")
    args = parser.parse_args()
    task_type = args.task_type
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    set_seed(config_data['seed'])
    model_name = config_data[task_type]['model_name']
    task_targets = config_data[task_type]['task_targets']
    lora_path = [get_loras_path(task_type, model_name)[item] for item in task_targets]
    with_pretrain_matrix = config_data[task_type]['with_pretrain_matrix']
    tokenizer = AutoTokenizer.from_pretrained(model_name) if 'blip' not in model_name else AutoProcessor.from_pretrained(model_name)
    save = config_data[task_type]['save'],

    # IterIS algorithm
    start_time = time.time() 
    model = update_param(
        task_targets=task_targets,
        lora_path=lora_path,
        model_name=model_name,
        with_pretrain_matrix=with_pretrain_matrix,
        max_iter=config_data[task_type]['max_iter'],
        max_length=config_data[task_type]['max_length'],
        lora_alpha=config_data[task_type]['lora_alpha'],
        alpha_1=config_data[task_type]['alpha_1'],
        alpha_2=config_data[task_type]['alpha_2'],
        reg_ceof=config_data[task_type]['reg_ceof'],
        rank=config_data[task_type]['rank'],
        samples_num=config_data[task_type]['samples_num'],
        manual_ceof=config_data[task_type]['manual_ceof'],
        if_divide=config_data[task_type]['if_divide'],
        if_balance=config_data[task_type]['if_balance'],
        inner_num=config_data[task_type]['inner_num'],
        outer_num=config_data[task_type]['outer_num'],
        seed=config_data['seed'],
        select_long=config_data[task_type]['select_long'],
        shuffle=config_data[task_type]['shuffle'],
    )
    if save == 1:
        torch.save(model, "merged_model/merged_model.pth")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    torch.cuda.empty_cache() 
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
    
    # model evaluation
    for task_name in task_targets:
        eval_iteris_model(
            model=model, 
            tokenizer=tokenizer,
            model_name=model_name,
            task_name=task_name,
            max_length=config_data[task_type]['max_length'], 
            per_device_eval_batch_size=config_data[task_type]['per_device_eval_batch_size'],
        )
    torch.cuda.empty_cache() 
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
    

if __name__ == "__main__":
    main()
