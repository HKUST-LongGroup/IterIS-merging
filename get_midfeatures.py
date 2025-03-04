import gc
import torch
import json
from functools import partial
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from safetensors import safe_open
from datasets import load_dataset, DatasetDict, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, BlipForConditionalGeneration
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from loras.train_GLUE_t5 import preprocess_function as preprocess_function_t5
from loras.train_EMOTION_t5_large import preprocess_function as preprocess_function_EMOTION
from loras.train_BLIP import ImageCaptioningDataset_Senticap
from loras.train_BLIP import ImageCaptioningDataset_FlickrStyle10k
SENTICAP_task_name = ["positive", "negative"]
FlickrStyle10k_task_name = ['roman', 'humor']
TASKS_blip_base = ['positive', 'negative', "roman", "humor"]

def get_lora_pos(lora_path):
    tensor_dict = safe_open(lora_path, framework='pt')
    lora_pos = []
    # tensor_name like this: 'base_model.model.decoder.block.1.layer.0.SelfAttention.q.lora_A.weight'
    for tensor_name in tensor_dict.keys():
        if tensor_name[17:-14] not in lora_pos:
            lora_pos.append(tensor_name[17:-14])
    # the elem in lora_pos like this: 'decoder.block.1.layer.0.SelfAttention.q'
    return lora_pos

def get_lora_matrix(model_name, load_tensor, idx_str, alpha, rank=8, no_weight=False):
    last = -7
    if no_weight == True:
        last = None
    for tensor_name in load_tensor.keys():
        if (idx_str[:last] in tensor_name) and (idx_str[:last]) and ('bias' not in idx_str) :
            if 't5' in model_name or 'bart' in model_name or 'blip' in model_name:
                tensor_name_A = 'base_model.model.' + idx_str[:last] + '.lora_A.weight'
                matrix_A = load_tensor.get_tensor(tensor_name_A)
                tensor_name_B = 'base_model.model.' + idx_str[:last] + '.lora_B.weight'
                matrix_B = load_tensor.get_tensor(tensor_name_B)
            return alpha / rank * (matrix_B @ matrix_A)

    return None

class BartWithHooks(BartForConditionalGeneration):
    def __init__(self, config, lora_path):
        super().__init__(config)
        self.inputs_to_track = {}
        self.layers_to_hook = get_lora_pos(lora_path)
        self.register_hooks()

    def hook_fn(self, module, inputs, outputs, layer_name):
        self.inputs_to_track[layer_name] = inputs[0].detach()#.cpu()

    def register_hooks(self):

        for layer_name in self.layers_to_hook:
            layer = dict(self.named_modules()).get(layer_name)
            if layer is not None:
                layer.register_forward_hook(lambda module, inputs, outputs, name=layer_name: 
                                            self.hook_fn(module, inputs, outputs, name))
            else:
                print(f"Layer {layer_name} not found.")

class T5WithHooks(T5ForConditionalGeneration):
    def __init__(self, config, lora_path):
        super().__init__(config)
        self.inputs_to_track = {}
        self.layers_to_hook = get_lora_pos(lora_path)
        self.register_hooks()

    def hook_fn(self, module, inputs, outputs, layer_name):
        self.inputs_to_track[layer_name] = inputs[0].detach()#.cpu()

    def register_hooks(self):

        for layer_name in self.layers_to_hook:
            layer = dict(self.named_modules()).get(layer_name)
            if layer is not None:
                layer.register_forward_hook(lambda module, inputs, outputs, name=layer_name: 
                                            self.hook_fn(module, inputs, outputs, name))
            else:
                print(f"Layer {layer_name} not found.")


class BlipWithHook(BlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)  # Call the parent class's __init__ with only the config
        self.inputs_to_track = {}
        self.layers_to_hook = get_lora_pos("loras/SENTICAP-lora-blip/positive/adapter_model.safetensors")
        self.register_hooks()

    def hook_fn(self, module, inputs, outputs, layer_name):
        # Ensure inputs[0] exists before accessing it
        if inputs and len(inputs) > 0:
            self.inputs_to_track[layer_name] = inputs[0].detach()#.cpu()
        else:
            print(f"No inputs found for layer: {layer_name}")

    def register_hooks(self):
        for layer_name in self.layers_to_hook:
            layer = dict(self.named_modules()).get(layer_name)
            if layer is not None:
                # Use partial to avoid lambda capturing issue
                hook = partial(self.hook_fn, layer_name=layer_name)
                layer.register_forward_hook(hook)
            else:
                print(f"Layer {layer_name} not found.")
            
def get_pretrain_matrix(keys, model_name):
    # Via the names of LoRAs, get the pretrain model matrix
    model = None
    if 't5' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to('cpu')
    elif 'bart' in model_name:
        model = BartForConditionalGeneration.from_pretrained(model_name).to('cpu')
    elif 'blip' in model_name:
        model = BlipForConditionalGeneration.from_pretrained(model_name).to('cpu')
        
    pretrain_matrix_dict = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            # name like this: 'decoder.block.1.layer.0.SelfAttention.q.weight'
            if name[:-7] in keys: #delete 'weight'
                pretrain_matrix_dict[name[:-7]] = param.to('cpu')
    return pretrain_matrix_dict

def select_long_data(dataset, key, samples_num, select_long, seed=42):
    return DatasetDict({
        key: Dataset.from_list(sorted(
            dataset[key].shuffle(seed).select(range(min(3000, len(dataset[key])))), 
            key=lambda x: sum(len(str(v)) for v in x.values()), 
            reverse=True)[:min(2000, len(dataset[key])*2//3)])
    }) if select_long else dataset
    
def balanced_sample(dataset, label_column, total_samples, seed):
    np.random.seed(seed)
    labels = dataset[label_column]
    unique_labels, _ = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    np.random.shuffle(unique_labels)
    samples_per_class_list = [total_samples // num_classes] * (num_classes-1) + \
    [total_samples - (total_samples // num_classes) * (num_classes-1)]
    sampled_indices = []
    for label, samples_per_class in zip(unique_labels, samples_per_class_list):
        label_indices = [i for i, l in enumerate(labels) if l == label]
        np.random.shuffle(label_indices)
        sampled_indices.extend(label_indices[:samples_per_class])
        np.random.shuffle(sampled_indices)
    return dataset.select(sampled_indices)

def get_samples(
    model_name, 
    tokenizer, 
    max_length,
    task_name,
    samples_num=10,
    select_long=40,
    seed=42,
    shuffle=False,
    if_balance=True,
):
    dataset_map = None
    
    if 't5-base' in model_name or 'bart' in model_name:
        task_actual = "mnli" if task_name == "mnli-mm" else task_name
        dataset = load_dataset("glue", task_actual)
        dataset = select_long_data(dataset, 'train', samples_num, select_long)
        if if_balance == False:
            dataset = DatasetDict({
                'train': dataset['train'].select(range(samples_num)) if shuffle==False else 
                dataset['train'].shuffle(seed=seed).select(range(samples_num)), 
            })
        else:
            dataselect = balanced_sample(
                dataset['train'], 
                label_column='label', 
                total_samples=samples_num,
                seed=seed
            )
            dataset = DatasetDict({
                'train': dataselect 
            })
        print("Mapping the data...", flush=True) 
        dataset_map = dataset.map(
            lambda examples: preprocess_function_t5(examples, task_name, tokenizer, max_length), 
            batched=True,
            batch_size=samples_num,
        )
    elif 't5-large' in model_name:
        dataset = load_dataset('json', data_files=f"data/Emotion_v2.5/{task_name}/train.json")
        dataset = select_long_data(dataset, 'train', samples_num, select_long)
        if if_balance == False:
            dataset = DatasetDict({
                'train': dataset['train'].select(range(samples_num)) if shuffle==False else 
                dataset['train'].shuffle(seed=seed).select(range(samples_num)), 
            })
        else:
            dataselect = balanced_sample(
                dataset['train'], 
                label_column='output', 
                total_samples=samples_num,
                seed=seed,
            )
            dataset = DatasetDict({
                'train': dataselect 
            })
        print("Mapping the data...", flush=True)
        dataset_map = dataset.map(
            lambda examples: preprocess_function_EMOTION(examples, task_name, tokenizer, max_length), 
            batched=True,
            batch_size=samples_num,
        )
    elif "blip" in model_name:
        # image caption task using blip has no parameters such as if_balance and select_long
        assert task_name in TASKS_blip_base
        dataset = load_dataset(
            f"data/SENTICAP/{task_name}" if task_name in SENTICAP_task_name else f"data/FlickrStyle10k/{task_name}"
        ) 
        dataset = DatasetDict({
            'train': dataset['train'].select(range(samples_num)) if shuffle==False else 
                dataset['train'].shuffle(seed=seed).select(range(samples_num)),
        })
        print("Mapping the data...", flush=True)
        if task_name in SENTICAP_task_name:
            train_dataset = ImageCaptioningDataset_Senticap(
                dataset=dataset, 
                processor=tokenizer, 
                task_name=task_name,
                max_length=max_length,
                datatype='train',
            )
        elif task_name in FlickrStyle10k_task_name:
            train_dataset = ImageCaptioningDataset_FlickrStyle10k(
                dataset=dataset, 
                processor=tokenizer, 
                task_name=task_name,
                max_length=max_length,
                datatype='train',
            )
        dataset_map = next(iter(DataLoader(train_dataset, shuffle=True, batch_size=samples_num)))
    return dataset_map

def merge_peft(model, model_name, lora_path, rank):
    tensor = safe_open(lora_path + "/adapter_model.safetensors", framework='pt')
    with open(lora_path + "/adapter_config.json", "r") as f:
        config_json = json.load(f)
    alpha = config_json["lora_alpha"]
    with torch.no_grad():
        for name, param in model.named_parameters():
            lora_matrix = get_lora_matrix(model_name, tensor, name, alpha, rank)
            if lora_matrix is not None:
                param.copy_(param + lora_matrix)
    return model


def get_midfeatures(
    lora_path, 
    rank, 
    model_name, 
    input_ids=None, # if blip, (pixel_values, input_ids, attention_mask)
    model=None,
    max_length=None, # only for blip
    **generation_kwargs,
):
    if model is not None:
        model.inputs_to_track.clear()
    elif 't5' in model_name:
        model = T5WithHooks.from_pretrained(model_name, lora_path=lora_path+"/adapter_model.safetensors")
        model = merge_peft(model, model_name, lora_path, rank).to('cuda')
    elif 'bart' in model_name:
        model = BartWithHooks.from_pretrained(model_name, lora_path=lora_path+"/adapter_model.safetensors")
        model = merge_peft(model, model_name, lora_path, rank).to('cuda')
    elif 'blip' in model_name:
        model = BlipWithHook.from_pretrained(model_name)
        model = merge_peft(model, model_name, lora_path, rank).to('cuda')
    
    with torch.no_grad():
        if 'blip' in model_name:
            outputs = model.generate(**input_ids, max_length=max_length)
        else:
            outputs = model.generate(input_ids, **generation_kwargs)
    return model, dict(model.inputs_to_track.items())
        
def get_all_midfeatures(
    lora_path,
    task_targets,
    model_name,
    max_length,
    seed,
    rank,
    samples_num,
    shuffle,
    select_long,
    if_balance=True,
    if_divide=False,
    inner_num=2,
    outer_num=10,
    **generation_kwargs,
):  
    tokenizer = AutoTokenizer.from_pretrained(model_name) if 'blip' not in model_name else AutoProcessor.from_pretrained(model_name) 
    midfeatures_list, tokenized_input = [], []
    input_ids = None
    if if_divide == False or 'blip' in model_name:
        for i in range(len(lora_path)):
            dataset = get_samples(
                model_name=model_name,
                tokenizer=tokenizer,
                max_length=max_length,
                task_name=task_targets[i],
                samples_num=samples_num,
                if_balance=if_balance,
                select_long=select_long,
                seed=seed,
                shuffle=shuffle,
            )
            if 'blip' not in model_name:
                input_ids = torch.tensor(dataset['train']['input_ids']).to('cuda')
            else:
                input_ids = {
                    "pixel_values": torch.tensor(dataset['pixel_values']).to('cuda'),
                    "input_ids": torch.tensor(dataset['input_ids']).to('cuda'), 
                    "attention_mask": torch.tensor(dataset['attention_mask']).to('cuda')
                }
            _, midfeature = get_midfeatures(
                rank=rank,
                input_ids=input_ids,
                max_length=max_length,
                lora_path=lora_path[i],
                model_name=model_name, 
                **generation_kwargs,
            ) 
            midfeatures_list.append(midfeature)
            tokenized_input.append(input_ids)
            print(f"{lora_path[i]}: midfeatures are all collected.")
    else:
        assert inner_num * outer_num == samples_num
        for i in range(len(lora_path)):
            dataset = get_samples(
                model_name=model_name,
                select_long=select_long,
                tokenizer=tokenizer,
                max_length=max_length,
                task_name=task_targets[i],
                samples_num=samples_num,
                if_balance=if_balance,
                shuffle=shuffle,
                seed=seed,
            )
            input_ids = torch.tensor(dataset['train']['input_ids'])
            model, midfeature = None, None
            for j in range(outer_num):
                model, midfeature_item = get_midfeatures(
                    rank=rank,
                    model=model,
                    max_length=max_length,
                    lora_path=lora_path[i],
                    model_name=model_name, 
                    input_ids=input_ids[j*inner_num: (j+1)*inner_num, :].to('cuda'),
                    **generation_kwargs,
                ) 
                midfeature = midfeature_item if j == 0 else {key: torch.cat([value, midfeature_item[key]], dim=0) for key, value in midfeature.items()}
                torch.cuda.empty_cache()
                gc.collect()
            midfeatures_list.append(midfeature)
            tokenized_input.append(input_ids)
            print(f"{lora_path[i]}: midfeatures are all collected.")
            
    lora_pos = midfeatures_list[0].keys()
    for item in midfeatures_list:
        assert item.keys() == lora_pos
    X_dict = {}
    for item in lora_pos:
        X_dict[item] = torch.cat(
            [midfeatures[item].unsqueeze(dim=1) for midfeatures in midfeatures_list],
            dim=1
        )
    return tokenized_input, X_dict



