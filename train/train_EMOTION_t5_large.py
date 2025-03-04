import os
import yaml
import torch
import random
import logging
import argparse
import numpy as np
from datasets import load_from_disk
from collections import Counter
from datasets import load_dataset, load_metric
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model, PeftModelForSeq2SeqLM, TaskType
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_metric, DatasetDict
from sklearn.metrics import matthews_corrcoef
import sacrebleu
import evaluate

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's random generator (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch's random generator (GPU)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Make sure CuDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False  # This ensures deterministic behavior

def preprocess_function(examples, task_name, tokenizer, max_length):
    model_inputs = tokenizer(
        examples['instruction'],
        truncation=True, 
        max_length=max_length,
        padding='max_length',
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['output'],
            max_length=max_length, 
            padding='max_length', 
            truncation=True
        ).input_ids
    labels = [[(item if item != tokenizer.pad_token_id else -100) for item in label] for label in labels]
    model_inputs['labels'] = labels
    return model_inputs

# Calculate accuracy, f1-score and loss
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = labels[:, 0]
    preds = preds[:, 1]
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='macro') 
    return {
        "accuracy": accuracy,
        "f1-score": f1,
    }

def main():
    # 0. Get config
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        '--config',
        type=str,
        default='config/EMOTION-t5-lora-train-config/dailydialog-lora-train.yaml',
        help="Path to the config file",
    )
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    # Set random seed
    seed = config_data['rand_seed']
    set_seed(seed)

    # Get lora config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config_data['lora_config']['r'],
        lora_alpha=config_data['lora_config']['lora_alpha'],
        lora_dropout=config_data['lora_config']['lora_dropout'],
        target_modules=config_data['lora_config']['target_modules']
    )   
    # Load dataset
    task_name = config_data['task_name']
    dataset_train = load_dataset('json', data_files=f"data/Emotion_v2.5/{task_name}/train.json")
    dataset_eval = load_dataset('json', data_files=f"data/Emotion_v2.5/{task_name}/test.json")
    max_length = config_data['training']['max_length']
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    tokenized_datasets_train = dataset_train.map(
        lambda examples: preprocess_function(
            examples, 
            task_name=task_name,
            tokenizer=tokenizer,
            max_length=max_length,
        ), 
        batched=True,
        batch_size=1024,
        load_from_cache_file=False,
    )
    tokenized_datasets_eval = dataset_eval.map(
        lambda examples: preprocess_function(
            examples, 
            task_name=task_name,
            tokenizer=tokenizer,
            max_length=max_length,
        ), 
        batched=True,
        batch_size=1024,
        load_from_cache_file=False,
    )
    features_to_keep = ['input_ids', 'attention_mask', 'labels']
    for key in tokenized_datasets_train.keys():
        tokenized_datasets_train[key] = tokenized_datasets_train[key].remove_columns(
            [col for col in tokenized_datasets_train[key].column_names if col not in features_to_keep]
        )
        tokenized_datasets_eval[key] = tokenized_datasets_eval[key].remove_columns(
            [col for col in tokenized_datasets_eval[key].column_names if col not in features_to_keep]
        )
    dataset = DatasetDict({
        'train': tokenized_datasets_train['train'], 
        'validation': tokenized_datasets_eval['train'],
    })  
    
    # Apply lora to the model
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", max_length=196)
    model = get_peft_model(model, lora_config).to('cuda')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) # Attention please
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=config_data['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config_data['training']['per_device_eval_batch_size'],
        num_train_epochs=config_data['training']['num_train_epochs'],
        learning_rate=config_data['training']['learning_rate'],
        warmup_steps=config_data['training']['warmup_steps'],
        weight_decay=config_data['training']['weight_decay'],
        output_dir=config_data['training']['output_dir'],
        logging_dir=config_data['training']['logging_dir'],
        logging_steps=config_data['training']['logging_steps'],
        do_train=config_data['training']['do_train'],
        evaluation_strategy=config_data['training']['evaluation_strategy'],
        eval_steps=config_data['training']['eval_steps'],
        save_steps=config_data['training']['save_steps'],
        label_names=config_data['training']['label_names'],
        greater_is_better=config_data['training']['greater_is_better'],
        load_best_model_at_end=config_data['training']['load_best_model_at_end'],
        eval_accumulation_steps=config_data['training']['eval_accumulation_steps'],
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args, 
        train_dataset=dataset["train"],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()

