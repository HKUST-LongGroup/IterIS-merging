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


GLUE_key_single = {
    'cola': {
        'input': 'sentence',
        'label': 'label',
    },
    'sst2': {
        'input': 'sentence',
        'label': 'label',
    },
}
GLUE_key_double = {
    'ax': {
        'input': ['premise', 'hypothesis'],
        'label': 'label',
    },
    'mnli': {
        'input': ['premise', 'hypothesis'],
        'label': 'label',
    },
    'mrpc': {
        'input': ['sentence1', 'sentence2'],
        'label': 'label',
    },
    'qnli': {
        'input': ['question', 'sentence'],
        'label': 'label',
    },
    'qqp': {
        'input': ['question1', 'question2'],
        'label': 'label',
    },
    'rte': {
        'input': ['sentence1', 'sentence2'],
        'label': 'label',
    },
    'stsb': {
        'input': ['sentence1', 'sentence2'],
        'label': 'label',
    },
    'wnli': {
        'input': ['sentence1', 'sentence2'],
        'label': 'label',
    },
}
label2text = {
    'cola': {1: 'yes', 0: 'no'},
    'sst2': {1: 'yes', 0: 'no'},
    'mnli': {0: 'yes', 1: 'maybe', 2: 'no'},
    'rte':  {0: 'yes', 1: 'no'},
    'wnli': {1: 'yes', 0: 'no'},
    'qqp':  {1: 'yes', 0: 'no'},
    'mrpc': {1: 'yes', 0: 'no'},
    'qnli': {0: 'yes', 1: 'no'},
}
# Set all the seeds the same
def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's random generator (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch's random generator (GPU)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Make sure CuDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False  # This ensures deterministic behavior

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's random generator (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch's random generator (GPU)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Make sure CuDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False  # This ensures deterministic behavior

def prompt_text(input1, task_name, input2=None):
    assert (task_name in GLUE_key_single) or (task_name in GLUE_key_double)
    if task_name == 'cola':
        return f"""
                    Instruction: Is the following sentence grammatically correct? Answer acceptable or unacceptable.
                    Input: {input1}
                    Answer: 
                """
    elif task_name == 'sst2':
        return f"""
                    Instruction: Does the following sentence express a positive or negative sentiment? Answer positive or negative.
                    Input: {input1}
                    Answer: 
                """
    elif task_name in ['rte', 'wnli']:
        return f"""
                    Instruction: Does Sentence1 imply Sentence2? Please answer yes or no.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name in ['mnli', 'mnli-mm']:
        return f"""
                    Instruction: Does Sentence1 imply Sentence2? Please answer yes, no or maybe.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name == 'mrpc':
        return f"""
                    Instruction: Is Sentence1 equivalent to Sentence2? Please answer yes or no.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name == "qnli":
        return f"""
                    Instruction: Given a question and a sentence, does the sentence contain the answer to the question? Please answer yes or no.
                    Input: Question: {input1}; Sentence: {input2}
                    Answer:
                """
    elif task_name == "qqp":
        return f"""
                    Instruction: Are Question 1 and Question 2 semantically equivalent? Please answer yes or no.
                    Input: Question1: {input1}; Question2: {input2}
                    Answer:
                """
    else:
        return None

def preprocess_function(examples, task_name, tokenizer, max_length):
    assert task_name in GLUE_key_double.keys() \
    or task_name in GLUE_key_single.keys()
    label_name = None
    if task_name in GLUE_key_double.keys():
        model_inputs = tokenizer(
            [
                prompt_text(input1, task_name, input2) for input1, input2 in 
                zip(
                    examples[GLUE_key_double[task_name]['input'][0]], 
                    examples[GLUE_key_double[task_name]['input'][1]],
                )
            ],
            truncation=True, 
            max_length=max_length,
            padding='max_length',
        )
        label_name = GLUE_key_double[task_name]['label']
    elif task_name in GLUE_key_single.keys():
        model_inputs = tokenizer(
            [
                prompt_text(input1, task_name) for input1 in examples[GLUE_key_single[task_name]['input']]
            ],
            truncation=True, 
            max_length=max_length,
            padding='max_length',
        )
        label_name = GLUE_key_single[task_name]['label']
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [f"{label2text[task_name][ex]}" for ex in examples[label_name]],
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
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = labels[:, 0]
    preds = preds[:, 1]
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='weighted') 
    MCC = matthews_corrcoef(labels, preds)
    return {
        "accuracy": accuracy,
        "f1-score": f1,
        "MCC": MCC,
    }

def main():
    # 0. Get config
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        '--config',
        type=str,
        default="config/GLUE-t5-lora-train-config/MNLI-lora-train.yaml",
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
    validation_key = "validation_mismatched" if task_name == "mnli-mm" else "validation_matched" if task_name == "mnli" else "validation"
    task_actual = "mnli" if task_name == "mnli-mm" else task_name
    metric = load_metric('glue', task_actual)
    dataset = load_dataset("glue", task_actual)
    print(dataset)
    dataset = DatasetDict({
        'train': dataset['train'], 
        validation_key: dataset[validation_key],
    })
    print(metric)
    dataset[validation_key] = dataset[validation_key].shuffle(seed=seed)
    dataset['train'] = dataset['train'].shuffle(seed=seed)

    # Load pre-train model and the checkpoints    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(
            examples, 
            task_name=task_name, 
            tokenizer=tokenizer,
            max_length=256,
        ),
        batched=True,
        batch_size=1024,
    )

    features_to_keep = ['input_ids', 'attention_mask', 'labels']
    for key in tokenized_datasets.keys():
        tokenized_datasets[key] = tokenized_datasets[key].remove_columns(
            [col for col in tokenized_datasets[key].column_names if col not in features_to_keep]
        )
    
    # Apply lora to the model
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", max_length=256)
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[validation_key],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()

