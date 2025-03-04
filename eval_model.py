import torch
import random
import nltk
import gc
import sacrebleu
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from loras.train_BLIP import ImageCaptioningDataset_Senticap_val, ImageCaptioningDataset_FlickrStyle10k
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from get_midfeatures import get_all_midfeatures
from get_midfeatures import get_samples
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
from loras.train_GLUE_t5 import compute_metrics as compute_metrics_t5
from loras.train_GLUE_t5 import preprocess_function as preprocess_function_t5
from loras.train_EMOTION_t5_large import preprocess_function as preprocess_function_t5_emotion
from eval_senti import get_emotion

seed = 42
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
GLUE_task_name = [
    "mnli", "mnli-mm", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc", "wnli",
]
positive_prompt = "(delicious healthy excellent favorite beautiful great clean best nice good awesome happy tasty interesting amazing sunny relaxing clear handsome pretty smiling)"
negative_prompt = "(disgusting ugly broken bad damaged lonely miserable stupid annoying dirty dead comfy horrible dangerous sad dying weird terrible crappy evil silly)"
roman_prompt = "attractive emotional beautiful"
humor_prompt = "amusing silly comic witty"
SENTICAP_task_name = ['positive', 'negative']
FlickrStyle10k_task_name = ["roman", "humor"]
TASKS_blip_base = ['positive', 'negative', "roman", "humor"]
EMOTION_task_name = [
        "crowdflower", "dailydialog", "emoint",
        "emotion-cause", "grounded_emotions", 
        "tales-emotion", "tec", "isear"
    ]
# Set all the seeds the same

def get_content_after_word(text, word):
    try:
        index = text.index(word)
        return text[index + len(word):].strip()
    except ValueError:
        return f"'{word}' not found in the text."

def set_seed(seed):
    """_summary_

    Args:
        seed (INT): Set the random seeds all the same 
    """
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's random generator (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch's random generator (GPU)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Make sure CuDNN uses deterministic algorithms
    torch.backends.cudnn.benchmark = False  # This ensures deterministic behavior

def calculate_div_n(captions, n):
    total_ngrams = 0
    unique_ngrams = set()
    
    for caption in captions:
        words = caption.split()
        ngrams = zip(*[words[i:] for i in range(n)])
        ngrams_list = list(ngrams)
        
        total_ngrams += len(ngrams_list)
        unique_ngrams.update(ngrams_list)
    
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def calculate_vocab(captions):
    vocab_set = set()
    for caption in captions:
        words = caption.split()  
        vocab_set.update(words)  
    return len(vocab_set)

def evaluate_bleu(model, tokenizer, tokenized_dataset, max_length):
    model.eval()
    decoded_preds = []
    decoded_labels = []
    with torch.no_grad():
        for batch in tqdm(tokenized_dataset, desc="Evaluating"):
            print(batch['input_ids'])
            print(batch['labels'])
            inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device) 
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
            labels = [torch.where(label != -100, label, 0) for label in labels]
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0) 
            outputs = model.generate(input_ids=inputs).to('cpu')
            decoded_preds += tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels += tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return bleu.score

def get_eval_data(
    model_name,
    task_name,
    tokenizer,
    max_length,
):
    dataset, tokenized_datasets = None, None
    if task_name in GLUE_task_name:
        task_actual = "mnli" if task_name == "mnli-mm" else task_name
        dataset = load_dataset("glue", task_actual)
        validation_key = "validation_mismatched" if task_name == "mnli-mm" else "validation_matched" if task_name == "mnli" else "validation"
        dataset = DatasetDict({
            'validation': dataset[validation_key], #.select(range(min(len(dataset[validation_key]), 2000))),
        })
        if 't5' in model_name or 'bart' in model_name:
            tokenized_datasets = dataset.map(
                lambda examples: preprocess_function_t5(
                    examples, 
                    task_name=task_name, 
                    tokenizer=tokenizer,
                    max_length=max_length,
                ),
                batched=True,
                batch_size=1024,
            )
    elif task_name in EMOTION_task_name:
        dataset = load_dataset('json', data_files=f"data/Emotion_v2.5/{task_name}/test.json")
        dataset = DatasetDict({
            'validation': dataset['train'], 
        })
        # tokenizer.pad_token = tokenizer.eos_token
        tokenized_datasets = dataset.map(
            lambda examples: preprocess_function_t5_emotion(
                examples, 
                task_name=task_name, 
                tokenizer=tokenizer,
                max_length=max_length,
            ), 
            batched=True,
            batch_size=1024,
            load_from_cache_file=False,
        )
    elif task_name in TASKS_blip_base:
        assert  'blip' in model_name
        dataset = load_dataset(f"data/SENTICAP/{task_name}" if task_name in SENTICAP_task_name else f"data/FlickrStyle10k/{task_name}")
        dataset = DatasetDict({
            'test': dataset['test'],
        })
        if task_name in SENTICAP_task_name:
            tokenized_datasets = ImageCaptioningDataset_Senticap_val(
                max_length=max_length,
                dataset=dataset, 
                processor=tokenizer, 
                task_name=task_name,
                datatype='test',
            )
        else:
            tokenized_datasets = ImageCaptioningDataset_FlickrStyle10k(
                max_length=max_length,
                dataset=dataset, 
                processor=tokenizer, 
                task_name=task_name,
                datatype='test',
            )

    return tokenized_datasets

def blip_eval(eval_dataloader, model, task_name, processor):
    preds_list = {}
    labels_list = {}
    num = 0
    if task_name in SENTICAP_task_name:
        generation_kwargs = {
            'max_new_tokens': 42,
            'no_repeat_ngram_size': 3, 
            'repetition_penalty': 1.2, 
            'num_beams': 4, 
            'num_beam_groups': 2,
            'top_k': 50, 
            'diversity_penalty': 0.8,
            'length_penalty': 1.0
        }
    elif task_name in FlickrStyle10k_task_name:
        generation_kwargs = {
            'max_new_tokens': 42,
            'no_repeat_ngram_size': 3, 
            'repetition_penalty': 1.2, 
            'num_beams': 4, 
            'num_beam_groups': 2,
            'top_k': 50, 
            'diversity_penalty': 0.8,
            'length_penalty': 1.0
        }
    for _, val_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Processing", unit="val_batch"):
        pixel_values = val_batch.pop("pixel_values").to(model.device)
        labels = val_batch.pop("labels")
        if task_name in SENTICAP_task_name:
            val_inputs = processor(
                text=[f"a photo that expresses {task_name} sentiments{positive_prompt if task_name=='positive' else negative_prompt} of "]*len(pixel_values), 
                return_tensors="pt", 
            ).to(model.device)
        elif task_name in FlickrStyle10k_task_name:
            text_temp = f"a romantic warm {roman_prompt} photo of " if task_name == "roman" else f"a humorous funny {humor_prompt} photo of "
            val_inputs = processor(
                text=[text_temp]*len(pixel_values), 
                return_tensors="pt", 
            ).to(model.device)
        out = model.generate(
            **val_inputs,
            pixel_values=pixel_values,
            **generation_kwargs,
        )
        for item_out, label in zip(processor.batch_decode(out, skip_special_tokens=True), labels):
            preds_list[num] = [get_content_after_word(item_out, ' of ')]
            if task_name in SENTICAP_task_name:
                labels_list[num] = [
                    get_content_after_word(s, ' of ') for s in processor.batch_decode(
                        label, 
                        skip_special_tokens=True
                    ) if s.strip() != ""
                ]
            else:
                labels_list[num] = [get_content_after_word(processor.decode(label, skip_special_tokens=True) , ' of ')]
            num += 1
    cider_scorer = Cider()
    rouge_scorer = Rouge()
    bleu_scorer = Bleu(n=4)
    cider, _ = cider_scorer.compute_score(labels_list, preds_list)
    rouge, _ = rouge_scorer.compute_score(labels_list, preds_list)
    bleu, _ = bleu_scorer.compute_score(labels_list, preds_list)
    acc = sum([get_emotion(pred[0], task_name) for _, pred in preds_list.items()])/len(preds_list) if task_name in SENTICAP_task_name else None
    preds_list = [pred[0] for _, pred in preds_list.items()]
    labels_list = [label[0] for _, label in labels_list.items()]
    div_1 = calculate_div_n(preds_list, 1)
    div_2 = calculate_div_n(preds_list, 2)
    v_size = calculate_vocab(preds_list)
    return {
        "bleu": bleu,
        "acc": acc,
        "div_1": div_1,
        "div_2": div_2,
        "vocab_size": v_size,
        "rougeL": rouge,
        "cider": cider,
    }

def eval_iteris_model(
    model, 
    model_name,
    task_name,
    tokenizer, 
    max_length=512, 
    per_device_eval_batch_size=4,
):
    gc.collect()
    eval_results = None
    tokenized_datasets = get_eval_data(
        model_name=model_name,
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if 't5' in model_name or ('bart' in model_name and task_name in GLUE_task_name):
        features_to_keep = ['input_ids', 'attention_mask', 'labels']
        for key in tokenized_datasets.keys():
            tokenized_datasets[key] = tokenized_datasets[key].remove_columns(
                [col for col in tokenized_datasets[key].column_names if col not in features_to_keep]
            )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model) # Attention please
        training_args = Seq2SeqTrainingArguments(
            per_device_eval_batch_size=per_device_eval_batch_size,
            do_eval=True,
            output_dir="loras",
            label_names=['labels'],
            eval_accumulation_steps=2,
            predict_with_generate=True,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_t5 if 't5' in model_name else compute_metrics_bart,
        )
        eval_results = trainer.evaluate()
    elif "blip" in model_name:
        eval_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=per_device_eval_batch_size)
        eval_results = blip_eval(
            eval_dataloader=eval_dataloader,
            model=model,
            task_name=task_name,
            processor=tokenizer,
        )
    gc.collect()
    print(f"------------{task_name}, {model_name} Eval results------------")
    print(eval_results, flush=True)