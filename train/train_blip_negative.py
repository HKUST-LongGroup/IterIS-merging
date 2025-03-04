import sacrebleu
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from eval_senti import get_emotion
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
#dataset = load_dataset("ybelkada/football-dataset", split="train")
dataset = load_dataset("data/SENTICAP/negative")
dataset = DatasetDict({
    'train': dataset['train'],
    'test': dataset['test'],
})
negative_prompt = "(disgusting ugly broken bad damaged lonely miserable stupid annoying dirty dead comfy horrible dangerous sad dying weird terrible crappy evil silly)"

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


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, datatype='train'):
        self.dataset = dataset[datatype]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"], 
            text=f"a photo that expresses negative sentiments{negative_prompt} of "+item["text"], 
            padding="max_length", 
            return_tensors="pt", 
            max_length=168,
        )
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

class ImageCaptioningDataset_val(Dataset):
    def __init__(self, dataset, processor, datatype='test'):
        self.dataset = dataset[datatype]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"],
            return_tensors="pt", 
        )
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()} # pixel_values
        text_list = [f"a photo that expresses negative sentiments{negative_prompt} of "+text for text in json.loads(item["text"])]
        if len(text_list) < 3:
            text_list = text_list + [""] * (3-len(text_list))
        encoding["labels"] = self.processor(
                text=text_list,
                padding="max_length",
                return_tensors="pt", 
                max_length=168,
                truncation=True,
            ).input_ids
        return encoding
        
seed = 42
lr = 5e-5
num_epoch = 40
per_eval_num = 32
per_train_num = 16

set_seed(seed)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = ImageCaptioningDataset(dataset, processor, 'train')
val_dataset = ImageCaptioningDataset_val(dataset, processor, 'test')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=per_train_num)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=per_eval_num)
config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["attention.self.value", "attention.self.key", "attention.self.query"]
)
print(len(val_dataloader))

model = get_peft_model(model, config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.print_trainable_parameters()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    print(f"############################Epoch: {epoch}, Training...############################", flush=True)
    loss_list = []
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Processing", unit="batch"):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        attention_mask = batch.pop("attention_mask").to(device)
        outputs = model(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            labels=input_ids,
            attention_mask=attention_mask
        )
        loss = outputs.loss
        loss.backward()
        loss_list.append(loss)
        optimizer.step()
        optimizer.zero_grad()
        if idx % 20 == 0:
            print(f"LOSS: {sum(loss_list)/len(loss_list)}", flush=True)
        if idx % int(len(train_dataloader)/4) == int(len(train_dataloader)/4)-1:
            print(f"############################Epoch: {epoch}, eval...############################", flush=True)
            preds_list = {}
            labels_list = {}
            num = 0
            for _, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Processing", unit="val_batch"):
                pixel_values = val_batch.pop("pixel_values").to(device)
                labels = val_batch.pop("labels")
                val_inputs = processor(
                    text=[f"a photo that expresses negative sentiments{negative_prompt} of "]*len(pixel_values), 
                    return_tensors="pt", 
                ).to(device)
                out = model.generate(
                    **val_inputs,
                    pixel_values=pixel_values, 
                    max_new_tokens=168,
                )
                for item_out, label in zip(processor.batch_decode(out, skip_special_tokens=True), labels):
                    preds_list[num] = [get_content_after_word(item_out, ' of ')]
                    labels_list[num] = [
                        get_content_after_word(s, ' of ') for s in processor.batch_decode(
                            label, 
                            skip_special_tokens=True
                        ) if s.strip() != ""
                    ]
                    num += 1
                
            print("#############################Eval metric#############################", flush=True)
            print(" ", flush=True)
            cider_scorer = Cider()
            rouge_scorer = Rouge()
            bleu_scorer = Bleu(n=4)
            cider, _ = cider_scorer.compute_score(labels_list, preds_list)
            rouge, _ = rouge_scorer.compute_score(labels_list, preds_list)
            bleu, _ = bleu_scorer.compute_score(labels_list, preds_list)
            for i in range(10):
                print(preds_list[i])
            classification_tokenizer = AutoTokenizer.from_pretrained("atharvapawar/Bert-Sentiment-Classification-pos-or-neg")
            classification_model = AutoModelForSequenceClassification.from_pretrained("atharvapawar/Bert-Sentiment-Classification-pos-or-neg", num_labels=2)
            
            acc = None
            task_name = 'negative'
            if task_name in ['positive', 'negative']:
                preds_list = [pred[0] for _, pred in preds_list.items()]
                preds_emo = []
                for i in range(0, len(preds_list), 32):
                    subset = preds_list[i:i + 32]
                    preds_temp = classification_tokenizer(subset, padding=True, truncation=True, return_tensors='pt')
                    classification_model.eval()
                    with torch.no_grad():
                        outputs = classification_model(**preds_temp)
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                    preds_emo += list(predictions)
                print(preds_emo)
                acc = sum(preds_emo)/len(preds_emo)
                acc = acc if task_name == 'positive' else (1-acc)

            print(f"bleu: {bleu}", flush=True)
            print(f"acc: {acc}", flush=True)
            print(f"rouge: {rouge}", flush=True)
            print(f"cider: {cider}", flush=True)
            print(" ")
            saving_path = f"loras/SENTICAP-lora-blip/negative/checkpoint-{epoch*(len(train_dataloader)-1)+idx}"
            model.save_pretrained(saving_path, safetensors=True)
            processor.save_pretrained(saving_path)
            print(f"Saved model checkpoint to {saving_path}", flush=True)
            

