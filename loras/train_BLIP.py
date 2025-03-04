import sacrebleu
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from eval_senti import get_emotion
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
#dataset = load_dataset("ybelkada/football-dataset", split="train")
positive_prompt = "(delicious healthy excellent favorite beautiful great clean best nice good awesome happy tasty interesting amazing sunny relaxing clear handsome pretty smiling)"
negative_prompt = "(disgusting ugly broken bad damaged lonely miserable stupid annoying dirty dead comfy horrible dangerous sad dying weird terrible crappy evil silly)"
roman_prompt = "attractive emotional beautiful"
humor_prompt = "amusing silly comic witty"

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

class ImageCaptioningDataset_Senticap(Dataset):
    def __init__(self, dataset, processor, max_length, task_name="positive", datatype='train'):
        self.dataset = dataset[datatype]
        self.max_length=max_length
        self.processor = processor
        self.task_name = task_name
        if "style_fusion" in self.task_name:
            self.prefix_str = f"a photo of "
        else:
            self.prefix_str = f"a photo that expresses {self.task_name} sentiments{positive_prompt if self.task_name=='positive' else negative_prompt} of "
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"], 
            text=self.prefix_str+item["text"], 
            padding="max_length", 
            return_tensors="pt", 
            truncation=True,
            max_length=self.max_length,
        )
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

class ImageCaptioningDataset_Senticap_val(Dataset):
    def __init__(self, dataset, processor, max_length, task_name="positive", datatype='train'):
        self.dataset = dataset[datatype]
        self.max_length=max_length
        self.processor = processor
        self.task_name = task_name
        if "style_fusion" in self.task_name:
            self.prefix_str = f"a photo of "
        else:
            self.prefix_str = f"a photo that expresses {self.task_name} sentiments{positive_prompt if self.task_name=='positive' else negative_prompt} of "

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
        text_list = [self.prefix_str+text for text in json.loads(item["text"])]
        if len(text_list) < 3:
            text_list = text_list + [""] * (3-len(text_list))
        encoding["labels"] = self.processor(
                text=text_list,
                padding="max_length",
                return_tensors="pt", 
                max_length=self.max_length,
                truncation=True,
            ).input_ids
        return encoding



class ImageCaptioningDataset_FlickrStyle10k(Dataset):
    def __init__(self, dataset, processor, max_length, task_name="humor", datatype='train'):
        self.dataset = dataset[datatype]
        self.max_length=max_length
        self.processor = processor
        self.task_name = task_name
        if "style_fusion" in self.task_name:
            self.prefix_str = f"a photo of "
        elif self.task_name == "humor":
            self.prefix_str = f"a humorous funny {humor_prompt} photo of "
        elif self.task_name == "roman":
            self.prefix_str = f"a romantic warm {roman_prompt} photo of "

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"], 
            text=self.prefix_str+item["text"], 
            padding="max_length", 
            return_tensors="pt", 
            truncation=True,
            max_length=self.max_length
        )
        encoding["labels"] = encoding.input_ids
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding