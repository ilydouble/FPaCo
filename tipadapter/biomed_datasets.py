import os
import random
import json
from collections import defaultdict
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Configuration for datasets
DATASET_CONFIGS = {
    'oral_cancer': {
        'classes': ['Normal', 'Oral Cancer'],
        'folder_name': 'oral_cancer_classification_dataset'
    },
    'aptos': {
        'classes': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
        'folder_name': 'aptos_classification_dataset'
    },
    'finger': {
        'classes': [
            'Mountain', 'Fire Earth', 'Earth', 'Wind', 'Ground', 'Dream', 
            'Fire', 'Fire Light', 'Water', 'Electricity', 'Drill', 'Light', 
            'Electricity with Wind', 'Rock', 'Stone', 'Fire Wood', 'Wood', 'Gold'
        ],
        'folder_name': 'fingerA'
    },
    'mias': {
        'classes': ['NORM', 'CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM'],
        'folder_name': 'mias_classification_dataset'
    },
    'octa': {
        'classes': ['AMD', 'CNV', 'CSC', 'DR', 'Normal', 'Others', 'RVO'],
        'folder_name': 'octa_classification_dataset'
    }
}

def load_prompts_from_json(json_path, dataset_name):
    """Load prompts from the unified JSON file."""
    if not os.path.exists(json_path):
        # Allow fallback relative to parent directory if script is nested
        fallback = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), json_path)
        if os.path.exists(fallback):
             json_path = fallback
        else:
             raise FileNotFoundError(f"Prompts file not found: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if dataset_name not in data:
        raise ValueError(f"Dataset {dataset_name} not found in {json_path}")
        
    class_prompts_map = data[dataset_name]
    # Keys are strings "0", "1", etc. Sort them numerically to ensure order
    sorted_keys = sorted(class_prompts_map.keys(), key=lambda x: int(x))
    
    prompts_list = []
    for k in sorted_keys:
        p_list = class_prompts_map[k]
        if isinstance(p_list, list):
            prompts_list.append(p_list) 
        else:
             prompts_list.append([str(p_list)])
             
    return prompts_list

from PIL import Image

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class BiomedDataset:
    def __init__(self, dataset_name, root, num_shots, preprocess, prompts_file='prompts/unified_prompts.json', combine_train_val=False):
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS.get(dataset_name)
        if not self.config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        self.root = root
        # Check if root points to specific dataset or needs appending
        full_path = os.path.join(root, self.config['folder_name'])
        if os.path.exists(full_path):
            self.dataset_dir = full_path
        elif os.path.exists(root) and 'train' in os.listdir(root):
             self.dataset_dir = root
        else:
             self.dataset_dir = full_path 

        self.image_dir = self.dataset_dir
        
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        
        self.train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        self.test_preprocess = preprocess

        # Load Datasets
        train_dir = os.path.join(self.image_dir, 'train')
        val_dir = os.path.join(self.image_dir, 'val')
        test_dir = os.path.join(self.image_dir, 'test')
        
        if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
            val_dir = test_dir # Use test as val if val missing
            
        self.train = datasets.ImageFolder(train_dir, transform=self.train_preprocess)
        
        # Robust loading:
        try:
             self.val = datasets.ImageFolder(val_dir, transform=self.test_preprocess)
        except Exception as e:
             # print(f"Val set error: {e}. Fallback to Test set.")
             self.val = datasets.ImageFolder(test_dir, transform=self.test_preprocess)
             
        self.test = datasets.ImageFolder(test_dir, transform=self.test_preprocess)

        if combine_train_val:
            print("Combining Train and Val sets for training...")
            self.train = torch.utils.data.ConcatDataset([self.train, self.val])
            self.val = self.test # Use test as val since val is merged

        
        # Important: Verify class order matches config['classes']
        # ImageFolder classes are sorted. 
        # If dataset folders are 'class_0', 'class_1', they sort correctly.
        # If they are semantic names, they sort alphabetically.
        # Our config['classes'] is essentially the list of semantic names.
        # We need to ensure self.train.classes (folder names) map to these semantic names correctly.
        # For 'finger', 'oral_cancer', folders are 'class_0', 'class_1' etc.
        # So we just need config['classes'] to be in order 0, 1, 2... labels.
        # My config above uses this assumption.
        
        # Use full prompts (e.g., "fingerprint pattern of type Stone") instead of raw class names ("Stone")
        # This aligns with BioMedCLIP Zero-Shot baseline performance.
        try:
             self.classnames = load_prompts_from_json(prompts_file, dataset_name)
             print(f"Loaded {len(self.classnames)} prompts for {dataset_name}")
        except Exception as e:
             print(f"Error loading prompts: {e}")
             self.classnames = [] # Validate later
             
        # Template: Tip-Adapter uses standard prompts usually "a photo of a {}". 
        # But we actully defined full sentence prompts in config. 
        # So our template should be just "{}".
        self.template = ["{}"] 

        # Few-shot sampling
        self.num_shots = num_shots
        if self.num_shots > 0:
            self._create_few_shot_subset()
            
        # Expose train_x (few-shot train set)
        self.train_x = self.train

    def _create_few_shot_subset(self):
        # 1. Collect all samples
        all_samples = []
        if isinstance(self.train, torch.utils.data.ConcatDataset):
            for d in self.train.datasets:
                if hasattr(d, 'imgs'):
                    all_samples.extend(d.imgs)
                # If nested ConcatDataset, this might fail, but one level is expected here.
        elif hasattr(self.train, 'imgs'):
            all_samples = self.train.imgs
            
        if not all_samples:
             print("Warning: No samples found for few-shot creation.")
             return

        split_by_label_dict = defaultdict(list)
        for i in range(len(all_samples)):
            path, label = all_samples[i]
            split_by_label_dict[label].append((path, label))
            
        imgs = []
        targets = []
        
        for label, items in split_by_label_dict.items():
            # If not enough samples, take all or repeat? 
            # Tip-Adapter often assumes enough samples. 
            # If < num_shots, take all with replacement or just all.
            # Let's take minimum(len(items), num_shots) to avoid error, 
            # but ideally we want exactly num_shots.
            
            k = min(len(items), self.num_shots)
            selected = random.sample(items, k)
            
            # If we strictly need num_shots, we might need to oversample if k < num_shots
            # But let's stick to simple sampling.
            
            imgs.extend(selected)
            targets.extend([label] * k)
            
        # Create new dataset inplace
        self.train = SimpleDataset(imgs, self.train_preprocess)
