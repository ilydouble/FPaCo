import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from llm_guide import LLMGuidance
# from grounding_dino_agent import GroundingDINOAgent
from florence2_agent import Florence2Agent
from heatmap_utils import generate_gaussian_heatmap
from dynamic_bpaco import DynamicBPaCoModel, compute_contrastive_loss
import torch.nn.functional as F

class HeatmapAgentDataset(Dataset):
    def __init__(self, dataset_dir, agent, image_size=224, prompt_cache=None):
        self.dataset_dir = Path(dataset_dir)
        self.agent = agent # Florence2Agent instance
        self.image_size = image_size
        self.prompt_cache = prompt_cache if prompt_cache else {}
        self.split = 'train'
        
        self.samples = []
        # Support Standard BPaCo dataset structure: split/class_x/img.png
        split_dir = self.dataset_dir / self.split
        if not split_dir.exists():
            # Try finding classes directly if split folder doesn't exist
            split_dir = self.dataset_dir
            
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        
        for d in class_dirs:
            class_idx = self.class_to_idx[d.name]
            for img_path in d.glob('*.*'):
                 if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                     self.samples.append((str(img_path), class_idx))
                     
        print(f"Loaded {len(self.samples)} samples from {self.dataset_dir}")
        
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(0) # Fallback
            
        # 1. Image Transform
        img_t = self.transform(pil_img)
        
        # 2. Agent Guidance (Generate Heatmap)
        # In real training, this should be pre-computed. For demo, we do it on-the-fly (slow!).
        # To speed up, we can use a fixed prompt per dataset or per class?
        # For generalization, let's use a generic prompt or from cache.
        prompt = self.prompt_cache.get('default', 'salient object')
        
        # Florence-2 Detection
        # Note: Doing this on-the-fly in __getitem__ is very slow because of model inference.
        # Ideally, Agent should be outside Dataloader or we pre-process.
        # For this script run, we will MOCK it if agent is not provided, 
        # OR we assume the agent object is lightweight (it is not).
        
        # CRITICAL: Passing huge model to Dataset and Multiprocessing Dataloader will crash/lag.
        # We will generate heatmap logic here but assuming single worker or pre-calc.
        # For now, let's assume num_workers=0.
        
        with torch.no_grad():
             # Resize for detection? Agent handles it.
             # We pass the PIL image.
             boxes, _, _ = self.agent.detect(pil_img, prompt)
             
        # Generate Heatmap
        # Boxes are in original image coordinates
        h, w = pil_img.height, pil_img.width
        heatmap = generate_gaussian_heatmap(h, w, boxes.cpu().numpy(), sigma=30)
        
        # Resize heatmap to model input size
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        heatmap_t = F.interpolate(heatmap_t, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        heatmap_t = heatmap_t.squeeze(0) # [1, 224, 224]
        
        return {
            'image': img_t,
            'label': label,
            'heatmap': heatmap_t
        }

class HeatClassificationAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize modules
        self.llm = LLMGuidance()
        # Initialize Florence-2
        self.detector = Florence2Agent(device=self.device)
        
        # Model
        self.model = DynamicBPaCoModel(num_classes=config['num_classes']).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.lambda_start = config['lambda_start']
        self.lambda_max = config['lambda_max']
        self.epochs = config['epochs']

    def get_lambda(self, epoch):
        return min(self.lambda_max, self.lambda_start + epoch * (self.lambda_max - self.lambda_start) / self.epochs)

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        lambda_val = self.get_lambda(epoch)
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            target_heatmaps = batch['heatmap'].to(self.device)
            
            features, logits, z = self.model(images, target_heatmaps, lambda_val=lambda_val)
            
            loss_cls = F.cross_entropy(logits, labels)
            loss_con = compute_contrastive_loss(z, labels, self.model.prototypes if hasattr(self.model, 'prototypes') else None, inter_class_penalty=True)
            loss_cam = self.model.get_gradcam_loss(target_heatmaps)
            
            loss = loss_cls + self.config['beta'] * loss_con + self.config['gamma'] * loss_cam
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch} Finished | Avg Loss: {total_loss/len(train_loader):.4f}")

    def run(self, dataset_path):
        print(f"Starting Training on {dataset_path}...")
        
        # Dataset
        # Note: num_workers=0 is crucial because we call cuda model in __getitem__
        dataset = HeatmapAgentDataset(dataset_path, self.agent_wrapper(), image_size=224, prompt_cache={'default': 'lesion, anomaly'})
        self.model.num_classes = len(dataset.class_dirs) if hasattr(dataset, 'class_dirs') else 10 # adjust
        
        train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        
        for epoch in range(self.epochs):
            self.train_epoch(epoch, train_loader)
            
    def agent_wrapper(self):
        # Return the detector instance to be used by dataset
        return self.detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    
    args = parser.parse_args()
    
    config = {
        'num_classes': 10, # Will be updated by dataset
        'lr': args.lr,
        'epochs': args.epochs,
        'lambda_start': 0.1,
        'lambda_max': 0.8,
        'beta': 1.0, 
        'gamma': 0.5,
        'batch_size': args.batch_size
    }
    
    agent = HeatClassificationAgent(config)
    
    # Update num_classes dynamically
    # A bit hacky but works for demo script
    temp_ds = HeatmapAgentDataset(args.dataset, agent.detector)
    agent.model.classifier = nn.Sequential(
        nn.Linear(agent.model.classifier[0].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, len(temp_ds.class_to_idx))
    ).to(agent.device)
    config['num_classes'] = len(temp_ds.class_to_idx)
    
    print(f"Detected {config['num_classes']} classes.")
    
    train_loader = DataLoader(temp_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    for epoch in range(args.epochs):
        agent.train_epoch(epoch, train_loader)
