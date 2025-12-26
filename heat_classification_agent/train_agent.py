import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from llm_guide import LLMGuidance
from grounding_dino_agent import GroundingDINOAgent
from heatmap_utils import generate_gaussian_heatmap
from dynamic_bpaco import DynamicBPaCoModel, compute_contrastive_loss

class HeatClassificationAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize modules
        self.llm = LLMGuidance()
        self.detector = GroundingDINOAgent(device=self.device)
        self.model = DynamicBPaCoModel(num_classes=config['num_classes']).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.lambda_start = config['lambda_start']
        self.lambda_max = config['lambda_max']
        self.epochs = config['epochs']

    def get_lambda(self, epoch):
        """Dynamic lambda scheduling: increases with epoch."""
        return min(self.lambda_max, self.lambda_start + epoch * (self.lambda_max - self.lambda_start) / self.epochs)

    def prepare_heatmaps(self, dataset_name, class_names):
        """
        Preprocessing Step: Generate heatmaps for the entire dataset using LLM + Grounding DINO.
        """
        print(f"Preprocessing Step 1 & 2 for {dataset_name}...")
        dataset_prompts = self.llm.generate_dino_prompts(dataset_name)
        # Use prompts to detect features and generate heatmaps
        
        # In a real scenario, we would iterate through the dataset and save heatmaps
        # For now, we assume heatmaps are generated on-the-fly or cached.
        pass

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        lambda_val = self.get_lambda(epoch)
        total_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            target_heatmaps = batch['heatmap'].to(self.device) # From Grounding DINO
            
            # Step 4.1: Forward with dynamic fusion
            features, logits, z = self.model(images, target_heatmaps, lambda_val=lambda_val)
            
            # Step 4.2: Loss Computation
            # Classification Loss
            loss_cls = F.cross_entropy(logits, labels)
            
            # Contrastive Loss (Step 4.4: Inter-class penalty inside)
            loss_con = compute_contrastive_loss(z, labels, self.model.prototypes, inter_class_penalty=True)
            
            # Grad-CAM Guidance (Step 4.2)
            loss_cam = self.model.get_gradcam_loss(target_heatmaps)
            
            loss = loss_cls + config['beta'] * loss_con + config['gamma'] * loss_cam
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | Lambda: {lambda_val:.3f} | Avg Loss: {total_loss/len(train_loader):.4f}")

    def run(self, train_loader, val_loader):
        print("Starting Heat Classification Agent Training...")
        for epoch in range(self.epochs):
            self.train_epoch(epoch, train_loader)
            # Validation logic...
            
if __name__ == "__main__":
    config = {
        'num_classes': 3, # Example
        'lr': 0.005,      # Optimal LR found in tuning
        'epochs': 50,
        'lambda_start': 0.1,
        'lambda_max': 0.8,
        'beta': 2.0,      # Optimal Beta found in tuning
        'gamma': 0.5      # Weight for Grad-CAM guidance
    }
    
    # Initialize Agent
    # agent = HeatClassificationAgent(config)
    # agent.run(train_loader, val_loader)
    print("Agent initialized with config:", config)
