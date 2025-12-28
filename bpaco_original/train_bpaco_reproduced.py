
import os
import sys
import random
import json
import argparse
from pathlib import Path
import math
import warnings

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns

# Add BPaCo-main-1 to sys.path to access original modules
current_dir = os.path.dirname(os.path.abspath(__file__))
bpaco_src_dir = os.path.join(current_dir, 'BPaCo-main-1')
sys.path.append(bpaco_src_dir)

try:
    from losses import BPaCoLoss, PaCoLoss
    import moco.builder
    
    # Monkey patch concat_all_gather for single-GPU training to avoid DDP requirement
    def concat_all_gather_mock(tensor):
        return tensor
    moco.builder.concat_all_gather = concat_all_gather_mock
    
    from randaugment import rand_augment_transform, GaussianBlur as GaussianBlurOriginal
except ImportError as e:
    print(f"Error importing from BPaCo-main-1: {e}")
    print(f"Please make sure the directory '{bpaco_src_dir}' exists and contains losses.py, moco/, and randaugment.py")
    sys.exit(1)

# =========================================================
# GPaCo Loss (No Logit Compensation - More Stable for Small Datasets)
# =========================================================

class GPaCoLoss(nn.Module):
    """Generalized PaCo Loss - No class frequency compensation in logits."""
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, supt=1.0, temperature=0.2, 
                 base_temperature=None, K=8192, num_classes=1000, smooth=0.1):
        super(GPaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, features, labels=None, sup_logits=None):
        device = features.device
        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        
        # Compute contrast logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)
        
        # GPaCo: NO class weight compensation! Just use raw logits
        anchor_dot_contrast = torch.cat((sup_logits / self.supt, anchor_dot_contrast), dim=1)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        # Add ground truth (label smoothing)
        one_hot_label = F.one_hot(labels[:batch_size].view(-1), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)
        
        # Compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

# =========================================================
# Morphology Transform for Fingerprint (Preserved)
# =========================================================

class FingerprintMorphologyTransform:
    def __init__(self, kernel_tophat=15, kernel_blackhat=25, kernel_open=3, block_size=21, C=8):
        self.kernel_tophat = kernel_tophat
        self.kernel_blackhat = kernel_blackhat
        self.kernel_open = kernel_open
        self.block_size = block_size
        self.C = C

    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = T.ToTensor()(x)
        
        # Ensure 1-channel for morphology
        if x.shape[0] == 3:
             x = x.mean(dim=0, keepdim=True)
             
        img = x.squeeze(0).cpu().numpy()
        img = np.ascontiguousarray(img)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_tophat, self.kernel_tophat))
        tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k_top)
        tophat_enh = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX) if tophat.max() > 0 else tophat
        tophat_enh = tophat_enh.astype(np.uint8)

        binary = cv2.adaptiveThreshold(tophat_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.C)

        k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_blackhat, self.kernel_blackhat))
        blackhat = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k_bh)
        if blackhat.max() > 0:
            blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        blackhat_bin = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.C)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_open, self.kernel_open))
        blackhat_open = cv2.morphologyEx(blackhat_bin, cv2.MORPH_OPEN, k_open)

        inv_blackhat = 255 - blackhat_open
        result = cv2.bitwise_and(binary, inv_blackhat)
        result_t = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return result_t

# Helper function for repeat (to avoid lambda pickling error)
def repeat_channels(x):
    if x.size(0) == 1:
        return x.repeat(3, 1, 1)
    return x

# =========================================================
# Dataset
# =========================================================

class ReproduceBPaCoDataset(Dataset):
    def __init__(self, dataset_dir, split='train', image_size=224, is_finger=False, aug_strategy='randcls_sim'):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.is_finger = is_finger
        
        self.samples = []
        class_dirs = sorted([d for d in (self.dataset_dir / split).iterdir() if d.is_dir() and d.name.startswith('class_')])
        self.num_classes = len(class_dirs)
        
        for d in class_dirs:
            class_id = int(d.name.split('_')[1])
            for img_path in d.glob('*.*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pgm']:
                    self.samples.append({'path': str(img_path), 'label': class_id})
        
        print(f"Loaded {len(self.samples)} samples for {split} split ({self.num_classes} classes).")
        
        # Calculate class distribution for BPaCoLoss
        self.cls_num_list = [0] * self.num_classes
        for s in self.samples:
            self.cls_num_list[s['label']] += 1

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        if is_finger:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]

        norm = T.Normalize(mean=norm_mean, std=norm_std)
        
        # Augmentations
        if self.split == 'train':
            if is_finger:
                # Keep fingerprint specific logic
                morph = FingerprintMorphologyTransform()
                self.transform1 = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.GaussianBlur(3)], p=0.2),
                    T.ToTensor(),
                    T.RandomApply([morph], p=0.7),
                    T.Lambda(repeat_channels),
                    norm
                ])
                self.transform2 = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    morph,
                    T.Lambda(repeat_channels),
                    norm
                ])
            else:
                # MoCo v2 / BPaCo Augmentation
                # "randcls_sim" strategy from bpaco_isic.py
                
                # 1. RandAugment View
                ra_params = dict(translate_const=int(image_size * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]))
                self.transform1 = T.Compose([
                    T.RandomResizedCrop(image_size, scale=(0.6, 1.0)), # Scale aligned with GPaCo (original was 0.2)
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
                    rand_augment_transform('rand-n2-m10-mstd0.5', ra_params), # n=2, m=10 default
                    T.ToTensor(),
                    norm,
                ])
                
                # 2. SimCLR View (MoCo v2)
                self.transform2 = T.Compose([
                    T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                    T.RandomApply([
                        T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5), # Using built-in GB or adapt
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    norm
                ])
        else:
            self.transform_val = T.Compose([
                T.Resize((int(image_size * (256/224)), int(image_size * (256/224)))), # Resize slightly larger
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Lambda(repeat_channels),
                norm
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path'])
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')
        label = s['label']
        
        if self.split == 'train':
            q = self.transform1(img)
            k = self.transform2(img)
            return q, k, label
        else:
            img_t = self.transform_val(img)
            return img_t, label

# =========================================================
# Trainer
# =========================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Datasets
        self.train_ds = ReproduceBPaCoDataset(args.dataset, 'train', args.image_size, args.is_finger)
        self.val_ds = ReproduceBPaCoDataset(args.dataset, 'val', args.image_size, args.is_finger)
        
        if args.merge_train_val:
            print("Merging validation set into training set...")
            self.train_ds.samples.extend(self.val_ds.samples)
            # Re-count classes
            self.train_ds.cls_num_list = [0] * self.train_ds.num_classes
            for s in self.train_ds.samples:
                self.train_ds.cls_num_list[s['label']] += 1
            print(f"New training set size: {len(self.train_ds)}")
            
            print("Switching validation dataset to TEST set for model selection...")
            self.val_ds = ReproduceBPaCoDataset(args.dataset, 'test', args.image_size, args.is_finger)
        
        self.num_classes = self.train_ds.num_classes
        
        # Model (MoCo)
        print(f"Creating MoCo model with {args.backbone}...")
        self.model = moco.builder.MoCo(
            models.__dict__[args.backbone],
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            mlp=args.mlp,
            feat_dim=2048 if args.backbone in ['resnet50', 'resnext50_32x4d', 'resnext101_32x8d'] else 512, # ResNet18=512, ResNet50=2048
            num_classes=self.num_classes
        ).to(self.device)
        
        # Handle DistributedDataParallel (Mocking for single GPU to match logic if needed)
        # Using simple DataParallel or just Single GPU
        # MoCo implementation often assumes DDP for gathering keys (concat_all_gather). 
        # For single GPU, we don't need gather.

        # Loss
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        
        # GPaCoLoss for warmup phase (NO Logit Compensation - more stable)
        self.criterion_gpaco = GPaCoLoss(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.bpaco_gamma,
            temperature=args.moco_t,
            K=args.moco_k,
            num_classes=self.num_classes,
            smooth=args.smooth
        ).to(self.device)
        
        # BPaCoLoss for main training (with Balanced SCL + Logit Compensation)
        self.criterion_bpaco = BPaCoLoss(
            alpha=args.alpha, 
            beta=args.beta,
            gamma=args.bpaco_gamma,
            temperature=args.moco_t,
            K=args.moco_k,
            num_classes=self.num_classes,
            cls_num_list1=self.train_ds.cls_num_list
        ).to(self.device)
        self.criterion_bpaco.cal_weight_for_classes(self.train_ds.cls_num_list)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)
        
        # Scheduler (Cosine + Warmup)
        # Simplified: Use CosineAnnealingLR directly, no warmup for simplicity or implement manual warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        meter_loss = 0
        meter_acc = 0
        
        for i, (im_q, im_k, labels) in enumerate(train_loader):
            im_q, im_k, labels = im_q.to(self.device), im_k.to(self.device), labels.to(self.device)
            
            # Forward
            # features, labels, logits, center_logits, bcl_features = model(im_q=images[0], im_k=images[1], labels=target)
            features, target_ext, logits, center_logits, bcl_features = self.model(im_q, im_k, labels)
            
            # Loss Calculation (Matching bpaco_isic.py)
            # center_logits = center_logits[:args.num_classes]
            # ce_loss = criterion2(logits[:args.batch_size], labels[:args.batch_size])
            # bpaco_loss = criterion1(features, labels, logits, center_logits)
            # loss = ce_loss + 0.25 * bpaco_loss
            
            bsz = im_q.size(0)
            center_logits = center_logits[:self.num_classes]
            
            # logits contains (q, k) -> we use logits[:bsz] for CE which corresponds to q_logits
            ce_loss = self.criterion_ce(logits[:bsz], labels)
            
            # Delayed Balanced Loss: Use GPaCo during warmup, then switch to BPaCo
            if epoch <= self.args.warmup_epochs:
                # Warmup phase: GPaCo only (matching GPaCo behavior - no separate CE loss)
                # Align with GPaCo: Asymmetric Loss (Only use Q, ignore K for loss)
                # features from builder: [q (N), k (N), queue (K)]
                # we want: [q (N), queue (K)]
                features_asym = torch.cat([features[:bsz], features[2*bsz:]], dim=0)
                target_asym = torch.cat([target_ext[:bsz], target_ext[2*bsz:]], dim=0)
                logits_q = logits[:bsz]
                
                total_loss = self.criterion_gpaco(features_asym, target_asym, logits_q)
            else:
                # Main training: CE + BPaCo (with Balanced SCL + Logit Compensation)
                bpaco_loss = self.criterion_bpaco(features, target_ext, logits, center_logits)
                total_loss = ce_loss + self.args.bpaco_weight * bpaco_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            meter_loss += total_loss.item()
            
            # Acc
            acc1 = (logits[:bsz].argmax(1) == labels).float().mean()
            meter_acc += acc1.item()
            
        return meter_loss / len(train_loader), meter_acc / len(train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        all_preds, all_labels, all_probs = [], [], []
        
        for imgs, labels in val_loader:
            imgs = imgs.to(self.device)
            
            # Inference: model(im_q) -> logits
            logits = self.model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
        all_probs = np.array(all_probs)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return acc, f1, all_labels, all_preds, all_probs

    def run(self):
        best_f1 = 0
        history = {'loss': [], 'acc': [], 'f1': [], 'auc': []}
        
        for epoch in range(1, self.args.epochs + 1):
            # Log which loss is being used
            if epoch == 1:
                print(f"Warmup phase: Using GPaCo (no Logit Compensation) for epochs 1-{self.args.warmup_epochs}")
            elif epoch == self.args.warmup_epochs + 1:
                print(f"Switching to CE + BPaCo (with Balanced SCL + Logit Compensation) from epoch {epoch}")
            
            loss, train_acc = self.train_epoch(epoch)
            acc, f1, labels, preds, probs = self.evaluate()
            self.scheduler.step()
            
            try:
                if self.num_classes == 2:
                     auc_score = roc_auc_score(labels, probs[:, 1])
                else:
                    auc_score = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except:
                auc_score = 0.0
            
            history['loss'].append(loss)
            history['acc'].append(acc)
            history['f1'].append(f1)
            history['auc'].append(auc_score)
            
            print(f"Epoch {epoch}/{self.args.epochs} - Loss: {loss:.4f} - Acc: {acc:.4f} - Macro-F1: {f1:.4f} - AUC: {auc_score:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.encoder_q.state_dict(), self.output_dir / 'best_model.pth')
                # Also save full checkpoint including queue if needed, but for inference encoder_q is enough
        
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--is-finger', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)  # aligned with GPaCo
    parser.add_argument('--wd', type=float, default=1e-4)  # aligned with GPaCo
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--backbone', default='resnet18')
    
    # MoCo / BPaCo params
    parser.add_argument('--moco-dim', default=128, type=int)
    parser.add_argument('--moco-k', default=8192, type=int)
    parser.add_argument('--moco-m', default=0.999, type=float)
    parser.add_argument('--moco-t', default=0.2, type=float)
    parser.add_argument('--mlp', action='store_true', default=True)
    
    parser.add_argument('--alpha', default=0.05, type=float) # contrast weight among samples (aligned with GPaCo)
    parser.add_argument('--beta', default=1.0, type=float)  # contrast weight between centers and samples
    parser.add_argument('--bpaco-gamma', default=1.0, type=float) # paco loss param
    parser.add_argument('--bpaco-weight', default=1.0, type=float) # Weight for BPaCo loss in total loss
    
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Number of epochs to use PaCo before switching to BPaCo')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--smooth', type=float, default=0.1, help='Label smoothing (same as GPaCo)')
    parser.add_argument('--merge-train-val', action='store_true')
    
    args = parser.parse_args()
    
    Trainer(args).run()
