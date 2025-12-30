import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import open_clip
from torchvision import datasets, transforms
import numpy as np
import random
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_prompts(json_path, dataset_name):
    # For CoOp, we typically just need class names, not full sentences.
    # CoOp learns the context.
    # We can extract class names from the folder structure.
    pass

class PromptLearner(nn.Module):
    def __init__(self, classnames, model, tokenizer, n_ctx=16, ctx_init="", device="cpu"):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.device = device
        
        # Access embedding layer of BioMedCLIP's text encoder (PubMedBERT)
        # BioMedCLIP uses: model.text (HF model)
        self.text_encoder = model.text
        try:
            # HF BERT structure: embeddings.word_embeddings
            # OpenCLIP HFTextEncoder structure: self.transformer holds the HF model
            if hasattr(self.text_encoder, 'transformer'):
                self.embeddings = self.text_encoder.transformer.embeddings.word_embeddings
            else:
                self.embeddings = self.text_encoder.embeddings.word_embeddings
                
            self.hidden_size = self.embeddings.embedding_dim
        except AttributeError:
             print(f"Available attributes in text_encoder: {dir(self.text_encoder)}")
             raise AttributeError("Could not access model.text.transformer.embeddings.word_embeddings. Verify model structure.")

        ctx_dim = self.hidden_size
        
        if ctx_init:
            print(f"Initializing context with: {ctx_init}")
            ctx_init = ctx_init.replace("_", " ")
            # Tokenize without special tokens
            ids = tokenizer(ctx_init, add_special_tokens=False)['input_ids']
            n_ctx = len(ids) # Overwrite n_ctx
            self.n_ctx = n_ctx
            with torch.no_grad():
                ctx_vectors = self.embeddings(torch.tensor(ids, device=device)).clone()
        else:
            print("Initializing generic context (random)")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors) # [n_ctx, dim]
        
        # Prepare Class Names Tokens
        self.class_name_tokens = []
        self.class_token_lens = []
        
        for name in classnames:
            name = name.replace("_", " ")
            ids = tokenizer(name, add_special_tokens=False)['input_ids']
            self.class_name_tokens.append(torch.tensor(ids, dtype=torch.long, device=device))
            self.class_token_lens.append(len(ids))

        # Get [CLS] and [SEP] tokens
        # Check specific tokenizer behavior
        # Simple/Safe way: use tokenizer.cls_token_id
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is None: # Fallback
             cls_id = tokenizer(tokenizer.cls_token)['input_ids'][0]
        if sep_id is None:
             sep_id = tokenizer(tokenizer.sep_token)['input_ids'][0]
             
        self.register_buffer('cls_token', torch.tensor([cls_id], dtype=torch.long, device=device))
        self.register_buffer('sep_token', torch.tensor([sep_id], dtype=torch.long, device=device))

    def forward(self):
        ctx = self.ctx # [n_ctx, dim]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # [C, n_ctx, dim]
            
        prompts_embeds = []
        
        cls_embed = self.embeddings(self.cls_token).unsqueeze(0).expand(self.n_cls, -1, -1) # [C, 1, dim]
        sep_embed = self.embeddings(self.sep_token).unsqueeze(0).expand(self.n_cls, -1, -1) # [C, 1, dim]
        
        batch_embeds = []
        
        for i in range(self.n_cls):
            c_cls = cls_embed[i]
            c_ctx = ctx[i]
            
            # Embed class name
            class_ids = self.class_name_tokens[i]
            c_class = self.embeddings(class_ids) # [len, dim]
            
            c_sep = sep_embed[i]
            
            # Concat: [CLS] [CTX] [CLASS] [SEP]
            combined = torch.cat([c_cls, c_ctx, c_class, c_sep], dim=0) # [L, dim]
            batch_embeds.append(combined)
        
        # Pad
        max_len = max([t.size(0) for t in batch_embeds])
        out_embeds = torch.zeros(self.n_cls, max_len, self.hidden_size, device=self.device)
        out_mask = torch.zeros(self.n_cls, max_len, dtype=torch.long, device=self.device)
        
        for i, embed in enumerate(batch_embeds):
            curr_len = embed.size(0)
            out_embeds[i, :curr_len, :] = embed
            out_mask[i, :curr_len] = 1
            
        return out_embeds, out_mask

class CustomCLIP(nn.Module):
    def __init__(self, classnames, model, tokenizer, n_ctx=16, ctx_init="", device="cpu"):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, model, tokenizer, n_ctx, ctx_init, device=device)
        self.image_encoder = model.visual
        self.text_encoder = model.text
        self.logit_scale = model.logit_scale
        
        # Handle Text Projection
        self.text_projection = None
        
        # 1. Check top-level model
        if hasattr(model, 'text_projection'):
            self.text_projection = model.text_projection
        # 2. Check text_encoder (HFTextEncoder wrapper) for 'proj'
        elif hasattr(self.text_encoder, 'proj'):
            self.text_projection = self.text_encoder.proj
        # 3. Check text_encoder for 'text_projection'
        elif hasattr(self.text_encoder, 'text_projection'):
            self.text_projection = self.text_encoder.text_projection
            
        if self.text_projection is not None:
             if hasattr(self.text_projection, 'shape'):
                 print(f"Found text_projection with shape: {self.text_projection.shape}")
             else:
                 print(f"Found text_projection of type: {type(self.text_projection)}")
        else:
            print("WARNING: text_projection NOT found. Available attributes in text_encoder:")
            print([a for a in dir(self.text_encoder) if not a.startswith('_')])
            
    def forward(self, image):
        # Image Features
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Text Features
        prompts_embeds, prompts_mask = self.prompt_learner() # [C, L, D] (embeds), [C, L] (mask)
        
        # Pass raw embeddings to HF BERT
        # We need to bypass OpenCLIP's wrapper and call the underlying transformer
        if hasattr(self.text_encoder, 'transformer'):
            transformer_model = self.text_encoder.transformer
        else:
            transformer_model = self.text_encoder
            
        # BERT forward accepts inputs_embeds
        text_outputs = transformer_model(inputs_embeds=prompts_embeds, attention_mask=prompts_mask)
        
        # Pooler output (CLS token)
        # BERT output has pooler_output at index 1 usually, or attribute pooler_output
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
             text_features = text_outputs.pooler_output
        else:
             # Fallback: take [CLS] token from last_hidden_state
             # last_hidden_state: [Batch, Seq, Hidden]
             text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Project text features if needed (OpenCLIP wrapper handles this usually)
        if self.text_projection is not None:
             # If projection is a module (e.g. Sequential, Linear), call it
             if isinstance(self.text_projection, nn.Module):
                 text_features = self.text_projection(text_features)
             else:
                 # Assume it's a tensor/parameter
                 # Ensure correct device/dtype
                 if self.text_projection.device != text_features.device:
                      self.text_projection = self.text_projection.to(text_features.device)
                 text_features = text_features @ self.text_projection
        else:
             print(f"Skipping projection. Text feats: {text_features.shape}")
             
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Sim
        logit_scale = self.logit_scale.exp()
        # Verify shapes
        if image_features.shape[1] != text_features.shape[1]:
            raise RuntimeError(f"Dimension mismatch: Image {image_features.shape}, Text {text_features.shape}")
            
        logits = logit_scale * image_features @ text_features.t()
        
        return logits

def get_few_shot_idx(dataset, shots):
    # Select 'shots' samples per class
    indices = []
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    
    for c in classes:
        c_idx = np.where(targets == c)[0]
        if len(c_idx) < shots:
             # Take all if less than shots
             indices.extend(c_idx)
        else:
             indices.extend(np.random.choice(c_idx, shots, replace=False))
             
    return indices

def load_class_names(json_path, dataset_name):
    # Load semantic class names from DPE prompts file
    if not os.path.exists(json_path):
        print(f"Prompts file {json_path} not found. Using default class names.")
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if dataset_name not in data:
        print(f"Dataset {dataset_name} not found in prompts file. Using default.")
        return None
        
    class_map = data[dataset_name]
    # Keys are strings "0", "1", etc.
    sorted_keys = sorted(class_map.keys(), key=lambda x: int(x))
    
    # Extract first prompt as class name
    class_names = []
    for k in sorted_keys:
        prompts = class_map[k]
        # Prompts are like "a photo of a normal fundus".
        # We can use the full sentence or try to extract the class name.
        # CoOp adds context [CTX] so "a photo of a" might be redundant if we want CoOp to learn it.
        # But user suggested "Unified initialization".
        # Let's use the full prompt string provided in JSON as the "Class Name" token sequence.
        # It's better than "class_0".
        if isinstance(prompts, list):
            class_names.append(prompts[0])
        else:
            class_names.append(str(prompts))
            
    return class_names

def main():
    args = get_arguments()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Device: {device}")
    
    # 1. Load BioMedCLIP
    print(f"Loading BioMedCLIP: {args.model}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    
    # Use standard HF Tokenizer for flexibility
    # Strip hf-hub: prefix if present
    hf_model_name = args.model.replace('hf-hub:', '')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    model.to(device)
    
    # 2. Dataset
    # Handle mapping
    dataset_mapping = {
        'finger': 'fingerprint',
        'oral_cancer': 'oral_cancer',
        'aptos': 'aptos',
        'mias': 'mias',
        'octa': 'octa'
    }
    
    ds_name = dataset_mapping.get(args.dataset, args.dataset)
    
    # Structure: dataset/train, dataset/test
    # Check for _classification_dataset suffix
    dataset_dir = os.path.join(args.data_root, f"{ds_name}_classification_dataset")
    if not os.path.exists(dataset_dir):
         # Try bare name
         dataset_dir = os.path.join(args.data_root, ds_name)
    
    if not os.path.exists(dataset_dir):
        # Last resort: try original args.dataset
        dataset_dir = os.path.join(args.data_root, f"{args.dataset}_classification_dataset")
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset dir not found for {args.dataset}. Checked {ds_name}_classification_dataset and others.")

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    if not os.path.exists(test_dir): test_dir = os.path.join(dataset_dir, 'val')
    
    print(f"Loading data from {dataset_dir}")
    full_train_dataset = datasets.ImageFolder(train_dir, transform=preprocess)
    test_dataset = datasets.ImageFolder(test_dir, transform=preprocess)
    
    # Load Semantic Class Names
    prompts_file = '../dpe/gpt3_prompts/prompts_simple.json'
    semantic_classes = load_class_names(prompts_file, args.dataset) # Use original arg name for json key
    
    if semantic_classes:
        print(f"Loaded semantic class names: {semantic_classes}")
        # Verify length matches
        if len(semantic_classes) != len(full_train_dataset.classes):
            print(f"Warning: Semantic classes {len(semantic_classes)} != Dataset classes {len(full_train_dataset.classes)}")
            print("Falling back to dataset folder names.")
            class_names = full_train_dataset.classes
        else:
            class_names = semantic_classes
    else:
        class_names = full_train_dataset.classes
    
    print(f"Classes: {class_names}")
    
    # Few-Shot Sampling
    if args.shots > 0:
        print(f"Sampling {args.shots} shots per class...")
        fs_indices = get_few_shot_idx(full_train_dataset, args.shots)
        train_dataset = Subset(full_train_dataset, fs_indices)
    else:
        train_dataset = full_train_dataset # Full training
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Model Wrap
    print("Building CustomCLIP (CoOp)...")
    coop_model = CustomCLIP(class_names, model, tokenizer, n_ctx=args.n_ctx, ctx_init=args.ctx_init, device=device)
    coop_model.to(device)
    
    # Freeze everything except prompt_learner
    for name, param in coop_model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, coop_model.parameters()), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4 # CoOp default
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    # 4. Training
    print(f"Start Training for {args.epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        coop_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = coop_model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        scheduler.step()
        
        # Evaluate? (Optional per epoch, or just final)
        # To save time, eval only every 10 epochs or last
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
             pass 
             
    # 5. Final Evaluation
    print("Evaluating...")
    coop_model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Zero-Shot Baseline? (User wants comparison usually)
    # But CoOp modifies model state (soft prompts). Baseline requires unmodified prompts.
    # We can't easily switch back without implementing logic. 
    # Just run CoOp eval.
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            logits = coop_model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    print(f"\nResults for {args.dataset}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    res = {
        'dataset': args.dataset,
        'acc': acc,
        'f1': f1,
        'auc': auc
    }
    with open(os.path.join(args.output_dir, f'results_{args.dataset}.json'), 'w') as f:
        json.dump(res, f, indent=4)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../datasets')
    parser.add_argument('--model', type=str, default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--output-dir', type=str, default='results_coop')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.002) 
    parser.add_argument('--n-ctx', type=int, default=16)
    parser.add_argument('--ctx-init', type=str, default="")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--shots', type=int, default=16)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
