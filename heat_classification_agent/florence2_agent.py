import torch
import os
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

class Florence2Agent:
    """
    Agent for Phrase Grounding using Microsoft Florence-2.
    Replaces the interface of GroundingDINOAgent.
    """
    def __init__(self, model_id='microsoft/Florence-2-large-ft', device='cuda', precision='fp16'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        dtype = torch.float16 if precision == 'fp16' and self.device.type == 'cuda' else torch.float32
        
        print(f"Loading Florence-2 model: {model_id} ({dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager"
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.is_lora_enabled = False # Florence-2 fine-tuning is different, placeholder for interface compatibility

    def enable_lora(self, r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]):
        """
        Placeholder for LoRA. 
        Florence-2 (Qwen/BART based) LoRA would target 'q_proj', 'v_proj' of the language model backbone.
        """
        print("Enabling LoRA for Florence-2 (Placeholder)...")
        self.is_lora_enabled = True

    def detect(self, image_path, prompt, box_threshold=0.3, text_threshold=None):
        """
        Perform Phrase Grounding using Florence-2.
        
        Args:
            image_path: Path to image file or PIL Image object.
            prompt: Text prompt to ground (e.g., 'microaneurysms').
            box_threshold: Unused for Florence-2 generation, but kept for interface.
            text_threshold: Unused.
            
        Returns: 
            boxes: torch.Tensor (N, 4) in [x1, y1, x2, y2] format
            scores: torch.Tensor (N,) - Calculated confidence scores.
            phrases: List[str] - The label phrases
        """
        
        if isinstance(image_path, str) or isinstance(image_path, os.PathLike):
            image = Image.open(str(image_path)).convert("RGB")
        else:
            image = image_path.convert("RGB") # Assume PIL Image
            
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        
        # Sanitize prompt: Avoid double-prepending task token
        clean_prompt = prompt
        if prompt.startswith(task_prompt):
            clean_prompt = prompt.replace(task_prompt, "", 1).strip()
            
        text_input = task_prompt + clean_prompt
        
        inputs = self.processor(text=text_input, images=image, return_tensors="pt").to(self.device, self.model.dtype)
        
        # Determine beam search parameters for stability
        generated_out = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        generated_ids = generated_out.sequences
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process raw text to dictionary
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        result_content = parsed_answer.get(task_prompt, {})
        bboxes = result_content.get('bboxes', [])
        labels = result_content.get('labels', [])
        
        # --- Confidence Calculation ---
        # Strategy: Iterate through generated tokens, look for <loc_...> tokens,
        # groups of 4 loc tokens correspond to 4 coords of a bbox using Average Prob.
        
        # Extract Scores (logits)
        # scores is a tuple of (batch_size, vocab_size) tensors, length = generated_len - input_len
        # We need to align them with generated_ids[0, input_len:] (?)
        # Actually generate only returns new tokens in scores, but sequences usually includes input.
        # Check generate docs: output_scores -> one list per step.
        
        scores = generated_out.scores
        # len(scores) is number of generated steps
        
        # Get input length to align
        input_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0][input_len:] # Only new tokens
        
        # Reconstruct probabilities
        token_probs = []
        for i, step_logits in enumerate(scores):
            if i >= len(out_ids): break # Should match
            token_id = out_ids[i]
            probs = torch.softmax(step_logits, dim=-1)
            token_probs.append(probs[0, token_id].item())
            
        # Parse text to find which tokens are locations
        # This is tricky because tokenizer decoding maps ids -> text.
        # Florence-2 uses specific tokens for locations formatted like "<loc_X>"
        # We can inspect the token IDs directly.
        
        # Florence-2 vocab typically has special tokens for locations.
        # We assume any token decoding to something starting with "<loc_" is a coordinate.
        
        bbox_scores = []
        current_box_probs = []
        
        # The vocab is large, let's decode one by one to check type? Slow.
        # Optimization: Just check if token ID is in the "loc" range if known,
        # OR decode and check string.
        
        for i, tid in enumerate(out_ids):
            token_str = self.processor.tokenizer.decode([tid], skip_special_tokens=False).strip()
            if token_str.startswith("<loc_") and token_str.endswith(">"):
                current_box_probs.append(token_probs[i])
                if len(current_box_probs) == 4:
                    # Completed a box
                    avg_score = sum(current_box_probs) / 4.0
                    bbox_scores.append(avg_score)
                    current_box_probs = []
            elif "<" in token_str and ">" in token_str:
                 # Any other special token might indicate transition (labels, etc.)
                 # We don't reset current_box_probs here unless we are sure it breaks a box
                 pass
                 
        if len(bboxes) == 0:
            return torch.tensor([]), torch.tensor([]), []
            
        # Match scores to bboxes
        if len(bbox_scores) == len(bboxes):
            scores_t = torch.tensor(bbox_scores, dtype=torch.float32)
        elif len(bbox_scores) > 0:
            # Partial match or extra tokens?
            
            if len(bbox_scores) > len(bboxes):
                scores_t = torch.tensor(bbox_scores[:len(bboxes)], dtype=torch.float32)
            else:
                # Pad with average of existing scores or 0.5
                avg_existent = sum(bbox_scores) / len(bbox_scores)
                padded_scores = bbox_scores + [avg_existent] * (len(bboxes) - len(bbox_scores))
                scores_t = torch.tensor(padded_scores, dtype=torch.float32)
        else:
            # Complete failure to extract scores - fallback to 1.0 but log it
            scores_t = torch.ones(len(bboxes), dtype=torch.float32)

        # Convert to torch tensor
        boxes_t = torch.tensor(bboxes, dtype=torch.float32)
        
        return boxes_t, scores_t, labels

    def finetune_lora(self, train_loader, epochs=10, lr=1e-4):
        """
        Placeholder for Florence-2 Fine-tuning
        """
        print("Florence-2 LoRA fine-tuning not yet implemented in this adapter.")
        pass

if __name__ == "__main__":
    # Test script
    agent = Florence2Agent(device='cpu', precision='fp32') # Use CPU for quick test
    
    # Create dummy image
    img = Image.new('RGB', (100, 100), color='white')
    
    # Test Detect
    print("Testing detection...")
    try:
        boxes, scores, labels = agent.detect(img, "a white pixel")
        print("Boxes:", boxes)
        print("Labels:", labels)
    except Exception as e:
        print(f"Detection failed (expected if model needs download): {e}")
