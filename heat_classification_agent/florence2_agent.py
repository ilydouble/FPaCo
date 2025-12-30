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
            logits: torch.Tensor (N,) - Florence-2 doesn't output confidence easily, returns ones.
            phrases: List[str] - The label phrases
        """
        
        if isinstance(image_path, str) or isinstance(image_path, os.PathLike):
            image = Image.open(str(image_path)).convert("RGB")
        else:
            image = image_path.convert("RGB") # Assume PIL Image
            
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = task_prompt + prompt
        
        inputs = self.processor(text=text_input, images=image, return_tensors="pt").to(self.device, self.model.dtype)
        
        # Determine beam search parameters for stability
        # For grounding, standard generation usually works.
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False,
            early_stopping=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process raw text to dictionary
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        # parsed_answer format for <CAPTION_TO_PHRASE_GROUNDING>:
        # {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['phrase1', ...]}}
        
        result_content = parsed_answer.get(task_prompt, {})
        bboxes = result_content.get('bboxes', [])
        labels = result_content.get('labels', [])
        
        if len(bboxes) == 0:
            return torch.tensor([]), torch.tensor([]), []
            
        # Convert to torch tensor
        boxes_t = torch.tensor(bboxes, dtype=torch.float32)
        
        # Create dummy logits (1.0) since Florence-2 is generative
        logits_t = torch.ones(len(bboxes), dtype=torch.float32)
        
        return boxes_t, logits_t, labels

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
