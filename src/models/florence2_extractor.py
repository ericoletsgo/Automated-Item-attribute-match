"""
Florence-2 model wrapper for product attribute extraction.
Handles loading, LoRA setup, and inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image


class Florence2Extractor:
    """Extracts product attributes from images using fine-tuned Florence-2."""

    def __init__(self, model_path=None, base_model="microsoft/Florence-2-large", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        if model_path:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def setup_for_training(cls, config):
        """Create a model ready for LoRA fine-tuning."""
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(
            config["model"]["name"], trust_remote_code=True
        )

        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"]["lora_alpha"],
            target_modules=config["model"]["lora_target_modules"],
            lora_dropout=config["model"]["lora_dropout"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, processor

    @torch.no_grad()
    def extract(self, image, entity_name=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        prompt = "<OCR>"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            early_stopping=True,
        )

        output_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return self._parse_output(output_text, entity_name)

    def _parse_output(self, text, entity_name=None):
        import re
        text = text.strip()

        match = re.match(r"([\d.]+)\s*(.+)", text)
        if match:
            return {
                "entity_name": entity_name or "unknown",
                "value": float(match.group(1)),
                "unit": match.group(2).strip(),
                "raw": text,
            }

        return {"entity_name": entity_name, "value": None, "unit": None, "raw": text}
