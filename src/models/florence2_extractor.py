"""
Florence-2 model wrapper for product attribute extraction.
Handles loading, LoRA setup, and inference.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image

UNIT_PATTERNS = [
    "kilogram", "kg", "gram", "g", "pound", "lb", "lbs", "ounce", "oz",
    "milligram", "mg", "ton",
    "centimetre", "centimeter", "cm", "millimetre", "millimeter", "mm",
    "metre", "meter", "m", "inch", "in", "foot", "feet", "ft", "yard",
    "volt", "v", "watt", "w", "ampere", "amp", "a",
    "litre", "liter", "l", "millilitre", "milliliter", "ml",
    "gallon", "gal", "quart", "qt", "pint", "pt", "cup",
    "fluid ounce", "fl oz",
]


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
    def _generate(self, image, prompt, max_tokens=256):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.device == "cpu":
            inputs = {k: v.float() if v.dtype == torch.float16 else v for k, v in inputs.items()}
            generated = self.model.float().generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=3,
                early_stopping=True,
            )
        else:
            with torch.amp.autocast("cuda"):
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=3,
                    early_stopping=True,
                )

        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]

    @torch.no_grad()
    def extract(self, image, entity_name=None):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        raw_text = self._generate(image, "<OCR>")
        return self._parse_output(raw_text, entity_name)

    @torch.no_grad()
    def extract_all_specs(self, image):
        """Extract specs using fine-tuned model + base model full OCR."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        finetuned_output = self._generate(image, "<OCR>")
        finetuned_specs = self._parse_structured_output(finetuned_output)
        if not finetuned_specs:
            finetuned_specs = self._parse_all_values(finetuned_output)

        self.model.disable_adapter_layers()
        ocr_text = self._generate(image, "<OCR>", max_tokens=512)
        caption = self._generate(image, "<MORE_DETAILED_CAPTION>", max_tokens=512)
        self.model.enable_adapter_layers()

        ocr_specs = self._parse_all_values(ocr_text + " " + caption)

        seen = set()
        all_specs = []
        for spec in finetuned_specs:
            key = (spec.get("value"), spec.get("unit", "").lower())
            if key not in seen:
                seen.add(key)
                all_specs.append(spec)
        for spec in ocr_specs:
            key = (spec.get("value"), spec.get("unit", "").lower())
            if key not in seen:
                seen.add(key)
                all_specs.append(spec)

        return {
            "finetuned": finetuned_output,
            "ocr_text": ocr_text,
            "caption": caption,
            "specs": all_specs,
        }

    def _parse_output(self, text, entity_name=None):
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

    def _parse_structured_output(self, text):
        """Parse 'entity_name: value unit | entity_name: value unit' format."""
        parts = text.split("|")
        specs = []
        for part in parts:
            part = part.strip()
            match = re.match(r"(\w+(?:\s\w+)*):\s*([\d,.]+)\s*(.+)", part)
            if match:
                try:
                    value = float(match.group(2).replace(",", ""))
                except ValueError:
                    continue
                specs.append({
                    "attribute": match.group(1).strip(),
                    "value": value,
                    "unit": match.group(3).strip(),
                })
        return specs

    def _parse_all_values(self, text):
        """Find all number+unit pairs in text."""
        units_re = "|".join(re.escape(u) for u in sorted(UNIT_PATTERNS, key=len, reverse=True))
        pattern = rf"([\d,.]+)\s*({units_re})\b"
        matches = re.findall(pattern, text, re.IGNORECASE)

        specs = []
        seen = set()
        for value_str, unit in matches:
            value_str = value_str.replace(",", "")
            try:
                value = float(value_str)
            except ValueError:
                continue
            key = (value, unit.lower())
            if key not in seen:
                seen.add(key)
                specs.append({"value": value, "unit": unit})

        return specs
