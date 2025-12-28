"""
T5 Fine-tuning for Machine Translation
Uses HuggingFace Transformers
"""
import torch
import torch.nn as nn
from typing import Optional, List, Dict
import warnings

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not installed. T5 model will not be available.")


class T5Translator(nn.Module):
    """
    T5 model wrapper for translation task
    """
    
    def __init__(
        self,
        model_name: str = 't5-base',
        max_length: int = 128,
        task_prefix: str = "translate Chinese to English: "
    ):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.task_prefix = task_prefix
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add Chinese vocabulary if needed (T5 is primarily English)
        # For production, consider using mT5 instead
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def encode_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of source (and optionally target) texts
        """
        # Add task prefix
        src_texts = [self.task_prefix + text for text in src_texts]
        
        # Encode source
        src_encoding = self.tokenizer(
            src_texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': src_encoding['input_ids'].to(device),
            'attention_mask': src_encoding['attention_mask'].to(device)
        }
        
        # Encode target if provided
        if tgt_texts is not None:
            tgt_encoding = self.tokenizer(
                tgt_texts,
                max_length=self.max_length,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            # Replace padding token id with -100 for loss computation
            labels = tgt_encoding['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels.to(device)
        
        return result
    
    def translate(
        self,
        src_texts: List[str],
        device: torch.device = torch.device('cpu'),
        beam_width: int = 5,
        max_length: int = 128
    ) -> List[str]:
        """
        Translate source texts to target language
        """
        self.model.eval()
        
        # Add task prefix
        src_texts = [self.task_prefix + text for text in src_texts]
        
        # Encode
        inputs = self.tokenizer(
            src_texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=beam_width,
                early_stopping=True
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return translations


class T5TranslationDataset(torch.utils.data.Dataset):
    """
    Dataset for T5 translation fine-tuning
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: T5Tokenizer,
        src_lang: str = 'zh',
        tgt_lang: str = 'en',
        max_length: int = 128,
        task_prefix: str = "translate Chinese to English: "
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.task_prefix = task_prefix
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = self.task_prefix + item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        
        # Encode source
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode target
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (replace pad tokens with -100)
        labels = tgt_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def create_t5_model(
    model_name: str = 't5-base',
    max_length: int = 128
) -> T5Translator:
    """Factory function to create T5 translator"""
    return T5Translator(
        model_name=model_name,
        max_length=max_length,
        task_prefix="translate Chinese to English: "
    )


if __name__ == "__main__":
    if HAS_TRANSFORMERS:
        # Test T5 model
        model = create_t5_model('t5-small')
        
        # Test translation
        src_texts = [
            "你好，世界！",
            "这是一个测试。"
        ]
        
        translations = model.translate(src_texts)
        print("Translations:")
        for src, tgt in zip(src_texts, translations):
            print(f"  {src} -> {tgt}")
    else:
        print("Transformers not installed, skipping T5 test")
