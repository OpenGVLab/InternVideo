import logging
import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
from transformers import LlamaForCausalLM, LlamaConfig

from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)


class LLaMA(nn.Module):
    def __init__(
            self,
            use_flash_attn: bool = True,
            transformer_width: int = 4096,
            llama_path: str = None,
            use_lora: bool = True,
            clip_embed_dim: int = 768,
        ):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn
        self.transformer_width = transformer_width
        
        """ text encoder of InternVL """
        llama_config = LlamaConfig.from_pretrained(llama_path, local_files_only=True)
        llama_config.causal = True
        llama_config.use_flash_attention = use_flash_attn
        # model = LlamaForCausalLM.from_pretrained(  # LLAMA model
        #     llama_path, torch_dtype=torch.float16, config=llama_config, local_files_only=True)
        model = LlamaForCausalLM(config=llama_config)
        if not use_lora:
            self.transformer = model.model
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            self.transformer = model.base_model.model.model
        
        self.transformer.gradient_checkpointing = True
        self.text_projection = nn.Parameter(torch.empty(transformer_width, clip_embed_dim))
            
    def forward(self, text):
        text_key_padding_mask = text > 0
        
        x = self.transformer(input_ids=text, attention_mask=text_key_padding_mask).last_hidden_state
        x = x[torch.arange(x.shape[0]), text_key_padding_mask.sum(1) - 1]
        x = x @ self.text_projection

        return x


class Tokenizer(nn.Module):
    def __init__(self, tokenizer_path="your_model_path/chinese_alpaca_lora_7b"):
        super(Tokenizer, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path, 
            local_files_only=True,
            legacy=False
        )
        self.tokenizer.pad_token = " "  # allow padding
        self.tokenizer.add_eos_token = True
    
    def forward(self, text):
        text = ["summarize:" + item for item in text]
        text = self.tokenizer(text, return_tensors="pt", max_length=80, truncation=True, padding="max_length").input_ids
        return text
