from .internlm2_packed_training_patch import replace_internlm2_attention_class
from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from .llama_packed_training_patch import replace_llama_attention_class
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import concat_pad_data_collator, pad_data_collator
from .qwen2_packed_training_patch import replace_qwen2_attention_class
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm',
           'replace_llama2_attn_with_flash_attn',
           'replace_train_sampler',
           'replace_train_dataloader',
           'replace_internlm2_attention_class',
           'replace_qwen2_attention_class',
           'replace_llama_attention_class',
           'pad_data_collator',
           'concat_pad_data_collator']
