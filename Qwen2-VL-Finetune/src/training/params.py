from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="/local/dev1/chaofeng/Qwen2.5-VL-3B-Instruct")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=True)
    freeze_llm: bool = field(default=True)
    tune_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    # self add 
    output_dir: Optional[str] = field(default='output/lora_vision_test')
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=64)

    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_ratio: float =  0.03 
    lr_scheduler_type: str =  "cosine"
    logging_steps: int = 1
    tf32: bool = True
    gradient_checkpointing: bool = True
    report_to: str = 'tensorboard'
    lazy_preprocess: bool = True
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 10
    dataloader_num_workers: int = 4





    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    vision_lora: bool = True
    use_dora: bool = False       #### 确定这个参数作用
    lora_rank: int = 64
    lora_alpha: int = 64         #### scale 缩放因子
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1   #### 确定什么作用


@dataclass
class DataArguments:
    data_path: str = field(
        default='/local/dev1/chaofeng/LLaVA-CC3M-Pretrain-595K/chat.json', 
        metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default='/local/dev1/chaofeng/LLaVA-CC3M-Pretrain-595K/images')
    
    image_min_pixels: Optional[int] = field(default=256 * 28 * 28)
    image_max_pixels: Optional[int] = field(default=1280 * 28 * 28)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    fps: float = 1.0


# @dataclass
# class ModelArguments:
#     model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


# @dataclass
# class TrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     adam_beta1: float = field(default=0.9)
#     adam_beta2: float = field(default=0.999)
#     adam_epsilon: float = field(default=1e-8)

#     freeze_vision_tower: bool = field(default=False)
#     freeze_llm: bool = field(default=False)
#     tune_merger: bool = field(default=False)
#     disable_flash_attn2: bool = field(default=False)

#     max_seq_length: int = field(
#         default=32768, # This is the default value of the qwen2-vl model
#         metadata={
#             "help":
#                 "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )

#     double_quant: bool = field(
#         default=True,
#         metadata={"help": "Compress the quantization statistics through double quantization."}
#     )
#     quant_type: str = field(
#         default="nf4",
#         metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
#     )
#     bits: int = field(
#         default=16,
#         metadata={"help": "How many bits to use."}
#     )
#     lora_enable: bool = False
#     vision_lora: bool = False
#     use_dora: bool = False
#     lora_rank: int = 64
#     lora_alpha: int = 16
#     lora_dropout: float = 0.05
#     lora_weight_path: str = ""
#     lora_bias: str = "none"
#     vision_lr: Optional[float] = None
#     merger_lr: Optional[float] = None
#     lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
#     num_lora_modules: int = -1


# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default=None, metadata={"help": "Path to the training data."}
#     )
#     lazy_preprocess: bool = False
#     image_folder: Optional[str] = field(default=None)
#     image_min_pixels: Optional[int] = field(default=3136)
#     image_max_pixels: Optional[int] = field(default=12845056)
#     video_min_pixels: Optional[int] = field(default=100352)
#     video_max_pixels: Optional[int] = field(default=602112)
#     fps: float = 1.0