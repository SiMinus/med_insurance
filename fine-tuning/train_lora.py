import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ================= 配置区域 =================
# 1. 路径配置
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"  # 基座模型路径
DATA_PATH = "./medical_insurance_finetune_final.json"              # 数据路径
OUTPUT_DIR = "./output_qwen_lora"                                  # 输出路径

# 2. 数据量控制 (调试用)
# 设置为 None 或 0 表示使用全部数据
# 设置为 50 表示只使用前 50 条数据进行快速测试
MAX_SAMPLES = None

# 3. 显存与量化配置
# RTX 4090 (24GB) 跑 7B LoRA:
# - 方案 A (推荐): BF16 原生加载。速度快，显存占用约 16-20GB (视 max_seq_length 而定)。
# - 方案 B: 4-bit 量化 (QLoRA)。显存占用极低 (约 6-8GB)，可以开大 batch_size，但训练略慢。
USE_4BIT_QUANTIZATION = False  # 如果显存 OOM (爆显存)，请改为 True

# 4. 训练参数
NUM_EPOCHS = 3
BATCH_SIZE = 4          # 如果不量化且显存紧张，调小这个值 (如 1 或 2)
GRADIENT_ACCUMULATION = 4 # 梯度累积，相当于总 batch_size = 4 * 4 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024   # 序列长度，越长越占显存

# ===========================================

def main():
    print(f"正在加载模型: {MODEL_PATH}")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 训练时通常 padding 在右侧

    # 2. 加载模型
    if USE_4BIT_QUANTIZATION:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16, # 4090 支持 BF16
            device_map="auto",
            trust_remote_code=True
        )
    
    # 开启梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,                   # LoRA 秩
        lora_alpha=32,          # LoRA alpha
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 全模块微调效果更好
    )

    # 4. 加载与处理数据
    print(f"正在加载数据: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 数据截断 (调试用)
    if MAX_SAMPLES and MAX_SAMPLES > 0:
        print(f"【调试模式】仅使用前 {MAX_SAMPLES} 条数据...")
        dataset = dataset.select(range(min(len(dataset), MAX_SAMPLES)))

    # 定义格式化函数：将 instruction/output 转为 Chat 格式
    def format_example(example):
        instruction = example['instruction']
        output = example['output']
        
        messages = [
            {"role": "system", "content": "你是医保政策助手，结合参考内容回答用户问题。请用中文简洁作答。"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        # 使用 tokenizer 的 chat template
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False,
            max_length=MAX_SEQ_LENGTH - 1, # 预留空间
            truncation=True
        )
        
        # 处理 Qwen 模板可能自带的尾部换行符 (token_id 198)
        # 确保数据以 <|im_end|> 结尾，而不是 \n
        if len(input_ids) > 0 and input_ids[-1] == 198: # 198 is '\n'
            input_ids.pop()
            
        # 强制添加 EOS token (<|im_end|>)
        if len(input_ids) > 0 and input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    # 预处理数据：生成 input_ids 列并移除原始列
    print("正在预处理数据...")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 打印前5条数据的 input_ids
    print("\n=== 调试：前5条数据的 input_ids ===")
    for i in range(min(5, len(dataset))):
        ids = dataset[i]['input_ids']
        print(f"Sample {i}: Length={len(ids)}")
        print(f"IDs: {ids}")
        # 尝试解码回文本以便观察
        decoded = tokenizer.decode(ids)
        print(f"Decoded: {decoded}")
        print("-" * 50)
    print("====================================\n")

    # 5. 训练参数配置
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,                # 频繁打印日志以便观察
        bf16=True,                      # 4090 必须开 BF16
        
        # 节省空间的关键配置
        save_strategy="no",             # 【关键】禁止中间 checkpoint
        save_only_model=True,           # 【关键】只保存模型权重，不保存优化器状态
        
        optim="paged_adamw_32bit",      # 使用分页优化器节省显存
        report_to="none",               # 不上传 wandb
        remove_unused_columns=False,    # 防止 dataset 列被删除导致报错
    )

    # 6. 初始化 Trainer
    # 使用 DataCollatorForCompletionOnlyLM 可能会有兼容性问题，这里改用默认的 DataCollatorForSeq2Seq
    from transformers import DataCollatorForSeq2Seq
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        # dataset_text_field="text", # 不再需要，因为我们已经手动 tokenize 了
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=args,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 显式指定 collator
    )

    # 7. 开始训练
    print("开始训练...")
    trainer.train()

    # 8. 保存最终模型
    print(f"训练完成，正在保存模型至 {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)
    print("保存成功！")

if __name__ == "__main__":
    main()
