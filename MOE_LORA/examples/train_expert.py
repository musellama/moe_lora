from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
from loguru import logger
import os
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

def train_expert_model(
    base_model_path: str,
    expert_name: str,
    expert_domain: str,
    training_data_path: str,
    output_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """训练专家模型
    
    Args:
        base_model_path: 基础模型路径
        expert_name: 专家模型名称
        expert_domain: 专家领域
        training_data_path: 训练数据路径
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 加载训练数据
    logger.info(f"加载训练数据: {training_data_path}")
    dataset = load_dataset("json", data_files=training_data_path)
    
    # 打印数据集结构
    logger.info(f"数据集结构: {dataset['train'].features}")
    
    # 数据预处理
    def preprocess_function(examples):
        # 假设数据中有 'content' 或 'message' 字段
        text_field = None
        for field in ['content', 'message', 'text', 'input', 'prompt']:
            if field in examples:
                text_field = field
                break
        
        if text_field is None:
            raise ValueError(f"找不到文本字段，可用的字段有: {list(examples.keys())}")
            
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )
    
    # 开始训练
    logger.info(f"开始训练专家模型: {expert_name}")
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    logger.info(f"专家模型已保存到: {output_dir}")

def main():
    # 设置路径
    base_model_path = "Qwen-2.5-0.5B"
    experts_dir = "../experts"
    data_dir = "data"  # 训练数据目录
    
    # 确保专家目录存在
    os.makedirs(experts_dir, exist_ok=True)
    
    # 训练代码专家模型
    code_expert_path = os.path.join(experts_dir, "code_expert")
    train_expert_model(
        base_model_path=base_model_path,
        expert_name="code_expert",
        expert_domain="programming",
        training_data_path=os.path.join(data_dir, "grok3-聊天900.json"),  # 使用实际的训练数据文件
        output_dir=code_expert_path
    )
    
    # 训练数学专家模型
    math_expert_path = os.path.join(experts_dir, "math_expert")
    train_expert_model(
        base_model_path=base_model_path,
        expert_name="math_expert",
        expert_domain="mathematics",
        training_data_path=os.path.join(data_dir, "grok3-聊天900.json"),  # 使用实际的训练数据文件
        output_dir=math_expert_path
    )

if __name__ == "__main__":
    main() 