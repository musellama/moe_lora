from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_lora.base.base_model import BaseModel
from moe_lora.expert.expert_model import ExpertModel, ExpertMetadata
from moe_lora.fusion.fusion_engine import FusionEngine
from moe_lora.scheduler.expert_scheduler import ExpertScheduler
from loguru import logger
import os

def main():
    # 设置模型路径
    base_model_path = "Qwen-2.5-0.5B"  # 基础模型路径
    experts_dir = "../experts"  # 专家模型目录
    
    # 确保专家模型目录存在
    os.makedirs(experts_dir, exist_ok=True)
    
    # 初始化基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True,
        cache_dir=base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=base_model_path
    )
    base_model = BaseModel(model, tokenizer)
    
    # 初始化融合引擎
    fusion_engine = FusionEngine()
    
    # 初始化专家调度器
    scheduler = ExpertScheduler(base_model, fusion_engine)
    
    # 创建并注册多个专家模型
    experts = [
        {
            "name": "code_expert",
            "description": "代码生成专家",
            "domain": "programming",
            "version": "1.0.0",
            "capabilities": ["code", "programming", "python"],
            "model_path": os.path.join(experts_dir, "code_expert")
        },
        {
            "name": "math_expert",
            "description": "数学计算专家",
            "domain": "mathematics",
            "version": "1.0.0",
            "capabilities": ["math", "calculation", "algebra"],
            "model_path": os.path.join(experts_dir, "math_expert")
        }
    ]
    
    # 注册专家模型
    for expert_info in experts:
        # 检查专家模型是否存在
        if os.path.exists(expert_info["model_path"]):
            # 加载已存在的专家模型
            expert_model = AutoModelForCausalLM.from_pretrained(
                expert_info["model_path"],
                trust_remote_code=True,
                device_map="auto",
                local_files_only=True
            )
        else:
            # 使用基础模型初始化新的专家模型
            logger.info(f"创建新的专家模型: {expert_info['name']}")
            expert_model = model
            
        # 创建专家元数据
        expert_metadata = ExpertMetadata(
            name=expert_info["name"],
            description=expert_info["description"],
            domain=expert_info["domain"],
            version=expert_info["version"],
            capabilities=expert_info["capabilities"],
            base_model_name=base_model_path
        )
        
        # 创建专家模型实例
        expert = ExpertModel(expert_model, expert_metadata)
        
        # 注册专家模型
        scheduler.register_expert(expert)
    
    # 处理输入
    input_text = "你好告诉我500*500/100+50*50=多少"
    fused_state = scheduler.process_input(input_text)
    
    # 使用融合后的模型进行推理
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = base_model.forward(**inputs)
    
    # 生成回复
    generated_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
    logger.info(f"生成的回复: {generated_text}")

if __name__ == "__main__":
    main() 