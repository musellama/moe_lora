from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_lora.base.base_model import BaseModel
from moe_lora.expert.expert_model import ExpertModel, ExpertMetadata
from moe_lora.fusion.fusion_engine import FusionEngine
from peft import PeftModel
import torch
from loguru import logger

def generate_response(model, tokenizer, prompt, max_length=2048):
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    try:
        # 设置模型路径
        base_model_path = "Qwen-2.5-0.5B"
        experts_dir = "../experts"
        
        # 加载基础模型
        logger.info("加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 初始化融合引擎
        fusion_engine = FusionEngine()
        
        # 加载专家模型
        logger.info("\n加载专家模型...")
        experts = []
        for expert_name in ["code_expert", "math_expert"]:
            try:
                expert_path = f"{experts_dir}/{expert_name}"
                # 加载LoRA模型
                expert_model = PeftModel.from_pretrained(
                    base_model,
                    expert_path,
                    device_map="auto"
                )
                
                # 创建专家模型实例
                expert = ExpertModel(
                    model=expert_model,
                    metadata=ExpertMetadata(
                        name=expert_name,
                        description=f"{expert_name} expert",
                        domain="general",
                        version="1.0.0",
                        capabilities=["general"],
                        base_model_name=base_model_path
                    )
                )
                experts.append(expert)
                logger.info(f"已加载专家模型: {expert_name}")
                
            except Exception as e:
                logger.error(f"加载专家模型 {expert_name} 失败: {str(e)}")
                continue
        
        if not experts:
            logger.error("没有成功加载任何专家模型")
            return
        
        # 应用专家模型
        logger.info("\n应用专家模型...")
        model = fusion_engine.apply_experts(base_model, experts)
        
        # 测试问答
        test_prompts = [
            "请用Python实现一个快速排序算法",
            "求解方程：2x + 5 = 13",
            "解释一下什么是机器学习"
        ]
        
        logger.info("\n开始测试问答...")
        for prompt in test_prompts:
            logger.info(f"\n问题: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            logger.info(f"回答: {response}")
            
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 