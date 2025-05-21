from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_lora.base.base_model import BaseModel
from moe_lora.expert.expert_model import ExpertModel, ExpertMetadata
from moe_lora.fusion.fusion_engine import FusionEngine
from peft import PeftModel
import torch
from loguru import logger

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params
    }

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
        
        # 计算基础模型参数量
        base_params = count_parameters(base_model)
        logger.info(f"基础模型参数量: {base_params['total']:,}")
        
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
                
                # 计算专家模型参数量
                expert_params = count_parameters(expert.model)
                logger.info(f"{expert_name} 总参数量: {expert_params['total']:,}")
                logger.info(f"{expert_name} 可训练参数量: {expert_params['trainable']:,}")
                
                # 获取LoRA权重
                lora_weights = expert.get_lora_weights()
                logger.info(f"{expert_name} LoRA权重数量: {len(lora_weights)}")
                lora_params = sum(w.numel() for w in lora_weights.values())
                logger.info(f"{expert_name} LoRA参数量: {lora_params:,}")
                
            except Exception as e:
                logger.error(f"加载专家模型 {expert_name} 失败: {str(e)}")
                continue
        
        if not experts:
            logger.error("没有成功加载任何专家模型")
            return
        
        # 应用专家模型
        logger.info("\n应用专家模型...")
        model = fusion_engine.apply_experts(base_model, experts)
        
        # 计算最终参数量
        final_params = count_parameters(model)
        logger.info(f"\n最终模型参数量: {final_params['total']:,}")
        
        # 验证参数量
        expected_params = base_params['total'] + sum(len(e.get_lora_weights()) for e in experts)
        logger.info(f"预期参数量: {expected_params:,}")
        logger.info(f"参数量匹配: {final_params['total'] == expected_params}")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 