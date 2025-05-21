from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_lora.base.base_model import BaseModel
from moe_lora.expert.expert_model import ExpertModel, ExpertMetadata
from moe_lora.fusion.fusion_engine import FusionEngine
from peft import PeftModel
import torch
from loguru import logger
import os

def generate_response(model, tokenizer, prompt, max_length=2048):
    """生成回答"""
    # 构建Qwen-2.5的标准格式
    messages = [
        {"role": "system", "content": "你是一个有感情的人。"},
        {"role": "user", "content": prompt}
    ]
    
    # 将消息转换为模型输入格式
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成回答
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        num_beams=4,
        do_sample=True,  # 启用采样
        temperature=0.7,  # 添加温度参数
        top_p=0.9,  # 添加top_p参数
        top_k=50,  # 添加top_k参数
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        length_penalty=1.0,
        early_stopping=True
    )
    
    # 解码回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只提取assistant的回答部分
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    else:
        # 如果没有找到assistant标记，则移除输入文本
        response = response.replace(input_text, "").strip()
    
    return response

def load_all_experts(base_model, experts_dir, base_model_path):
    """加载所有专家模型"""
    experts = []
    # 设置警告过滤器
    import warnings
    warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")
    warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute in the model")
    
    for expert_name in os.listdir(experts_dir):
        expert_path = os.path.join(experts_dir, expert_name)
        # 跳过非目录和非专家模型
        if not os.path.isdir(expert_path):
            continue
        if expert_name == os.path.basename(base_model_path):  # 跳过基础模型目录
            logger.info(f"跳过基础模型目录: {expert_name}")
            continue
            
        try:
            # 检查是否存在adapter_config.json
            if not os.path.exists(os.path.join(expert_path, "adapter_config.json")):
                logger.warning(f"跳过非LoRA模型目录: {expert_name}")
                continue
                
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
            logger.warning(f"加载专家模型 {expert_name} 失败")
            continue
            
    # 恢复警告设置
    warnings.resetwarnings()
    return experts

def main():
    try:
        # 设置模型路径
        base_model_path = "Qwen2.5-1.5B-Instruct"
        experts_dir = "../experts"
        
        # 加载基础模型
        logger.info("加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=base_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=base_model_path
        )
        
        # 计算基础模型参数量
        base_params = sum(p.numel() for p in base_model.parameters())
        logger.info(f"基础模型参数量: {base_params:,}")
        
        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载所有专家模型
        logger.info("\n加载专家模型...")
        experts = load_all_experts(base_model, experts_dir, base_model_path)
        
        # 初始化模型
        if experts:
            # 初始化融合引擎，设置较小的alpha值
            fusion_engine = FusionEngine(alpha=0.5)
            # 应用专家模型
            logger.info("\n应用专家模型...")
            model = fusion_engine.apply_experts(base_model, experts)
            
            # 计算基础模型参数量
            base_params = sum(p.numel() for p in base_model.parameters())
            
            # 显示每个LoRA的参数
            logger.info("\n各LoRA模型参数统计:")
            total_lora_params = 0
            for expert in experts:
                # 只计算LoRA特有的参数
                expert_lora_params = sum(p.numel() for name, p in expert.model.named_parameters() 
                                      if 'lora_' in name)
                total_lora_params += expert_lora_params
                logger.info(f"- {expert.metadata.name}: {expert_lora_params:,}")
            
            # 计算融合后的总参数量
            fused_total_params = base_params + total_lora_params
            
            # 显示总体参数统计
            logger.info(f"\n总体参数统计:")
            logger.info(f"- 基础模型参数量: {base_params:,}")
            logger.info(f"- 所有LoRA总参数量: {total_lora_params:,}")
            logger.info(f"- 融合后总参数量: {fused_total_params:,} (基础模型 + {len(experts)}个LoRA)")
        else:
            logger.warning("没有成功加载任何专家模型，将使用基础模型")
            model = base_model  # 使用基础模型
        
        # 交互式问答
        logger.info("\n开始交互式问答（输入'退出'结束对话）...")
        while True:
            prompt = input("\n请输入您的问题: ")
            if prompt.lower() in ['退出', 'exit', 'quit']:
                break
                
            try:
                response = generate_response(model, tokenizer, prompt)
                print(f"\n回答: {response}")
            except Exception as e:
                logger.error(f"生成回答时出错: {str(e)}")
                print("抱歉，生成回答时出现错误，请重试。")
            
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 