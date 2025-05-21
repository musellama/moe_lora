from typing import Dict, Any, Optional, List
import torch
from transformers import PreTrainedModel
from loguru import logger
from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from peft import PeftModel

class ExpertMetadata(PydanticBaseModel):
    """专家模型元数据"""
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 允许任意类型
    
    name: str
    description: str
    domain: str
    version: str
    capabilities: List[str]
    base_model_name: str
    lora_config: Optional[Dict[str, Any]] = None  # LoRA配置
    lora_weights: Optional[Dict[str, torch.Tensor]] = None  # LoRA权重

class ExpertModel:
    """专家模型类，负责管理特定领域的专家模型"""
    
    def __init__(
        self,
        model: PeftModel,
        metadata: ExpertMetadata,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.metadata = metadata
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 加载LoRA权重
        self._load_lora_weights()
        
        logger.info(f"专家模型 {metadata.name} 已加载到 {device}")
    
    def _load_lora_weights(self):
        """加载LoRA权重"""
        try:
            # 从PeftModel中获取LoRA权重
            lora_weights = {}
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    lora_weights[name] = param.data.clone()
            
            if lora_weights:
                self.metadata.lora_weights = lora_weights
                logger.info(f"已加载 {len(lora_weights)} 个LoRA权重")
            else:
                logger.warning("未找到LoRA权重")
                
        except Exception as e:
            logger.error(f"加载LoRA权重失败: {str(e)}")
            raise
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """获取专家模型状态"""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def get_parameter_mapping(self) -> Dict[str, str]:
        """获取参数映射关系"""
        return {
            name: name for name, _ in self.model.named_parameters()
        }
    
    def is_compatible_with(self, base_model_name: str) -> bool:
        """检查是否与基础模型兼容"""
        return self.metadata.base_model_name == base_model_name
    
    def get_capabilities(self) -> list[str]:
        """获取专家模型的能力列表"""
        return self.metadata.capabilities 

    def get_lora_weights(self) -> Dict[str, torch.Tensor]:
        """获取LoRA权重"""
        if self.metadata.lora_weights is None:
            raise ValueError("LoRA权重未加载")
        return self.metadata.lora_weights
        
    def get_lora_config(self) -> Dict[str, Any]:
        """获取LoRA配置"""
        if self.metadata.lora_config is None:
            raise ValueError("LoRA配置未设置")
        return self.metadata.lora_config 