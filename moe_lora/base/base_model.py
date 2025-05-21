from typing import Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from loguru import logger

class BaseModel:
    """基础模型类，负责管理预训练大模型和状态"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
        
        # 冻结基础模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        logger.info(f"基础模型已加载到 {device}")
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """获取模型当前状态"""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def inject_state(self, state: Dict[str, torch.Tensor]) -> None:
        """注入新的状态到模型中"""
        for name, param in self.model.named_parameters():
            if name in state:
                param.data.copy_(state[name])
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """执行前向传播"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, **kwargs)
        return outputs
    
    def get_parameter_mapping(self) -> Dict[str, str]:
        """获取参数映射关系"""
        return {
            name: name for name, _ in self.model.named_parameters()
        } 