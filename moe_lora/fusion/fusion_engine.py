from typing import Dict, List, Optional
import torch
from loguru import logger
from ..expert.expert_model import ExpertModel
from transformers import PreTrainedModel

class FusionStrategy:
    """融合策略基类"""
    def __init__(self, name: str):
        self.name = name
    
    def fuse(self, base_state: Dict[str, torch.Tensor], expert_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class WeightedSumFusion(FusionStrategy):
    """加权求和融合策略"""
    def __init__(self):
        super().__init__("weighted_sum")
    
    def fuse(self, base_state: Dict[str, torch.Tensor], expert_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        fused_state = {}
        for param_name in base_state.keys():
            fused_state[param_name] = base_state[param_name].clone()
            for expert_state, weight in zip(expert_states, weights):
                if param_name in expert_state:
                    fused_state[param_name] += weight * (expert_state[param_name] - base_state[param_name])
        return fused_state

class AttentionFusion(FusionStrategy):
    """注意力融合策略"""
    def __init__(self):
        super().__init__("attention")
    
    def fuse(self, base_state: Dict[str, torch.Tensor], expert_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        # 实现注意力机制的融合策略
        # TODO: 实现具体的注意力融合逻辑
        raise NotImplementedError

class FusionEngine:
    """融合引擎，负责将专家模型与基础模型进行融合"""
    
    def __init__(self, alpha: float = 0.3):
        """
        初始化融合引擎
        
        Args:
            alpha: LoRA权重融合系数，范围[0,1]，默认0.3
                  值越小，基础模型权重占比越大
        """
        self.alpha = alpha
        logger.info(f"初始化融合引擎，LoRA融合系数: {alpha}")
    
    def apply_experts(self, base_model: PreTrainedModel, experts: List[ExpertModel]) -> PreTrainedModel:
        """应用专家模型
        
        Args:
            base_model: 基础模型
            experts: 专家模型列表
            
        Returns:
            融合后的模型
        """
        try:
            # 获取基础模型状态
            base_state = {
                name: param.data.clone()
                for name, param in base_model.named_parameters()
            }
            
            # 对每个专家模型进行融合
            for expert in experts:
                if not expert.is_compatible_with(base_model.config._name_or_path):
                    logger.warning(f"专家模型 {expert.metadata.name} 与基础模型不兼容，跳过")
                    continue
                
                # 获取LoRA权重
                lora_weights = expert.get_lora_weights()
                
                # 融合权重
                for name, param in base_model.named_parameters():
                    if name in lora_weights:
                        # 使用alpha控制融合比例
                        base_state[name] = base_state[name] * (1 - self.alpha) + lora_weights[name] * self.alpha
                        logger.debug(f"融合参数 {name}")
            
            # 更新模型参数
            for name, param in base_model.named_parameters():
                if name in base_state:
                    param.data.copy_(base_state[name])
            
            # 计算总参数量
            total_params = sum(p.numel() for p in base_model.parameters())
            lora_params = sum(
                sum(p.numel() for name, p in expert.model.named_parameters() if "lora" in name.lower())
                for expert in experts
            )
            logger.info(f"基础模型参数量: {total_params:,}")
            logger.info(f"LoRA参数量: {lora_params:,}")
            logger.info(f"融合后总参数量: {total_params + lora_params:,} (基础模型 + {len(experts)}个LoRA)")
            
            return base_model
            
        except Exception as e:
            logger.error(f"融合专家模型时出错: {str(e)}")
            raise 