from typing import Dict, List, Optional, Tuple
import torch
from loguru import logger
from ..expert.expert_model import ExpertModel
from ..base.base_model import BaseModel
from ..fusion.fusion_engine import FusionEngine

class ExpertScheduler:
    """专家调度器，负责管理专家模型的选择和资源调度"""
    
    def __init__(
        self,
        base_model: BaseModel,
        fusion_engine: FusionEngine,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model = base_model
        self.fusion_engine = fusion_engine
        self.device = device
        self.experts: Dict[str, ExpertModel] = {}
        self.expert_states: Dict[str, Dict[str, torch.Tensor]] = {}
        
    def register_expert(self, expert: ExpertModel) -> None:
        """注册专家模型"""
        if not expert.is_compatible_with(self.base_model.model.name_or_path):
            raise ValueError(f"专家模型 {expert.metadata.name} 与基础模型不兼容")
            
        self.experts[expert.metadata.name] = expert
        self.expert_states[expert.metadata.name] = expert.get_state()
        logger.info(f"已注册专家模型: {expert.metadata.name}")
    
    def select_experts(self, input_text: str, top_k: int = 2) -> List[Tuple[ExpertModel, float]]:
        """基于输入文本选择相关专家模型"""
        # TODO: 实现基于语义理解的专家选择逻辑
        # 这里使用简单的示例实现
        selected_experts = []
        for expert in self.experts.values():
            # 简单的关键词匹配示例
            if any(cap in input_text.lower() for cap in expert.get_capabilities()):
                selected_experts.append((expert, 1.0))
        
        # 按权重排序并返回top_k个专家
        selected_experts.sort(key=lambda x: x[1], reverse=True)
        return selected_experts[:top_k]
    
    def fuse_experts(
        self,
        selected_experts: List[Tuple[ExpertModel, float]],
        fusion_strategy: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """融合选中的专家模型"""
        base_state = self.base_model.get_state()
        expert_states = [self.expert_states[expert.metadata.name] for expert, _ in selected_experts]
        weights = [weight for _, weight in selected_experts]
        
        return self.fusion_engine.fuse(
            base_state=base_state,
            expert_states=expert_states,
            weights=weights,
            strategy=fusion_strategy
        )
    
    def process_input(self, input_text: str) -> Dict[str, torch.Tensor]:
        """处理输入文本"""
        # 选择相关专家
        selected_experts = self.select_experts(input_text)
        if not selected_experts:
            logger.warning("未找到相关专家模型，使用基础模型")
            return self.base_model.get_state()
        
        # 融合专家模型
        fused_state = self.fuse_experts(selected_experts)
        
        # 注入融合后的状态
        self.base_model.inject_state(fused_state)
        
        return fused_state 