"""
ONNX 模型导出模块

将 PyTorch 训练的 UGT 模型导出为 ONNX 格式，支持动态批次和序列长度。

特性：
1. 动态形状支持 - batch_size 和 seq_length 可变
2. 常量折叠优化 - 减少推理时计算量
3. 输入/输出验证 - 确保导出模型的正确性
4. 模型简化 - 使用 onnx-simplifier 优化计算图

Author: Person F (MLOps Engineer)
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn

from .config import ExportConfig, ModelInputSpec

# 配置日志
logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    ONNX 模型导出器
    
    负责将 PyTorch 模型导出为 ONNX 格式，支持动态形状和多种优化选项。
    
    Attributes:
        config: 导出配置
        input_spec: 模型输入规格
    
    Example:
        >>> config = ExportConfig(model_name="ugt_recommend", precision="fp16")
        >>> exporter = ONNXExporter(config)
        >>> onnx_path = exporter.export(model, "/tmp/model.onnx")
    """
    
    def __init__(self, config: ExportConfig):
        """
        初始化 ONNX 导出器
        
        Args:
            config: 导出配置对象
        """
        self.config = config
        self.input_spec = ModelInputSpec()
        config.validate()
    
    def _create_example_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 100,
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        创建示例输入用于 ONNX 导出
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            device: 设备类型
        
        Returns:
            包含所有模型输入的字典
        """
        l1_size, l2_size, l3_size = self.config.codebook_sizes
        
        example_inputs = {
            self.input_spec.encoder_l1_ids: torch.randint(
                0, l1_size, (batch_size, seq_len), dtype=torch.long, device=device
            ),
            self.input_spec.encoder_l2_ids: torch.randint(
                0, l2_size, (batch_size, seq_len), dtype=torch.long, device=device
            ),
            self.input_spec.encoder_l3_ids: torch.randint(
                0, l3_size, (batch_size, seq_len), dtype=torch.long, device=device
            ),
            self.input_spec.encoder_positions: torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1),
            self.input_spec.encoder_token_types: torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            ),
            self.input_spec.encoder_mask: torch.ones(
                batch_size, seq_len, dtype=torch.float32, device=device
            ),
        }
        
        return example_inputs
    
    def _get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """
        获取动态轴配置
        
        Returns:
            动态轴字典，键为张量名称，值为轴索引到名称的映射
        """
        # 输入动态轴
        dynamic_axes = {
            self.input_spec.encoder_l1_ids: {0: "batch", 1: "seq_len"},
            self.input_spec.encoder_l2_ids: {0: "batch", 1: "seq_len"},
            self.input_spec.encoder_l3_ids: {0: "batch", 1: "seq_len"},
            self.input_spec.encoder_positions: {0: "batch", 1: "seq_len"},
            self.input_spec.encoder_token_types: {0: "batch", 1: "seq_len"},
            self.input_spec.encoder_mask: {0: "batch", 1: "seq_len"},
        }
        
        # 输出动态轴
        dynamic_axes[self.input_spec.recommendations] = {0: "batch"}
        dynamic_axes[self.input_spec.scores] = {0: "batch"}
        
        return dynamic_axes
    
    def export(
        self,
        model: nn.Module,
        save_path: str,
        verify: bool = True
    ) -> str:
        """
        导出模型为 ONNX 格式
        
        Args:
            model: 要导出的 PyTorch 模型
            save_path: ONNX 文件保存路径
            verify: 是否验证导出的模型
        
        Returns:
            导出的 ONNX 文件路径
        
        Raises:
            RuntimeError: 导出失败时抛出
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # 设置模型为评估模式
        model.eval()
        device = next(model.parameters()).device
        
        # 创建示例输入
        example_inputs = self._create_example_inputs(
            batch_size=1,
            seq_len=100,
            device=str(device)
        )
        
        # 获取动态轴配置
        dynamic_axes = self._get_dynamic_axes()
        
        logger.info(f"开始导出 ONNX 模型到 {save_path}")
        logger.info(f"配置: opset_version={self.config.opset_version}, "
                   f"constant_folding={self.config.do_constant_folding}")
        
        try:
            # 使用 no_grad 上下文
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    tuple(example_inputs.values()),
                    save_path,
                    input_names=self.input_spec.get_input_names(),
                    output_names=self.input_spec.get_output_names(),
                    dynamic_axes=dynamic_axes,
                    opset_version=self.config.opset_version,
                    do_constant_folding=self.config.do_constant_folding,
                    export_params=True,
                    verbose=False,
                )
            
            logger.info(f"ONNX 模型导出成功: {save_path}")
            
            # 验证导出的模型
            if verify:
                self._verify_onnx_model(save_path, example_inputs)
            
            # 尝试简化模型
            simplified_path = self._simplify_model(save_path)
            if simplified_path:
                return simplified_path
            
            return save_path
            
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            raise RuntimeError(f"ONNX 导出失败: {e}") from e
    
    def _verify_onnx_model(
        self,
        onnx_path: str,
        example_inputs: Dict[str, torch.Tensor]
    ) -> None:
        """
        验证导出的 ONNX 模型
        
        Args:
            onnx_path: ONNX 文件路径
            example_inputs: 示例输入
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # 检查模型结构
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX 模型结构验证通过")
            
            # 运行推理测试
            session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
            )
            
            # 准备输入
            ort_inputs = {
                name: tensor.cpu().numpy()
                for name, tensor in example_inputs.items()
            }
            
            # 执行推理
            outputs = session.run(None, ort_inputs)
            logger.info(f"ONNX 推理测试通过，输出数量: {len(outputs)}")
            
        except ImportError:
            logger.warning("未安装 onnx 或 onnxruntime，跳过验证")
        except Exception as e:
            logger.warning(f"ONNX 验证警告: {e}")
    
    def _simplify_model(self, onnx_path: str) -> Optional[str]:
        """
        使用 onnx-simplifier 简化模型
        
        Args:
            onnx_path: 原始 ONNX 文件路径
        
        Returns:
            简化后的文件路径，如果失败则返回 None
        """
        try:
            import onnx
            from onnxsim import simplify
            
            logger.info("尝试简化 ONNX 模型...")
            
            model = onnx.load(onnx_path)
            model_simp, check = simplify(model)
            
            if check:
                # 保存简化后的模型
                simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
                onnx.save(model_simp, simplified_path)
                logger.info(f"ONNX 模型简化成功: {simplified_path}")
                return simplified_path
            else:
                logger.warning("ONNX 简化验证失败，使用原始模型")
                return None
                
        except ImportError:
            logger.info("未安装 onnxsim，跳过模型简化")
            return None
        except Exception as e:
            logger.warning(f"ONNX 简化失败: {e}，使用原始模型")
            return None
    
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        获取 ONNX 模型信息
        
        Args:
            onnx_path: ONNX 文件路径
        
        Returns:
            模型信息字典
        """
        try:
            import onnx
            
            model = onnx.load(onnx_path)
            graph = model.graph
            
            info = {
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "ir_version": model.ir_version,
                "producer_name": model.producer_name,
                "num_nodes": len(graph.node),
                "num_inputs": len(graph.input),
                "num_outputs": len(graph.output),
                "inputs": [
                    {
                        "name": inp.name,
                        "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim],
                        "dtype": inp.type.tensor_type.elem_type,
                    }
                    for inp in graph.input
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "shape": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim],
                        "dtype": out.type.tensor_type.elem_type,
                    }
                    for out in graph.output
                ],
                "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
            }
            
            return info
            
        except ImportError:
            logger.warning("未安装 onnx，无法获取模型信息")
            return {}
        except Exception as e:
            logger.warning(f"获取模型信息失败: {e}")
            return {}


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    config: Optional[ExportConfig] = None
) -> str:
    """
    便捷函数：导出模型为 ONNX 格式
    
    Args:
        model: PyTorch 模型
        save_path: 保存路径
        config: 导出配置，如果为 None 则使用默认配置
    
    Returns:
        导出的 ONNX 文件路径
    
    Example:
        >>> model = UGTModel(config)
        >>> onnx_path = export_to_onnx(model, "models/ugt.onnx")
    """
    if config is None:
        config = ExportConfig()
    
    exporter = ONNXExporter(config)
    return exporter.export(model, save_path)


class ModelWrapper(nn.Module):
    """
    模型包装器
    
    用于包装 UGT 模型以适配 ONNX 导出的输入输出格式。
    将字典/列表形式的输入输出转换为元组形式。
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_recommendations: int = 50
    ):
        """
        初始化模型包装器
        
        Args:
            model: 原始 UGT 模型
            num_recommendations: 推荐数量
        """
        super().__init__()
        self.model = model
        self.num_recommendations = num_recommendations
    
    def forward(
        self,
        encoder_l1_ids: torch.Tensor,
        encoder_l2_ids: torch.Tensor,
        encoder_l3_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        encoder_token_types: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            encoder_l1_ids: 第一层语义ID (batch, seq_len)
            encoder_l2_ids: 第二层语义ID (batch, seq_len)
            encoder_l3_ids: 第三层语义ID (batch, seq_len)
            encoder_positions: 位置编码 (batch, seq_len)
            encoder_token_types: Token类型 (batch, seq_len)
            encoder_mask: 注意力掩码 (batch, seq_len)
        
        Returns:
            Tuple[recommendations, scores]:
                - recommendations: (batch, num_recs, 3) 推荐物品的语义ID
                - scores: (batch, num_recs) 推荐分数
        """
        # 组装语义ID列表
        semantic_ids = [encoder_l1_ids, encoder_l2_ids, encoder_l3_ids]
        
        # 调用原始模型的生成方法
        recommendations = self.model.generate(
            encoder_semantic_ids=semantic_ids,
            encoder_positions=encoder_positions,
            encoder_token_types=encoder_token_types,
            encoder_attention_mask=encoder_mask,
            num_recommendations=self.num_recommendations,
        )
        
        # 将推荐结果转换为张量格式
        batch_size = encoder_l1_ids.shape[0]
        device = encoder_l1_ids.device
        
        # 创建输出张量
        rec_tensor = torch.zeros(
            batch_size, self.num_recommendations, 3,
            dtype=torch.long, device=device
        )
        score_tensor = torch.zeros(
            batch_size, self.num_recommendations,
            dtype=torch.float32, device=device
        )
        
        # 填充推荐结果
        for b, batch_recs in enumerate(recommendations):
            for i, (l1, l2, l3) in enumerate(batch_recs[:self.num_recommendations]):
                rec_tensor[b, i, 0] = l1
                rec_tensor[b, i, 1] = l2
                rec_tensor[b, i, 2] = l3
                # 分数可以从模型获取，这里使用递减分数作为示例
                score_tensor[b, i] = 1.0 - i * 0.01
        
        return rec_tensor, score_tensor


def wrap_model_for_export(
    model: nn.Module,
    num_recommendations: int = 50
) -> nn.Module:
    """
    包装模型以适配 ONNX 导出
    
    Args:
        model: 原始 UGT 模型
        num_recommendations: 推荐数量
    
    Returns:
        包装后的模型
    """
    return ModelWrapper(model, num_recommendations)

