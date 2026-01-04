"""
Triton Inference Server 配置生成模块

自动生成 Triton 模型仓库结构和配置文件。

特性：
1. 自动生成 config.pbtxt
2. 支持动态批处理配置
3. 支持多 GPU 实例配置
4. 生成完整的模型仓库目录结构

Author: Person F (MLOps Engineer)
"""

import os
import shutil
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import ExportConfig, TritonConfig, ModelInputSpec

# 配置日志
logger = logging.getLogger(__name__)


class TritonConfigGenerator:
    """
    Triton 配置生成器
    
    负责生成 Triton Inference Server 所需的配置文件和目录结构。
    
    Attributes:
        export_config: 导出配置
        triton_config: Triton 服务配置
        input_spec: 模型输入规格
    
    Example:
        >>> generator = TritonConfigGenerator(export_config, triton_config)
        >>> generator.generate("./model_repository", "model.plan")
    """
    
    def __init__(
        self,
        export_config: ExportConfig,
        triton_config: Optional[TritonConfig] = None
    ):
        """
        初始化配置生成器
        
        Args:
            export_config: 导出配置
            triton_config: Triton 配置，如果为 None 则使用默认配置
        """
        self.export_config = export_config
        self.triton_config = triton_config or TritonConfig()
        self.input_spec = ModelInputSpec()
    
    def generate(
        self,
        model_repository: str,
        model_file: Optional[str] = None,
        version: int = 1
    ) -> str:
        """
        生成完整的 Triton 模型仓库
        
        Args:
            model_repository: 模型仓库根目录
            model_file: 模型文件路径（TensorRT 引擎或 ONNX）
            version: 模型版本号
        
        Returns:
            生成的 config.pbtxt 文件路径
        """
        model_name = self.export_config.model_name
        model_dir = Path(model_repository) / model_name
        version_dir = model_dir / str(version)
        
        # 创建目录结构
        version_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建模型仓库目录: {model_dir}")
        
        # 复制模型文件
        if model_file and os.path.exists(model_file):
            if model_file.endswith(".plan"):
                dest_file = version_dir / "model.plan"
            elif model_file.endswith(".onnx"):
                dest_file = version_dir / "model.onnx"
            else:
                dest_file = version_dir / Path(model_file).name
            
            shutil.copy2(model_file, dest_file)
            logger.info(f"复制模型文件: {model_file} -> {dest_file}")
        
        # 生成 config.pbtxt
        config_content = self._generate_config_pbtxt()
        config_path = model_dir / "config.pbtxt"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"生成 Triton 配置文件: {config_path}")
        
        return str(config_path)
    
    def _generate_config_pbtxt(self) -> str:
        """
        生成 config.pbtxt 配置文件内容
        
        Returns:
            配置文件内容字符串
        """
        config_parts = []
        
        # 基本配置
        config_parts.append(f'name: "{self.export_config.model_name}"')
        config_parts.append(f'platform: "{self.triton_config.platform}"')
        config_parts.append(f'max_batch_size: {self.export_config.max_batch_size}')
        config_parts.append('')
        
        # 输入配置
        config_parts.append('input [')
        inputs = self._generate_input_configs()
        config_parts.append(',\n'.join(inputs))
        config_parts.append(']')
        config_parts.append('')
        
        # 输出配置
        config_parts.append('output [')
        outputs = self._generate_output_configs()
        config_parts.append(',\n'.join(outputs))
        config_parts.append(']')
        config_parts.append('')
        
        # 动态批处理配置
        if self.triton_config.enable_dynamic_batching:
            config_parts.append('dynamic_batching {')
            batch_sizes = ', '.join(str(s) for s in self.triton_config.preferred_batch_sizes)
            config_parts.append(f'  preferred_batch_size: [ {batch_sizes} ]')
            config_parts.append(f'  max_queue_delay_microseconds: {self.triton_config.max_queue_delay_us}')
            config_parts.append('}')
            config_parts.append('')
        
        # 实例组配置
        config_parts.append('instance_group [')
        gpus = ', '.join(str(g) for g in self.triton_config.gpus)
        config_parts.append('  {')
        config_parts.append(f'    count: {self.triton_config.instance_count}')
        config_parts.append(f'    kind: {self.triton_config.instance_kind}')
        config_parts.append(f'    gpus: [ {gpus} ]')
        config_parts.append('  }')
        config_parts.append(']')
        config_parts.append('')
        
        # 响应缓存配置
        if self.triton_config.response_cache_byte_size > 0:
            config_parts.append('response_cache {')
            config_parts.append(f'  enable: true')
            config_parts.append('}')
            config_parts.append('')
        
        # 版本策略
        config_parts.append('version_policy {')
        config_parts.append('  latest {')
        config_parts.append('    num_versions: 2')
        config_parts.append('  }')
        config_parts.append('}')
        
        return '\n'.join(config_parts)
    
    def _generate_input_configs(self) -> List[str]:
        """生成输入张量配置"""
        inputs = []
        
        # 语义 ID 输入（INT64）
        for name in [
            self.input_spec.encoder_l1_ids,
            self.input_spec.encoder_l2_ids,
            self.input_spec.encoder_l3_ids,
            self.input_spec.encoder_positions,
            self.input_spec.encoder_token_types,
        ]:
            inputs.append(
                f'  {{ name: "{name}", data_type: TYPE_INT64, dims: [ -1 ] }}'
            )
        
        # 掩码输入（FP32）
        inputs.append(
            f'  {{ name: "{self.input_spec.encoder_mask}", data_type: TYPE_FP32, dims: [ -1 ] }}'
        )
        
        return inputs
    
    def _generate_output_configs(self) -> List[str]:
        """生成输出张量配置"""
        num_recs = self.export_config.num_recommendations
        
        outputs = [
            f'  {{ name: "{self.input_spec.recommendations}", '
            f'data_type: TYPE_INT64, dims: [ {num_recs}, 3 ] }}',
            f'  {{ name: "{self.input_spec.scores}", '
            f'data_type: TYPE_FP32, dims: [ {num_recs} ] }}'
        ]
        
        return outputs
    
    def generate_ensemble_config(
        self,
        model_repository: str,
        encoder_name: str = "ugt_encoder",
        decoder_name: str = "ugt_decoder"
    ) -> str:
        """
        生成 Ensemble 模型配置（编码器 + 解码器分离部署）
        
        Args:
            model_repository: 模型仓库路径
            encoder_name: 编码器模型名称
            decoder_name: 解码器模型名称
        
        Returns:
            Ensemble 配置文件路径
        """
        ensemble_name = f"{self.export_config.model_name}_ensemble"
        ensemble_dir = Path(model_repository) / ensemble_name
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        config_content = self._generate_ensemble_pbtxt(encoder_name, decoder_name)
        config_path = ensemble_dir / "config.pbtxt"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"生成 Ensemble 配置文件: {config_path}")
        return str(config_path)
    
    def _generate_ensemble_pbtxt(
        self,
        encoder_name: str,
        decoder_name: str
    ) -> str:
        """生成 Ensemble 配置内容"""
        ensemble_name = f"{self.export_config.model_name}_ensemble"
        
        config = f'''name: "{ensemble_name}"
platform: "ensemble"
max_batch_size: {self.export_config.max_batch_size}

input [
  {{ name: "encoder_l1_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_l2_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_l3_ids", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_positions", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_token_types", data_type: TYPE_INT64, dims: [ -1 ] }},
  {{ name: "encoder_mask", data_type: TYPE_FP32, dims: [ -1 ] }}
]

output [
  {{ name: "recommendations", data_type: TYPE_INT64, dims: [ {self.export_config.num_recommendations}, 3 ] }},
  {{ name: "scores", data_type: TYPE_FP32, dims: [ {self.export_config.num_recommendations} ] }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "{encoder_name}"
      model_version: -1
      input_map {{
        key: "encoder_l1_ids"
        value: "encoder_l1_ids"
      }}
      input_map {{
        key: "encoder_l2_ids"
        value: "encoder_l2_ids"
      }}
      input_map {{
        key: "encoder_l3_ids"
        value: "encoder_l3_ids"
      }}
      input_map {{
        key: "encoder_positions"
        value: "encoder_positions"
      }}
      input_map {{
        key: "encoder_token_types"
        value: "encoder_token_types"
      }}
      input_map {{
        key: "encoder_mask"
        value: "encoder_mask"
      }}
      output_map {{
        key: "encoder_output"
        value: "encoder_hidden"
      }}
    }},
    {{
      model_name: "{decoder_name}"
      model_version: -1
      input_map {{
        key: "encoder_hidden"
        value: "encoder_hidden"
      }}
      output_map {{
        key: "recommendations"
        value: "recommendations"
      }}
      output_map {{
        key: "scores"
        value: "scores"
      }}
    }}
  ]
}}
'''
        return config


def generate_triton_config(
    model_name: str,
    config: Optional[ExportConfig] = None,
    triton_config: Optional[TritonConfig] = None
) -> str:
    """
    便捷函数：生成 Triton 配置内容
    
    Args:
        model_name: 模型名称
        config: 导出配置
        triton_config: Triton 配置
    
    Returns:
        配置文件内容字符串
    
    Example:
        >>> config_str = generate_triton_config("ugt_recommend")
        >>> print(config_str)
    """
    if config is None:
        config = ExportConfig(model_name=model_name)
    
    generator = TritonConfigGenerator(config, triton_config)
    return generator._generate_config_pbtxt()


def setup_model_repository(
    model_repository: str,
    model_file: str,
    export_config: Optional[ExportConfig] = None,
    triton_config: Optional[TritonConfig] = None
) -> Dict[str, str]:
    """
    便捷函数：设置完整的模型仓库
    
    Args:
        model_repository: 模型仓库根目录
        model_file: 模型文件路径
        export_config: 导出配置
        triton_config: Triton 配置
    
    Returns:
        包含生成文件路径的字典
    
    Example:
        >>> paths = setup_model_repository(
        ...     "./model_repo",
        ...     "./models/ugt.plan"
        ... )
    """
    if export_config is None:
        export_config = ExportConfig()
    
    generator = TritonConfigGenerator(export_config, triton_config)
    config_path = generator.generate(model_repository, model_file)
    
    model_name = export_config.model_name
    model_dir = Path(model_repository) / model_name
    
    return {
        "config_path": config_path,
        "model_dir": str(model_dir),
        "model_repository": model_repository,
    }


class TritonModelValidator:
    """
    Triton 模型配置验证器
    
    验证模型仓库配置的正确性和完整性。
    """
    
    def __init__(self, model_repository: str):
        """
        初始化验证器
        
        Args:
            model_repository: 模型仓库根目录
        """
        self.model_repository = Path(model_repository)
    
    def validate(self, model_name: str) -> Dict[str, Any]:
        """
        验证模型配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            验证结果字典
        """
        model_dir = self.model_repository / model_name
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
        }
        
        # 检查模型目录
        if not model_dir.exists():
            results["valid"] = False
            results["errors"].append(f"模型目录不存在: {model_dir}")
            return results
        
        # 检查配置文件
        config_path = model_dir / "config.pbtxt"
        if not config_path.exists():
            results["valid"] = False
            results["errors"].append(f"配置文件不存在: {config_path}")
        else:
            results["info"]["config_path"] = str(config_path)
        
        # 检查版本目录
        version_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not version_dirs:
            results["valid"] = False
            results["errors"].append("没有找到版本目录")
        else:
            results["info"]["versions"] = [d.name for d in version_dirs]
            
            # 检查每个版本的模型文件
            for version_dir in version_dirs:
                model_files = list(version_dir.glob("model.*"))
                if not model_files:
                    results["warnings"].append(
                        f"版本 {version_dir.name} 中没有找到模型文件"
                    )
                else:
                    results["info"][f"version_{version_dir.name}_files"] = [
                        f.name for f in model_files
                    ]
        
        return results
    
    def validate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        验证仓库中的所有模型
        
        Returns:
            所有模型的验证结果
        """
        results = {}
        
        if not self.model_repository.exists():
            return {"error": f"模型仓库不存在: {self.model_repository}"}
        
        for model_dir in self.model_repository.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                results[model_dir.name] = self.validate(model_dir.name)
        
        return results

