"""
ONNX 推理模块
=============

封装 ONNX Runtime 推理功能，支持从 word2vec.onnx 模型获取嵌入向量。

ONNX 模型说明:
- 输入: token_ids (int64, shape: [batch_size])
- 输出: embeddings (float32, shape: [batch_size, 128])

使用方法:
    # 获取单个 Token 的嵌入向量
    embedding = get_embedding("MOV_tt0111161")
    
    # 批量获取
    model = OnnxEmbeddingModel()
    embeddings = model.get_embeddings(["MOV_tt0111161", "ACT_nm0000001"])
"""
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import streamlit as st

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DataFiles


# =============================================================================
# ONNX 模型类
# =============================================================================

class OnnxEmbeddingModel:
    """
    ONNX 嵌入模型封装类
    
    使用 ONNX Runtime 加载 word2vec.onnx 模型，
    提供 Token 到嵌入向量的推理功能。
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        初始化模型
        
        Args:
            model_path: ONNX 模型文件路径，默认使用配置中的路径
        """
        self.model_path = model_path or DataFiles.WORD2VEC_ONNX
        self._session = None
        self._token_to_id = None
        self._id_to_token = None
        
    @property
    def session(self):
        """
        延迟加载 ONNX Session
        
        使用 @property 实现延迟加载，避免在导入时就加载模型。
        """
        if self._session is None:
            self._session = self._load_model()
        return self._session
    
    @property
    def token_to_id(self):
        """延迟加载 Token→ID 映射"""
        if self._token_to_id is None:
            from .data_loader import load_token_to_id
            self._token_to_id = load_token_to_id()
        return self._token_to_id
    
    @property
    def id_to_token(self):
        """延迟加载 ID→Token 映射"""
        if self._id_to_token is None:
            from .data_loader import load_id_to_token
            self._id_to_token = load_id_to_token()
        return self._id_to_token
    
    def _load_model(self):
        """
        加载 ONNX 模型
        
        Returns:
            onnxruntime.InferenceSession 对象
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            st.error(f"请安装 onnxruntime: pip install onnxruntime (错误: {e})")
            return None
        
        if not self.model_path.exists():
            st.error(f"ONNX 模型文件不存在: {self.model_path}")
            return None
        
        try:
            # 创建推理会话
            # 使用 CPU 执行，确保兼容性
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                str(self.model_path),
                session_options,
                providers=["CPUExecutionProvider"],
            )
            
            return session
        except Exception as e:
            st.error(f"加载 ONNX 模型失败: {e}")
            return None
    
    def get_embedding_by_id(self, token_id: int) -> Optional[np.ndarray]:
        """
        根据 ID 获取嵌入向量
        
        Args:
            token_id: Token 的数字 ID
            
        Returns:
            128维嵌入向量，如果失败返回 None
        """
        if self.session is None:
            return None
        
        try:
            # 准备输入
            input_ids = np.array([token_id], dtype=np.int64)
            
            # 执行推理
            outputs = self.session.run(
                None,
                {"token_ids": input_ids}
            )
            
            # 返回嵌入向量
            return outputs[0][0]
        
        except Exception as e:
            st.warning(f"ONNX 推理失败: {e}")
            return None
    
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """
        根据 Token 获取嵌入向量
        
        Args:
            token: Token 字符串，如 "MOV_tt0111161"
            
        Returns:
            128维嵌入向量，如果 Token 不存在返回 None
        """
        if token not in self.token_to_id:
            return None
        
        token_id = self.token_to_id[token]
        return self.get_embedding_by_id(token_id)
    
    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        """
        批量获取多个 Token 的嵌入向量
        
        Args:
            tokens: Token 列表
            
        Returns:
            shape (n_tokens, 128) 的嵌入矩阵
        """
        if self.session is None:
            return np.array([])
        
        # 转换为 ID
        token_ids = []
        valid_indices = []
        
        for i, token in enumerate(tokens):
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
                valid_indices.append(i)
        
        if not token_ids:
            return np.array([])
        
        try:
            # 准备输入
            input_ids = np.array(token_ids, dtype=np.int64)
            
            # 执行推理
            outputs = self.session.run(
                None,
                {"token_ids": input_ids}
            )
            
            return outputs[0]
        
        except Exception as e:
            st.warning(f"批量 ONNX 推理失败: {e}")
            return np.array([])
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._session is not None
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型输入输出信息的字典
        """
        if self.session is None:
            return {}
        
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "inputs": [
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type,
                }
                for inp in inputs
            ],
            "outputs": [
                {
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type,
                }
                for out in outputs
            ],
            "model_path": str(self.model_path),
            "model_size_mb": round(self.model_path.stat().st_size / (1024**2), 2),
        }


# =============================================================================
# 便捷函数
# =============================================================================

# 全局模型实例（延迟初始化）
_model_instance: Optional[OnnxEmbeddingModel] = None


def get_model() -> OnnxEmbeddingModel:
    """
    获取全局模型实例
    
    使用单例模式，确保模型只加载一次。
    
    Returns:
        OnnxEmbeddingModel 实例
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = OnnxEmbeddingModel()
    
    return _model_instance


def get_embedding(token: str) -> Optional[np.ndarray]:
    """
    获取单个 Token 的嵌入向量（便捷函数）
    
    Args:
        token: Token 字符串
        
    Returns:
        128维嵌入向量
    """
    model = get_model()
    return model.get_embedding(token)


def get_embeddings_batch(tokens: List[str]) -> np.ndarray:
    """
    批量获取嵌入向量（便捷函数）
    
    Args:
        tokens: Token 列表
        
    Returns:
        嵌入矩阵
    """
    model = get_model()
    return model.get_embeddings(tokens)


# =============================================================================
# 相似度搜索（结合 ONNX 推理）
# =============================================================================

def find_similar_with_onnx(
    query_token: str,
    k: int = 10,
    entity_type_filter: Optional[str] = None,
) -> List[dict]:
    """
    使用 ONNX 推理 + 相似度搜索找到相似 Token
    
    这个函数结合了 ONNX 推理和相似度计算，
    适用于需要实时推理的场景。
    
    Args:
        query_token: 查询 Token
        k: 返回数量
        entity_type_filter: 实体类型过滤
        
    Returns:
        相似结果列表
    """
    from .data_loader import load_embeddings_npy, load_token_to_id, load_id_to_token
    from .similarity import find_top_k_similar
    
    model = get_model()
    
    # 获取查询向量
    query_vec = model.get_embedding(query_token)
    if query_vec is None:
        return []
    
    # 加载所有嵌入和 Token
    embeddings = load_embeddings_npy()
    id_to_token = load_id_to_token()
    tokens = [id_to_token.get(i, f"<UNK_{i}>") for i in range(len(embeddings))]
    
    # 执行相似度搜索
    results = find_top_k_similar(
        query_vec,
        embeddings,
        tokens,
        k=k,
        exclude_self=True,
        query_token=query_token,
        entity_type_filter=entity_type_filter,
    )
    
    return results

