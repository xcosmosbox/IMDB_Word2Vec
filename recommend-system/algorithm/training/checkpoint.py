"""
检查点管理模块

实现训练检查点的保存、加载和管理：
- 自动删除旧检查点
- 保留最佳模型
- 支持断点续训

对应架构文档: 第八章 训练流程
"""

import os
import shutil
import json
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime

import torch


logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """检查点信息"""
    path: str
    step: int
    epoch: int
    loss: float
    timestamp: str
    is_best: bool = False


class CheckpointManager:
    """
    检查点管理器
    
    功能：
    - 保存和加载检查点
    - 自动管理检查点数量
    - 跟踪最佳模型
    - 支持分布式训练
    """
    
    CHECKPOINT_FILE = "checkpoint.pt"
    METADATA_FILE = "metadata.json"
    LATEST_LINK = "latest"
    BEST_LINK = "best"
    
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 3,
        keep_best: bool = True,
    ):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 检查点保存目录
            max_checkpoints: 最多保留的检查点数量
            keep_best: 是否额外保留最佳检查点
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查点历史记录
        self.checkpoint_history: List[CheckpointInfo] = []
        self._load_history()
    
    def save(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        step: int,
        epoch: int = 0,
        loss: float = 0.0,
        is_best: bool = False,
    ) -> str:
        """
        保存检查点
        
        Args:
            checkpoint: 检查点字典
            path: 保存路径
            step: 当前步数
            epoch: 当前轮数
            loss: 当前损失
            is_best: 是否是最佳模型
        
        Returns:
            保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存检查点文件
        checkpoint_path = os.path.join(path, self.CHECKPOINT_FILE)
        torch.save(checkpoint, checkpoint_path)
        
        # 保存元数据
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best,
        }
        metadata_path = os.path.join(path, self.METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 记录检查点信息
        info = CheckpointInfo(
            path=path,
            step=step,
            epoch=epoch,
            loss=loss,
            timestamp=metadata["timestamp"],
            is_best=is_best,
        )
        self.checkpoint_history.append(info)
        
        # 更新最新链接
        self._update_link(path, self.LATEST_LINK)
        
        # 如果是最佳模型，更新最佳链接
        if is_best:
            self._update_link(path, self.BEST_LINK)
            # 标记之前的最佳为非最佳
            for ckpt in self.checkpoint_history[:-1]:
                ckpt.is_best = False
        
        # 清理旧检查点
        self._cleanup_checkpoints()
        
        # 保存历史记录
        self._save_history()
        
        logger.info(f"检查点已保存: {path} (step={step}, loss={loss:.4f})")
        
        return path
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径或链接名（如 "latest", "best"）
        
        Returns:
            检查点字典
        """
        # 处理链接
        if path == self.LATEST_LINK or path == self.BEST_LINK:
            link_path = os.path.join(self.save_dir, path)
            if os.path.islink(link_path):
                path = os.readlink(link_path)
            elif os.path.isdir(link_path):
                pass
            else:
                raise FileNotFoundError(f"检查点链接不存在: {link_path}")
        
        checkpoint_path = os.path.join(path, self.CHECKPOINT_FILE)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        logger.info(f"检查点已加载: {path}")
        
        return checkpoint
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """加载最新检查点"""
        latest_path = os.path.join(self.save_dir, self.LATEST_LINK)
        if os.path.exists(latest_path):
            return self.load(latest_path)
        return None
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        best_path = os.path.join(self.save_dir, self.BEST_LINK)
        if os.path.exists(best_path):
            return self.load(best_path)
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """获取最新检查点路径"""
        if self.checkpoint_history:
            return self.checkpoint_history[-1].path
        return None
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """获取最佳检查点路径"""
        for ckpt in reversed(self.checkpoint_history):
            if ckpt.is_best:
                return ckpt.path
        return None
    
    def _update_link(self, target: str, link_name: str) -> None:
        """更新符号链接"""
        link_path = os.path.join(self.save_dir, link_name)
        
        # 删除旧链接
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            shutil.rmtree(link_path)
        
        # 在 Windows 上使用目录复制代替符号链接
        try:
            os.symlink(os.path.abspath(target), link_path)
        except OSError:
            # Windows 可能不支持符号链接
            shutil.copytree(target, link_path)
    
    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # 保留最佳检查点
        best_path = self.get_best_checkpoint_path()
        
        # 按时间排序，删除最旧的
        checkpoints_to_remove = []
        non_best_checkpoints = [
            ckpt for ckpt in self.checkpoint_history 
            if not ckpt.is_best
        ]
        
        while len(non_best_checkpoints) > self.max_checkpoints - (1 if self.keep_best else 0):
            oldest = non_best_checkpoints.pop(0)
            checkpoints_to_remove.append(oldest)
        
        # 删除检查点
        for ckpt in checkpoints_to_remove:
            if os.path.exists(ckpt.path) and ckpt.path != best_path:
                try:
                    shutil.rmtree(ckpt.path)
                    logger.info(f"已删除旧检查点: {ckpt.path}")
                except Exception as e:
                    logger.warning(f"删除检查点失败: {ckpt.path}, 错误: {e}")
            
            self.checkpoint_history.remove(ckpt)
    
    def _save_history(self) -> None:
        """保存检查点历史"""
        history_path = os.path.join(self.save_dir, "checkpoint_history.json")
        history = [
            {
                "path": ckpt.path,
                "step": ckpt.step,
                "epoch": ckpt.epoch,
                "loss": ckpt.loss,
                "timestamp": ckpt.timestamp,
                "is_best": ckpt.is_best,
            }
            for ckpt in self.checkpoint_history
        ]
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _load_history(self) -> None:
        """加载检查点历史"""
        history_path = os.path.join(self.save_dir, "checkpoint_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            self.checkpoint_history = [
                CheckpointInfo(
                    path=ckpt["path"],
                    step=ckpt["step"],
                    epoch=ckpt.get("epoch", 0),
                    loss=ckpt.get("loss", 0.0),
                    timestamp=ckpt.get("timestamp", ""),
                    is_best=ckpt.get("is_best", False),
                )
                for ckpt in history
            ]


def save_model_for_inference(
    model: torch.nn.Module,
    save_path: str,
    config: Optional[Dict] = None,
) -> str:
    """
    保存用于推理的模型
    
    只保存模型权重，不包含优化器状态等训练信息
    
    Args:
        model: 模型
        save_path: 保存路径
        config: 模型配置（可选）
    
    Returns:
        保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(save_path, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # 保存配置
    if config is not None:
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"推理模型已保存: {save_path}")
    
    return save_path


def load_model_for_inference(
    model_class: type,
    load_path: str,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    加载用于推理的模型
    
    Args:
        model_class: 模型类
        load_path: 加载路径
        device: 设备
    
    Returns:
        模型实例
    """
    # 加载配置
    config_path = os.path.join(load_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = model_class(**config)
    else:
        model = model_class()
    
    # 加载权重
    model_path = os.path.join(load_path, "model.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"推理模型已加载: {load_path}")
    
    return model

