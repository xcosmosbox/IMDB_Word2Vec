"""
训练模块完整单元测试

测试内容：
- 配置类测试
- 数据集测试
- 损失函数测试
- 优化器测试
- 调度器测试
- 训练器测试
- 检查点管理测试
- 评估指标测试
- 分布式训练测试

运行方法:
    pytest test_training.py -v
"""

import os
import sys
import tempfile
import json
import shutil
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Mock 类和辅助函数
# ============================================================================

class MockModel(nn.Module):
    """Mock 模型用于测试"""
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(100000, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.l1_head = nn.Linear(d_model, 1024)
        self.l2_head = nn.Linear(d_model, 4096)
        self.l3_head = nn.Linear(d_model, 16384)
    
    def forward(
        self,
        encoder_semantic_ids=None,
        encoder_positions=None,
        encoder_token_types=None,
        encoder_attention_mask=None,
        decoder_semantic_ids=None,
        decoder_positions=None,
        decoder_token_types=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 获取批次信息
        if encoder_semantic_ids is not None:
            batch_size = encoder_semantic_ids[0].shape[0]
            device = encoder_semantic_ids[0].device
        else:
            batch_size = 4
            device = torch.device("cpu")
        
        # 确定序列长度
        if decoder_semantic_ids is not None:
            seq_len = decoder_semantic_ids[0].shape[1]
        elif encoder_semantic_ids is not None:
            seq_len = encoder_semantic_ids[0].shape[1]
        else:
            seq_len = 32
        
        # 生成随机输出
        hidden = torch.randn(batch_size, seq_len, 512, device=device)
        
        return {
            "l1_logits": self.l1_head(hidden),
            "l2_logits": self.l2_head(hidden),
            "l3_logits": self.l3_head(hidden),
            "aux_loss": torch.tensor(0.1, device=device),
            "encoder_output": hidden,
            "user_repr": hidden[:, 0, :],
            "item_repr": hidden[:, -1, :],
        }


class MockDataset(Dataset):
    """Mock 数据集用于测试"""
    
    def __init__(self, size: int = 100, seq_len: int = 32):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "encoder_semantic_ids": [
                torch.randint(0, 1024, (self.seq_len,)),
                torch.randint(0, 4096, (self.seq_len,)),
                torch.randint(0, 16384, (self.seq_len,)),
            ],
            "encoder_positions": torch.arange(self.seq_len),
            "encoder_token_types": torch.zeros(self.seq_len, dtype=torch.long),
            "encoder_mask": torch.ones(self.seq_len, dtype=torch.long),
            "decoder_semantic_ids": [
                torch.randint(0, 1024, (self.seq_len,)),
                torch.randint(0, 4096, (self.seq_len,)),
                torch.randint(0, 16384, (self.seq_len,)),
            ],
            "decoder_positions": torch.arange(self.seq_len),
            "decoder_token_types": torch.ones(self.seq_len, dtype=torch.long),
            "decoder_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": [
                torch.randint(0, 1024, (self.seq_len,)),
                torch.randint(0, 4096, (self.seq_len,)),
                torch.randint(0, 16384, (self.seq_len,)),
            ],
        }


class MockPreferenceDataset(Dataset):
    """Mock 偏好数据集用于测试"""
    
    def __init__(self, size: int = 100, seq_len: int = 32):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "encoder_semantic_ids": [
                torch.randint(0, 1024, (self.seq_len,)),
                torch.randint(0, 4096, (self.seq_len,)),
                torch.randint(0, 16384, (self.seq_len,)),
            ],
            "encoder_positions": torch.arange(self.seq_len),
            "encoder_token_types": torch.zeros(self.seq_len, dtype=torch.long),
            "encoder_mask": torch.ones(self.seq_len, dtype=torch.long),
            "chosen_ids": torch.tensor([
                torch.randint(0, 1024, (1,)).item(),
                torch.randint(0, 4096, (1,)).item(),
                torch.randint(0, 16384, (1,)).item(),
            ]),
            "rejected_ids": torch.tensor([
                torch.randint(0, 1024, (1,)).item(),
                torch.randint(0, 4096, (1,)).item(),
                torch.randint(0, 16384, (1,)).item(),
            ]),
        }


# ============================================================================
# 配置类测试
# ============================================================================

class TestConfig:
    """测试配置类"""
    
    def test_training_config_default(self):
        """测试默认训练配置"""
        from config import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.batch_size == 256
        assert config.learning_rate == 1e-4
        assert config.max_epochs == 10
        assert config.warmup_steps == 10000
        assert config.fp16 == True
    
    def test_stage1_config(self):
        """测试阶段 1 配置"""
        from config import Stage1Config
        
        config = Stage1Config()
        
        assert config.lambda_contrastive == 0.0
        assert config.lambda_preference == 0.0
        assert config.max_epochs == 5
    
    def test_stage2_config(self):
        """测试阶段 2 配置"""
        from config import Stage2Config
        
        config = Stage2Config()
        
        assert config.lambda_contrastive == 0.1
        assert config.lambda_preference == 0.0
        assert config.max_epochs == 3
        assert config.learning_rate == 5e-5
    
    def test_stage3_config(self):
        """测试阶段 3 配置"""
        from config import Stage3Config
        
        config = Stage3Config()
        
        assert config.lambda_contrastive == 0.1
        assert config.lambda_preference == 0.1
        assert config.dpo_beta == 0.1
        assert config.max_epochs == 2
    
    def test_config_validation(self):
        """测试配置验证"""
        from config import TrainingConfig
        
        # 测试无效的层次化损失权重
        with pytest.raises(ValueError):
            config = TrainingConfig(
                l1_loss_weight=0.5,
                l2_loss_weight=0.5,
                l3_loss_weight=0.5,  # 总和不为 1
            )
        
        # 测试 FP16 和 BF16 冲突
        with pytest.raises(ValueError):
            config = TrainingConfig(fp16=True, bf16=True)
    
    def test_effective_batch_size(self):
        """测试有效批次大小计算"""
        from config import TrainingConfig
        
        config = TrainingConfig(
            batch_size=64,
            gradient_accumulation_steps=4,
        )
        
        assert config.effective_batch_size == 256
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        from config import TrainingConfig
        
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "batch_size" in config_dict
        assert "learning_rate" in config_dict


# ============================================================================
# 数据集测试
# ============================================================================

class TestDataset:
    """测试数据集类"""
    
    def test_recommend_dataset_from_jsonl(self):
        """测试从 JSONL 文件加载数据集"""
        from dataset import RecommendDataset
        
        # 创建临时数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "encoder_l1_ids": [1, 2, 3],
                "encoder_l2_ids": [10, 20, 30],
                "encoder_l3_ids": [100, 200, 300],
                "encoder_positions": [0, 1, 2],
                "encoder_token_types": [0, 1, 1],
                "encoder_mask": [1, 1, 1],
                "decoder_l1_ids": [4, 5],
                "decoder_l2_ids": [40, 50],
                "decoder_l3_ids": [400, 500],
                "decoder_positions": [0, 1],
                "decoder_token_types": [1, 1],
                "decoder_mask": [1, 1],
                "labels_l1": [5, 6],
                "labels_l2": [50, 60],
                "labels_l3": [500, 600],
            }
            f.write(json.dumps(sample) + '\n')
            f.write(json.dumps(sample) + '\n')
            temp_path = f.name
        
        try:
            dataset = RecommendDataset(
                data_path=temp_path,
                max_encoder_length=10,
                max_decoder_length=5,
            )
            
            assert len(dataset) == 2
            
            item = dataset[0]
            assert "encoder_semantic_ids" in item
            assert "decoder_semantic_ids" in item
            assert "labels" in item
            assert len(item["encoder_semantic_ids"]) == 3
        finally:
            os.unlink(temp_path)
    
    def test_data_collator(self):
        """测试数据整理器"""
        from dataset import DataCollator
        
        collator = DataCollator()
        dataset = MockDataset(size=4)
        
        batch = [dataset[i] for i in range(4)]
        collated = collator(batch)
        
        assert "encoder_semantic_ids" in collated
        assert len(collated["encoder_semantic_ids"]) == 3
        assert collated["encoder_semantic_ids"][0].shape[0] == 4  # batch_size
    
    def test_preference_dataset(self):
        """测试偏好数据集"""
        dataset = MockPreferenceDataset(size=10)
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert "chosen_ids" in item
        assert "rejected_ids" in item
        assert item["chosen_ids"].shape == (3,)


# ============================================================================
# 损失函数测试
# ============================================================================

class TestLoss:
    """测试损失函数"""
    
    def test_ntp_loss(self):
        """测试 NTP 损失"""
        from loss import NextTokenPredictionLoss
        
        loss_fn = NextTokenPredictionLoss()
        
        batch_size, seq_len = 4, 16
        l1_logits = torch.randn(batch_size, seq_len, 1024)
        l2_logits = torch.randn(batch_size, seq_len, 4096)
        l3_logits = torch.randn(batch_size, seq_len, 16384)
        labels_l1 = torch.randint(0, 1024, (batch_size, seq_len))
        labels_l2 = torch.randint(0, 4096, (batch_size, seq_len))
        labels_l3 = torch.randint(0, 16384, (batch_size, seq_len))
        
        loss, metrics = loss_fn(
            l1_logits, l2_logits, l3_logits,
            labels_l1, labels_l2, labels_l3,
        )
        
        assert loss.shape == ()
        assert loss > 0
        assert "ntp_loss" in metrics
        assert "ntp_l1_loss" in metrics
        assert "ntp_l1_acc" in metrics
    
    def test_contrastive_loss(self):
        """测试对比学习损失"""
        from loss import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        batch_size = 8
        user_repr = torch.randn(batch_size, 512)
        item_repr = torch.randn(batch_size, 512)
        
        loss, metrics = loss_fn(user_repr, item_repr)
        
        assert loss.shape == ()
        assert loss > 0
        assert "contrastive_loss" in metrics
        assert "contrastive_u2i_acc" in metrics
    
    def test_dpo_loss(self):
        """测试 DPO 损失"""
        from loss import DPOLoss
        
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        
        batch_size = 8
        chosen_logps = torch.randn(batch_size)
        rejected_logps = torch.randn(batch_size)
        
        loss, metrics = loss_fn(chosen_logps, rejected_logps)
        
        assert loss.shape == ()
        assert "dpo_loss" in metrics
        assert "dpo_accuracy" in metrics
    
    def test_unified_loss(self):
        """测试统一损失"""
        from loss import UnifiedLoss
        
        loss_fn = UnifiedLoss(
            lambda_contrastive=0.1,
            lambda_preference=0.0,
            lambda_moe_balance=0.01,
        )
        
        batch_size, seq_len = 4, 16
        model_outputs = {
            "l1_logits": torch.randn(batch_size, seq_len, 1024),
            "l2_logits": torch.randn(batch_size, seq_len, 4096),
            "l3_logits": torch.randn(batch_size, seq_len, 16384),
            "user_repr": torch.randn(batch_size, 512),
            "item_repr": torch.randn(batch_size, 512),
        }
        labels = {
            "l1": torch.randint(0, 1024, (batch_size, seq_len)),
            "l2": torch.randint(0, 4096, (batch_size, seq_len)),
            "l3": torch.randint(0, 16384, (batch_size, seq_len)),
        }
        aux_loss = torch.tensor(0.1)
        
        losses = loss_fn(model_outputs, labels, aux_loss)
        
        assert "total_loss" in losses
        assert "ntp_loss" in losses
        assert "contrastive_loss" in losses
        assert losses["total_loss"] > 0


# ============================================================================
# 优化器测试
# ============================================================================

class TestOptimizer:
    """测试优化器"""
    
    def test_create_adamw_optimizer(self):
        """测试创建 AdamW 优化器"""
        from optimizer import create_optimizer
        
        model = MockModel()
        optimizer = create_optimizer(
            model=model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # decay 和 no_decay
    
    def test_optimizer_step(self):
        """测试优化器步骤"""
        from optimizer import create_optimizer
        
        model = MockModel()
        optimizer = create_optimizer(model=model, learning_rate=1e-3)
        
        # 前向传播
        output = model()
        loss = output["l1_logits"].mean()
        
        # 反向传播
        loss.backward()
        
        # 保存参数副本
        old_params = [p.clone() for p in model.parameters() if p.requires_grad]
        
        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
        
        # 验证参数已更新
        new_params = [p for p in model.parameters() if p.requires_grad]
        for old, new in zip(old_params, new_params):
            assert not torch.equal(old, new)


# ============================================================================
# 调度器测试
# ============================================================================

class TestScheduler:
    """测试学习率调度器"""
    
    def test_cosine_scheduler(self):
        """测试余弦调度器"""
        from scheduler import create_scheduler
        
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type="cosine",
            total_steps=1000,
            warmup_steps=100,
            min_lr_ratio=0.1,
        )
        
        # 测试预热阶段
        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr < 1e-3
        
        # 模拟训练步骤
        for _ in range(100):
            scheduler.step()
        
        # 预热结束后应该接近初始学习率
        warmup_end_lr = scheduler.get_last_lr()[0]
        assert warmup_end_lr > initial_lr
    
    def test_linear_scheduler(self):
        """测试线性调度器"""
        from scheduler import create_scheduler
        
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_type="linear",
            total_steps=1000,
            warmup_steps=100,
        )
        
        lrs = []
        for _ in range(1000):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        # 验证学习率在下降
        assert lrs[-1] < lrs[100]


# ============================================================================
# 检查点测试
# ============================================================================

class TestCheckpoint:
    """测试检查点管理"""
    
    def test_save_and_load_checkpoint(self):
        """测试保存和加载检查点"""
        from checkpoint import CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, max_checkpoints=3)
            
            # 创建模型和优化器
            model = MockModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            # 保存检查点
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": 100,
                "current_epoch": 1,
            }
            
            save_path = os.path.join(tmpdir, "test_checkpoint")
            manager.save(checkpoint, save_path, step=100)
            
            # 加载检查点
            loaded = manager.load(save_path)
            
            assert loaded["global_step"] == 100
            assert loaded["current_epoch"] == 1
    
    def test_checkpoint_cleanup(self):
        """测试检查点清理"""
        from checkpoint import CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, max_checkpoints=2)
            
            # 保存多个检查点
            for i in range(5):
                checkpoint = {"step": i}
                save_path = os.path.join(tmpdir, f"checkpoint_{i}")
                manager.save(checkpoint, save_path, step=i)
            
            # 验证只保留最新的 2 个
            assert len(manager.checkpoint_history) <= 2


# ============================================================================
# 评估指标测试
# ============================================================================

class TestMetrics:
    """测试评估指标"""
    
    def test_recall_at_k(self):
        """测试 Recall@K"""
        from metrics import recall_at_k
        
        predictions = [
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(4, 4, 4), (5, 5, 5), (6, 6, 6)],
        ]
        ground_truth = [(1, 1, 1), (5, 5, 5)]
        
        recall = recall_at_k(predictions, ground_truth, k=2)
        assert recall == 0.5
        
        recall = recall_at_k(predictions, ground_truth, k=3)
        assert recall == 1.0
    
    def test_ndcg_at_k(self):
        """测试 NDCG@K"""
        from metrics import ndcg_at_k
        import math
        
        predictions = [
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
        ]
        ground_truth = [(1, 1, 1)]
        
        ndcg = ndcg_at_k(predictions, ground_truth, k=3)
        expected = 1.0 / math.log2(2)  # 在位置 0
        assert abs(ndcg - expected) < 1e-6
    
    def test_mrr(self):
        """测试 MRR"""
        from metrics import mrr
        
        predictions = [
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(4, 4, 4), (5, 5, 5), (6, 6, 6)],
        ]
        ground_truth = [(2, 2, 2), (4, 4, 4)]
        
        mrr_value = mrr(predictions, ground_truth)
        expected = (1/2 + 1/1) / 2  # 第一个在位置 1，第二个在位置 0
        assert abs(mrr_value - expected) < 1e-6
    
    def test_metrics_calculator(self):
        """测试指标计算器"""
        from metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        predictions = [
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(4, 4, 4), (5, 5, 5), (6, 6, 6)],
        ]
        ground_truth = [(1, 1, 1), (5, 5, 5)]
        
        calculator.add_batch(predictions, ground_truth)
        metrics = calculator.compute(k_values=[2, 3])
        
        assert "recall@2" in metrics
        assert "recall@3" in metrics
        assert "ndcg@2" in metrics
        assert "mrr" in metrics


# ============================================================================
# 训练器测试
# ============================================================================

class TestTrainer:
    """测试训练器"""
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        from config import TrainingConfig
        from trainer import Trainer
        
        model = MockModel()
        config = TrainingConfig(
            batch_size=4,
            max_epochs=1,
            max_steps=10,
            logging_steps=5,
            save_steps=10,
            eval_steps=10,
            output_dir=tempfile.mkdtemp(),
        )
        train_dataset = MockDataset(size=20)
        
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
    
    def test_train_epoch(self):
        """测试训练一个 epoch"""
        from config import TrainingConfig
        from trainer import Trainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            config = TrainingConfig(
                batch_size=4,
                max_epochs=1,
                gradient_accumulation_steps=1,
                logging_steps=5,
                save_steps=100,
                eval_steps=100,
                fp16=False,
                output_dir=tmpdir,
            )
            train_dataset = MockDataset(size=20)
            
            trainer = Trainer(
                model=model,
                config=config,
                train_dataset=train_dataset,
            )
            
            metrics = trainer.train_epoch()
            
            assert "loss" in metrics
            assert "ntp_loss" in metrics
            assert metrics["loss"] > 0
    
    def test_save_load_checkpoint(self):
        """测试保存和加载检查点"""
        from config import TrainingConfig
        from trainer import Trainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            config = TrainingConfig(
                batch_size=4,
                max_epochs=1,
                fp16=False,
                output_dir=tmpdir,
            )
            train_dataset = MockDataset(size=20)
            
            trainer = Trainer(
                model=model,
                config=config,
                train_dataset=train_dataset,
            )
            
            # 保存检查点
            checkpoint_path = os.path.join(tmpdir, "test_ckpt")
            trainer.save_checkpoint(checkpoint_path)
            
            assert os.path.exists(os.path.join(checkpoint_path, "checkpoint.pt"))
            
            # 加载检查点
            trainer.load_checkpoint(checkpoint_path)


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_training_pipeline(self):
        """测试完整训练流程"""
        from config import Stage1Config
        from trainer import Trainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            config = Stage1Config(
                batch_size=4,
                max_epochs=1,
                max_steps=5,
                gradient_accumulation_steps=1,
                logging_steps=2,
                save_steps=5,
                eval_steps=5,
                fp16=False,
                output_dir=tmpdir,
            )
            train_dataset = MockDataset(size=20)
            eval_dataset = MockDataset(size=10)
            
            trainer = Trainer(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            
            result = trainer.train()
            
            assert "best_eval_loss" in result
            assert os.path.exists(os.path.join(tmpdir, "final_model"))
    
    def test_three_stage_training(self):
        """测试三阶段训练流程"""
        from config import Stage1Config, Stage2Config, Stage3Config
        from trainer import Trainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 阶段 1
            model = MockModel()
            config1 = Stage1Config(
                batch_size=4,
                max_epochs=1,
                max_steps=3,
                gradient_accumulation_steps=1,
                fp16=False,
                output_dir=os.path.join(tmpdir, "stage1"),
            )
            trainer1 = Trainer(
                model=model,
                config=config1,
                train_dataset=MockDataset(size=20),
            )
            trainer1.train()
            
            # 阶段 2
            config2 = Stage2Config(
                batch_size=4,
                max_epochs=1,
                max_steps=3,
                gradient_accumulation_steps=1,
                fp16=False,
                output_dir=os.path.join(tmpdir, "stage2"),
            )
            trainer2 = Trainer(
                model=model,
                config=config2,
                train_dataset=MockDataset(size=20),
            )
            trainer2.train()
            
            # 阶段 3
            config3 = Stage3Config(
                batch_size=4,
                max_epochs=1,
                max_steps=3,
                gradient_accumulation_steps=1,
                fp16=False,
                output_dir=os.path.join(tmpdir, "stage3"),
            )
            trainer3 = Trainer(
                model=model,
                config=config3,
                train_dataset=MockDataset(size=20),
            )
            trainer3.train()
            
            # 验证所有阶段都完成
            assert os.path.exists(os.path.join(tmpdir, "stage1", "final_model"))
            assert os.path.exists(os.path.join(tmpdir, "stage2", "final_model"))
            assert os.path.exists(os.path.join(tmpdir, "stage3", "final_model"))


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

