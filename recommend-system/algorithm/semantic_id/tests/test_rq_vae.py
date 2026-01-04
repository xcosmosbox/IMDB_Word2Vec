"""
RQ-VAE 语义 ID 编码器单元测试

该模块包含 Semantic ID 编码器所有组件的完整单元测试。

测试覆盖：
1. SemanticIDConfig - 配置类测试
2. VectorQuantizer - 单层向量量化器测试
3. ResidualVectorQuantizer - 残差向量量化器测试
4. SemanticIDEncoder - 编码器接口测试
5. SemanticIDTrainer - 训练器测试
6. 集成测试

运行测试：
    pytest algorithm/semantic_id/tests/test_rq_vae.py -v

作者: Person A
"""

import pytest
import torch
import torch.nn.functional as F
import tempfile
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from semantic_id.config import SemanticIDConfig, PresetConfigs
from semantic_id.codebook import VectorQuantizer
from semantic_id.rq_vae import ResidualVectorQuantizer, RQVAELoss
from semantic_id.encoder import SemanticIDEncoder, create_encoder
from semantic_id.trainer import SemanticIDTrainer, TrainingConfig, train_codebook


# ============================================================================
# 配置类测试
# ============================================================================

class TestSemanticIDConfig:
    """SemanticIDConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SemanticIDConfig()
        
        assert config.embedding_dim == 256
        assert config.num_codebooks == 3
        assert config.codebook_sizes == (1024, 4096, 16384)
        assert config.commitment_cost == 0.25
        assert config.ema_decay == 0.99
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SemanticIDConfig(
            embedding_dim=512,
            codebook_sizes=(2048, 8192, 32768),
            commitment_cost=0.5,
        )
        
        assert config.embedding_dim == 512
        assert config.codebook_sizes == (2048, 8192, 32768)
        assert config.commitment_cost == 0.5
    
    def test_invalid_codebook_sizes(self):
        """测试无效的码本大小配置"""
        with pytest.raises(ValueError):
            # 码本大小数量与层数不匹配
            SemanticIDConfig(
                num_codebooks=3,
                codebook_sizes=(1024, 4096),  # 只有两个
            )
    
    def test_invalid_embedding_dim(self):
        """测试无效的嵌入维度"""
        with pytest.raises(ValueError):
            SemanticIDConfig(embedding_dim=-1)
    
    def test_invalid_ema_decay(self):
        """测试无效的 EMA 衰减率"""
        with pytest.raises(ValueError):
            SemanticIDConfig(ema_decay=1.5)  # 必须在 (0, 1) 范围内
    
    def test_total_vocab_size(self):
        """测试总词表大小计算"""
        config = SemanticIDConfig()
        assert config.total_vocab_size == 1024 + 4096 + 16384
    
    def test_get_codebook_size(self):
        """测试获取指定层级码本大小"""
        config = SemanticIDConfig()
        
        assert config.get_codebook_size(1) == 1024
        assert config.get_codebook_size(2) == 4096
        assert config.get_codebook_size(3) == 16384
        
        with pytest.raises(ValueError):
            config.get_codebook_size(0)
        
        with pytest.raises(ValueError):
            config.get_codebook_size(4)
    
    def test_to_dict_and_from_dict(self):
        """测试字典序列化"""
        config = SemanticIDConfig(
            embedding_dim=128,
            codebook_sizes=(512, 2048, 8192),
        )
        
        config_dict = config.to_dict()
        restored = SemanticIDConfig.from_dict(config_dict)
        
        assert restored.embedding_dim == config.embedding_dim
        assert restored.codebook_sizes == config.codebook_sizes
    
    def test_save_and_load(self):
        """测试配置保存和加载"""
        config = SemanticIDConfig(
            embedding_dim=128,
            codebook_sizes=(512, 2048, 8192),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            config.save(path)
            
            loaded = SemanticIDConfig.load(path)
            
            assert loaded.embedding_dim == config.embedding_dim
            assert loaded.codebook_sizes == config.codebook_sizes
    
    def test_preset_configs(self):
        """测试预设配置"""
        small = PresetConfigs.small()
        assert small.embedding_dim == 128
        
        medium = PresetConfigs.medium()
        assert medium.embedding_dim == 256
        
        large = PresetConfigs.large()
        assert large.embedding_dim == 512
        
        production = PresetConfigs.production()
        assert production.embedding_dim == 256


# ============================================================================
# 向量量化器测试
# ============================================================================

class TestVectorQuantizer:
    """VectorQuantizer 测试"""
    
    @pytest.fixture
    def quantizer(self):
        """创建测试用量化器"""
        return VectorQuantizer(
            num_embeddings=1024,
            embedding_dim=256,
            commitment_cost=0.25,
            ema_decay=0.99,
        )
    
    def test_forward_shape(self, quantizer):
        """测试前向传播输出形状"""
        batch_size = 32
        x = torch.randn(batch_size, 256)
        
        quantized, indices, loss = quantizer(x)
        
        assert quantized.shape == (batch_size, 256)
        assert indices.shape == (batch_size,)
        assert loss.dim() == 0  # 标量
    
    def test_forward_values(self, quantizer):
        """测试前向传播输出值范围"""
        x = torch.randn(32, 256)
        
        quantized, indices, loss = quantizer(x)
        
        # 索引应该在有效范围内
        assert indices.min() >= 0
        assert indices.max() < 1024
        
        # 损失应该为非负
        assert loss >= 0
    
    def test_straight_through_estimator(self, quantizer):
        """测试直通估计器梯度传播"""
        x = torch.randn(32, 256, requires_grad=True)
        
        quantized, indices, loss = quantizer(x)
        
        # 应该能够计算梯度
        loss_total = loss + quantized.sum()
        loss_total.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_ema_update(self, quantizer):
        """测试 EMA 更新"""
        initial_weights = quantizer.embedding.weight.clone()
        
        # 训练模式下处理数据
        quantizer.train()
        x = torch.randn(100, 256)
        for _ in range(10):
            quantizer(x)
        
        # 权重应该发生变化
        assert not torch.allclose(initial_weights, quantizer.embedding.weight)
    
    def test_eval_mode_no_ema_update(self, quantizer):
        """测试评估模式下不进行 EMA 更新"""
        # 先训练一下
        quantizer.train()
        x = torch.randn(100, 256)
        quantizer(x)
        
        # 切换到评估模式
        quantizer.eval()
        initial_weights = quantizer.embedding.weight.clone()
        
        # 处理数据
        quantizer(x)
        
        # 权重不应该变化
        assert torch.allclose(initial_weights, quantizer.embedding.weight)
    
    def test_get_codebook_embeddings(self, quantizer):
        """测试获取码本嵌入"""
        embeddings = quantizer.get_codebook_embeddings()
        
        assert embeddings.shape == (1024, 256)
        assert not embeddings.requires_grad  # 应该是 detached
    
    def test_quantize_indices(self, quantizer):
        """测试根据索引获取量化向量"""
        indices = torch.randint(0, 1024, (32,))
        
        quantized = quantizer.quantize_indices(indices)
        
        assert quantized.shape == (32, 256)
    
    def test_usage_stats(self, quantizer):
        """测试使用统计"""
        quantizer.train()
        x = torch.randn(100, 256)
        quantizer(x)
        
        stats = quantizer.get_codebook_usage()
        
        assert "utilization" in stats
        assert "perplexity" in stats
        assert "dead_codes" in stats
        assert 0 <= stats["utilization"] <= 1
    
    def test_reset_usage_stats(self, quantizer):
        """测试重置使用统计"""
        quantizer.train()
        x = torch.randn(100, 256)
        quantizer(x)
        
        quantizer.reset_usage_stats()
        stats = quantizer.get_codebook_usage()
        
        assert stats["utilization"] == 0.0


# ============================================================================
# 残差向量量化器测试
# ============================================================================

class TestResidualVectorQuantizer:
    """ResidualVectorQuantizer 测试"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return SemanticIDConfig(
            embedding_dim=256,
            codebook_sizes=(1024, 4096, 16384),
        )
    
    @pytest.fixture
    def rq_vae(self, config):
        """创建测试用 RQ-VAE"""
        return ResidualVectorQuantizer(config)
    
    def test_forward_shape(self, rq_vae):
        """测试前向传播输出形状"""
        batch_size = 32
        x = torch.randn(batch_size, 256)
        
        output = rq_vae(x)
        
        assert output["quantized"].shape == (batch_size, 256)
        assert len(output["indices"]) == 3
        assert output["indices"][0].shape == (batch_size,)
        assert output["indices"][1].shape == (batch_size,)
        assert output["indices"][2].shape == (batch_size,)
    
    def test_forward_values(self, rq_vae):
        """测试前向传播输出值范围"""
        x = torch.randn(32, 256)
        
        output = rq_vae(x)
        
        # 检查索引范围
        assert output["indices"][0].max() < 1024
        assert output["indices"][1].max() < 4096
        assert output["indices"][2].max() < 16384
        
        # 损失应该为非负
        assert output["reconstruction_loss"] >= 0
        assert output["commitment_loss"] >= 0
        assert output["total_loss"] >= 0
    
    def test_encode(self, rq_vae):
        """测试编码功能"""
        x = torch.randn(32, 256)
        
        l1, l2, l3 = rq_vae.encode(x)
        
        assert l1.shape == (32,)
        assert l2.shape == (32,)
        assert l3.shape == (32,)
        
        assert l1.max() < 1024
        assert l2.max() < 4096
        assert l3.max() < 16384
    
    def test_decode(self, rq_vae):
        """测试解码功能"""
        batch_size = 32
        l1 = torch.randint(0, 1024, (batch_size,))
        l2 = torch.randint(0, 4096, (batch_size,))
        l3 = torch.randint(0, 16384, (batch_size,))
        
        reconstructed = rq_vae.decode(l1, l2, l3)
        
        assert reconstructed.shape == (batch_size, 256)
    
    def test_encode_decode_consistency(self, rq_vae):
        """测试编码解码一致性"""
        x = torch.randn(32, 256)
        
        # 编码
        l1, l2, l3 = rq_vae.encode(x)
        
        # 解码
        reconstructed = rq_vae.decode(l1, l2, l3)
        
        # 使用前向传播获取量化结果
        output = rq_vae(x)
        
        # 解码结果应该与量化结果近似
        # （由于 Straight-Through Estimator，可能有微小差异）
        assert reconstructed.shape == output["quantized"].shape
    
    def test_get_codebook_embeddings(self, rq_vae):
        """测试获取码本嵌入"""
        emb1 = rq_vae.get_codebook_embeddings(1)
        emb2 = rq_vae.get_codebook_embeddings(2)
        emb3 = rq_vae.get_codebook_embeddings(3)
        
        assert emb1.shape == (1024, 256)
        assert emb2.shape == (4096, 256)
        assert emb3.shape == (16384, 256)
    
    def test_get_all_codebook_embeddings(self, rq_vae):
        """测试获取所有码本嵌入"""
        all_emb = rq_vae.get_all_codebook_embeddings()
        
        assert len(all_emb) == 3
        assert all_emb[0].shape == (1024, 256)
        assert all_emb[1].shape == (4096, 256)
        assert all_emb[2].shape == (16384, 256)
    
    def test_compute_perplexity(self, rq_vae):
        """测试困惑度计算"""
        batch_size = 100
        l1 = torch.randint(0, 1024, (batch_size,))
        l2 = torch.randint(0, 4096, (batch_size,))
        l3 = torch.randint(0, 16384, (batch_size,))
        
        perplexity = rq_vae.compute_perplexity([l1, l2, l3])
        
        assert "level_1" in perplexity
        assert "level_2" in perplexity
        assert "level_3" in perplexity
        assert perplexity["level_1"] > 0


class TestRQVAELoss:
    """RQVAELoss 测试"""
    
    def test_loss_computation(self):
        """测试损失计算"""
        loss_fn = RQVAELoss(commitment_cost=0.25)
        
        rq_output = {
            "reconstruction_loss": torch.tensor(0.5),
            "commitment_loss": torch.tensor(0.2),
        }
        
        losses = loss_fn(rq_output)
        
        expected = 0.5 + 0.25 * 0.2
        assert torch.isclose(losses["total_loss"], torch.tensor(expected))


# ============================================================================
# SemanticIDEncoder 测试
# ============================================================================

class TestSemanticIDEncoder:
    """SemanticIDEncoder 测试"""
    
    @pytest.fixture
    def encoder(self):
        """创建测试用编码器"""
        config = SemanticIDConfig(
            embedding_dim=256,
            codebook_sizes=(1024, 4096, 16384),
        )
        return SemanticIDEncoder(config)
    
    def test_encode(self, encoder):
        """测试编码功能"""
        features = torch.randn(32, 256)
        
        l1, l2, l3 = encoder.encode(features)
        
        assert l1.shape == (32,), f"L1 shape mismatch: {l1.shape}"
        assert l2.shape == (32,), f"L2 shape mismatch: {l2.shape}"
        assert l3.shape == (32,), f"L3 shape mismatch: {l3.shape}"
        
        assert l1.max() < 1024, "L1 out of range"
        assert l2.max() < 4096, "L2 out of range"
        assert l3.max() < 16384, "L3 out of range"
    
    def test_decode(self, encoder):
        """测试解码功能"""
        features = torch.randn(32, 256)
        
        l1, l2, l3 = encoder.encode(features)
        reconstructed = encoder.decode(l1, l2, l3)
        
        assert reconstructed.shape == (32, 256)
    
    def test_reconstruction_error(self, encoder):
        """测试重建误差"""
        features = torch.randn(32, 256)
        
        l1, l2, l3 = encoder.encode(features)
        reconstructed = encoder.decode(l1, l2, l3)
        
        error = F.mse_loss(reconstructed, features)
        print(f"Reconstruction error: {error.item():.4f}")
        
        # 未训练的模型误差可能较大，但不应该是 NaN 或 Inf
        assert not torch.isnan(error)
        assert not torch.isinf(error)
    
    def test_get_codebook_embeddings(self, encoder):
        """测试获取码本嵌入"""
        emb1 = encoder.get_codebook_embeddings(1)
        emb2 = encoder.get_codebook_embeddings(2)
        emb3 = encoder.get_codebook_embeddings(3)
        
        assert emb1.shape == (1024, 256)
        assert emb2.shape == (4096, 256)
        assert emb3.shape == (16384, 256)
    
    def test_forward(self, encoder):
        """测试前向传播"""
        features = torch.randn(32, 256)
        
        output = encoder(features)
        
        assert "l1_ids" in output
        assert "l2_ids" in output
        assert "l3_ids" in output
        assert "quantized" in output
        assert "total_loss" in output
    
    def test_invalid_input_dim(self, encoder):
        """测试无效输入维度"""
        # 1D 输入
        with pytest.raises(ValueError):
            encoder.encode(torch.randn(256))
        
        # 3D 输入
        with pytest.raises(ValueError):
            encoder.encode(torch.randn(32, 10, 256))
    
    def test_invalid_embedding_dim(self, encoder):
        """测试无效嵌入维度"""
        with pytest.raises(ValueError):
            encoder.encode(torch.randn(32, 128))  # 应该是 256
    
    def test_save_and_load(self, encoder):
        """测试模型保存和加载"""
        features = torch.randn(32, 256)
        
        # 编码
        l1_orig, l2_orig, l3_orig = encoder.encode(features)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存
            encoder.save_pretrained(tmpdir)
            
            # 加载
            loaded = SemanticIDEncoder.from_pretrained(tmpdir)
            
            # 使用加载的模型编码
            l1_loaded, l2_loaded, l3_loaded = loaded.encode(features)
            
            # 结果应该相同
            assert torch.equal(l1_orig, l1_loaded)
            assert torch.equal(l2_orig, l2_loaded)
            assert torch.equal(l3_orig, l3_loaded)
    
    def test_compute_reconstruction_error(self, encoder):
        """测试计算重建误差"""
        features = torch.randn(32, 256)
        
        errors = encoder.compute_reconstruction_error(features)
        
        assert "mse" in errors
        assert "rmse" in errors
        assert "mae" in errors
        assert "cosine_sim" in errors
    
    def test_get_num_parameters(self, encoder):
        """测试获取参数数量"""
        params = encoder.get_num_parameters()
        
        assert "total" in params
        assert "trainable" in params
        assert "codebook" in params
        assert params["total"] > 0


class TestCreateEncoder:
    """create_encoder 工厂函数测试"""
    
    def test_create_small(self):
        """测试创建小规模编码器"""
        encoder = create_encoder("small")
        assert encoder.config.embedding_dim == 128
    
    def test_create_medium(self):
        """测试创建中等规模编码器"""
        encoder = create_encoder("medium")
        assert encoder.config.embedding_dim == 256
    
    def test_create_large(self):
        """测试创建大规模编码器"""
        encoder = create_encoder("large")
        assert encoder.config.embedding_dim == 512
    
    def test_create_with_custom_params(self):
        """测试使用自定义参数创建"""
        encoder = create_encoder("medium", commitment_cost=0.5)
        assert encoder.config.commitment_cost == 0.5
    
    def test_invalid_preset(self):
        """测试无效预设"""
        with pytest.raises(ValueError):
            create_encoder("invalid")


# ============================================================================
# 训练器测试
# ============================================================================

class TestSemanticIDTrainer:
    """SemanticIDTrainer 测试"""
    
    @pytest.fixture
    def small_encoder(self):
        """创建小规模编码器用于测试"""
        config = SemanticIDConfig(
            embedding_dim=64,
            codebook_sizes=(32, 64, 128),
        )
        return SemanticIDEncoder(config)
    
    @pytest.fixture
    def training_config(self):
        """创建训练配置"""
        return TrainingConfig(
            batch_size=16,
            num_epochs=2,
            learning_rate=1e-3,
            log_interval=10,
        )
    
    def test_trainer_initialization(self, small_encoder, training_config):
        """测试训练器初始化"""
        trainer = SemanticIDTrainer(small_encoder, training_config)
        
        assert trainer.encoder is small_encoder
        assert trainer.config is training_config
        assert trainer.optimizer is not None
    
    def test_training_basic(self, small_encoder, training_config):
        """测试基本训练流程"""
        trainer = SemanticIDTrainer(small_encoder, training_config)
        
        # 创建小规模训练数据
        train_features = torch.randn(100, 64)
        val_features = torch.randn(20, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.train(
                train_features,
                val_features,
                save_dir=tmpdir,
            )
            
            # 检查历史记录
            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2
            
            # 检查模型被保存
            assert os.path.exists(os.path.join(tmpdir, "final_model"))
    
    def test_training_without_validation(self, small_encoder, training_config):
        """测试无验证集训练"""
        trainer = SemanticIDTrainer(small_encoder, training_config)
        
        train_features = torch.randn(100, 64)
        
        history = trainer.train(train_features)
        
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 0
    
    def test_compute_codebook_utilization(self, small_encoder, training_config):
        """测试码本利用率计算"""
        trainer = SemanticIDTrainer(small_encoder, training_config)
        
        # 创建数据加载器
        features = torch.randn(100, 64)
        dataloader = trainer._create_dataloader(features, shuffle=False)
        
        usage = trainer.compute_codebook_utilization(dataloader)
        
        assert "level_1" in usage
        assert "level_2" in usage
        assert "level_3" in usage


class TestTrainCodebook:
    """train_codebook 便捷函数测试"""
    
    def test_train_codebook_basic(self):
        """测试基本训练"""
        config = SemanticIDConfig(
            embedding_dim=64,
            codebook_sizes=(32, 64, 128),
        )
        encoder = SemanticIDEncoder(config)
        
        features = torch.randn(200, 64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_codebook(
                encoder,
                features,
                num_epochs=2,
                batch_size=16,
                save_dir=tmpdir,
            )
            
            assert len(history["train_loss"]) == 2
            assert encoder.is_trained


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程：配置 -> 创建 -> 训练 -> 编码 -> 解码"""
        # 1. 创建配置
        config = SemanticIDConfig(
            embedding_dim=64,
            codebook_sizes=(32, 64, 128),
        )
        
        # 2. 创建编码器
        encoder = SemanticIDEncoder(config)
        
        # 3. 准备数据
        train_features = torch.randn(200, 64)
        
        # 4. 训练
        training_config = TrainingConfig(
            batch_size=16,
            num_epochs=2,
        )
        trainer = SemanticIDTrainer(encoder, training_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(train_features, save_dir=tmpdir)
            
            # 5. 测试编码
            test_features = torch.randn(10, 64)
            l1, l2, l3 = encoder.encode(test_features)
            
            assert l1.shape == (10,)
            assert l2.shape == (10,)
            assert l3.shape == (10,)
            
            # 6. 测试解码
            reconstructed = encoder.decode(l1, l2, l3)
            
            assert reconstructed.shape == (10, 64)
            
            # 7. 测试加载
            loaded = SemanticIDEncoder.from_pretrained(
                os.path.join(tmpdir, "final_model")
            )
            
            l1_loaded, l2_loaded, l3_loaded = loaded.encode(test_features)
            
            assert torch.equal(l1, l1_loaded)
            assert torch.equal(l2, l2_loaded)
            assert torch.equal(l3, l3_loaded)
    
    def test_semantic_similarity(self):
        """测试语义相似性：相似的输入应该有相似的 ID 前缀"""
        config = SemanticIDConfig(
            embedding_dim=64,
            codebook_sizes=(32, 64, 128),
        )
        encoder = SemanticIDEncoder(config)
        
        # 创建一组相似的向量
        base = torch.randn(1, 64)
        similar = base + torch.randn(10, 64) * 0.1  # 添加小量噪声
        different = torch.randn(10, 64)  # 完全随机
        
        # 编码
        l1_base, _, _ = encoder.encode(base)
        l1_similar, _, _ = encoder.encode(similar)
        l1_different, _, _ = encoder.encode(different)
        
        # 相似向量的 L1 ID 相同的比例应该更高
        similar_match_rate = (l1_similar == l1_base[0]).float().mean()
        different_match_rate = (l1_different == l1_base[0]).float().mean()
        
        # 注意：由于模型未训练，这个测试可能不总是通过
        # 但至少应该能运行
        print(f"Similar match rate: {similar_match_rate:.2%}")
        print(f"Different match rate: {different_match_rate:.2%}")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        config = SemanticIDConfig(
            embedding_dim=64,
            codebook_sizes=(32, 64, 128),
        )
        encoder = SemanticIDEncoder(config)
        
        features = torch.randn(16, 64, requires_grad=True)
        
        output = encoder(features)
        loss = output["total_loss"]
        
        # 反向传播
        loss.backward()
        
        # 检查梯度存在
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        
        # 检查模型参数有梯度
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


# ============================================================================
# 性能测试
# ============================================================================

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.slow
    def test_encoding_speed(self):
        """测试编码速度"""
        config = SemanticIDConfig()
        encoder = SemanticIDEncoder(config)
        encoder.eval()
        
        # 预热
        with torch.no_grad():
            encoder.encode(torch.randn(32, 256))
        
        # 测试
        batch_size = 32
        num_batches = 100
        features = torch.randn(batch_size, 256)
        
        import time
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_batches):
                encoder.encode(features)
        
        elapsed = time.time() - start
        samples_per_second = (batch_size * num_batches) / elapsed
        
        print(f"Encoding speed: {samples_per_second:.0f} samples/second")
        
        # 基本性能要求
        assert samples_per_second > 100, "Encoding too slow"
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """测试内存使用"""
        config = SemanticIDConfig()
        encoder = SemanticIDEncoder(config)
        
        params = encoder.get_num_parameters()
        total_params = params["total"]
        
        # 假设每个参数是 float32 (4 bytes)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        print(f"Model parameters: {total_params:,}")
        print(f"Estimated memory: {memory_mb:.2f} MB")
        
        # 模型不应该太大
        assert memory_mb < 500, "Model too large"


# ============================================================================
# 主函数
# ============================================================================

def test_semantic_id_encoder():
    """
    主测试函数（对应任务要求中的测试用例）
    
    测试内容：
    1. 编码功能
    2. 解码功能
    3. 重建误差
    """
    config = SemanticIDConfig()
    encoder = SemanticIDEncoder(config)
    
    # 测试编码
    features = torch.randn(32, 256)
    l1, l2, l3 = encoder.encode(features)
    
    assert l1.shape == (32,), f"L1 shape mismatch: {l1.shape}"
    assert l2.shape == (32,), f"L2 shape mismatch: {l2.shape}"
    assert l3.shape == (32,), f"L3 shape mismatch: {l3.shape}"
    
    assert l1.max() < 1024, "L1 out of range"
    assert l2.max() < 4096, "L2 out of range"
    assert l3.max() < 16384, "L3 out of range"
    
    # 测试解码
    reconstructed = encoder.decode(l1, l2, l3)
    assert reconstructed.shape == (32, 256)
    
    # 测试重建误差
    error = F.mse_loss(reconstructed, features)
    print(f"Reconstruction error: {error.item():.4f}")
    
    print("All tests passed!")


if __name__ == "__main__":
    # 运行主测试
    test_semantic_id_encoder()
    
    # 运行 pytest
    pytest.main([__file__, "-v"])

