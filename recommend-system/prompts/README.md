# Go 后端开发提示词

## 📋 下一阶段：Go 后端补充模块

算法模块已 100% 完成，现在进入 Go 后端补充模块的开发。

---

## 👥 任务分配

| 角色 | 文件 | 模块 | 职责 |
|------|------|------|------|
| **Person A** | `person_a_user_service.md` | User Service | 用户服务 API + 业务逻辑 |
| **Person B** | `person_b_item_service.md` | Item Service | 物品服务 API + 向量搜索 |
| **Person C** | `person_c_feature_service.md` | Feature Service | 特征提取 + 存储 + Token 化 |
| **Person D** | `person_d_coldstart_llm.md` | Cold Start + LLM | 冷启动服务 + 大模型集成 |
| **Person E** | `person_e_grpc_proto.md` | gRPC & Proto | 服务间通信 + Proto 定义 |
| **Person F** | `person_f_testing.md` | Testing | 单元测试 + 集成测试 + 性能测试 |

---

## 🗂️ 开发目标

```
待开发模块：
├── api/user/v1/              → Person A
├── api/item/v1/              → Person B
├── cmd/user-service/         → Person A
├── cmd/item-service/         → Person B
├── internal/llm/             → Person D
├── internal/service/coldstart/ → Person D
├── internal/service/feature/   → Person C
├── internal/service/item/      → Person B
├── internal/service/user/      → Person A
├── internal/grpc/            → Person E
├── proto/                    → Person E
├── tests/unit/               → Person F
├── tests/integration/        → Person F
└── tests/benchmark/          → Person F
```

---

## 🚀 开发流程

1. **阅读提示词文件** - 理解任务要求
2. **创建代码文件** - 在对应目录下实现
3. **编写测试** - 确保功能正确
4. **提交代码** - 遵循提交规范

---

## 📅 时间线

```
Week 1: 并行开发（6人同时进行）
Week 2: 模块联调
Week 3: 集成测试 + 性能测试
```

---

## 🔗 依赖关系

```
Person A (User)     ← 独立
Person B (Item)     ← 独立
Person C (Feature)  ← 依赖 A, B 的接口
Person D (ColdStart)← 独立
Person E (gRPC)     ← 独立
Person F (Testing)  ← 依赖所有模块完成
```

---

## 📝 注意事项

1. 遵循现有代码风格和目录结构
2. 使用中文注释
3. 测试覆盖率 >= 80%
4. API 响应时间 <= 200ms

