# AI Safety & Model Defense

## 概述
AI Safety 模块专注于大模型应用的安全，包括输入端的 Prompt 注入防御、输出端的内容审核与隐私保护，以及模型鲁棒性检测。

## 目录结构
- `prompt-guard/`: 输入防御
    - `injection_detector.py`: 基于规则与启发式的 Prompt Injection 检测。
- `content-moderation/`: 内容审核
    - `toxicity_classifier.py`: 毒性/不安全内容检测。
    - `pii_leakage.py`: 输出端敏感信息 (PII) 扫描与屏蔽。
- `adversarial/`: 对抗防御
    - `perturbation.py`: 简单的对抗扰动检测。

## 功能特性

### 1. Prompt Injection 检测
- 拦截 "Ignore previous instructions", "System override" 等常见攻击模式。
- 检查异常长度和复杂结构的 Prompt。

### 2. 输出隐私保护 (DLP)
- 自动检测并打码模型输出中的手机号、邮箱、身份证等 PII。
- 只有经过清洗的输出才能返回给用户。

### 3. 内容合规
- 过滤暴力、色情、仇恨言论。
- 确保生成内容符合社区准则。

## 使用指南

### 输入检测
```python
detector = InjectionDetector()
is_unsafe, conf, reason = detector.detect(user_input)
if is_unsafe:
    block_request()
```

### 输出清洗
```python
pii_scanner = PIILeakageDetector()
safe_response = pii_scanner.sanitize(llm_response)
```

## 测试
```bash
python -m unittest discover .
```

