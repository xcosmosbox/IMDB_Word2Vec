# Person E: AI 安全与模型防护

## 你的角色
你是一名 AI 安全研究员，负责实现生成式推荐系统的 **AI 安全模块**。你的目标是保护大模型免受攻击，并确保输出内容的安全性。

---

## ⚠️ 重要：标准驱动开发

**开始编码前，必须先阅读安全标准契约：**

```
security/SECURITY_STANDARDS.md
```

你需要遵循的标准：
- **输入防御**: Prompt Injection, Jailbreak 检测
- **输出审核**: 敏感内容, PII 泄露检测
- **对抗防御**: 输入扰动检测

---

## 你的任务

```
security/ai-safety/
├── prompt-guard/           # 提示词防御
│   ├── injection_detector.py  # 注入攻击检测
│   ├── jailbreak_filter.py    # 越狱攻击过滤
│   └── sanitizer.py           # 输入清洗
├── content-moderation/     # 内容审核
│   ├── toxicity_classifier.py # 毒性检测
│   ├── pii_leakage.py         # 输出 PII 检测
│   └── bias_detector.py       # 偏见检测
└── adversarial/            # 对抗样本防御
    ├── perturbation.py        # 扰动检测
    └── robustness_test.py     # 鲁棒性测试套件
```

---

## 1. 提示词注入检测 (prompt-guard/injection_detector.py)

```python
"""
Prompt Injection Detector

检测输入中是否包含试图操纵 LLM 行为的指令
"""

import re
from typing import List, Tuple

class InjectionDetector:
    """
    基于规则和模型的注入检测
    """
    
    PATTERNS = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are now",
        r"roleplay",
        r"override",
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
    
    def detect(self, text: str) -> Tuple[bool, float, str]:
        """
        检测文本
        
        Returns:
            (is_injection, confidence, reason)
        """
        # 1. 规则匹配
        for pattern in self.patterns:
            if pattern.search(text):
                return True, 1.0, f"Matched pattern: {pattern.pattern}"
        
        # 2. 长度与复杂度检查
        if len(text) > 1000 and "{" in text and "}" in text:
             return True, 0.6, "Complex structure suspicious"
             
        # TODO: 3. 模型检测 (BERT / Deberta)
        
        return False, 0.0, ""
```

## 2. PII 泄露检测 (content-moderation/pii_leakage.py)

实现一个检测器，检查模型输出是否包含：
- 手机号 (`\d{11}`)
- 邮箱
- 身份证号
- 信用卡号

如果检测到，应立即屏蔽或脱敏。

## 3. 内容安全过滤器 (content-moderation/toxicity_classifier.py)

集成开源模型（如 Detoxify 或 OpenAI Moderation API）对输出内容进行评分，拦截高风险内容（暴力、色情、仇恨言论）。

## 输出要求

请输出完整的 AI 安全模块代码：
1. 提示词注入检测器
2. 输出 PII 扫描与脱敏
3. 简单的内容毒性分类器封装
4. 对抗样本检测逻辑

