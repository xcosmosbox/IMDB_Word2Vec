# Person E: 数据质量 (Data Quality)

## 你的角色
你是一名数据工程师，负责实现生成式推荐系统的 **数据质量模块**，包括数据验证、质量监控、异常检测、数据剖析等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
data-pipeline/interfaces.py
```

你需要实现的接口：

```python
class DataValidatorInterface(ABC):
    @abstractmethod
    def validate(self, data: Dict) -> DataQualityReport:
        pass
    
    @abstractmethod
    def add_rule(self, rule_name, rule_func, level) -> None:
        pass

class DataQualityMonitorInterface(ABC):
    @abstractmethod
    def start_monitoring(self) -> None:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def set_alert_threshold(self, metric_name, threshold, comparison) -> None:
        pass

class DataProfilerInterface(ABC):
    @abstractmethod
    def profile(self, data: List[Dict]) -> Dict:
        pass
    
    @abstractmethod
    def detect_drift(self, baseline, current) -> Dict:
        pass
```

---

## 技术栈

- **验证框架**: Great Expectations, Pydantic
- **剖析**: ydata-profiling (pandas-profiling)
- **监控**: Prometheus
- **可视化**: Grafana

---

## 你的任务

```
data-pipeline/data-quality/
├── validators/
│   ├── __init__.py
│   ├── base.py               # 验证器基类
│   ├── schema_validator.py   # Schema 验证
│   ├── rule_validator.py     # 规则验证
│   ├── statistical.py        # 统计验证
│   └── custom_rules.py       # 自定义规则
├── monitors/
│   ├── __init__.py
│   ├── quality_monitor.py    # 质量监控
│   ├── metrics.py            # 质量指标
│   └── alerts.py             # 告警管理
├── profilers/
│   ├── __init__.py
│   ├── data_profiler.py      # 数据剖析
│   └── drift_detector.py     # 漂移检测
├── reports/
│   ├── __init__.py
│   ├── report_generator.py   # 报告生成
│   └── templates/
│       └── quality_report.html
└── tests/
    ├── test_validators.py
    ├── test_monitors.py
    └── test_profilers.py
```

---

## 1. 规则验证器 (validators/rule_validator.py)

```python
"""
规则验证器

基于可配置规则的数据验证
"""

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
import re

from ..interfaces import (
    DataValidatorInterface,
    DataQualityReport,
    DataQualityLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """验证规则"""
    name: str
    check: Callable[[Dict[str, Any]], bool]
    level: DataQualityLevel
    description: str = ""
    tags: List[str] = field(default_factory=list)


class RuleBasedValidator(DataValidatorInterface):
    """
    规则验证器
    
    使用示例:
        validator = RuleBasedValidator()
        
        # 添加规则
        validator.add_rule(
            "user_id_required",
            lambda d: bool(d.get("user_id")),
            DataQualityLevel.CRITICAL,
            "user_id must be present"
        )
        
        # 验证数据
        report = validator.validate(data)
    """
    
    # 预定义规则
    PRESET_RULES = {
        'not_null': lambda field: lambda d: d.get(field) is not None,
        'not_empty': lambda field: lambda d: bool(d.get(field)),
        'positive': lambda field: lambda d: (d.get(field) or 0) > 0,
        'in_range': lambda field, min_v, max_v: lambda d: min_v <= (d.get(field) or 0) <= max_v,
        'matches_pattern': lambda field, pattern: lambda d: bool(re.match(pattern, str(d.get(field, '')))),
        'in_list': lambda field, allowed: lambda d: d.get(field) in allowed,
        'type_check': lambda field, expected_type: lambda d: isinstance(d.get(field), expected_type),
    }
    
    def __init__(self):
        self._rules: List[ValidationRule] = []
    
    def add_rule(
        self,
        rule_name: str,
        rule_func: Callable[[Dict[str, Any]], bool],
        level: DataQualityLevel,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """添加验证规则"""
        rule = ValidationRule(
            name=rule_name,
            check=rule_func,
            level=level,
            description=description,
            tags=tags or [],
        )
        self._rules.append(rule)
        logger.debug(f"Added rule: {rule_name}")
    
    def add_preset_rule(
        self,
        preset: str,
        rule_name: str,
        level: DataQualityLevel,
        *args,
        **kwargs,
    ) -> None:
        """添加预设规则"""
        if preset not in self.PRESET_RULES:
            raise ValueError(f"Unknown preset: {preset}")
        
        rule_func = self.PRESET_RULES[preset](*args)
        self.add_rule(rule_name, rule_func, level, f"{preset} check for {args}")
    
    def get_rules(self) -> List[str]:
        """获取所有规则名"""
        return [rule.name for rule in self._rules]
    
    def validate(self, data: Dict[str, Any]) -> DataQualityReport:
        """
        验证单条数据
        
        Returns:
            DataQualityReport: 验证报告
        """
        failed_rules = []
        metrics = {
            'total_rules': len(self._rules),
            'passed_rules': 0,
            'failed_rules': 0,
        }
        
        highest_level = DataQualityLevel.LOW
        
        for rule in self._rules:
            try:
                passed = rule.check(data)
                if passed:
                    metrics['passed_rules'] += 1
                else:
                    metrics['failed_rules'] += 1
                    failed_rules.append(rule.name)
                    
                    # 更新最高级别
                    if self._level_priority(rule.level) > self._level_priority(highest_level):
                        highest_level = rule.level
                        
            except Exception as e:
                metrics['failed_rules'] += 1
                failed_rules.append(f"{rule.name} (error: {e})")
        
        passed = metrics['failed_rules'] == 0
        
        return DataQualityReport(
            check_name="rule_validation",
            level=highest_level if not passed else DataQualityLevel.LOW,
            passed=passed,
            message=f"Failed rules: {failed_rules}" if failed_rules else "All rules passed",
            metrics=metrics,
            timestamp=datetime.now(),
        )
    
    def validate_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> List[DataQualityReport]:
        """批量验证"""
        return [self.validate(d) for d in data]
    
    def get_batch_summary(
        self,
        reports: List[DataQualityReport]
    ) -> Dict[str, Any]:
        """获取批量验证摘要"""
        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        
        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'critical_failures': sum(
                1 for r in reports 
                if not r.passed and r.level == DataQualityLevel.CRITICAL
            ),
        }
    
    def _level_priority(self, level: DataQualityLevel) -> int:
        """获取级别优先级"""
        priorities = {
            DataQualityLevel.LOW: 0,
            DataQualityLevel.MEDIUM: 1,
            DataQualityLevel.HIGH: 2,
            DataQualityLevel.CRITICAL: 3,
        }
        return priorities.get(level, 0)


class SchemaValidator(DataValidatorInterface):
    """
    Schema 验证器
    
    基于 JSON Schema 的验证
    """
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self._rules = []
    
    def validate(self, data: Dict[str, Any]) -> DataQualityReport:
        """验证数据是否符合 Schema"""
        import jsonschema
        
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return DataQualityReport(
                check_name="schema_validation",
                level=DataQualityLevel.LOW,
                passed=True,
                message="Schema validation passed",
            )
        except jsonschema.ValidationError as e:
            return DataQualityReport(
                check_name="schema_validation",
                level=DataQualityLevel.CRITICAL,
                passed=False,
                message=f"Schema validation failed: {e.message}",
                metrics={'error_path': list(e.absolute_path)},
            )
    
    def validate_batch(self, data: List[Dict[str, Any]]) -> List[DataQualityReport]:
        return [self.validate(d) for d in data]
    
    def add_rule(self, *args, **kwargs):
        pass
    
    def get_rules(self) -> List[str]:
        return ["schema_validation"]
```

---

## 2. 质量监控 (monitors/quality_monitor.py)

```python
"""
数据质量监控

实时监控数据质量指标
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import logging
import time

from prometheus_client import Counter, Gauge, Histogram

from ..interfaces import DataQualityMonitorInterface, DataQualityLevel

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """告警阈值"""
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    level: DataQualityLevel = DataQualityLevel.HIGH


@dataclass
class Alert:
    """告警"""
    metric_name: str
    threshold: AlertThreshold
    current_value: float
    triggered_at: datetime
    level: DataQualityLevel


class DataQualityMonitor(DataQualityMonitorInterface):
    """
    数据质量监控
    
    使用示例:
        monitor = DataQualityMonitor()
        
        # 设置告警阈值
        monitor.set_alert_threshold("null_rate", 0.05, "gt")
        monitor.set_alert_threshold("duplicate_rate", 0.01, "gt")
        
        # 启动监控
        monitor.start_monitoring()
        
        # 记录指标
        monitor.record_metric("null_rate", 0.03)
        
        # 获取告警
        alerts = monitor.get_alerts()
    """
    
    def __init__(
        self,
        check_interval: int = 60,
        alert_cooldown: int = 300,
    ):
        self.check_interval = check_interval
        self.alert_cooldown = alert_cooldown
        
        self._metrics: Dict[str, float] = {}
        self._metric_history: Dict[str, List[tuple]] = {}
        self._thresholds: Dict[str, AlertThreshold] = {}
        self._alerts: List[Alert] = []
        self._last_alert_time: Dict[str, datetime] = {}
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Prometheus 指标
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """初始化 Prometheus 指标"""
        self.prom_quality_score = Gauge(
            'data_quality_score',
            'Overall data quality score',
        )
        
        self.prom_null_rate = Gauge(
            'data_null_rate',
            'Null value rate',
            ['field']
        )
        
        self.prom_validation_failures = Counter(
            'data_validation_failures_total',
            'Total validation failures',
            ['rule', 'level']
        )
        
        self.prom_records_processed = Counter(
            'data_records_processed_total',
            'Total records processed',
        )
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Data quality monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Data quality monitoring stopped")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            self._check_thresholds()
            time.sleep(self.check_interval)
    
    def _check_thresholds(self):
        """检查阈值"""
        with self._lock:
            for metric_name, threshold in self._thresholds.items():
                current_value = self._metrics.get(metric_name)
                
                if current_value is None:
                    continue
                
                # 检查是否触发
                triggered = self._compare(current_value, threshold.threshold, threshold.comparison)
                
                if triggered:
                    # 检查冷却时间
                    last_alert = self._last_alert_time.get(metric_name)
                    if last_alert and (datetime.now() - last_alert).seconds < self.alert_cooldown:
                        continue
                    
                    alert = Alert(
                        metric_name=metric_name,
                        threshold=threshold,
                        current_value=current_value,
                        triggered_at=datetime.now(),
                        level=threshold.level,
                    )
                    
                    self._alerts.append(alert)
                    self._last_alert_time[metric_name] = datetime.now()
                    
                    logger.warning(f"Alert triggered: {metric_name} = {current_value} ({threshold.comparison} {threshold.threshold})")
    
    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        """比较值和阈值"""
        comparisons = {
            'gt': lambda v, t: v > t,
            'lt': lambda v, t: v < t,
            'eq': lambda v, t: v == t,
            'gte': lambda v, t: v >= t,
            'lte': lambda v, t: v <= t,
        }
        return comparisons.get(comparison, lambda v, t: False)(value, threshold)
    
    def record_metric(self, name: str, value: float) -> None:
        """记录指标"""
        with self._lock:
            self._metrics[name] = value
            
            if name not in self._metric_history:
                self._metric_history[name] = []
            
            self._metric_history[name].append((datetime.now(), value))
            
            # 保留最近 1 小时的历史
            cutoff = datetime.now() - timedelta(hours=1)
            self._metric_history[name] = [
                (t, v) for t, v in self._metric_history[name] if t > cutoff
            ]
    
    def get_metrics(self) -> Dict[str, float]:
        """获取所有指标"""
        with self._lock:
            return self._metrics.copy()
    
    def get_metric_history(self, name: str) -> List[tuple]:
        """获取指标历史"""
        with self._lock:
            return self._metric_history.get(name, []).copy()
    
    def set_alert_threshold(
        self,
        metric_name: str,
        threshold: float,
        comparison: str,
        level: DataQualityLevel = DataQualityLevel.HIGH,
    ) -> None:
        """设置告警阈值"""
        with self._lock:
            self._thresholds[metric_name] = AlertThreshold(
                metric_name=metric_name,
                threshold=threshold,
                comparison=comparison,
                level=level,
            )
        logger.debug(f"Set threshold: {metric_name} {comparison} {threshold}")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警"""
        with self._lock:
            return [
                {
                    'metric_name': a.metric_name,
                    'threshold': a.threshold.threshold,
                    'comparison': a.threshold.comparison,
                    'current_value': a.current_value,
                    'triggered_at': a.triggered_at.isoformat(),
                    'level': a.level.value,
                }
                for a in self._alerts
            ]
    
    def clear_alerts(self) -> None:
        """清除告警"""
        with self._lock:
            self._alerts.clear()
```

---

## 3. 数据剖析器 (profilers/data_profiler.py)

```python
"""
数据剖析器

自动分析数据集特征
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from scipy import stats

from ..interfaces import DataProfilerInterface

logger = logging.getLogger(__name__)


class DataProfiler(DataProfilerInterface):
    """
    数据剖析器
    
    使用示例:
        profiler = DataProfiler()
        profile = profiler.profile(data)
        
        # 检测漂移
        drift = profiler.detect_drift(baseline_profile, current_profile)
    """
    
    def profile(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        剖析数据
        
        Returns:
            Dict: 包含统计信息、分布、异常检测等
        """
        if not data:
            return {'error': 'Empty dataset'}
        
        df = pd.DataFrame(data)
        
        profile = {
            'dataset': {
                'rows': len(df),
                'columns': len(df.columns),
                'profiled_at': datetime.now().isoformat(),
            },
            'columns': {},
        }
        
        for column in df.columns:
            col_data = df[column]
            col_profile = self._profile_column(col_data)
            profile['columns'][column] = col_profile
        
        # 数据集级别统计
        profile['dataset']['total_nulls'] = int(df.isnull().sum().sum())
        profile['dataset']['null_rate'] = float(df.isnull().mean().mean())
        profile['dataset']['duplicate_rows'] = int(df.duplicated().sum())
        profile['dataset']['duplicate_rate'] = float(df.duplicated().mean())
        
        return profile
    
    def _profile_column(self, col: pd.Series) -> Dict[str, Any]:
        """剖析单列"""
        profile = {
            'dtype': str(col.dtype),
            'count': int(col.count()),
            'null_count': int(col.isnull().sum()),
            'null_rate': float(col.isnull().mean()),
            'unique_count': int(col.nunique()),
            'unique_rate': float(col.nunique() / len(col)) if len(col) > 0 else 0,
        }
        
        # 数值型列
        if pd.api.types.is_numeric_dtype(col):
            non_null = col.dropna()
            if len(non_null) > 0:
                profile['numeric'] = {
                    'mean': float(non_null.mean()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'median': float(non_null.median()),
                    'q1': float(non_null.quantile(0.25)),
                    'q3': float(non_null.quantile(0.75)),
                    'skewness': float(non_null.skew()),
                    'kurtosis': float(non_null.kurtosis()),
                    'zeros': int((non_null == 0).sum()),
                    'negatives': int((non_null < 0).sum()),
                }
                
                # 异常值检测 (IQR 方法)
                q1, q3 = non_null.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                
                profile['numeric']['outlier_count'] = int(len(outliers))
                profile['numeric']['outlier_rate'] = float(len(outliers) / len(non_null))
        
        # 字符串列
        elif col.dtype == 'object':
            non_null = col.dropna().astype(str)
            if len(non_null) > 0:
                lengths = non_null.str.len()
                profile['string'] = {
                    'avg_length': float(lengths.mean()),
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'empty_count': int((non_null == '').sum()),
                }
                
                # Top 值
                value_counts = non_null.value_counts()
                profile['top_values'] = value_counts.head(10).to_dict()
        
        # 日期列
        elif pd.api.types.is_datetime64_any_dtype(col):
            non_null = col.dropna()
            if len(non_null) > 0:
                profile['datetime'] = {
                    'min': str(non_null.min()),
                    'max': str(non_null.max()),
                    'range_days': (non_null.max() - non_null.min()).days,
                }
        
        return profile
    
    def compare_profiles(
        self,
        profile1: Dict[str, Any],
        profile2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """比较两个数据剖析结果"""
        comparison = {
            'dataset': {},
            'columns': {},
        }
        
        # 数据集级别比较
        for key in ['rows', 'null_rate', 'duplicate_rate']:
            if key in profile1['dataset'] and key in profile2['dataset']:
                v1 = profile1['dataset'][key]
                v2 = profile2['dataset'][key]
                comparison['dataset'][key] = {
                    'before': v1,
                    'after': v2,
                    'change': v2 - v1 if isinstance(v1, (int, float)) else None,
                    'change_pct': (v2 - v1) / v1 * 100 if v1 and isinstance(v1, (int, float)) else None,
                }
        
        # 列级别比较
        common_columns = set(profile1['columns'].keys()) & set(profile2['columns'].keys())
        
        for col in common_columns:
            col1 = profile1['columns'][col]
            col2 = profile2['columns'][col]
            
            col_comparison = {}
            
            # 比较数值统计
            if 'numeric' in col1 and 'numeric' in col2:
                for metric in ['mean', 'std', 'min', 'max']:
                    v1 = col1['numeric'].get(metric)
                    v2 = col2['numeric'].get(metric)
                    if v1 is not None and v2 is not None:
                        col_comparison[metric] = {
                            'before': v1,
                            'after': v2,
                            'change_pct': (v2 - v1) / abs(v1) * 100 if v1 != 0 else None,
                        }
            
            comparison['columns'][col] = col_comparison
        
        return comparison
    
    def detect_drift(
        self,
        baseline_profile: Dict[str, Any],
        current_profile: Dict[str, Any],
        threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        检测数据漂移
        
        使用 KS 检验和其他统计方法
        
        Returns:
            Dict: 列名 -> 漂移分数
        """
        drift_scores = {}
        
        common_columns = (
            set(baseline_profile['columns'].keys()) &
            set(current_profile['columns'].keys())
        )
        
        for col in common_columns:
            baseline_col = baseline_profile['columns'][col]
            current_col = current_profile['columns'][col]
            
            drift_score = 0.0
            
            # 数值列：比较均值和标准差变化
            if 'numeric' in baseline_col and 'numeric' in current_col:
                baseline_mean = baseline_col['numeric'].get('mean', 0)
                current_mean = current_col['numeric'].get('mean', 0)
                baseline_std = baseline_col['numeric'].get('std', 1) or 1
                
                # 标准化均值变化
                mean_drift = abs(current_mean - baseline_mean) / baseline_std
                
                # 标准差比值
                current_std = current_col['numeric'].get('std', 1) or 1
                std_ratio = max(current_std / baseline_std, baseline_std / current_std)
                
                drift_score = (mean_drift + std_ratio - 1) / 2
            
            # 类别列：比较分布变化
            elif 'top_values' in baseline_col and 'top_values' in current_col:
                baseline_dist = baseline_col['top_values']
                current_dist = current_col['top_values']
                
                # 计算 JS 散度
                all_values = set(baseline_dist.keys()) | set(current_dist.keys())
                
                if all_values:
                    baseline_total = sum(baseline_dist.values())
                    current_total = sum(current_dist.values())
                    
                    divergence = 0
                    for val in all_values:
                        p = baseline_dist.get(val, 0) / baseline_total if baseline_total else 0
                        q = current_dist.get(val, 0) / current_total if current_total else 0
                        m = (p + q) / 2
                        
                        if p > 0 and m > 0:
                            divergence += p * np.log(p / m)
                        if q > 0 and m > 0:
                            divergence += q * np.log(q / m)
                    
                    drift_score = divergence / 2
            
            drift_scores[col] = float(drift_score)
        
        return drift_scores
```

---

## 注意事项

1. 规则可组合、可复用
2. 监控指标使用 Prometheus 导出
3. 告警有冷却期避免重复
4. 剖析支持增量更新
5. 漂移检测使用多种统计方法

## 输出要求

请输出完整的可运行代码，包含：
1. 规则验证器
2. Schema 验证器
3. 质量监控
4. 数据剖析器
5. 完整测试

