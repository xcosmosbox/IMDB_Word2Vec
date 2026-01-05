"""
日志格式验证测试

测试日志格式是否符合 interfaces.yaml 中定义的契约
"""

import json
import unittest
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import re


class LogRecord:
    """日志记录数据类"""
    
    # 必需字段 (来自 interfaces.yaml)
    REQUIRED_FIELDS = {'timestamp', 'level', 'service', 'trace_id', 'message'}
    
    # 可选字段 (来自 interfaces.yaml)
    OPTIONAL_FIELDS = {'user_id', 'request_id', 'duration_ms', 'error_stack'}
    
    # 有效的日志级别 (来自 interfaces.yaml)
    VALID_LEVELS = {'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'}
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LogRecord':
        """从 JSON 字符串创建日志记录"""
        return cls(json.loads(json_str))
    
    def validate(self) -> List[str]:
        """验证日志记录，返回错误列表"""
        errors = []
        
        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in self.data:
                errors.append(f"缺少必需字段: {field}")
            elif self.data[field] is None:
                errors.append(f"必需字段不能为 None: {field}")
        
        # 验证 timestamp 格式
        if 'timestamp' in self.data:
            if not self._validate_timestamp(self.data['timestamp']):
                errors.append(f"无效的 timestamp 格式: {self.data['timestamp']}")
        
        # 验证 level
        if 'level' in self.data:
            level = self.data['level']
            if isinstance(level, str):
                level = level.upper()
            if level not in self.VALID_LEVELS:
                errors.append(f"无效的日志级别: {self.data['level']}")
        
        # 验证 duration_ms (如果存在)
        if 'duration_ms' in self.data:
            duration = self.data['duration_ms']
            if not isinstance(duration, (int, float)):
                errors.append(f"duration_ms 应该是数字: {type(duration)}")
            elif duration < 0:
                errors.append(f"duration_ms 不能为负数: {duration}")
        
        return errors
    
    def _validate_timestamp(self, timestamp: str) -> bool:
        """验证时间戳格式"""
        # 支持的格式
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',           # RFC3339 with microseconds
            '%Y-%m-%dT%H:%M:%SZ',              # RFC3339 without microseconds
            '%Y-%m-%dT%H:%M:%S.%f%z',          # RFC3339 with timezone
            '%Y-%m-%dT%H:%M:%S%z',             # RFC3339 with timezone, no ms
            '%Y-%m-%d %H:%M:%S.%f',            # Datetime with microseconds
            '%Y-%m-%d %H:%M:%S',               # Standard datetime
        ]
        
        for fmt in formats:
            try:
                datetime.strptime(timestamp, fmt)
                return True
            except ValueError:
                continue
        
        # 尝试 ISO 格式
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except ValueError:
            pass
        
        return False
    
    def is_error(self) -> bool:
        """检查是否是错误日志"""
        level = self.data.get('level', '').upper()
        return level in {'ERROR', 'FATAL'}
    
    def get_trace_id(self) -> Optional[str]:
        """获取 trace ID"""
        return self.data.get('trace_id')


class TestLogFormat(unittest.TestCase):
    """日志格式测试"""
    
    def test_valid_log_record(self):
        """测试有效的日志记录"""
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'recommend-service',
            'trace_id': 'abc123',
            'message': 'Processing recommendation request'
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 0, f"不应有错误: {errors}")
    
    def test_missing_required_fields(self):
        """测试缺少必需字段"""
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO'
            # 缺少 service, trace_id, message
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 3, "应该有 3 个缺失字段错误")
    
    def test_invalid_level(self):
        """测试无效的日志级别"""
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INVALID',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'Test message'
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn('无效的日志级别', errors[0])
    
    def test_all_valid_levels(self):
        """测试所有有效的日志级别"""
        for level in LogRecord.VALID_LEVELS:
            log = LogRecord({
                'timestamp': '2025-01-05T10:30:00.123456Z',
                'level': level,
                'service': 'test-service',
                'trace_id': 'abc123',
                'message': 'Test message'
            })
            
            errors = log.validate()
            self.assertEqual(len(errors), 0, f"级别 {level} 应该有效")
    
    def test_case_insensitive_level(self):
        """测试日志级别大小写不敏感"""
        for level in ['info', 'INFO', 'Info']:
            log = LogRecord({
                'timestamp': '2025-01-05T10:30:00.123456Z',
                'level': level,
                'service': 'test-service',
                'trace_id': 'abc123',
                'message': 'Test message'
            })
            
            errors = log.validate()
            self.assertEqual(len(errors), 0, f"级别 {level} 应该有效")
    
    def test_invalid_timestamp(self):
        """测试无效的时间戳"""
        log = LogRecord({
            'timestamp': 'invalid-timestamp',
            'level': 'INFO',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'Test message'
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn('无效的 timestamp', errors[0])
    
    def test_valid_timestamp_formats(self):
        """测试各种有效的时间戳格式"""
        valid_timestamps = [
            '2025-01-05T10:30:00.123456Z',
            '2025-01-05T10:30:00Z',
            '2025-01-05T10:30:00.123456+08:00',
            '2025-01-05 10:30:00.123456',
            '2025-01-05 10:30:00',
        ]
        
        for ts in valid_timestamps:
            log = LogRecord({
                'timestamp': ts,
                'level': 'INFO',
                'service': 'test-service',
                'trace_id': 'abc123',
                'message': 'Test message'
            })
            
            errors = log.validate()
            self.assertEqual(len(errors), 0, f"时间戳 {ts} 应该有效")
    
    def test_optional_fields(self):
        """测试可选字段"""
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'recommend-service',
            'trace_id': 'abc123',
            'message': 'Processing request',
            'user_id': 'user_123',
            'request_id': 'req_456',
            'duration_ms': 150.5
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 0)
    
    def test_invalid_duration_ms(self):
        """测试无效的 duration_ms"""
        # 非数字类型
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'Test message',
            'duration_ms': 'not a number'
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn('duration_ms 应该是数字', errors[0])
        
        # 负数
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'Test message',
            'duration_ms': -100
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn('duration_ms 不能为负数', errors[0])
    
    def test_is_error(self):
        """测试错误日志检测"""
        error_log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'ERROR',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'An error occurred'
        })
        
        self.assertTrue(error_log.is_error())
        
        fatal_log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'FATAL',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'A fatal error occurred'
        })
        
        self.assertTrue(fatal_log.is_error())
        
        info_log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'INFO',
            'service': 'test-service',
            'trace_id': 'abc123',
            'message': 'Just info'
        })
        
        self.assertFalse(info_log.is_error())
    
    def test_from_json(self):
        """测试从 JSON 创建日志记录"""
        json_str = '''{
            "timestamp": "2025-01-05T10:30:00.123456Z",
            "level": "INFO",
            "service": "recommend-service",
            "trace_id": "abc123",
            "message": "Test message"
        }'''
        
        log = LogRecord.from_json(json_str)
        errors = log.validate()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(log.data['service'], 'recommend-service')
    
    def test_error_with_stack(self):
        """测试带堆栈的错误日志"""
        log = LogRecord({
            'timestamp': '2025-01-05T10:30:00.123456Z',
            'level': 'ERROR',
            'service': 'recommend-service',
            'trace_id': 'abc123',
            'message': 'Database connection failed',
            'error_stack': '''Traceback (most recent call last):
  File "app.py", line 100, in connect
    conn = psycopg2.connect(...)
psycopg2.OperationalError: connection refused'''
        })
        
        errors = log.validate()
        self.assertEqual(len(errors), 0)
        self.assertTrue(log.is_error())
        self.assertIn('error_stack', log.data)


class TestLogFormatExamples(unittest.TestCase):
    """实际日志格式示例测试"""
    
    def test_go_service_log(self):
        """测试 Go 服务日志格式"""
        log_json = '''
        {
            "timestamp": "2025-01-05T10:30:00.123456Z",
            "level": "INFO",
            "service": "recommend-service",
            "trace_id": "trace-abc-123",
            "message": "Received recommendation request",
            "user_id": "user_12345",
            "request_id": "req_67890",
            "duration_ms": 45.2,
            "method": "POST",
            "path": "/api/v1/recommend",
            "status_code": 200
        }
        '''
        
        log = LogRecord.from_json(log_json)
        errors = log.validate()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(log.data['service'], 'recommend-service')
        self.assertFalse(log.is_error())
    
    def test_python_inference_log(self):
        """测试 Python 推理服务日志格式"""
        log_json = '''
        {
            "timestamp": "2025-01-05T10:30:00.123456Z",
            "level": "INFO",
            "service": "ugt-inference",
            "trace_id": "trace-def-456",
            "message": "Inference completed",
            "model": "ugt-v1",
            "batch_size": 32,
            "latency_ms": 85.5,
            "gpu_memory_mb": 8192,
            "status": "success"
        }
        '''
        
        log = LogRecord.from_json(log_json)
        errors = log.validate()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(log.data['service'], 'ugt-inference')
    
    def test_error_log_with_stack_trace(self):
        """测试带堆栈跟踪的错误日志"""
        log_json = '''
        {
            "timestamp": "2025-01-05T10:30:00.123456Z",
            "level": "ERROR",
            "service": "user-service",
            "trace_id": "trace-ghi-789",
            "message": "Failed to fetch user data",
            "user_id": "user_99999",
            "error_stack": "Error: User not found\\n    at UserService.getUser (user.go:150)\\n    at main (main.go:50)"
        }
        '''
        
        log = LogRecord.from_json(log_json)
        errors = log.validate()
        
        self.assertEqual(len(errors), 0)
        self.assertTrue(log.is_error())
        self.assertIn('error_stack', log.data)
    
    def test_warning_log(self):
        """测试警告日志"""
        log_json = '''
        {
            "timestamp": "2025-01-05T10:30:00.123456Z",
            "level": "WARN",
            "service": "item-service",
            "trace_id": "trace-jkl-012",
            "message": "Cache miss, falling back to database",
            "request_id": "req_11111"
        }
        '''
        
        log = LogRecord.from_json(log_json)
        errors = log.validate()
        
        self.assertEqual(len(errors), 0)
        self.assertFalse(log.is_error())


class TestLogLabels(unittest.TestCase):
    """日志标签测试 (符合 interfaces.yaml)"""
    
    # 有效的标签 (来自 interfaces.yaml)
    VALID_LABELS = {'app', 'env', 'pod', 'namespace'}
    
    def test_valid_labels(self):
        """测试有效的标签"""
        labels = {
            'app': 'recommend-service',
            'env': 'production',
            'pod': 'recommend-service-abc123',
            'namespace': 'recommend-prod'
        }
        
        for label, value in labels.items():
            self.assertIn(label, self.VALID_LABELS)
            self.assertIsInstance(value, str)
            self.assertTrue(len(value) > 0)
    
    def test_label_values_format(self):
        """测试标签值格式"""
        # 标签值应该是有效的 Prometheus 标签值
        valid_values = [
            'recommend-service',
            'production',
            'recommend-prod',
            'pod-abc-123',
        ]
        
        label_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_-]*$')
        
        for value in valid_values:
            # 标签值可以包含字母、数字、下划线和连字符
            self.assertTrue(
                all(c.isalnum() or c in '-_' for c in value),
                f"标签值 {value} 包含无效字符"
            )


if __name__ == '__main__':
    unittest.main()

