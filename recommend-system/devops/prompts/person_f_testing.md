# Person F: 性能测试

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **性能测试套件**，包括负载测试、压力测试、基准测试和性能报告。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
testing:
  load_scenarios:
    - name: baseline
      rps: 100
      duration: 5m
    - name: stress
      rps: 1000
      duration: 10m
  
  sla:
    availability: 99.9%
    p50_latency: 50ms
    p99_latency: 200ms
    error_rate: 0.1%
```

---

## 你的任务

```
devops/testing/
├── load/
│   ├── k6/
│   │   ├── scenarios/
│   │   │   ├── baseline.js
│   │   │   ├── stress.js
│   │   │   └── spike.js
│   │   ├── lib/
│   │   │   ├── api.js
│   │   │   └── utils.js
│   │   └── config.js
│   ├── locust/
│   │   ├── locustfile.py
│   │   └── config.py
│   └── scripts/
│       └── run-tests.sh
├── benchmark/
│   ├── api-benchmark.go
│   ├── inference-benchmark.py
│   └── database-benchmark.sql
└── reports/
    ├── templates/
    │   └── report.html
    └── generate-report.py
```

---

## 1. K6 负载测试 - 基线场景 (k6/scenarios/baseline.js)

```javascript
/**
 * 基线负载测试
 * 
 * 目标: 验证系统在正常负载下的性能
 * RPS: 100
 * 持续时间: 5 分钟
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomItem, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// 自定义指标
const errorRate = new Rate('error_rate');
const recommendLatency = new Trend('recommend_latency');
const searchLatency = new Trend('search_latency');
const feedbackCounter = new Counter('feedback_count');

// 配置
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// 测试数据
const USER_IDS = Array.from({ length: 1000 }, (_, i) => `user_${i}`);
const ITEM_IDS = Array.from({ length: 10000 }, (_, i) => `item_${i}`);
const SEARCH_TERMS = ['action', 'comedy', 'drama', 'sci-fi', 'horror', 'romance'];

// 负载配置
export const options = {
  scenarios: {
    baseline: {
      executor: 'constant-arrival-rate',
      rate: 100,           // 100 RPS
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],      // 错误率 < 1%
    http_req_duration: ['p(95)<500'],     // P95 < 500ms
    'recommend_latency': ['p(99)<200'],   // 推荐 P99 < 200ms
    'search_latency': ['p(99)<300'],      // 搜索 P99 < 300ms
    'error_rate': ['rate<0.001'],         // 自定义错误率 < 0.1%
  },
};

// 公共请求头
const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_KEY}`,
};

// 主测试函数
export default function () {
  // 随机选择测试场景
  const scenario = randomItem(['recommend', 'search', 'feedback', 'detail']);
  
  switch (scenario) {
    case 'recommend':
      testRecommendations();
      break;
    case 'search':
      testSearch();
      break;
    case 'feedback':
      testFeedback();
      break;
    case 'detail':
      testItemDetail();
      break;
  }
  
  sleep(randomIntBetween(1, 3));
}

// 测试推荐接口
function testRecommendations() {
  const userId = randomItem(USER_IDS);
  const payload = JSON.stringify({
    user_id: userId,
    limit: 20,
    scene: 'home',
  });
  
  const startTime = Date.now();
  
  const response = http.post(`${BASE_URL}/api/v1/recommend`, payload, { headers });
  
  const duration = Date.now() - startTime;
  recommendLatency.add(duration);
  
  const success = check(response, {
    'recommend status is 200': (r) => r.status === 200,
    'recommend has recommendations': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.recommendations && body.data.recommendations.length > 0;
      } catch {
        return false;
      }
    },
    'recommend latency < 200ms': () => duration < 200,
  });
  
  errorRate.add(!success);
}

// 测试搜索接口
function testSearch() {
  const query = randomItem(SEARCH_TERMS);
  const startTime = Date.now();
  
  const response = http.get(`${BASE_URL}/api/v1/items/search?q=${query}&limit=20`, { headers });
  
  const duration = Date.now() - startTime;
  searchLatency.add(duration);
  
  const success = check(response, {
    'search status is 200': (r) => r.status === 200,
    'search has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && Array.isArray(body.data);
      } catch {
        return false;
      }
    },
    'search latency < 300ms': () => duration < 300,
  });
  
  errorRate.add(!success);
}

// 测试反馈接口
function testFeedback() {
  const payload = JSON.stringify({
    user_id: randomItem(USER_IDS),
    item_id: randomItem(ITEM_IDS),
    action: randomItem(['click', 'view', 'like']),
  });
  
  const response = http.post(`${BASE_URL}/api/v1/feedback`, payload, { headers });
  
  const success = check(response, {
    'feedback status is 200 or 204': (r) => r.status === 200 || r.status === 204,
  });
  
  if (success) {
    feedbackCounter.add(1);
  }
  errorRate.add(!success);
}

// 测试物品详情接口
function testItemDetail() {
  const itemId = randomItem(ITEM_IDS);
  
  const response = http.get(`${BASE_URL}/api/v1/items/${itemId}`, { headers });
  
  const success = check(response, {
    'detail status is 200': (r) => r.status === 200,
    'detail has item data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.id;
      } catch {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

// 测试生命周期钩子
export function setup() {
  console.log('Starting baseline load test...');
  console.log(`Target: ${BASE_URL}`);
  
  // 验证目标服务可用
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error('Target service is not healthy');
  }
  
  return { startTime: Date.now() };
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(2)} seconds`);
}
```

---

## 2. K6 压力测试 (k6/scenarios/stress.js)

```javascript
/**
 * 压力测试
 * 
 * 目标: 找到系统的性能极限
 * RPS: 逐步增加到 1000+
 * 持续时间: 10 分钟
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

export const options = {
  scenarios: {
    stress: {
      executor: 'ramping-arrival-rate',
      startRate: 50,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 1000,
      stages: [
        { duration: '1m', target: 100 },   // 预热
        { duration: '2m', target: 300 },   // 增加负载
        { duration: '2m', target: 500 },   // 中等负载
        { duration: '2m', target: 800 },   // 高负载
        { duration: '2m', target: 1000 },  // 峰值负载
        { duration: '1m', target: 0 },     // 恢复
      ],
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],        // 允许 5% 错误率
    http_req_duration: ['p(95)<1000'],     // P95 < 1s
    'error_rate': ['rate<0.05'],
  },
};

const headers = {
  'Content-Type': 'application/json',
};

export default function () {
  const userId = `user_${Math.floor(Math.random() * 10000)}`;
  
  const payload = JSON.stringify({
    user_id: userId,
    limit: 20,
    scene: 'home',
  });
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/api/v1/recommend`, payload, { headers });
  const duration = Date.now() - startTime;
  
  responseTime.add(duration);
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
  });
  
  errorRate.add(!success);
  
  sleep(0.1);
}

// 自定义摘要报告
export function handleSummary(data) {
  return {
    'stress-test-summary.json': JSON.stringify(data, null, 2),
    stdout: generateTextSummary(data),
  };
}

function generateTextSummary(data) {
  const metrics = data.metrics;
  
  return `
================================================================================
                          STRESS TEST SUMMARY
================================================================================

Duration: ${(data.state.testRunDurationMs / 1000).toFixed(2)}s
VUs Peak: ${data.metrics.vus_max?.values?.max || 'N/A'}

HTTP Requests:
  Total: ${metrics.http_reqs?.values?.count || 0}
  Rate: ${(metrics.http_reqs?.values?.rate || 0).toFixed(2)}/s

Response Time:
  Avg: ${(metrics.http_req_duration?.values?.avg || 0).toFixed(2)}ms
  P50: ${(metrics.http_req_duration?.values['p(50)'] || 0).toFixed(2)}ms
  P90: ${(metrics.http_req_duration?.values['p(90)'] || 0).toFixed(2)}ms
  P95: ${(metrics.http_req_duration?.values['p(95)'] || 0).toFixed(2)}ms
  P99: ${(metrics.http_req_duration?.values['p(99)'] || 0).toFixed(2)}ms
  Max: ${(metrics.http_req_duration?.values?.max || 0).toFixed(2)}ms

Errors:
  Failed Requests: ${metrics.http_req_failed?.values?.passes || 0}
  Error Rate: ${((metrics.error_rate?.values?.rate || 0) * 100).toFixed(4)}%

Thresholds:
${Object.entries(data.metrics)
  .filter(([_, v]) => v.thresholds)
  .map(([name, v]) => {
    const passed = Object.values(v.thresholds).every(t => t.ok);
    return `  ${passed ? '✓' : '✗'} ${name}`;
  })
  .join('\n')}

================================================================================
`;
}
```

---

## 3. Locust 负载测试 (locust/locustfile.py)

```python
"""
Locust 负载测试脚本
支持 Web UI 和分布式测试
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import random
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendUser(HttpUser):
    """模拟推荐系统用户"""
    
    wait_time = between(1, 3)  # 请求间隔 1-3 秒
    
    def on_start(self):
        """用户启动时调用"""
        self.user_id = f"user_{random.randint(1, 100000)}"
        self.headers = {
            "Content-Type": "application/json",
        }
    
    @task(5)  # 权重 5
    def get_recommendations(self):
        """获取推荐"""
        payload = {
            "user_id": self.user_id,
            "limit": 20,
            "scene": random.choice(["home", "search", "detail"]),
        }
        
        with self.client.post(
            "/api/v1/recommend",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("data", {}).get("recommendations"):
                    response.success()
                else:
                    response.failure("No recommendations returned")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)  # 权重 3
    def search_items(self):
        """搜索物品"""
        query = random.choice([
            "action", "comedy", "drama", "thriller",
            "sci-fi", "horror", "romance", "documentary"
        ])
        
        with self.client.get(
            f"/api/v1/items/search?q={query}&limit=20",
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)  # 权重 2
    def submit_feedback(self):
        """提交反馈"""
        payload = {
            "user_id": self.user_id,
            "item_id": f"item_{random.randint(1, 10000)}",
            "action": random.choice(["view", "click", "like"]),
        }
        
        with self.client.post(
            "/api/v1/feedback",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 204]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)  # 权重 1
    def get_item_detail(self):
        """获取物品详情"""
        item_id = f"item_{random.randint(1, 10000)}"
        
        with self.client.get(
            f"/api/v1/items/{item_id}",
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # 物品不存在是正常的
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class AdminUser(HttpUser):
    """模拟管理员用户（低频）"""
    
    wait_time = between(5, 10)
    weight = 1  # 低权重
    
    @task
    def get_dashboard(self):
        """获取仪表盘数据"""
        self.client.get("/api/admin/v1/dashboard")
    
    @task
    def list_users(self):
        """列出用户"""
        self.client.get("/api/admin/v1/users?page=1&page_size=20")


# 测试事件钩子
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时"""
    logger.info("Load test starting...")
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running in distributed mode")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时"""
    logger.info("Load test completed")
    
    # 打印统计摘要
    stats = environment.stats.total
    logger.info(f"Total requests: {stats.num_requests}")
    logger.info(f"Failures: {stats.num_failures}")
    logger.info(f"Avg response time: {stats.avg_response_time:.2f}ms")
    logger.info(f"Requests/s: {stats.total_rps:.2f}")
```

---

## 4. 性能报告生成器 (reports/generate-report.py)

```python
#!/usr/bin/env python3
"""
性能测试报告生成器
"""

import json
import os
from datetime import datetime
from pathlib import Path
from jinja2 import Template
import argparse


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>性能测试报告 - {{ report_time }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 32px; font-weight: bold; }
        .metric-label { font-size: 14px; opacity: 0.9; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .pass { color: #4CAF50; }
        .fail { color: #f44336; }
        .chart { height: 300px; margin: 20px 0; }
        .sla-table td:last-child { font-weight: bold; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>🚀 性能测试报告</h1>
        <p>生成时间: {{ report_time }} | 测试类型: {{ test_type }} | 持续时间: {{ duration }}s</p>
        
        <h2>📊 概览</h2>
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{{ total_requests | format_number }}</div>
                <div class="metric-label">总请求数</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="metric-value">{{ rps | round(2) }}</div>
                <div class="metric-label">RPS</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
                <div class="metric-value">{{ avg_latency | round(2) }}ms</div>
                <div class="metric-label">平均延迟</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
                <div class="metric-value">{{ error_rate | round(4) }}%</div>
                <div class="metric-label">错误率</div>
            </div>
        </div>
        
        <h2>⏱️ 延迟分布</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>值</th>
            </tr>
            <tr><td>P50</td><td>{{ p50 | round(2) }}ms</td></tr>
            <tr><td>P90</td><td>{{ p90 | round(2) }}ms</td></tr>
            <tr><td>P95</td><td>{{ p95 | round(2) }}ms</td></tr>
            <tr><td>P99</td><td>{{ p99 | round(2) }}ms</td></tr>
            <tr><td>Max</td><td>{{ max_latency | round(2) }}ms</td></tr>
        </table>
        
        <h2>🎯 SLA 检查</h2>
        <table class="sla-table">
            <tr>
                <th>指标</th>
                <th>目标</th>
                <th>实际</th>
                <th>状态</th>
            </tr>
            {% for check in sla_checks %}
            <tr>
                <td>{{ check.name }}</td>
                <td>{{ check.target }}</td>
                <td>{{ check.actual }}</td>
                <td class="{{ 'pass' if check.passed else 'fail' }}">
                    {{ '✓ PASS' if check.passed else '✗ FAIL' }}
                </td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>📈 请求趋势</h2>
        <canvas id="rpsChart" class="chart"></canvas>
        
        <h2>📋 阈值检查</h2>
        <table>
            <tr>
                <th>阈值</th>
                <th>状态</th>
            </tr>
            {% for threshold in thresholds %}
            <tr>
                <td>{{ threshold.name }}</td>
                <td class="{{ 'pass' if threshold.passed else 'fail' }}">
                    {{ '✓ PASS' if threshold.passed else '✗ FAIL' }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <script>
        // RPS 趋势图
        new Chart(document.getElementById('rpsChart'), {
            type: 'line',
            data: {
                labels: {{ rps_trend_labels | tojson }},
                datasets: [{
                    label: 'RPS',
                    data: {{ rps_trend_values | tojson }},
                    borderColor: '#667eea',
                    fill: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
    </script>
</body>
</html>
"""


def format_number(value):
    """格式化数字"""
    return "{:,}".format(int(value))


def generate_report(data_file: str, output_file: str):
    """生成 HTML 报告"""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    # 提取指标
    http_reqs = metrics.get('http_reqs', {}).get('values', {})
    http_duration = metrics.get('http_req_duration', {}).get('values', {})
    http_failed = metrics.get('http_req_failed', {}).get('values', {})
    
    # SLA 检查
    sla_checks = [
        {
            'name': '可用性',
            'target': '≥ 99.9%',
            'actual': f"{(1 - http_failed.get('rate', 0)) * 100:.2f}%",
            'passed': http_failed.get('rate', 0) < 0.001,
        },
        {
            'name': 'P50 延迟',
            'target': '≤ 50ms',
            'actual': f"{http_duration.get('p(50)', 0):.2f}ms",
            'passed': http_duration.get('p(50)', 0) <= 50,
        },
        {
            'name': 'P99 延迟',
            'target': '≤ 200ms',
            'actual': f"{http_duration.get('p(99)', 0):.2f}ms",
            'passed': http_duration.get('p(99)', 0) <= 200,
        },
        {
            'name': '错误率',
            'target': '≤ 0.1%',
            'actual': f"{http_failed.get('rate', 0) * 100:.4f}%",
            'passed': http_failed.get('rate', 0) <= 0.001,
        },
    ]
    
    # 阈值检查
    thresholds = []
    for name, metric in metrics.items():
        if metric.get('thresholds'):
            for t_name, t_val in metric['thresholds'].items():
                thresholds.append({
                    'name': f"{name}: {t_name}",
                    'passed': t_val.get('ok', False),
                })
    
    # 渲染模板
    template = Template(REPORT_TEMPLATE)
    template.globals['format_number'] = format_number
    
    html = template.render(
        report_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        test_type='Load Test',
        duration=data.get('state', {}).get('testRunDurationMs', 0) / 1000,
        total_requests=http_reqs.get('count', 0),
        rps=http_reqs.get('rate', 0),
        avg_latency=http_duration.get('avg', 0),
        error_rate=http_failed.get('rate', 0) * 100,
        p50=http_duration.get('p(50)', 0),
        p90=http_duration.get('p(90)', 0),
        p95=http_duration.get('p(95)', 0),
        p99=http_duration.get('p(99)', 0),
        max_latency=http_duration.get('max', 0),
        sla_checks=sla_checks,
        thresholds=thresholds,
        rps_trend_labels=['0s', '30s', '60s', '90s', '120s'],  # 示例
        rps_trend_values=[50, 80, 100, 95, 100],  # 示例
    )
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate performance test report')
    parser.add_argument('input', help='K6 JSON output file')
    parser.add_argument('-o', '--output', default='report.html', help='Output HTML file')
    
    args = parser.parse_args()
    generate_report(args.input, args.output)
```

---

## 5. 测试运行脚本 (load/scripts/run-tests.sh)

```bash
#!/bin/bash
# =============================================================================
# 性能测试运行脚本
# =============================================================================

set -euo pipefail

# 配置
K6_VERSION="0.47.0"
BASE_URL="${BASE_URL:-http://localhost:8080}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 检查 K6 安装
check_k6() {
    if ! command -v k6 &> /dev/null; then
        error "K6 not found. Please install K6 first."
        echo "  brew install k6  # macOS"
        echo "  docker pull grafana/k6  # Docker"
        exit 1
    fi
    log "K6 version: $(k6 version)"
}

# 健康检查
health_check() {
    log "Checking target health: $BASE_URL/health"
    if ! curl -sf "$BASE_URL/health" > /dev/null; then
        error "Target is not healthy"
        exit 1
    fi
    log "Target is healthy"
}

# 运行基线测试
run_baseline() {
    log "Running baseline test..."
    
    mkdir -p "$RESULTS_DIR"
    
    k6 run \
        --out json="$RESULTS_DIR/baseline_${TIMESTAMP}.json" \
        --env BASE_URL="$BASE_URL" \
        scenarios/baseline.js \
        2>&1 | tee "$RESULTS_DIR/baseline_${TIMESTAMP}.log"
    
    log "Baseline test completed"
}

# 运行压力测试
run_stress() {
    log "Running stress test..."
    
    mkdir -p "$RESULTS_DIR"
    
    k6 run \
        --out json="$RESULTS_DIR/stress_${TIMESTAMP}.json" \
        --env BASE_URL="$BASE_URL" \
        scenarios/stress.js \
        2>&1 | tee "$RESULTS_DIR/stress_${TIMESTAMP}.log"
    
    log "Stress test completed"
}

# 运行峰值测试
run_spike() {
    log "Running spike test..."
    
    mkdir -p "$RESULTS_DIR"
    
    k6 run \
        --out json="$RESULTS_DIR/spike_${TIMESTAMP}.json" \
        --env BASE_URL="$BASE_URL" \
        scenarios/spike.js \
        2>&1 | tee "$RESULTS_DIR/spike_${TIMESTAMP}.log"
    
    log "Spike test completed"
}

# 生成报告
generate_reports() {
    log "Generating reports..."
    
    for json_file in "$RESULTS_DIR"/*.json; do
        if [[ -f "$json_file" ]]; then
            html_file="${json_file%.json}.html"
            python3 ../reports/generate-report.py "$json_file" -o "$html_file"
        fi
    done
    
    log "Reports generated in $RESULTS_DIR"
}

# 主函数
main() {
    local test_type="${1:-baseline}"
    
    log "Starting performance test suite"
    log "Test type: $test_type"
    log "Target: $BASE_URL"
    
    check_k6
    health_check
    
    case "$test_type" in
        baseline)
            run_baseline
            ;;
        stress)
            run_stress
            ;;
        spike)
            run_spike
            ;;
        all)
            run_baseline
            run_stress
            run_spike
            ;;
        *)
            error "Unknown test type: $test_type"
            echo "Usage: $0 [baseline|stress|spike|all]"
            exit 1
            ;;
    esac
    
    generate_reports
    
    log "All tests completed!"
}

main "$@"
```

---

## 注意事项

1. K6 脚本使用 ES6 模块语法
2. 自定义指标监控关键性能点
3. 压力测试逐步增加负载
4. 报告包含 SLA 检查结果
5. 支持 CI/CD 集成

## 输出要求

请输出完整的性能测试套件，包含：
1. K6 测试脚本
2. Locust 测试脚本
3. 报告生成器
4. 运行脚本

