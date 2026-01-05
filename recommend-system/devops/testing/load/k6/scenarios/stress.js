/**
 * 压力测试
 * 
 * 目标: 找到系统的性能极限，验证系统在高负载下的稳定性
 * RPS: 逐步增加到 1000
 * 持续时间: 10 分钟
 * 
 * 阶段:
 * 1. 预热 (1min): 50 → 100 RPS
 * 2. 增加负载 (2min): 100 → 300 RPS
 * 3. 中等负载 (2min): 300 → 500 RPS
 * 4. 高负载 (2min): 500 → 800 RPS
 * 5. 峰值负载 (2min): 800 → 1000 RPS
 * 6. 恢复 (1min): 1000 → 0 RPS
 */

import { sleep, group } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';
import { loadScenarios, getEnvConfig } from '../config.js';
import { ApiClient, getRecommendations, searchItems, submitFeedback, getItemDetail, healthCheck } from '../lib/api.js';
import {
  randomUserId,
  randomItemId,
  randomSearchTerm,
  randomScene,
  randomFeedbackAction,
  randomContext,
  weightedRandom,
  randomSleep,
} from '../lib/utils.js';

// ============================================================================
// 自定义指标
// ============================================================================

const stressResponseTime = new Trend('stress_response_time', true);
const stressErrorRate = new Rate('stress_error_rate');
const stressRequestCount = new Counter('stress_request_count');
const degradationDetected = new Counter('degradation_detected');

// ============================================================================
// 测试配置
// ============================================================================

export const options = {
  scenarios: {
    stress: loadScenarios.stress,
  },
  
  thresholds: {
    // 压力测试允许更高的错误率和延迟
    http_req_failed: ['rate<0.05'],           // 允许 5% 错误率
    http_req_duration: ['p(95)<1000'],        // P95 < 1s
    'stress_response_time': ['p(99)<2000'],   // P99 < 2s
    'stress_error_rate': ['rate<0.05'],       // 错误率 < 5%
  },
  
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  
  tags: {
    testType: 'stress',
    environment: __ENV.TEST_ENV || 'local',
  },
};

// ============================================================================
// 测试生命周期
// ============================================================================

export function setup() {
  console.log('========================================');
  console.log('        STRESS TEST STARTING           ');
  console.log('========================================');
  
  const envConfig = getEnvConfig();
  console.log(`Target Environment: ${__ENV.TEST_ENV || 'local'}`);
  console.log(`Base URL: ${envConfig.baseUrl}`);
  console.log(`Peak RPS: 1000`);
  console.log(`Duration: 10 minutes`);
  console.log('');
  console.log('Stages:');
  console.log('  1. Warm-up (1m): 50 → 100 RPS');
  console.log('  2. Ramp-up (2m): 100 → 300 RPS');
  console.log('  3. Medium (2m): 300 → 500 RPS');
  console.log('  4. High (2m): 500 → 800 RPS');
  console.log('  5. Peak (2m): 800 → 1000 RPS');
  console.log('  6. Recovery (1m): 1000 → 0 RPS');
  console.log('');
  
  // 健康检查
  const client = new ApiClient();
  const health = healthCheck(client);
  
  if (!health.success) {
    throw new Error('Target service is not healthy! Aborting test.');
  }
  
  console.log('Health check passed. Starting stress test...');
  console.log('========================================');
  
  return {
    startTime: Date.now(),
    envConfig: envConfig,
    stageMetrics: [],
  };
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('========================================');
  console.log('        STRESS TEST COMPLETED          ');
  console.log('========================================');
  console.log(`Total Duration: ${duration.toFixed(2)} seconds`);
  console.log('========================================');
}

// ============================================================================
// 主测试函数
// ============================================================================

export default function (data) {
  const client = new ApiClient();
  
  // 执行混合负载场景
  const scenario = weightedRandom({
    recommend: 6,       // 60% 推荐
    search: 2,          // 20% 搜索
    feedback: 1,        // 10% 反馈
    item_detail: 1,     // 10% 物品详情
  });
  
  let success = false;
  let latency = 0;
  
  group(`stress_${scenario}`, () => {
    const startTime = Date.now();
    
    switch (scenario) {
      case 'recommend':
        const recResult = stressTestRecommendations(client);
        success = recResult.success;
        latency = recResult.latency;
        break;
        
      case 'search':
        const searchResult = stressTestSearch(client);
        success = searchResult.success;
        latency = searchResult.latency;
        break;
        
      case 'feedback':
        const fbResult = stressTestFeedback(client);
        success = fbResult.success;
        latency = fbResult.latency;
        break;
        
      case 'item_detail':
        const itemResult = stressTestItemDetail(client);
        success = itemResult.success;
        latency = itemResult.latency;
        break;
    }
  });
  
  // 记录指标
  stressResponseTime.add(latency);
  stressErrorRate.add(!success);
  stressRequestCount.add(1);
  
  // 检测性能降级
  if (latency > 1000) {
    degradationDetected.add(1);
  }
  
  // 短暂间隔
  sleep(0.1);
}

// ============================================================================
// 压力测试场景
// ============================================================================

function stressTestRecommendations(client) {
  const userId = randomUserId();
  
  return getRecommendations(client, {
    userId: userId,
    limit: 20,
    scene: randomScene(),
    context: randomContext(),
  });
}

function stressTestSearch(client) {
  return searchItems(client, {
    query: randomSearchTerm(),
    limit: 20,
  });
}

function stressTestFeedback(client) {
  return submitFeedback(client, {
    userId: randomUserId(),
    itemId: randomItemId(),
    action: randomFeedbackAction(),
  });
}

function stressTestItemDetail(client) {
  return getItemDetail(client, randomItemId());
}

// ============================================================================
// 自定义摘要报告
// ============================================================================

export function handleSummary(data) {
  const summary = generateStressSummary(data);
  
  return {
    'stress-test-summary.json': JSON.stringify(data, null, 2),
    stdout: summary,
  };
}

function generateStressSummary(data) {
  const metrics = data.metrics;
  const httpReqs = metrics.http_reqs?.values || {};
  const httpDuration = metrics.http_req_duration?.values || {};
  const httpFailed = metrics.http_req_failed?.values || {};
  const stressTime = metrics.stress_response_time?.values || {};
  const degradation = metrics.degradation_detected?.values || {};
  
  // 计算性能等级
  const maxLatency = httpDuration.max || 0;
  const errorRate = (httpFailed.rate || 0) * 100;
  
  let performanceGrade = 'A';
  if (maxLatency > 5000 || errorRate > 10) {
    performanceGrade = 'F';
  } else if (maxLatency > 2000 || errorRate > 5) {
    performanceGrade = 'D';
  } else if (maxLatency > 1000 || errorRate > 2) {
    performanceGrade = 'C';
  } else if (maxLatency > 500 || errorRate > 1) {
    performanceGrade = 'B';
  }
  
  // 阈值检查结果
  const thresholdResults = [];
  for (const [name, metric] of Object.entries(metrics)) {
    if (metric.thresholds) {
      for (const [threshold, result] of Object.entries(metric.thresholds)) {
        thresholdResults.push({
          name: `${name}: ${threshold}`,
          passed: result.ok,
        });
      }
    }
  }
  
  return `
╔══════════════════════════════════════════════════════════════════════════════╗
║                           STRESS TEST SUMMARY                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Test Duration: ${((data.state?.testRunDurationMs || 0) / 1000).toFixed(2).padEnd(15)}s                                     ║
║  Peak VUs: ${(metrics.vus_max?.values?.max || 0).toString().padEnd(20)}                                       ║
║  Performance Grade: ${performanceGrade.padEnd(15)}                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            THROUGHPUT                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Requests: ${(httpReqs.count || 0).toString().padEnd(15)}                                      ║
║  Peak RPS: ${(httpReqs.rate || 0).toFixed(2).padEnd(20)} req/s                                  ║
║  Degradation Events: ${(degradation.count || 0).toString().padEnd(15)}                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            LATENCY (ms)                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Average: ${(httpDuration.avg || 0).toFixed(2).padEnd(15)}                                           ║
║  Median: ${(httpDuration.med || 0).toFixed(2).padEnd(16)}                                           ║
║  P90: ${(httpDuration['p(90)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  P95: ${(httpDuration['p(95)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  P99: ${(httpDuration['p(99)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  Max: ${(httpDuration.max || 0).toFixed(2).padEnd(19)}                                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            ERROR ANALYSIS                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Failed Requests: ${(httpFailed.passes || 0).toString().padEnd(15)}                                   ║
║  Error Rate: ${errorRate.toFixed(4).padEnd(18)}%                                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            THRESHOLD CHECKS                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
${thresholdResults.slice(0, 5).map(t => `║  ${t.passed ? '✓' : '✗'} ${t.name.substring(0, 65).padEnd(65)}  ║`).join('\n')}
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ANALYSIS:                                                                   ║
║  ${getAnalysisMessage(performanceGrade, maxLatency, errorRate).padEnd(74)}║
╚══════════════════════════════════════════════════════════════════════════════╝
`;
}

function getAnalysisMessage(grade, maxLatency, errorRate) {
  switch (grade) {
    case 'A':
      return 'Excellent! System handled stress test with minimal degradation.';
    case 'B':
      return 'Good. System showed acceptable performance under stress.';
    case 'C':
      return 'Warning: Performance degradation detected. Review bottlenecks.';
    case 'D':
      return 'Poor: Significant degradation. System may need optimization.';
    case 'F':
      return 'Critical: System failed under stress. Immediate action required.';
    default:
      return 'Unable to determine performance grade.';
  }
}

