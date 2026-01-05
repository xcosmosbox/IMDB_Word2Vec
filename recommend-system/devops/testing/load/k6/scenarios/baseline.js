/**
 * 基线负载测试
 * 
 * 目标: 验证系统在正常负载下的性能
 * RPS: 100
 * 持续时间: 5 分钟
 * 
 * SLA 目标 (来自 interfaces.yaml):
 * - 可用性: ≥ 99.9%
 * - P50 延迟: ≤ 50ms
 * - P99 延迟: ≤ 200ms
 * - 错误率: ≤ 0.1%
 */

import { sleep } from 'k6';
import { loadScenarios, k6Thresholds, getEnvConfig } from '../config.js';
import { ApiClient, getRecommendations, searchItems, submitFeedback, getItemDetail, healthCheck } from '../lib/api.js';
import {
  randomUserId,
  randomItemId,
  randomSearchTerm,
  randomScene,
  randomFeedbackAction,
  randomContext,
  selectScenario,
  recordRequest,
  checkSlaViolation,
  randomSleep,
} from '../lib/utils.js';

// ============================================================================
// 测试配置
// ============================================================================

export const options = {
  scenarios: {
    baseline: loadScenarios.baseline,
  },
  thresholds: k6Thresholds,
  
  // 输出配置
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  
  // 标签
  tags: {
    testType: 'baseline',
    environment: __ENV.TEST_ENV || 'local',
  },
};

// ============================================================================
// 测试生命周期
// ============================================================================

/**
 * 测试初始化 (每个 VU 执行一次)
 */
export function setup() {
  console.log('========================================');
  console.log('       BASELINE LOAD TEST STARTING     ');
  console.log('========================================');
  
  const envConfig = getEnvConfig();
  console.log(`Target Environment: ${__ENV.TEST_ENV || 'local'}`);
  console.log(`Base URL: ${envConfig.baseUrl}`);
  console.log(`Target RPS: 100`);
  console.log(`Duration: 5 minutes`);
  
  // 创建 API 客户端进行健康检查
  const client = new ApiClient();
  const health = healthCheck(client);
  
  if (!health.success) {
    throw new Error('Target service is not healthy! Aborting test.');
  }
  
  console.log('Health check passed. Starting test...');
  console.log('========================================');
  
  return {
    startTime: Date.now(),
    envConfig: envConfig,
  };
}

/**
 * 测试结束清理
 */
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('========================================');
  console.log('       BASELINE LOAD TEST COMPLETED    ');
  console.log('========================================');
  console.log(`Total Duration: ${duration.toFixed(2)} seconds`);
  console.log('========================================');
}

// ============================================================================
// 主测试函数
// ============================================================================

export default function (data) {
  const client = new ApiClient();
  
  // 根据业务场景分布选择测试用例
  const scenario = selectScenario();
  
  switch (scenario) {
    case 'recommend':
      testRecommendations(client);
      break;
    case 'search':
      testSearch(client);
      break;
    case 'feedback':
      testFeedback(client);
      break;
    case 'item_detail':
      testItemDetail(client);
      break;
    default:
      testRecommendations(client);
  }
  
  // 模拟用户思考时间
  sleep(randomSleep(0.5, 2));
}

// ============================================================================
// 测试场景函数
// ============================================================================

/**
 * 测试推荐接口
 */
function testRecommendations(client) {
  const userId = randomUserId();
  const scene = randomScene();
  const context = randomContext();
  
  const result = getRecommendations(client, {
    userId: userId,
    limit: 20,
    scene: scene,
    context: context,
  });
  
  // 记录指标
  recordRequest(result.success, result.latency, 'recommend');
  checkSlaViolation(result.latency, 200);
}

/**
 * 测试搜索接口
 */
function testSearch(client) {
  const query = randomSearchTerm();
  
  const result = searchItems(client, {
    query: query,
    limit: 20,
  });
  
  // 记录指标
  recordRequest(result.success, result.latency, 'search');
  checkSlaViolation(result.latency, 300);
}

/**
 * 测试反馈接口
 */
function testFeedback(client) {
  const userId = randomUserId();
  const itemId = randomItemId();
  const action = randomFeedbackAction();
  
  const result = submitFeedback(client, {
    userId: userId,
    itemId: itemId,
    action: action,
  });
  
  // 记录指标
  recordRequest(result.success, result.latency, 'feedback');
  checkSlaViolation(result.latency, 100);
}

/**
 * 测试物品详情接口
 */
function testItemDetail(client) {
  const itemId = randomItemId();
  
  const result = getItemDetail(client, itemId);
  
  // 记录指标
  recordRequest(result.success, result.latency, 'item_detail');
  checkSlaViolation(result.latency, 200);
}

// ============================================================================
// 自定义摘要报告
// ============================================================================

export function handleSummary(data) {
  const summary = generateBaselineSummary(data);
  
  return {
    'baseline-summary.json': JSON.stringify(data, null, 2),
    stdout: summary,
  };
}

function generateBaselineSummary(data) {
  const metrics = data.metrics;
  const httpReqs = metrics.http_reqs?.values || {};
  const httpDuration = metrics.http_req_duration?.values || {};
  const httpFailed = metrics.http_req_failed?.values || {};
  
  // 检查 SLA
  const slaChecks = [
    {
      name: '可用性',
      target: '≥ 99.9%',
      actual: `${((1 - (httpFailed.rate || 0)) * 100).toFixed(4)}%`,
      passed: (httpFailed.rate || 0) < 0.001,
    },
    {
      name: 'P50 延迟',
      target: '≤ 50ms',
      actual: `${(httpDuration['p(50)'] || 0).toFixed(2)}ms`,
      passed: (httpDuration['p(50)'] || 0) <= 50,
    },
    {
      name: 'P99 延迟',
      target: '≤ 200ms',
      actual: `${(httpDuration['p(99)'] || 0).toFixed(2)}ms`,
      passed: (httpDuration['p(99)'] || 0) <= 200,
    },
    {
      name: '错误率',
      target: '≤ 0.1%',
      actual: `${((httpFailed.rate || 0) * 100).toFixed(4)}%`,
      passed: (httpFailed.rate || 0) <= 0.001,
    },
  ];
  
  const allPassed = slaChecks.every(c => c.passed);
  
  return `
╔══════════════════════════════════════════════════════════════════════════════╗
║                          BASELINE TEST SUMMARY                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Test Duration: ${((data.state?.testRunDurationMs || 0) / 1000).toFixed(2).padEnd(20)}s                                  ║
║  Peak VUs: ${(metrics.vus_max?.values?.max || 0).toString().padEnd(20)}                                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            HTTP REQUESTS                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Requests: ${(httpReqs.count || 0).toString().padEnd(15)}                                      ║
║  Request Rate: ${(httpReqs.rate || 0).toFixed(2).padEnd(17)} req/s                                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            LATENCY (ms)                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Average: ${(httpDuration.avg || 0).toFixed(2).padEnd(15)}                                           ║
║  P50: ${(httpDuration['p(50)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  P90: ${(httpDuration['p(90)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  P95: ${(httpDuration['p(95)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  P99: ${(httpDuration['p(99)'] || 0).toFixed(2).padEnd(19)}                                           ║
║  Max: ${(httpDuration.max || 0).toFixed(2).padEnd(19)}                                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            SLA CHECKS                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
${slaChecks.map(c => `║  ${c.passed ? '✓' : '✗'} ${c.name.padEnd(12)} Target: ${c.target.padEnd(12)} Actual: ${c.actual.padEnd(15)}║`).join('\n')}
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Overall Result: ${allPassed ? '✓ ALL SLA CHECKS PASSED' : '✗ SOME SLA CHECKS FAILED'}                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
`;
}

