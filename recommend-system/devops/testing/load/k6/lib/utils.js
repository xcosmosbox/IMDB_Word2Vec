/**
 * K6 测试工具函数库
 * 
 * 提供随机数据生成、指标收集、结果验证等辅助功能
 */

import { Counter, Rate, Trend, Gauge } from 'k6/metrics';
import { testData } from '../config.js';

// ============================================================================
// 自定义指标
// ============================================================================

// 延迟指标
export const recommendLatency = new Trend('recommend_latency', true);
export const searchLatency = new Trend('search_latency', true);
export const feedbackLatency = new Trend('feedback_latency', true);
export const itemDetailLatency = new Trend('item_detail_latency', true);

// 计数器
export const successCounter = new Counter('successful_requests');
export const failureCounter = new Counter('failed_requests');
export const feedbackCounter = new Counter('feedback_count');
export const recommendCounter = new Counter('recommend_count');

// 错误率
export const errorRate = new Rate('error_rate');
export const slaViolationRate = new Rate('sla_violation_rate');

// 状态指标
export const currentVUs = new Gauge('current_vus');
export const requestsPerSecond = new Gauge('requests_per_second');

// ============================================================================
// 随机数据生成器
// ============================================================================

/**
 * 生成随机整数
 */
export function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * 生成随机用户 ID
 */
export function randomUserId() {
  const id = randomInt(testData.userIdRange.min, testData.userIdRange.max);
  return `user_${id}`;
}

/**
 * 生成随机物品 ID
 */
export function randomItemId() {
  const id = randomInt(testData.itemIdRange.min, testData.itemIdRange.max);
  return `item_${id}`;
}

/**
 * 从数组中随机选择一个元素
 */
export function randomItem(arr) {
  return arr[randomInt(0, arr.length - 1)];
}

/**
 * 随机选择搜索词
 */
export function randomSearchTerm() {
  return randomItem(testData.searchTerms);
}

/**
 * 随机选择推荐场景
 */
export function randomScene() {
  return randomItem(testData.recommendScenes);
}

/**
 * 随机选择反馈动作
 */
export function randomFeedbackAction() {
  return randomItem(testData.feedbackActions);
}

/**
 * 生成随机用户元数据
 */
export function randomUserMetadata() {
  const ages = ['18-24', '25-34', '35-44', '45-54', '55+'];
  const genders = ['male', 'female', 'other'];
  const regions = ['north', 'south', 'east', 'west', 'central'];
  
  return {
    age_group: randomItem(ages),
    gender: randomItem(genders),
    region: randomItem(regions),
    is_new_user: Math.random() > 0.8,
  };
}

/**
 * 生成随机上下文数据
 */
export function randomContext() {
  const devices = ['mobile', 'desktop', 'tablet'];
  const platforms = ['ios', 'android', 'web'];
  const hours = randomInt(0, 23);
  
  return {
    device: randomItem(devices),
    platform: randomItem(platforms),
    hour_of_day: hours,
    is_weekend: Math.random() > 0.7,
    session_duration: randomInt(0, 3600),
  };
}

/**
 * 生成批量用户ID
 */
export function generateUserIds(count) {
  return Array.from({ length: count }, () => randomUserId());
}

/**
 * 生成批量物品ID
 */
export function generateItemIds(count) {
  return Array.from({ length: count }, () => randomItemId());
}

// ============================================================================
// 指标记录器
// ============================================================================

/**
 * 记录请求结果
 */
export function recordRequest(success, latency, type = 'general') {
  if (success) {
    successCounter.add(1);
  } else {
    failureCounter.add(1);
  }
  
  errorRate.add(!success);
  
  // 按类型记录延迟
  switch (type) {
    case 'recommend':
      recommendLatency.add(latency);
      recommendCounter.add(1);
      break;
    case 'search':
      searchLatency.add(latency);
      break;
    case 'feedback':
      feedbackLatency.add(latency);
      feedbackCounter.add(1);
      break;
    case 'item_detail':
      itemDetailLatency.add(latency);
      break;
  }
}

/**
 * 检查 SLA 违规
 * 
 * @param {number} latency - 请求延迟 (ms)
 * @param {number} threshold - SLA 阈值 (ms)
 */
export function checkSlaViolation(latency, threshold = 200) {
  const violated = latency > threshold;
  slaViolationRate.add(violated);
  return violated;
}

// ============================================================================
// 响应验证器
// ============================================================================

/**
 * 验证响应结构
 */
export function validateResponse(response, expectedStatus = 200) {
  const checks = {
    statusOk: response.status === expectedStatus,
    hasBody: response.body && response.body.length > 0,
    isJson: false,
    hasData: false,
    noError: false,
  };
  
  try {
    const body = JSON.parse(response.body);
    checks.isJson = true;
    checks.hasData = body.data !== undefined;
    checks.noError = !body.error;
  } catch (e) {
    // JSON 解析失败
  }
  
  return checks;
}

/**
 * 验证推荐响应
 */
export function validateRecommendResponse(response) {
  const baseChecks = validateResponse(response);
  
  if (!baseChecks.isJson) {
    return { ...baseChecks, hasRecommendations: false, recommendCount: 0 };
  }
  
  try {
    const body = JSON.parse(response.body);
    const recommendations = body.data?.recommendations || [];
    
    return {
      ...baseChecks,
      hasRecommendations: recommendations.length > 0,
      recommendCount: recommendations.length,
    };
  } catch (e) {
    return { ...baseChecks, hasRecommendations: false, recommendCount: 0 };
  }
}

/**
 * 验证搜索响应
 */
export function validateSearchResponse(response) {
  const baseChecks = validateResponse(response);
  
  if (!baseChecks.isJson) {
    return { ...baseChecks, hasResults: false, resultCount: 0 };
  }
  
  try {
    const body = JSON.parse(response.body);
    const results = body.data || [];
    
    return {
      ...baseChecks,
      hasResults: results.length > 0,
      resultCount: results.length,
    };
  } catch (e) {
    return { ...baseChecks, hasResults: false, resultCount: 0 };
  }
}

// ============================================================================
// 加权随机选择器
// ============================================================================

/**
 * 加权随机选择
 * 
 * @param {Object} weights - 权重对象 { choice1: weight1, choice2: weight2, ... }
 * @returns {string} 选中的项
 */
export function weightedRandom(weights) {
  const entries = Object.entries(weights);
  const totalWeight = entries.reduce((sum, [_, w]) => sum + w, 0);
  
  let random = Math.random() * totalWeight;
  
  for (const [choice, weight] of entries) {
    random -= weight;
    if (random <= 0) {
      return choice;
    }
  }
  
  return entries[entries.length - 1][0];
}

/**
 * 场景选择器
 * 
 * 根据真实业务场景分布选择测试用例
 */
export function selectScenario() {
  const weights = {
    recommend: 5,      // 50% 推荐请求
    search: 2,         // 20% 搜索请求
    feedback: 2,       // 20% 反馈请求
    item_detail: 1,    // 10% 物品详情
  };
  
  return weightedRandom(weights);
}

// ============================================================================
// 睡眠与节流
// ============================================================================

/**
 * 随机睡眠 (思考时间)
 */
export function randomSleep(minSec = 1, maxSec = 3) {
  const sleepTime = randomInt(minSec * 1000, maxSec * 1000) / 1000;
  return sleepTime;
}

/**
 * 指数退避睡眠
 */
export function exponentialBackoff(attempt, baseMs = 100, maxMs = 5000) {
  const delay = Math.min(baseMs * Math.pow(2, attempt), maxMs);
  const jitter = delay * 0.1 * Math.random();
  return (delay + jitter) / 1000;
}

// ============================================================================
// 日期时间工具
// ============================================================================

/**
 * 获取当前时间戳
 */
export function getCurrentTimestamp() {
  return Date.now();
}

/**
 * 格式化持续时间
 */
export function formatDuration(ms) {
  if (ms < 1000) {
    return `${ms.toFixed(2)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else {
    return `${(ms / 60000).toFixed(2)}m`;
  }
}

/**
 * 格式化字节数
 */
export function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (bytes >= 1024 && i < units.length - 1) {
    bytes /= 1024;
    i++;
  }
  return `${bytes.toFixed(2)} ${units[i]}`;
}

// ============================================================================
// 报告辅助
// ============================================================================

/**
 * 生成测试摘要
 */
export function generateSummary(data) {
  const metrics = data.metrics || {};
  
  const summary = {
    duration: data.state?.testRunDurationMs || 0,
    totalRequests: metrics.http_reqs?.values?.count || 0,
    rps: metrics.http_reqs?.values?.rate || 0,
    avgLatency: metrics.http_req_duration?.values?.avg || 0,
    p50Latency: metrics.http_req_duration?.values?.['p(50)'] || 0,
    p90Latency: metrics.http_req_duration?.values?.['p(90)'] || 0,
    p95Latency: metrics.http_req_duration?.values?.['p(95)'] || 0,
    p99Latency: metrics.http_req_duration?.values?.['p(99)'] || 0,
    errorRate: metrics.http_req_failed?.values?.rate || 0,
    maxVUs: metrics.vus_max?.values?.max || 0,
  };
  
  return summary;
}

/**
 * 检查所有阈值是否通过
 */
export function checkThresholds(data) {
  const results = [];
  
  for (const [metricName, metric] of Object.entries(data.metrics || {})) {
    if (metric.thresholds) {
      for (const [threshold, result] of Object.entries(metric.thresholds)) {
        results.push({
          metric: metricName,
          threshold: threshold,
          passed: result.ok,
        });
      }
    }
  }
  
  return {
    results,
    allPassed: results.every(r => r.passed),
    passedCount: results.filter(r => r.passed).length,
    failedCount: results.filter(r => !r.passed).length,
  };
}

// ============================================================================
// 默认导出
// ============================================================================

export default {
  // 自定义指标
  recommendLatency,
  searchLatency,
  feedbackLatency,
  itemDetailLatency,
  successCounter,
  failureCounter,
  feedbackCounter,
  recommendCounter,
  errorRate,
  slaViolationRate,
  currentVUs,
  requestsPerSecond,
  
  // 随机数据生成
  randomInt,
  randomUserId,
  randomItemId,
  randomItem,
  randomSearchTerm,
  randomScene,
  randomFeedbackAction,
  randomUserMetadata,
  randomContext,
  generateUserIds,
  generateItemIds,
  
  // 指标记录
  recordRequest,
  checkSlaViolation,
  
  // 响应验证
  validateResponse,
  validateRecommendResponse,
  validateSearchResponse,
  
  // 工具函数
  weightedRandom,
  selectScenario,
  randomSleep,
  exponentialBackoff,
  getCurrentTimestamp,
  formatDuration,
  formatBytes,
  generateSummary,
  checkThresholds,
};

