/**
 * 峰值测试 (Spike Test)
 * 
 * 目标: 测试系统对突发流量的应对能力
 * 特点: 流量快速上升到峰值 5000 RPS，然后快速恢复
 * 持续时间: 2 分钟
 * 
 * 场景:
 * 1. 预热 (30s): 100 RPS
 * 2. 突发 (10s): 100 → 5000 RPS
 * 3. 峰值 (60s): 5000 RPS 维持
 * 4. 恢复 (20s): 5000 → 100 RPS
 */

import { sleep, group, fail } from 'k6';
import { Trend, Counter, Rate, Gauge } from 'k6/metrics';
import { loadScenarios, getEnvConfig, slaThresholds } from '../config.js';
import { ApiClient, getRecommendations, searchItems, submitFeedback, healthCheck } from '../lib/api.js';
import {
  randomUserId,
  randomItemId,
  randomSearchTerm,
  randomScene,
  randomFeedbackAction,
  randomContext,
  weightedRandom,
} from '../lib/utils.js';

// ============================================================================
// 自定义指标
// ============================================================================

const spikeLatency = new Trend('spike_latency', true);
const spikeErrorRate = new Rate('spike_error_rate');
const spikeRequests = new Counter('spike_requests');
const circuitBreaker = new Counter('circuit_breaker_triggered');
const recoveryTime = new Gauge('recovery_time_ms');

// 各阶段指标
const warmupLatency = new Trend('warmup_latency', true);
const spikePhaseLatency = new Trend('spike_phase_latency', true);
const peakLatency = new Trend('peak_latency', true);
const recoveryLatency = new Trend('recovery_latency', true);

// ============================================================================
// 测试配置
// ============================================================================

export const options = {
  scenarios: {
    spike: loadScenarios.spike,
  },
  
  thresholds: {
    // 峰值测试的阈值更宽松
    http_req_failed: ['rate<0.10'],           // 允许 10% 错误率
    http_req_duration: ['p(95)<3000'],        // P95 < 3s
    'spike_latency': ['p(99)<5000'],          // P99 < 5s
    'spike_error_rate': ['rate<0.10'],        // 错误率 < 10%
    
    // 恢复阶段需要快速恢复正常
    'recovery_latency': ['p(95)<500'],        // 恢复后 P95 < 500ms
  },
  
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  
  tags: {
    testType: 'spike',
    environment: __ENV.TEST_ENV || 'local',
  },
};

// ============================================================================
// 阶段配置
// ============================================================================

const STAGES = {
  WARMUP: { start: 0, end: 30, rps: 100 },
  SPIKE: { start: 30, end: 40, rps: 5000 },
  PEAK: { start: 40, end: 100, rps: 5000 },
  RECOVERY: { start: 100, end: 120, rps: 100 },
};

// ============================================================================
// 测试生命周期
// ============================================================================

export function setup() {
  console.log('========================================');
  console.log('          SPIKE TEST STARTING          ');
  console.log('========================================');
  
  const envConfig = getEnvConfig();
  console.log(`Target Environment: ${__ENV.TEST_ENV || 'local'}`);
  console.log(`Base URL: ${envConfig.baseUrl}`);
  console.log(`Peak RPS: 5000`);
  console.log(`Duration: 2 minutes`);
  console.log('');
  console.log('Stages:');
  console.log('  1. Warm-up (30s): 100 RPS');
  console.log('  2. Spike (10s): 100 → 5000 RPS');
  console.log('  3. Peak (60s): 5000 RPS');
  console.log('  4. Recovery (20s): 5000 → 100 RPS');
  console.log('');
  console.log('⚠️  WARNING: This test generates extreme load!');
  console.log('');
  
  // 健康检查
  const client = new ApiClient();
  const health = healthCheck(client);
  
  if (!health.success) {
    throw new Error('Target service is not healthy! Aborting test.');
  }
  
  console.log('Health check passed. Starting spike test...');
  console.log('========================================');
  
  return {
    startTime: Date.now(),
    envConfig: envConfig,
    recoveryStartTime: null,
    normalLatencyBaseline: null,
  };
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('========================================');
  console.log('          SPIKE TEST COMPLETED         ');
  console.log('========================================');
  console.log(`Total Duration: ${duration.toFixed(2)} seconds`);
  console.log('========================================');
}

// ============================================================================
// 主测试函数
// ============================================================================

export default function (data) {
  const client = new ApiClient();
  const elapsedSeconds = (Date.now() - data.startTime) / 1000;
  
  // 确定当前阶段
  const stage = getCurrentStage(elapsedSeconds);
  
  // 执行测试请求
  const result = executeRequest(client, stage);
  
  // 记录指标
  spikeLatency.add(result.latency);
  spikeErrorRate.add(!result.success);
  spikeRequests.add(1);
  
  // 按阶段记录延迟
  switch (stage) {
    case 'warmup':
      warmupLatency.add(result.latency);
      // 记录基线延迟
      if (data.normalLatencyBaseline === null && result.success) {
        data.normalLatencyBaseline = result.latency;
      }
      break;
    case 'spike':
    case 'peak':
      spikePhaseLatency.add(result.latency);
      // 检测熔断
      if (result.latency > 5000 || !result.success) {
        circuitBreaker.add(1);
      }
      break;
    case 'recovery':
      recoveryLatency.add(result.latency);
      // 记录恢复时间
      if (data.recoveryStartTime === null) {
        data.recoveryStartTime = Date.now();
      }
      if (result.latency < 200 && result.success) {
        const recTime = Date.now() - data.recoveryStartTime;
        recoveryTime.add(recTime);
      }
      break;
  }
  
  // 非常短的间隔以产生高负载
  sleep(0.01);
}

// ============================================================================
// 辅助函数
// ============================================================================

function getCurrentStage(elapsedSeconds) {
  if (elapsedSeconds < STAGES.WARMUP.end) {
    return 'warmup';
  } else if (elapsedSeconds < STAGES.SPIKE.end) {
    return 'spike';
  } else if (elapsedSeconds < STAGES.PEAK.end) {
    return 'peak';
  } else {
    return 'recovery';
  }
}

function executeRequest(client, stage) {
  // 峰值和恢复阶段使用更简单的请求以减少服务器压力
  const scenario = stage === 'peak' || stage === 'spike'
    ? weightedRandom({ recommend: 8, search: 2 })  // 峰值时主要测试核心接口
    : weightedRandom({ recommend: 5, search: 3, feedback: 2 });  // 正常分布
  
  let result;
  
  group(`spike_${stage}_${scenario}`, () => {
    switch (scenario) {
      case 'recommend':
        result = getRecommendations(client, {
          userId: randomUserId(),
          limit: 10,  // 减少返回数量以降低压力
          scene: randomScene(),
          context: randomContext(),
        });
        break;
        
      case 'search':
        result = searchItems(client, {
          query: randomSearchTerm(),
          limit: 10,
        });
        break;
        
      case 'feedback':
        result = submitFeedback(client, {
          userId: randomUserId(),
          itemId: randomItemId(),
          action: randomFeedbackAction(),
        });
        break;
        
      default:
        result = getRecommendations(client, {
          userId: randomUserId(),
          limit: 10,
          scene: 'home',
        });
    }
  });
  
  return result;
}

// ============================================================================
// 自定义摘要报告
// ============================================================================

export function handleSummary(data) {
  const summary = generateSpikeSummary(data);
  
  return {
    'spike-test-summary.json': JSON.stringify(data, null, 2),
    stdout: summary,
  };
}

function generateSpikeSummary(data) {
  const metrics = data.metrics;
  const httpReqs = metrics.http_reqs?.values || {};
  const httpDuration = metrics.http_req_duration?.values || {};
  const httpFailed = metrics.http_req_failed?.values || {};
  const warmup = metrics.warmup_latency?.values || {};
  const peak = metrics.peak_latency?.values || {};
  const recovery = metrics.recovery_latency?.values || {};
  const cb = metrics.circuit_breaker_triggered?.values || {};
  const recTime = metrics.recovery_time_ms?.values || {};
  
  // 计算恢复能力评分
  const peakP99 = peak['p(99)'] || 0;
  const recoveryP95 = recovery['p(95)'] || 0;
  const errorRate = (httpFailed.rate || 0) * 100;
  const cbTriggers = cb.count || 0;
  
  let resilienceScore = 100;
  if (peakP99 > 5000) resilienceScore -= 30;
  else if (peakP99 > 3000) resilienceScore -= 15;
  if (recoveryP95 > 500) resilienceScore -= 20;
  else if (recoveryP95 > 200) resilienceScore -= 10;
  if (errorRate > 10) resilienceScore -= 30;
  else if (errorRate > 5) resilienceScore -= 15;
  if (cbTriggers > 100) resilienceScore -= 20;
  
  resilienceScore = Math.max(0, resilienceScore);
  
  let resilienceGrade = 'A';
  if (resilienceScore < 50) resilienceGrade = 'F';
  else if (resilienceScore < 60) resilienceGrade = 'D';
  else if (resilienceScore < 70) resilienceGrade = 'C';
  else if (resilienceScore < 85) resilienceGrade = 'B';
  
  return `
╔══════════════════════════════════════════════════════════════════════════════╗
║                            SPIKE TEST SUMMARY                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Test Duration: ${((data.state?.testRunDurationMs || 0) / 1000).toFixed(2).padEnd(15)}s                                     ║
║  Peak VUs: ${(metrics.vus_max?.values?.max || 0).toString().padEnd(20)}                                       ║
║  Resilience Score: ${resilienceScore}/100 (Grade: ${resilienceGrade})                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            THROUGHPUT                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Requests: ${(httpReqs.count || 0).toString().padEnd(15)}                                      ║
║  Peak RPS: ${(httpReqs.rate || 0).toFixed(2).padEnd(20)} req/s                                  ║
║  Circuit Breaker Triggers: ${(cbTriggers).toString().padEnd(10)}                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            PHASE LATENCY (ms)                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                      Warmup       Peak        Recovery                       ║
║  Average:         ${(warmup.avg || 0).toFixed(2).padEnd(12)}${(peak.avg || 0).toFixed(2).padEnd(12)}${(recovery.avg || 0).toFixed(2).padEnd(12)}     ║
║  P95:             ${(warmup['p(95)'] || 0).toFixed(2).padEnd(12)}${(peak['p(95)'] || 0).toFixed(2).padEnd(12)}${(recovery['p(95)'] || 0).toFixed(2).padEnd(12)}     ║
║  P99:             ${(warmup['p(99)'] || 0).toFixed(2).padEnd(12)}${(peak['p(99)'] || 0).toFixed(2).padEnd(12)}${(recovery['p(99)'] || 0).toFixed(2).padEnd(12)}     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            ERROR ANALYSIS                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Errors: ${(httpFailed.passes || 0).toString().padEnd(18)}                                    ║
║  Error Rate: ${errorRate.toFixed(4).padEnd(20)}%                                       ║
║  Max Latency: ${(httpDuration.max || 0).toFixed(2).padEnd(17)}ms                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                            RECOVERY ANALYSIS                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Recovery Time: ${(recTime.value || 0).toFixed(2).padEnd(16)}ms                                    ║
║  Recovery P95 Latency: ${(recoveryP95).toFixed(2).padEnd(10)}ms                                    ║
║  ${getRecoveryAnalysis(recoveryP95, recTime.value || 0).padEnd(74)}║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RECOMMENDATIONS:                                                            ║
${getRecommendations(resilienceScore, peakP99, recoveryP95, errorRate)}║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
`;
}

function getRecoveryAnalysis(recoveryP95, recoveryTime) {
  if (recoveryP95 < 100 && recoveryTime < 10000) {
    return '✓ Excellent recovery! System quickly stabilized after spike.';
  } else if (recoveryP95 < 300) {
    return '○ Good recovery. System returned to normal within acceptable time.';
  } else if (recoveryP95 < 500) {
    return '△ Slow recovery. Consider improving auto-scaling configuration.';
  } else {
    return '✗ Poor recovery. System struggled to return to normal operation.';
  }
}

function getRecommendations(score, peakP99, recoveryP95, errorRate) {
  const recommendations = [];
  
  if (peakP99 > 3000) {
    recommendations.push('  - Consider implementing request queuing or rate limiting');
  }
  if (recoveryP95 > 300) {
    recommendations.push('  - Review auto-scaling policies for faster scale-down');
  }
  if (errorRate > 5) {
    recommendations.push('  - Implement circuit breaker pattern to prevent cascading failures');
  }
  if (score < 70) {
    recommendations.push('  - Add more replicas or increase resource limits');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('  ✓ System shows good resilience. No immediate actions required.');
  }
  
  return recommendations.map(r => `║${r.padEnd(76)}║\n`).join('');
}

