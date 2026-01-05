/**
 * K6 负载测试全局配置
 * 
 * 基于 devops/interfaces.yaml 中定义的 SLA 契约:
 * - 可用性: 99.9%
 * - P50 延迟: 50ms
 * - P99 延迟: 200ms
 * - 错误率: 0.1%
 */

// ============================================================================
// 环境配置
// ============================================================================

export const environments = {
  local: {
    baseUrl: 'http://localhost:8080',
    recommendService: 'http://localhost:8080',
    userService: 'http://localhost:8081',
    itemService: 'http://localhost:8082',
    inferenceService: 'http://localhost:50051',
  },
  dev: {
    baseUrl: 'http://recommend-dev.internal:8080',
    recommendService: 'http://recommend-service.recommend-dev:8080',
    userService: 'http://user-service.recommend-dev:8081',
    itemService: 'http://item-service.recommend-dev:8082',
    inferenceService: 'http://ugt-inference.recommend-dev:50051',
  },
  prod: {
    baseUrl: 'http://recommend.internal:8080',
    recommendService: 'http://recommend-service.recommend-prod:8080',
    userService: 'http://user-service.recommend-prod:8081',
    itemService: 'http://item-service.recommend-prod:8082',
    inferenceService: 'http://ugt-inference.recommend-prod:50051',
  },
};

// 获取当前环境配置
export function getEnvConfig() {
  const env = __ENV.TEST_ENV || 'local';
  return environments[env] || environments.local;
}

// ============================================================================
// SLA 阈值配置 (来自 interfaces.yaml)
// ============================================================================

export const slaThresholds = {
  // 可用性: 99.9% (错误率 < 0.1%)
  availability: 0.999,
  errorRate: 0.001,
  
  // 延迟阈值
  p50Latency: 50,   // 50ms
  p90Latency: 100,  // 100ms
  p95Latency: 150,  // 150ms
  p99Latency: 200,  // 200ms
  
  // 吞吐量 (根据场景不同)
  minRps: {
    baseline: 100,
    stress: 1000,
    spike: 5000,
  },
};

// K6 阈值配置 (用于 options.thresholds)
export const k6Thresholds = {
  // HTTP 请求
  http_req_failed: ['rate<0.001'],          // 错误率 < 0.1%
  http_req_duration: [
    'p(50)<50',                              // P50 < 50ms
    'p(90)<100',                             // P90 < 100ms
    'p(95)<150',                             // P95 < 150ms
    'p(99)<200',                             // P99 < 200ms
  ],
  
  // 自定义指标阈值
  recommend_latency: ['p(99)<200'],
  search_latency: ['p(99)<300'],
  feedback_latency: ['p(99)<100'],
  error_rate: ['rate<0.001'],
};

// ============================================================================
// 负载场景配置 (来自 interfaces.yaml)
// ============================================================================

export const loadScenarios = {
  // 基线测试: 100 RPS, 5分钟
  baseline: {
    executor: 'constant-arrival-rate',
    rate: 100,
    timeUnit: '1s',
    duration: '5m',
    preAllocatedVUs: 50,
    maxVUs: 200,
  },
  
  // 压力测试: 逐步增加到 1000 RPS, 10分钟
  stress: {
    executor: 'ramping-arrival-rate',
    startRate: 50,
    timeUnit: '1s',
    preAllocatedVUs: 100,
    maxVUs: 1000,
    stages: [
      { duration: '1m', target: 100 },
      { duration: '2m', target: 300 },
      { duration: '2m', target: 500 },
      { duration: '2m', target: 800 },
      { duration: '2m', target: 1000 },
      { duration: '1m', target: 0 },
    ],
  },
  
  // 峰值测试: 突发 5000 RPS, 2分钟
  spike: {
    executor: 'ramping-arrival-rate',
    startRate: 100,
    timeUnit: '1s',
    preAllocatedVUs: 200,
    maxVUs: 2000,
    stages: [
      { duration: '30s', target: 100 },   // 预热
      { duration: '10s', target: 5000 },  // 突发到峰值
      { duration: '1m', target: 5000 },   // 维持峰值
      { duration: '20s', target: 100 },   // 快速恢复
    ],
  },
};

// ============================================================================
// API 端点配置
// ============================================================================

export const apiEndpoints = {
  // 健康检查
  health: '/health',
  ready: '/ready',
  
  // 推荐服务 API
  recommend: {
    getRecommendations: '/api/v1/recommend',
    getPersonalized: '/api/v1/recommend/personalized',
    getSimilar: '/api/v1/recommend/similar',
    getColdStart: '/api/v1/recommend/coldstart',
  },
  
  // 用户服务 API
  user: {
    getProfile: '/api/v1/users/{userId}',
    updateProfile: '/api/v1/users/{userId}',
    getHistory: '/api/v1/users/{userId}/history',
    getPreferences: '/api/v1/users/{userId}/preferences',
  },
  
  // 物品服务 API
  item: {
    getItem: '/api/v1/items/{itemId}',
    search: '/api/v1/items/search',
    getCategories: '/api/v1/items/categories',
    getPopular: '/api/v1/items/popular',
  },
  
  // 反馈 API
  feedback: {
    submit: '/api/v1/feedback',
    batch: '/api/v1/feedback/batch',
  },
  
  // 管理 API
  admin: {
    dashboard: '/api/admin/v1/dashboard',
    users: '/api/admin/v1/users',
    items: '/api/admin/v1/items',
    metrics: '/api/admin/v1/metrics',
  },
};

// ============================================================================
// 测试数据配置
// ============================================================================

export const testData = {
  // 用户ID范围
  userIdRange: {
    min: 1,
    max: 100000,
  },
  
  // 物品ID范围
  itemIdRange: {
    min: 1,
    max: 1000000,
  },
  
  // 搜索关键词
  searchTerms: [
    'action', 'comedy', 'drama', 'thriller', 'sci-fi',
    'horror', 'romance', 'documentary', 'animation', 'adventure',
    'mystery', 'fantasy', 'crime', 'biography', 'history',
  ],
  
  // 推荐场景
  recommendScenes: ['home', 'search', 'detail', 'cart', 'checkout'],
  
  // 反馈动作类型
  feedbackActions: ['view', 'click', 'like', 'share', 'purchase', 'skip'],
  
  // 分页参数
  pagination: {
    defaultLimit: 20,
    maxLimit: 100,
    defaultOffset: 0,
  },
};

// ============================================================================
// 请求头配置
// ============================================================================

export const defaultHeaders = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'User-Agent': 'K6-LoadTest/1.0',
};

export function getAuthHeaders(apiKey) {
  return {
    ...defaultHeaders,
    'Authorization': `Bearer ${apiKey || __ENV.API_KEY || 'test-api-key'}`,
  };
}

// ============================================================================
// 报告配置
// ============================================================================

export const reportConfig = {
  outputDir: './results',
  formats: ['json', 'html', 'junit'],
  includeMetrics: [
    'http_reqs',
    'http_req_duration',
    'http_req_failed',
    'http_req_blocked',
    'http_req_connecting',
    'http_req_tls_handshaking',
    'http_req_sending',
    'http_req_waiting',
    'http_req_receiving',
    'vus',
    'vus_max',
    'iterations',
    'iteration_duration',
  ],
};

// ============================================================================
// 导出默认配置
// ============================================================================

export default {
  environments,
  getEnvConfig,
  slaThresholds,
  k6Thresholds,
  loadScenarios,
  apiEndpoints,
  testData,
  defaultHeaders,
  getAuthHeaders,
  reportConfig,
};

