/**
 * K6 API 调用辅助函数库
 * 
 * 封装推荐系统各服务的 API 调用
 */

import http from 'k6/http';
import { check } from 'k6';
import { getEnvConfig, getAuthHeaders, apiEndpoints } from '../config.js';

// ============================================================================
// 基础 API 客户端
// ============================================================================

/**
 * API 客户端类
 */
export class ApiClient {
  constructor(options = {}) {
    const envConfig = getEnvConfig();
    this.baseUrl = options.baseUrl || envConfig.baseUrl;
    this.headers = getAuthHeaders(options.apiKey);
    this.timeout = options.timeout || '30s';
  }
  
  /**
   * GET 请求
   */
  get(path, params = {}) {
    const url = this.buildUrl(path, params);
    return http.get(url, {
      headers: this.headers,
      timeout: this.timeout,
      tags: { api: path },
    });
  }
  
  /**
   * POST 请求
   */
  post(path, body, params = {}) {
    const url = this.buildUrl(path, params);
    return http.post(url, JSON.stringify(body), {
      headers: this.headers,
      timeout: this.timeout,
      tags: { api: path },
    });
  }
  
  /**
   * PUT 请求
   */
  put(path, body, params = {}) {
    const url = this.buildUrl(path, params);
    return http.put(url, JSON.stringify(body), {
      headers: this.headers,
      timeout: this.timeout,
      tags: { api: path },
    });
  }
  
  /**
   * DELETE 请求
   */
  delete(path, params = {}) {
    const url = this.buildUrl(path, params);
    return http.del(url, null, {
      headers: this.headers,
      timeout: this.timeout,
      tags: { api: path },
    });
  }
  
  /**
   * 构建完整 URL
   */
  buildUrl(path, params = {}) {
    let url = `${this.baseUrl}${path}`;
    
    // 替换路径参数
    Object.keys(params).forEach(key => {
      url = url.replace(`{${key}}`, params[key]);
    });
    
    return url;
  }
}

// ============================================================================
// 推荐服务 API
// ============================================================================

/**
 * 获取个性化推荐
 * 
 * @param {ApiClient} client - API 客户端
 * @param {Object} params - 请求参数
 * @returns {Object} 响应结果和检查状态
 */
export function getRecommendations(client, params) {
  const { userId, limit = 20, scene = 'home', context = {} } = params;
  
  const payload = {
    user_id: userId,
    limit: limit,
    scene: scene,
    context: context,
  };
  
  const response = client.post(apiEndpoints.recommend.getRecommendations, payload);
  
  const success = check(response, {
    'recommend: status is 200': (r) => r.status === 200,
    'recommend: has recommendations': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && 
               body.data.recommendations && 
               body.data.recommendations.length > 0;
      } catch {
        return false;
      }
    },
    'recommend: response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  return {
    response,
    success,
    latency: response.timings.duration,
  };
}

/**
 * 获取相似推荐
 */
export function getSimilarItems(client, params) {
  const { itemId, limit = 10 } = params;
  
  const payload = {
    item_id: itemId,
    limit: limit,
  };
  
  const response = client.post(apiEndpoints.recommend.getSimilar, payload);
  
  const success = check(response, {
    'similar: status is 200': (r) => r.status === 200,
    'similar: has items': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && Array.isArray(body.data);
      } catch {
        return false;
      }
    },
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 冷启动推荐
 */
export function getColdStartRecommendations(client, params) {
  const { userId, metadata = {} } = params;
  
  const payload = {
    user_id: userId,
    user_metadata: metadata,
  };
  
  const response = client.post(apiEndpoints.recommend.getColdStart, payload);
  
  const success = check(response, {
    'coldstart: status is 200': (r) => r.status === 200,
    'coldstart: has recommendations': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.recommendations;
      } catch {
        return false;
      }
    },
  });
  
  return { response, success, latency: response.timings.duration };
}

// ============================================================================
// 用户服务 API
// ============================================================================

/**
 * 获取用户信息
 */
export function getUserProfile(client, userId) {
  const response = client.get(apiEndpoints.user.getProfile, { userId });
  
  const success = check(response, {
    'user profile: status is 200': (r) => r.status === 200,
    'user profile: has user data': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.id;
      } catch {
        return false;
      }
    },
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 获取用户历史
 */
export function getUserHistory(client, userId, limit = 50) {
  const response = client.get(
    `${apiEndpoints.user.getHistory}?limit=${limit}`, 
    { userId }
  );
  
  const success = check(response, {
    'user history: status is 200': (r) => r.status === 200,
  });
  
  return { response, success, latency: response.timings.duration };
}

// ============================================================================
// 物品服务 API
// ============================================================================

/**
 * 获取物品详情
 */
export function getItemDetail(client, itemId) {
  const response = client.get(apiEndpoints.item.getItem, { itemId });
  
  const success = check(response, {
    'item detail: status is 200 or 404': (r) => r.status === 200 || r.status === 404,
    'item detail: has item data if 200': (r) => {
      if (r.status === 404) return true;
      try {
        const body = JSON.parse(r.body);
        return body.data && body.data.id;
      } catch {
        return false;
      }
    },
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 搜索物品
 */
export function searchItems(client, params) {
  const { query, limit = 20, offset = 0, filters = {} } = params;
  
  let url = `${apiEndpoints.item.search}?q=${encodeURIComponent(query)}&limit=${limit}&offset=${offset}`;
  
  // 添加过滤条件
  Object.keys(filters).forEach(key => {
    url += `&${key}=${encodeURIComponent(filters[key])}`;
  });
  
  const response = client.get(url);
  
  const success = check(response, {
    'search: status is 200': (r) => r.status === 200,
    'search: has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.data && Array.isArray(body.data);
      } catch {
        return false;
      }
    },
    'search: response time < 300ms': (r) => r.timings.duration < 300,
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 获取热门物品
 */
export function getPopularItems(client, category = null, limit = 20) {
  let url = `${apiEndpoints.item.getPopular}?limit=${limit}`;
  if (category) {
    url += `&category=${encodeURIComponent(category)}`;
  }
  
  const response = client.get(url);
  
  const success = check(response, {
    'popular: status is 200': (r) => r.status === 200,
  });
  
  return { response, success, latency: response.timings.duration };
}

// ============================================================================
// 反馈服务 API
// ============================================================================

/**
 * 提交用户反馈
 */
export function submitFeedback(client, params) {
  const { userId, itemId, action, timestamp = Date.now(), metadata = {} } = params;
  
  const payload = {
    user_id: userId,
    item_id: itemId,
    action: action,
    timestamp: timestamp,
    metadata: metadata,
  };
  
  const response = client.post(apiEndpoints.feedback.submit, payload);
  
  const success = check(response, {
    'feedback: status is 200 or 204': (r) => r.status === 200 || r.status === 204,
    'feedback: response time < 100ms': (r) => r.timings.duration < 100,
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 批量提交反馈
 */
export function submitBatchFeedback(client, feedbacks) {
  const response = client.post(apiEndpoints.feedback.batch, { feedbacks });
  
  const success = check(response, {
    'batch feedback: status is 200': (r) => r.status === 200,
  });
  
  return { response, success, latency: response.timings.duration };
}

// ============================================================================
// 健康检查 API
// ============================================================================

/**
 * 健康检查
 */
export function healthCheck(client) {
  const response = client.get(apiEndpoints.health);
  
  const success = check(response, {
    'health: status is 200': (r) => r.status === 200,
  });
  
  return { response, success };
}

/**
 * 就绪检查
 */
export function readinessCheck(client) {
  const response = client.get(apiEndpoints.ready);
  
  const success = check(response, {
    'ready: status is 200': (r) => r.status === 200,
  });
  
  return { response, success };
}

// ============================================================================
// 管理 API
// ============================================================================

/**
 * 获取仪表盘数据
 */
export function getDashboard(client) {
  const response = client.get(apiEndpoints.admin.dashboard);
  
  const success = check(response, {
    'dashboard: status is 200': (r) => r.status === 200,
  });
  
  return { response, success, latency: response.timings.duration };
}

/**
 * 列出用户
 */
export function listUsers(client, page = 1, pageSize = 20) {
  const response = client.get(
    `${apiEndpoints.admin.users}?page=${page}&page_size=${pageSize}`
  );
  
  const success = check(response, {
    'list users: status is 200': (r) => r.status === 200,
  });
  
  return { response, success, latency: response.timings.duration };
}

// ============================================================================
// 默认导出
// ============================================================================

export default {
  ApiClient,
  getRecommendations,
  getSimilarItems,
  getColdStartRecommendations,
  getUserProfile,
  getUserHistory,
  getItemDetail,
  searchItems,
  getPopularItems,
  submitFeedback,
  submitBatchFeedback,
  healthCheck,
  readinessCheck,
  getDashboard,
  listUsers,
};

