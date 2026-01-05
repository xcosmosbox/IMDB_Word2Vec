"""
Locust 负载测试脚本

支持 Web UI 和分布式测试
用于生成式推荐系统的性能测试

用法:
  # 本地 Web UI 模式
  locust -f locustfile.py --host http://localhost:8080
  
  # 无头模式
  locust -f locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 5m
  
  # 分布式模式 (Master)
  locust -f locustfile.py --master --host http://localhost:8080
  
  # 分布式模式 (Worker)
  locust -f locustfile.py --worker --master-host <master-ip>
"""

import random
import time
import json
import logging
from typing import Dict, Any, Optional, List

from locust import HttpUser, task, between, events, tag
from locust.runners import MasterRunner, WorkerRunner
from locust.env import Environment

from config import (
    get_environment,
    get_auth_headers,
    ENDPOINTS,
    TEST_DATA,
    SLA,
    TASK_WEIGHTS,
    REPORT_CONFIG,
)

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 辅助函数
# ============================================================================

def random_user_id() -> str:
    """生成随机用户 ID"""
    return f"user_{random.randint(TEST_DATA.user_id_min, TEST_DATA.user_id_max)}"


def random_item_id() -> str:
    """生成随机物品 ID"""
    return f"item_{random.randint(TEST_DATA.item_id_min, TEST_DATA.item_id_max)}"


def random_search_term() -> str:
    """获取随机搜索词"""
    return random.choice(TEST_DATA.search_terms)


def random_scene() -> str:
    """获取随机推荐场景"""
    return random.choice(TEST_DATA.recommend_scenes)


def random_feedback_action() -> str:
    """获取随机反馈动作"""
    return random.choice(TEST_DATA.feedback_actions)


def random_context() -> Dict[str, Any]:
    """生成随机上下文"""
    return {
        "device": random.choice(["mobile", "desktop", "tablet"]),
        "platform": random.choice(["ios", "android", "web"]),
        "hour_of_day": random.randint(0, 23),
        "is_weekend": random.random() > 0.7,
        "session_duration": random.randint(0, 3600),
    }


# ============================================================================
# 推荐系统用户类
# ============================================================================

class RecommendUser(HttpUser):
    """模拟推荐系统普通用户"""
    
    # 请求间隔 1-3 秒
    wait_time = between(1, 3)
    
    # 用户权重 (普通用户较多)
    weight = 10
    
    def on_start(self):
        """用户启动时初始化"""
        self.user_id = random_user_id()
        self.headers = get_auth_headers()
        self.session_start = time.time()
        logger.debug(f"User {self.user_id} started")
    
    def on_stop(self):
        """用户停止时清理"""
        session_duration = time.time() - self.session_start
        logger.debug(f"User {self.user_id} stopped after {session_duration:.2f}s")
    
    # ========================================================================
    # 推荐相关任务
    # ========================================================================
    
    @task(TASK_WEIGHTS.get_recommendations)
    @tag("recommend", "core")
    def get_recommendations(self):
        """获取个性化推荐"""
        payload = {
            "user_id": self.user_id,
            "limit": TEST_DATA.default_limit,
            "scene": random_scene(),
            "context": random_context(),
        }
        
        with self.client.post(
            ENDPOINTS.recommend,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/recommend",
            catch_response=True,
        ) as response:
            self._validate_recommendation_response(response)
    
    @task(1)
    @tag("recommend", "similar")
    def get_similar_items(self):
        """获取相似物品推荐"""
        item_id = random_item_id()
        payload = {
            "item_id": item_id,
            "limit": 10,
        }
        
        with self.client.post(
            ENDPOINTS.recommend_similar,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/recommend/similar",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    # ========================================================================
    # 搜索相关任务
    # ========================================================================
    
    @task(TASK_WEIGHTS.search_items)
    @tag("search", "core")
    def search_items(self):
        """搜索物品"""
        query = random_search_term()
        
        with self.client.get(
            f"{ENDPOINTS.item_search}?q={query}&limit={TEST_DATA.default_limit}",
            headers=self.headers,
            name="GET /api/v1/items/search",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data.get("data"), list):
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status {response.status_code}")
    
    # ========================================================================
    # 反馈相关任务
    # ========================================================================
    
    @task(TASK_WEIGHTS.submit_feedback)
    @tag("feedback", "core")
    def submit_feedback(self):
        """提交用户反馈"""
        payload = {
            "user_id": self.user_id,
            "item_id": random_item_id(),
            "action": random_feedback_action(),
            "timestamp": int(time.time() * 1000),
        }
        
        with self.client.post(
            ENDPOINTS.feedback,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/feedback",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 204]:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    # ========================================================================
    # 物品详情任务
    # ========================================================================
    
    @task(TASK_WEIGHTS.get_item_detail)
    @tag("item", "detail")
    def get_item_detail(self):
        """获取物品详情"""
        item_id = random_item_id()
        
        with self.client.get(
            ENDPOINTS.item_detail.format(item_id=item_id),
            headers=self.headers,
            name="GET /api/v1/items/{item_id}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # 物品不存在是正常的
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _validate_recommendation_response(self, response):
        """验证推荐响应"""
        if response.status_code != 200:
            response.failure(f"Status {response.status_code}")
            return
        
        try:
            data = response.json()
            recommendations = data.get("data", {}).get("recommendations", [])
            
            if recommendations and len(recommendations) > 0:
                # 检查延迟 SLA
                latency = response.elapsed.total_seconds() * 1000
                if latency > SLA.p99_latency:
                    response.failure(f"Latency {latency:.2f}ms > SLA {SLA.p99_latency}ms")
                else:
                    response.success()
            else:
                response.failure("No recommendations returned")
        except json.JSONDecodeError:
            response.failure("Invalid JSON response")


# ============================================================================
# 冷启动用户类
# ============================================================================

class ColdStartUser(HttpUser):
    """模拟新用户 (冷启动场景)"""
    
    wait_time = between(2, 5)
    weight = 2  # 较低权重
    
    def on_start(self):
        """新用户初始化"""
        # 使用较大的 ID 范围模拟新用户
        self.user_id = f"new_user_{random.randint(1000000, 9999999)}"
        self.headers = get_auth_headers()
        self.first_interactions = 0
    
    @task(5)
    @tag("coldstart")
    def get_coldstart_recommendations(self):
        """获取冷启动推荐"""
        metadata = {
            "age_group": random.choice(["18-24", "25-34", "35-44", "45-54", "55+"]),
            "gender": random.choice(["male", "female", "other"]),
            "region": random.choice(["north", "south", "east", "west"]),
        }
        
        payload = {
            "user_id": self.user_id,
            "user_metadata": metadata,
            "limit": 20,
        }
        
        with self.client.post(
            ENDPOINTS.recommend_coldstart,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/recommend/coldstart",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(3)
    @tag("coldstart", "feedback")
    def submit_first_feedback(self):
        """模拟新用户首次交互"""
        payload = {
            "user_id": self.user_id,
            "item_id": random_item_id(),
            "action": random.choice(["view", "click"]),
            "timestamp": int(time.time() * 1000),
        }
        
        with self.client.post(
            ENDPOINTS.feedback,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/feedback (coldstart)",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 204]:
                self.first_interactions += 1
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# ============================================================================
# 管理员用户类
# ============================================================================

class AdminUser(HttpUser):
    """模拟管理员用户 (低频操作)"""
    
    wait_time = between(5, 10)
    weight = 1  # 最低权重
    
    def on_start(self):
        """管理员初始化"""
        self.headers = get_auth_headers()
        # 添加管理员角色
        self.headers["X-User-Role"] = "admin"
    
    @task(TASK_WEIGHTS.get_dashboard)
    @tag("admin", "dashboard")
    def get_dashboard(self):
        """获取仪表盘数据"""
        with self.client.get(
            ENDPOINTS.admin_dashboard,
            headers=self.headers,
            name="GET /api/admin/v1/dashboard",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 403:
                response.success()  # 权限不足也是预期的
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(TASK_WEIGHTS.list_users)
    @tag("admin", "users")
    def list_users(self):
        """列出用户"""
        page = random.randint(1, 10)
        
        with self.client.get(
            f"{ENDPOINTS.admin_users}?page={page}&page_size=20",
            headers=self.headers,
            name="GET /api/admin/v1/users",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 403]:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    @tag("admin", "metrics")
    def get_metrics(self):
        """获取系统指标"""
        with self.client.get(
            ENDPOINTS.admin_metrics,
            headers=self.headers,
            name="GET /api/admin/v1/metrics",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 403]:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# ============================================================================
# 高并发用户类 (用于压力测试)
# ============================================================================

class HighLoadUser(HttpUser):
    """高负载用户 (用于压力/峰值测试)"""
    
    # 更短的等待时间
    wait_time = between(0.1, 0.5)
    weight = 0  # 默认禁用，需要手动启用
    
    def on_start(self):
        self.user_id = random_user_id()
        self.headers = get_auth_headers()
    
    @task(10)
    @tag("stress", "recommend")
    def rapid_recommendations(self):
        """快速获取推荐"""
        payload = {
            "user_id": self.user_id,
            "limit": 10,  # 减少返回数量
            "scene": "home",
        }
        
        with self.client.post(
            ENDPOINTS.recommend,
            json=payload,
            headers=self.headers,
            name="POST /api/v1/recommend (stress)",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# ============================================================================
# 事件钩子
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """测试开始时"""
    logger.info("=" * 60)
    logger.info("       LOCUST LOAD TEST STARTING")
    logger.info("=" * 60)
    
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running in DISTRIBUTED mode (Master)")
    elif isinstance(environment.runner, WorkerRunner):
        logger.info("Running in DISTRIBUTED mode (Worker)")
    else:
        logger.info("Running in STANDALONE mode")
    
    env_config = get_environment()
    logger.info(f"Environment: {env_config.name}")
    logger.info(f"Target URL: {env_config.base_url}")
    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """测试结束时"""
    logger.info("=" * 60)
    logger.info("       LOCUST LOAD TEST COMPLETED")
    logger.info("=" * 60)
    
    # 输出统计摘要
    stats = environment.stats.total
    
    logger.info(f"Total Requests: {stats.num_requests}")
    logger.info(f"Total Failures: {stats.num_failures}")
    logger.info(f"Failure Rate: {stats.fail_ratio * 100:.2f}%")
    logger.info(f"Average Response Time: {stats.avg_response_time:.2f}ms")
    logger.info(f"Requests/s: {stats.total_rps:.2f}")
    
    # SLA 检查
    logger.info("")
    logger.info("SLA Checks:")
    
    # 可用性检查
    availability = (1 - stats.fail_ratio) * 100
    availability_passed = availability >= SLA.availability * 100
    logger.info(f"  Availability: {availability:.2f}% "
                f"(Target: {SLA.availability * 100}%) "
                f"{'✓ PASS' if availability_passed else '✗ FAIL'}")
    
    # 延迟检查
    p50 = stats.get_response_time_percentile(0.5) or 0
    p99 = stats.get_response_time_percentile(0.99) or 0
    
    p50_passed = p50 <= SLA.p50_latency
    p99_passed = p99 <= SLA.p99_latency
    
    logger.info(f"  P50 Latency: {p50:.2f}ms "
                f"(Target: {SLA.p50_latency}ms) "
                f"{'✓ PASS' if p50_passed else '✗ FAIL'}")
    logger.info(f"  P99 Latency: {p99:.2f}ms "
                f"(Target: {SLA.p99_latency}ms) "
                f"{'✓ PASS' if p99_passed else '✗ FAIL'}")
    
    # 错误率检查
    error_rate = stats.fail_ratio
    error_passed = error_rate <= SLA.error_rate
    logger.info(f"  Error Rate: {error_rate * 100:.4f}% "
                f"(Target: {SLA.error_rate * 100}%) "
                f"{'✓ PASS' if error_passed else '✗ FAIL'}")
    
    logger.info("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, 
               context, exception, start_time, url, **kwargs):
    """每个请求完成时 (用于自定义指标收集)"""
    # 可以在这里添加自定义指标收集逻辑
    pass


@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response, 
                       exception, context, **kwargs):
    """请求失败时"""
    logger.warning(f"Request failed: {name} - {exception}")


# ============================================================================
# 自定义形状类 (可选)
# ============================================================================

class StagesShape:
    """
    自定义负载形状类
    
    用于实现更复杂的负载模式，如阶梯式增加、峰值测试等
    
    使用方法:
    locust -f locustfile.py --host http://localhost:8080 StagesShape
    """
    
    stages = [
        {"duration": 60, "users": 100, "spawn_rate": 10},   # 预热
        {"duration": 120, "users": 300, "spawn_rate": 20},  # 增加
        {"duration": 120, "users": 500, "spawn_rate": 30},  # 中等
        {"duration": 120, "users": 800, "spawn_rate": 40},  # 高负载
        {"duration": 120, "users": 1000, "spawn_rate": 50}, # 峰值
        {"duration": 60, "users": 100, "spawn_rate": 100},  # 恢复
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        cumulative_time = 0
        for stage in self.stages:
            cumulative_time += stage["duration"]
            if run_time < cumulative_time:
                return (stage["users"], stage["spawn_rate"])
        
        return None  # 测试结束


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Locust Load Test for Generative Recommendation System")
    print("")
    print("Usage:")
    print("  locust -f locustfile.py --host http://localhost:8080")
    print("")
    print("Available User Classes:")
    print("  - RecommendUser: Normal user behavior (weight: 10)")
    print("  - ColdStartUser: New user cold start (weight: 2)")
    print("  - AdminUser: Admin operations (weight: 1)")
    print("  - HighLoadUser: Stress testing (weight: 0, enable manually)")
    print("")
    print("Run with 'locust' command to start the test.")
    sys.exit(0)

