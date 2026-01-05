# =============================================================================
# CI/CD Makefile
# 
# 用途: 提供常用的构建、测试、部署命令
# 用法: make [target]
# =============================================================================

.PHONY: help build test lint deploy rollback clean docker docker-push \
        build-go build-python build-frontend \
        test-go test-python test-frontend test-integration \
        deploy-dev deploy-staging deploy-prod \
        lint-go lint-python lint-frontend \
        security-scan coverage

# 默认目标
.DEFAULT_GOAL := help

# =============================================================================
# 变量定义
# =============================================================================

# 版本信息
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
GIT_COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

# Docker 配置
REGISTRY ?= localhost:5000
DOCKER_BUILDKIT ?= 1

# Go 配置
GO_VERSION ?= 1.21
GOOS ?= linux
GOARCH ?= amd64
CGO_ENABLED ?= 0

# 目录配置
PROJECT_ROOT := $(shell cd ../.. && pwd)
SCRIPTS_DIR := ./scripts
BIN_DIR := ../../bin
COVERAGE_DIR := ../../coverage

# 服务列表
GO_SERVICES := recommend-service user-service item-service
DOCKER_IMAGES := recommend-service user-service item-service ugt-inference user-app admin-app

# 颜色定义
COLOR_RESET := \033[0m
COLOR_GREEN := \033[0;32m
COLOR_BLUE := \033[0;34m
COLOR_YELLOW := \033[1;33m

# =============================================================================
# 帮助信息
# =============================================================================

help: ## 显示帮助信息
	@echo ""
	@echo "$(COLOR_BLUE)生成式推荐系统 - CI/CD 命令$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)构建命令:$(COLOR_RESET)"
	@grep -E '^(build|docker)[^:]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_GREEN)测试命令:$(COLOR_RESET)"
	@grep -E '^(test|lint|coverage|security)[^:]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_GREEN)部署命令:$(COLOR_RESET)"
	@grep -E '^(deploy|rollback)[^:]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_GREEN)其他命令:$(COLOR_RESET)"
	@grep -E '^(clean|setup|check)[^:]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_YELLOW)变量:$(COLOR_RESET)"
	@echo "  VERSION=$(VERSION)"
	@echo "  REGISTRY=$(REGISTRY)"
	@echo "  GOOS=$(GOOS) GOARCH=$(GOARCH)"
	@echo ""

# =============================================================================
# 构建命令
# =============================================================================

build: build-go build-python build-frontend ## 构建所有组件
	@echo "$(COLOR_GREEN)✓ 所有组件构建完成$(COLOR_RESET)"

build-go: ## 构建 Go 服务
	@echo "$(COLOR_BLUE)构建 Go 服务...$(COLOR_RESET)"
	@cd ../.. && \
	for service in $(GO_SERVICES); do \
		echo "  构建 $$service..."; \
		CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="-s -w -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)" \
		-o bin/$$service ./cmd/$$service; \
	done
	@echo "$(COLOR_GREEN)✓ Go 服务构建完成$(COLOR_RESET)"

build-python: ## 构建 Python 包
	@echo "$(COLOR_BLUE)构建 Python 包...$(COLOR_RESET)"
	@cd ../../algorithm && pip install build wheel -q && python -m build --wheel
	@echo "$(COLOR_GREEN)✓ Python 包构建完成$(COLOR_RESET)"

build-frontend: ## 构建前端应用
	@echo "$(COLOR_BLUE)构建前端应用...$(COLOR_RESET)"
	@cd ../../frontend/user-app && npm ci --silent && npm run build
	@cd ../../frontend/admin && npm ci --silent && npm run build
	@echo "$(COLOR_GREEN)✓ 前端应用构建完成$(COLOR_RESET)"

# =============================================================================
# Docker 命令
# =============================================================================

docker: ## 构建 Docker 镜像
	@echo "$(COLOR_BLUE)构建 Docker 镜像...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/build.sh docker --version $(VERSION) --registry $(REGISTRY)

docker-push: docker ## 构建并推送 Docker 镜像
	@echo "$(COLOR_BLUE)推送 Docker 镜像...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/build.sh docker --version $(VERSION) --registry $(REGISTRY) --push

docker-build-%: ## 构建单个 Docker 镜像 (例: docker-build-recommend-service)
	@echo "$(COLOR_BLUE)构建 $* 镜像...$(COLOR_RESET)"
	@cd ../.. && docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(REGISTRY)/$*:$(VERSION) \
		-t $(REGISTRY)/$*:latest \
		-f deployments/docker/Dockerfile$(if $(filter recommend-service,$*),,.$*) .

# =============================================================================
# 测试命令
# =============================================================================

test: test-go test-python test-frontend ## 运行所有测试
	@echo "$(COLOR_GREEN)✓ 所有测试完成$(COLOR_RESET)"

test-go: ## 运行 Go 测试
	@echo "$(COLOR_BLUE)运行 Go 测试...$(COLOR_RESET)"
	@cd ../.. && go test -v -race -short ./...
	@echo "$(COLOR_GREEN)✓ Go 测试完成$(COLOR_RESET)"

test-python: ## 运行 Python 测试
	@echo "$(COLOR_BLUE)运行 Python 测试...$(COLOR_RESET)"
	@cd ../../algorithm && pytest -v --ignore=serving/model_repository
	@echo "$(COLOR_GREEN)✓ Python 测试完成$(COLOR_RESET)"

test-frontend: ## 运行前端测试
	@echo "$(COLOR_BLUE)运行前端测试...$(COLOR_RESET)"
	@cd ../../frontend/user-app && npm run test -- --run || true
	@cd ../../frontend/admin && npm run test -- --run || true
	@echo "$(COLOR_GREEN)✓ 前端测试完成$(COLOR_RESET)"

test-integration: ## 运行集成测试
	@echo "$(COLOR_BLUE)运行集成测试...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/test.sh all --integration

coverage: ## 生成测试覆盖率报告
	@echo "$(COLOR_BLUE)生成覆盖率报告...$(COLOR_RESET)"
	@mkdir -p $(COVERAGE_DIR)
	@bash $(SCRIPTS_DIR)/test.sh all --coverage
	@echo "$(COLOR_GREEN)✓ 覆盖率报告: $(COVERAGE_DIR)/index.html$(COLOR_RESET)"

# =============================================================================
# 代码检查命令
# =============================================================================

lint: lint-go lint-python lint-frontend ## 运行所有代码检查
	@echo "$(COLOR_GREEN)✓ 所有代码检查完成$(COLOR_RESET)"

lint-go: ## 运行 Go 代码检查
	@echo "$(COLOR_BLUE)检查 Go 代码...$(COLOR_RESET)"
	@cd ../.. && golangci-lint run --timeout=5m
	@echo "$(COLOR_GREEN)✓ Go 代码检查完成$(COLOR_RESET)"

lint-python: ## 运行 Python 代码检查
	@echo "$(COLOR_BLUE)检查 Python 代码...$(COLOR_RESET)"
	@cd ../../algorithm && \
		flake8 . --max-line-length=100 --exclude=__pycache__,serving/model_repository && \
		black --check . && \
		isort --check-only .
	@echo "$(COLOR_GREEN)✓ Python 代码检查完成$(COLOR_RESET)"

lint-frontend: ## 运行前端代码检查
	@echo "$(COLOR_BLUE)检查前端代码...$(COLOR_RESET)"
	@cd ../../frontend/user-app && npm run lint
	@cd ../../frontend/admin && npm run lint
	@echo "$(COLOR_GREEN)✓ 前端代码检查完成$(COLOR_RESET)"

security-scan: ## 运行安全扫描
	@echo "$(COLOR_BLUE)运行安全扫描...$(COLOR_RESET)"
	@cd ../.. && govulncheck ./... || true
	@cd ../../algorithm && pip-audit -r requirements.txt || true
	@echo "$(COLOR_GREEN)✓ 安全扫描完成$(COLOR_RESET)"

# =============================================================================
# 部署命令
# =============================================================================

deploy-dev: ## 部署到开发环境
	@echo "$(COLOR_BLUE)部署到开发环境...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/deploy.sh dev --version $(VERSION) --registry $(REGISTRY)

deploy-staging: ## 部署到预发布环境
	@echo "$(COLOR_BLUE)部署到预发布环境...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/deploy.sh staging --version $(VERSION) --registry $(REGISTRY)

deploy-prod: ## 部署到生产环境 (金丝雀)
	@echo "$(COLOR_YELLOW)⚠️  即将部署到生产环境$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/deploy.sh prod --version $(VERSION) --registry $(REGISTRY) --canary

deploy-prod-full: ## 部署到生产环境 (全量)
	@echo "$(COLOR_YELLOW)⚠️  即将全量部署到生产环境$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/deploy.sh prod --version $(VERSION) --registry $(REGISTRY)

# =============================================================================
# 回滚命令
# =============================================================================

rollback-dev: ## 回滚开发环境
	@echo "$(COLOR_BLUE)回滚开发环境...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/rollback.sh dev

rollback-staging: ## 回滚预发布环境
	@echo "$(COLOR_BLUE)回滚预发布环境...$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/rollback.sh staging

rollback-prod: ## 回滚生产环境
	@echo "$(COLOR_YELLOW)⚠️  即将回滚生产环境$(COLOR_RESET)"
	@bash $(SCRIPTS_DIR)/rollback.sh prod

rollback-history: ## 显示部署历史
	@bash $(SCRIPTS_DIR)/rollback.sh dev --history

# =============================================================================
# 其他命令
# =============================================================================

setup: ## 安装开发依赖
	@echo "$(COLOR_BLUE)安装开发依赖...$(COLOR_RESET)"
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install golang.org/x/vuln/cmd/govulncheck@latest
	@pip install flake8 black isort mypy pytest pytest-cov pip-audit
	@echo "$(COLOR_GREEN)✓ 开发依赖安装完成$(COLOR_RESET)"

check: ## 检查环境配置
	@echo "$(COLOR_BLUE)检查环境配置...$(COLOR_RESET)"
	@echo "Go: $$(go version 2>/dev/null || echo '未安装')"
	@echo "Python: $$(python3 --version 2>/dev/null || echo '未安装')"
	@echo "Node: $$(node --version 2>/dev/null || echo '未安装')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo '未安装')"
	@echo "kubectl: $$(kubectl version --client --short 2>/dev/null || echo '未安装')"
	@echo "$(COLOR_GREEN)✓ 环境检查完成$(COLOR_RESET)"

clean: ## 清理构建产物
	@echo "$(COLOR_BLUE)清理构建产物...$(COLOR_RESET)"
	@rm -rf $(BIN_DIR)/* $(COVERAGE_DIR)/*
	@cd ../../frontend/user-app && rm -rf dist node_modules/.cache
	@cd ../../frontend/admin && rm -rf dist node_modules/.cache
	@cd ../../algorithm && rm -rf dist build *.egg-info __pycache__ .pytest_cache
	@echo "$(COLOR_GREEN)✓ 清理完成$(COLOR_RESET)"

# =============================================================================
# CI/CD 流水线命令 (用于 CI 环境)
# =============================================================================

ci-lint: lint ## CI: 代码检查
ci-test: test coverage ## CI: 测试和覆盖率
ci-build: build docker ## CI: 构建
ci-deploy: docker-push ## CI: 部署 (推送镜像)

# 完整 CI 流水线
ci: ci-lint ci-test ci-build ## 运行完整 CI 流水线
	@echo "$(COLOR_GREEN)✓ CI 流水线完成$(COLOR_RESET)"

