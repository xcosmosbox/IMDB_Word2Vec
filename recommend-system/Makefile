# 生成式推荐系统 Makefile

.PHONY: all build run test clean docker lint fmt help

# 版本信息
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME ?= $(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME)"

# Go 参数
GOCMD := go
GOBUILD := $(GOCMD) build
GOTEST := $(GOCMD) test
GOVET := $(GOCMD) vet
GOFMT := gofmt
GOMOD := $(GOCMD) mod

# 项目参数
BINARY_NAME := recommend-service
CMD_DIR := ./cmd/recommend-service
OUTPUT_DIR := ./bin

# Docker 参数
DOCKER_IMAGE := recommend-service
DOCKER_TAG := $(VERSION)

## all: 构建所有服务
all: build

## build: 构建服务
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(OUTPUT_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME) $(CMD_DIR)
	@echo "Build complete: $(OUTPUT_DIR)/$(BINARY_NAME)"

## build-linux: 交叉编译 Linux 版本
build-linux:
	@echo "Building $(BINARY_NAME) for Linux..."
	@mkdir -p $(OUTPUT_DIR)
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME)-linux $(CMD_DIR)
	@echo "Build complete: $(OUTPUT_DIR)/$(BINARY_NAME)-linux"

## run: 运行服务
run: build
	@echo "Starting $(BINARY_NAME)..."
	$(OUTPUT_DIR)/$(BINARY_NAME) -config configs/config.yaml

## test: 运行测试
test:
	@echo "Running tests..."
	$(GOTEST) -v -race -cover ./...

## test-coverage: 生成测试覆盖率报告
test-coverage:
	@echo "Running tests with coverage..."
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

## lint: 代码检查
lint:
	@echo "Running linters..."
	$(GOVET) ./...
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed, skipping..."; \
	fi

## fmt: 格式化代码
fmt:
	@echo "Formatting code..."
	$(GOFMT) -w -s .

## mod: 管理依赖
mod:
	@echo "Tidying modules..."
	$(GOMOD) tidy
	$(GOMOD) verify

## clean: 清理构建产物
clean:
	@echo "Cleaning..."
	@rm -rf $(OUTPUT_DIR)
	@rm -f coverage.out coverage.html
	@echo "Clean complete"

## docker-build: 构建 Docker 镜像
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-f deployments/docker/Dockerfile .
	@echo "Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

## docker-push: 推送 Docker 镜像
docker-push:
	@echo "Pushing Docker image..."
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

## docker-run: 运行 Docker 容器
docker-run:
	@echo "Running Docker container..."
	docker run -d --name $(BINARY_NAME) \
		-p 8080:8080 -p 9090:9090 -p 9091:9091 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

## compose-up: 启动 docker-compose
compose-up:
	@echo "Starting services..."
	cd deployments/docker && docker-compose up -d

## compose-down: 停止 docker-compose
compose-down:
	@echo "Stopping services..."
	cd deployments/docker && docker-compose down

## compose-logs: 查看日志
compose-logs:
	cd deployments/docker && docker-compose logs -f recommend-service

## init-db: 初始化数据库
init-db:
	@echo "Initializing database..."
	psql -h localhost -U postgres -d recommend -f scripts/init_db.sql

## proto: 生成 gRPC 代码 (如果使用 protobuf)
proto:
	@echo "Generating protobuf code..."
	protoc --go_out=. --go-grpc_out=. api/**/*.proto

## swagger: 生成 API 文档
swagger:
	@echo "Generating Swagger docs..."
	swag init -g cmd/recommend-service/main.go -o docs/swagger

## k8s-deploy: 部署到 Kubernetes
k8s-deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployments/kubernetes/

## k8s-delete: 从 Kubernetes 删除
k8s-delete:
	@echo "Deleting from Kubernetes..."
	kubectl delete -f deployments/kubernetes/

## help: 显示帮助信息
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'

# 默认目标
.DEFAULT_GOAL := help

