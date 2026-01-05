# Person A: CI/CD 流水线

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **CI/CD 流水线**，包括 GitHub Actions 配置、构建流程、自动化测试和部署流程。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
cicd:
  workflows:
    - name: ci.yaml           # 持续集成
    - name: cd-dev.yaml       # 开发环境部署
    - name: cd-prod.yaml      # 生产环境部署
  
  artifacts:
    - recommend-service:latest
    - user-service:latest
    - item-service:latest
    - ugt-inference:latest
```

---

## 你的任务

```
devops/ci-cd/
├── .github/
│   └── workflows/
│       ├── ci.yaml              # 持续集成
│       ├── cd-dev.yaml          # 开发环境部署
│       ├── cd-prod.yaml         # 生产环境部署
│       ├── security-scan.yaml   # 安全扫描
│       └── release.yaml         # 版本发布
├── scripts/
│   ├── build.sh                 # 构建脚本
│   ├── test.sh                  # 测试脚本
│   ├── deploy.sh                # 部署脚本
│   └── rollback.sh              # 回滚脚本
└── Makefile                     # 常用命令
```

---

## 1. 持续集成 (ci.yaml)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  GO_VERSION: '1.21'
  PYTHON_VERSION: '3.10'
  NODE_VERSION: '20'

jobs:
  # ==========================================================================
  # Go 后端检查
  # ==========================================================================
  go-lint:
    name: Go Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ env.GO_VERSION }}
      
      - name: golangci-lint
        uses: golangci/golangci-lint-action@v3
        with:
          version: latest
          working-directory: ./recommend-system

  go-test:
    name: Go Test
    runs-on: ubuntu-latest
    needs: go-lint
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: recommend_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ env.GO_VERSION }}
      
      - name: Run tests
        working-directory: ./recommend-system
        run: |
          go test -v -race -coverprofile=coverage.out ./...
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./recommend-system/coverage.out
          flags: go

  go-build:
    name: Go Build
    runs-on: ubuntu-latest
    needs: go-test
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ env.GO_VERSION }}
      
      - name: Build
        working-directory: ./recommend-system
        run: |
          CGO_ENABLED=0 GOOS=linux go build -o bin/recommend-service ./cmd/recommend-service
          CGO_ENABLED=0 GOOS=linux go build -o bin/user-service ./cmd/user-service
          CGO_ENABLED=0 GOOS=linux go build -o bin/item-service ./cmd/item-service
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: go-binaries
          path: recommend-system/bin/

  # ==========================================================================
  # Python 算法检查
  # ==========================================================================
  python-lint:
    name: Python Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install flake8 black isort mypy
      
      - name: Lint
        working-directory: ./recommend-system/algorithm
        run: |
          flake8 . --max-line-length=100
          black --check .
          isort --check-only .

  python-test:
    name: Python Test
    runs-on: ubuntu-latest
    needs: python-lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        working-directory: ./recommend-system/algorithm
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        working-directory: ./recommend-system/algorithm
        run: |
          pytest --cov=. --cov-report=xml -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./recommend-system/algorithm/coverage.xml
          flags: python

  # ==========================================================================
  # 前端检查
  # ==========================================================================
  frontend-lint:
    name: Frontend Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Install dependencies
        working-directory: ./recommend-system/frontend/user-app
        run: npm ci
      
      - name: Lint
        working-directory: ./recommend-system/frontend/user-app
        run: npm run lint

  frontend-test:
    name: Frontend Test
    runs-on: ubuntu-latest
    needs: frontend-lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Install & Test (User App)
        working-directory: ./recommend-system/frontend/user-app
        run: |
          npm ci
          npm run test -- --coverage
      
      - name: Install & Test (Admin)
        working-directory: ./recommend-system/frontend/admin
        run: |
          npm ci
          npm run test -- --coverage

  frontend-build:
    name: Frontend Build
    runs-on: ubuntu-latest
    needs: frontend-test
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Build User App
        working-directory: ./recommend-system/frontend/user-app
        run: |
          npm ci
          npm run build
      
      - name: Build Admin
        working-directory: ./recommend-system/frontend/admin
        run: |
          npm ci
          npm run build
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: frontend-dist
          path: |
            recommend-system/frontend/user-app/dist
            recommend-system/frontend/admin/dist

  # ==========================================================================
  # Docker 构建
  # ==========================================================================
  docker-build:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: [go-build, python-test, frontend-build]
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ secrets.DOCKER_REGISTRY_URL }}
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./recommend-system
          file: ./recommend-system/deployments/docker/Dockerfile
          push: true
          tags: |
            ${{ secrets.DOCKER_REGISTRY_URL }}/recommend-service:${{ github.sha }}
            ${{ secrets.DOCKER_REGISTRY_URL }}/recommend-service:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 2. 开发环境部署 (cd-dev.yaml)

```yaml
name: CD - Development

on:
  push:
    branches: [develop]
  workflow_dispatch:

jobs:
  deploy:
    name: Deploy to Dev
    runs-on: ubuntu-latest
    environment: development
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure kubeconfig
        run: |
          echo "${{ secrets.KUBECONFIG_DEV }}" | base64 -d > $HOME/.kube/config
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/recommend-service \
            recommend-service=${{ secrets.DOCKER_REGISTRY_URL }}/recommend-service:${{ github.sha }} \
            -n recommend-dev
          
          kubectl rollout status deployment/recommend-service -n recommend-dev --timeout=5m
      
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: "Dev deployment: ${{ job.status }}"
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## 3. 生产环境部署 (cd-prod.yaml)

```yaml
name: CD - Production

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true

jobs:
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: security-scan
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure kubeconfig
        run: |
          echo "${{ secrets.KUBECONFIG_PROD }}" | base64 -d > $HOME/.kube/config
      
      - name: Get version
        id: version
        run: |
          if [ "${{ github.event_name }}" == "release" ]; then
            echo "version=${{ github.event.release.tag_name }}" >> $GITHUB_OUTPUT
          else
            echo "version=${{ github.event.inputs.version }}" >> $GITHUB_OUTPUT
          fi
      
      # 金丝雀发布
      - name: Canary Deploy (5%)
        run: |
          kubectl apply -f devops/kubernetes/overlays/prod/canary.yaml
          kubectl set image deployment/recommend-service-canary \
            recommend-service=${{ secrets.DOCKER_REGISTRY_URL }}/recommend-service:${{ steps.version.outputs.version }} \
            -n recommend-prod
          
          sleep 300  # 观察 5 分钟
      
      - name: Check Canary Metrics
        run: |
          # 检查错误率
          ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])" | jq '.data.result[0].value[1]')
          if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
            echo "Error rate too high: $ERROR_RATE"
            exit 1
          fi
      
      - name: Full Deploy
        run: |
          kubectl set image deployment/recommend-service \
            recommend-service=${{ secrets.DOCKER_REGISTRY_URL }}/recommend-service:${{ steps.version.outputs.version }} \
            -n recommend-prod
          
          kubectl rollout status deployment/recommend-service -n recommend-prod --timeout=10m
      
      - name: Remove Canary
        if: success()
        run: |
          kubectl delete deployment recommend-service-canary -n recommend-prod --ignore-not-found
      
      - name: Rollback on Failure
        if: failure()
        run: |
          kubectl rollout undo deployment/recommend-service -n recommend-prod
      
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: "Production deployment ${{ steps.version.outputs.version }}: ${{ job.status }}"
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## 4. Makefile

```makefile
.PHONY: build test lint deploy

# 变量
VERSION ?= $(shell git describe --tags --always --dirty)
REGISTRY ?= your-registry.com

# 构建
build:
	docker build -t $(REGISTRY)/recommend-service:$(VERSION) -f deployments/docker/Dockerfile .

build-all:
	docker build -t $(REGISTRY)/recommend-service:$(VERSION) -f deployments/docker/Dockerfile .
	docker build -t $(REGISTRY)/user-service:$(VERSION) -f deployments/docker/Dockerfile.user .
	docker build -t $(REGISTRY)/item-service:$(VERSION) -f deployments/docker/Dockerfile.item .
	docker build -t $(REGISTRY)/ugt-inference:$(VERSION) -f algorithm/Dockerfile .

# 测试
test:
	cd recommend-system && go test -v -race ./...
	cd algorithm && pytest -v

test-coverage:
	cd recommend-system && go test -coverprofile=coverage.out ./...
	cd algorithm && pytest --cov=. --cov-report=html

# 代码检查
lint:
	cd recommend-system && golangci-lint run
	cd algorithm && flake8 . && black --check . && isort --check-only .
	cd frontend/user-app && npm run lint
	cd frontend/admin && npm run lint

# 部署
deploy-dev:
	kubectl apply -k devops/kubernetes/overlays/dev

deploy-prod:
	kubectl apply -k devops/kubernetes/overlays/prod

# 回滚
rollback:
	kubectl rollout undo deployment/recommend-service -n $(NAMESPACE)

# 清理
clean:
	rm -rf bin/ dist/ coverage.out
	docker system prune -f
```

---

## 注意事项

1. 所有密钥通过 GitHub Secrets 管理
2. 使用多阶段构建减小镜像大小
3. 金丝雀发布确保生产安全
4. 失败自动回滚
5. 通知到 Slack

## 输出要求

请输出完整的可运行配置，包含：
1. 所有 GitHub Actions workflow
2. 构建和部署脚本
3. Makefile
4. 文档说明

