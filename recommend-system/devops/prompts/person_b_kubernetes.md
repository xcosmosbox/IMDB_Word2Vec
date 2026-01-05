# Person B: Kubernetes 配置增强

## 你的角色
你是一名 DevOps 工程师，负责实现生成式推荐系统的 **完整 Kubernetes 配置**，包括基础资源、Service Mesh、Ingress、配置管理等。

---

## ⚠️ 重要：接口驱动开发

**开始编码前，必须先阅读接口定义文件：**

```
devops/interfaces.yaml
```

你需要实现的契约：

```yaml
kubernetes:
  namespaces:
    - recommend-dev
    - recommend-prod
  
  services:
    recommend-service:
      http: 8080
      grpc: 9090
      metrics: 9091
```

---

## 你的任务

```
devops/kubernetes/
├── base/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── kustomization.yaml
├── overlays/
│   ├── dev/
│   │   ├── kustomization.yaml
│   │   ├── patches/
│   │   └── resources/
│   └── prod/
│       ├── kustomization.yaml
│       ├── patches/
│       ├── canary.yaml
│       └── resources/
├── istio/
│   ├── gateway.yaml
│   ├── virtual-service.yaml
│   ├── destination-rule.yaml
│   └── authorization-policy.yaml
└── ingress/
    ├── ingress.yaml
    └── certificate.yaml
```

---

## 1. Base Deployment (base/deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommend-service
  labels:
    app: recommend-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommend-service
  template:
    metadata:
      labels:
        app: recommend-service
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9091"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: recommend-service
      
      # 反亲和性 - 分布在不同节点
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: recommend-service
                topologyKey: kubernetes.io/hostname
      
      # 初始化容器
      initContainers:
        - name: wait-for-db
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              until nc -z postgres-service 5432; do
                echo "Waiting for PostgreSQL..."
                sleep 2
              done
              echo "PostgreSQL is ready!"
      
      containers:
        - name: recommend-service
          image: recommend-service:latest
          imagePullPolicy: Always
          
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: grpc
              containerPort: 9090
              protocol: TCP
            - name: metrics
              containerPort: 9091
              protocol: TCP
          
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: app-secrets
          
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          
          # 存活探针
          livenessProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          # 就绪探针
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          
          # 启动探针
          startupProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 30
          
          # 安全上下文
          securityContext:
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000
            capabilities:
              drop:
                - ALL
          
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /var/cache
      
      volumes:
        - name: tmp
          emptyDir: {}
        - name: cache
          emptyDir:
            sizeLimit: 1Gi
      
      # 优雅关闭
      terminationGracePeriodSeconds: 30
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ugt-inference
  labels:
    app: ugt-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ugt-inference
  template:
    metadata:
      labels:
        app: ugt-inference
    spec:
      # GPU 节点选择
      nodeSelector:
        nvidia.com/gpu: "true"
      
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      
      containers:
        - name: ugt-inference
          image: ugt-inference:latest
          
          ports:
            - name: grpc
              containerPort: 50051
            - name: metrics
              containerPort: 9094
          
          resources:
            requests:
              cpu: "2000m"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4000m"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: MODEL_PATH
              value: "/models/ugt"
          
          volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true
      
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
```

---

## 2. ConfigMap (base/configmap.yaml)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # 应用配置
  LOG_LEVEL: "info"
  LOG_FORMAT: "json"
  ENABLE_TRACING: "true"
  TRACING_ENDPOINT: "http://jaeger-collector:14268/api/traces"
  
  # 缓存配置
  CACHE_TTL: "300"
  CACHE_MAX_SIZE: "10000"
  
  # 限流配置
  RATE_LIMIT_RPS: "1000"
  RATE_LIMIT_BURST: "2000"
  
  # 特性开关
  FEATURE_NEW_ALGORITHM: "false"
  FEATURE_AB_TEST: "true"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: db-config
data:
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "recommend"
  
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  
  MILVUS_HOST: "milvus-service"
  MILVUS_PORT: "19530"
```

---

## 3. Service (base/service.yaml)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: recommend-service
  labels:
    app: recommend-service
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8080
      targetPort: http
      protocol: TCP
    - name: grpc
      port: 9090
      targetPort: grpc
      protocol: TCP
    - name: metrics
      port: 9091
      targetPort: metrics
      protocol: TCP
  selector:
    app: recommend-service
---
apiVersion: v1
kind: Service
metadata:
  name: ugt-inference
  labels:
    app: ugt-inference
spec:
  type: ClusterIP
  ports:
    - name: grpc
      port: 50051
      targetPort: grpc
      protocol: TCP
  selector:
    app: ugt-inference
```

---

## 4. HPA (base/hpa.yaml)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: recommend-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recommend-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # 自定义指标
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

---

## 5. Istio Virtual Service (istio/virtual-service.yaml)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: recommend-vs
spec:
  hosts:
    - recommend-service
    - api.recommend.example.com
  gateways:
    - recommend-gateway
  http:
    # 金丝雀路由
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: recommend-service
            subset: canary
    
    # A/B 测试
    - match:
        - headers:
            x-user-group:
              exact: "experiment"
      route:
        - destination:
            host: recommend-service
            subset: v2
    
    # 默认路由
    - route:
        - destination:
            host: recommend-service
            subset: stable
          weight: 95
        - destination:
            host: recommend-service
            subset: canary
          weight: 5
      
      # 超时和重试
      timeout: 10s
      retries:
        attempts: 3
        perTryTimeout: 3s
        retryOn: gateway-error,connect-failure,refused-stream
      
      # 熔断
      fault:
        delay:
          percentage:
            value: 0.1
          fixedDelay: 5s
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: recommend-dr
spec:
  host: recommend-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  
  subsets:
    - name: stable
      labels:
        version: v1
    - name: canary
      labels:
        version: canary
    - name: v2
      labels:
        version: v2
```

---

## 6. Ingress (ingress/ingress.yaml)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: recommend-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.recommend.example.com
        - admin.recommend.example.com
      secretName: recommend-tls
  
  rules:
    - host: api.recommend.example.com
      http:
        paths:
          - path: /api/v1
            pathType: Prefix
            backend:
              service:
                name: recommend-service
                port:
                  number: 8080
    
    - host: admin.recommend.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: admin-frontend
                port:
                  number: 80
```

---

## 注意事项

1. 使用 Kustomize 管理环境差异
2. 配置合理的资源限制
3. 实现优雅关闭
4. 配置 Pod 反亲和性
5. GPU 工作负载特殊处理

## 输出要求

请输出完整的 K8s 配置，包含：
1. 所有 base 资源
2. dev/prod overlay
3. Istio 配置
4. Ingress 配置

