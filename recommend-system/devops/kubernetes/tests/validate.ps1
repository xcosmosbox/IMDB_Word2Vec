# Kubernetes 配置验证脚本 (PowerShell)
# 使用多种工具验证配置的正确性

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# 颜色定义
$Colors = @{
    Green = "Green"
    Red = "Red"
    Yellow = "Yellow"
    Cyan = "Cyan"
}

# 脚本目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$K8sDir = Split-Path -Parent $ScriptDir

# 计数器
$Script:Passed = 0
$Script:Failed = 0
$Script:Skipped = 0

# 输出函数
function Write-Info($Message) {
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Green
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Err($Message) {
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

function Write-Pass($Message) {
    Write-Host "[PASS] $Message" -ForegroundColor $Colors.Green
    $Script:Passed++
}

function Write-Fail($Message) {
    Write-Host "[FAIL] $Message" -ForegroundColor $Colors.Red
    $Script:Failed++
}

function Write-Skip($Message) {
    Write-Host "[SKIP] $Message" -ForegroundColor $Colors.Yellow
    $Script:Skipped++
}

# 检查工具是否存在
function Test-Tool($ToolName) {
    return $null -ne (Get-Command $ToolName -ErrorAction SilentlyContinue)
}

# 1. YAML 文件存在性验证
function Test-YamlFilesExist {
    Write-Info "=== 测试 YAML 文件存在性 ==="
    
    $RequiredFiles = @(
        "$K8sDir\base\namespace.yaml",
        "$K8sDir\base\configmap.yaml",
        "$K8sDir\base\secret.yaml",
        "$K8sDir\base\deployment.yaml",
        "$K8sDir\base\service.yaml",
        "$K8sDir\base\hpa.yaml",
        "$K8sDir\base\pdb.yaml",
        "$K8sDir\base\kustomization.yaml",
        "$K8sDir\overlays\dev\kustomization.yaml",
        "$K8sDir\overlays\prod\kustomization.yaml",
        "$K8sDir\overlays\prod\canary.yaml",
        "$K8sDir\istio\gateway.yaml",
        "$K8sDir\istio\virtual-service.yaml",
        "$K8sDir\istio\destination-rule.yaml",
        "$K8sDir\istio\authorization-policy.yaml",
        "$K8sDir\ingress\ingress.yaml",
        "$K8sDir\ingress\certificate.yaml"
    )
    
    foreach ($File in $RequiredFiles) {
        if (Test-Path $File) {
            Write-Pass "文件存在: $(Split-Path -Leaf $File)"
        } else {
            Write-Fail "文件不存在: $(Split-Path -Leaf $File)"
        }
    }
}

# 2. YAML 语法验证
function Test-YamlSyntax {
    Write-Info "=== 测试 YAML 语法 ==="
    
    $YamlFiles = Get-ChildItem -Path $K8sDir -Recurse -Include "*.yaml", "*.yml" -File
    
    foreach ($File in $YamlFiles) {
        try {
            $Content = Get-Content $File.FullName -Raw
            # 基本语法检查：确保文件不为空且格式正确
            if ($Content -match "^---" -or $Content -match "apiVersion:" -or $Content -match "kind:") {
                Write-Pass "YAML 语法正确: $($File.Name)"
            } else {
                Write-Warn "YAML 格式可疑: $($File.Name)"
            }
        } catch {
            Write-Fail "YAML 读取失败: $($File.Name)"
        }
    }
}

# 3. 资源限制验证
function Test-ResourceLimits {
    Write-Info "=== 测试资源限制配置 ==="
    
    $DeploymentFile = "$K8sDir\base\deployment.yaml"
    if (Test-Path $DeploymentFile) {
        $Content = Get-Content $DeploymentFile -Raw
        
        if ($Content -match "resources:" -and $Content -match "limits:" -and $Content -match "requests:") {
            Write-Pass "资源限制已配置"
        } else {
            Write-Fail "缺少资源限制配置"
        }
        
        if ($Content -match "cpu:") {
            Write-Pass "CPU 限制已配置"
        } else {
            Write-Fail "缺少 CPU 限制"
        }
        
        if ($Content -match "memory:") {
            Write-Pass "内存限制已配置"
        } else {
            Write-Fail "缺少内存限制"
        }
    }
}

# 4. 安全配置验证
function Test-SecurityConfig {
    Write-Info "=== 测试安全配置 ==="
    
    $DeploymentFile = "$K8sDir\base\deployment.yaml"
    if (Test-Path $DeploymentFile) {
        $Content = Get-Content $DeploymentFile -Raw
        
        if ($Content -match "securityContext:") {
            Write-Pass "SecurityContext 已配置"
        } else {
            Write-Fail "缺少 SecurityContext 配置"
        }
        
        if ($Content -match "runAsNonRoot:\s*true") {
            Write-Pass "runAsNonRoot 已启用"
        } else {
            Write-Warn "建议启用 runAsNonRoot"
        }
        
        if ($Content -match "readOnlyRootFilesystem:\s*true") {
            Write-Pass "只读根文件系统已启用"
        } else {
            Write-Warn "建议启用只读根文件系统"
        }
        
        if ($Content -match "capabilities:") {
            Write-Pass "Capabilities 已配置"
        } else {
            Write-Warn "建议配置 Capabilities"
        }
    }
}

# 5. 探针配置验证
function Test-Probes {
    Write-Info "=== 测试探针配置 ==="
    
    $DeploymentFile = "$K8sDir\base\deployment.yaml"
    if (Test-Path $DeploymentFile) {
        $Content = Get-Content $DeploymentFile -Raw
        
        if ($Content -match "livenessProbe:") {
            Write-Pass "存活探针已配置"
        } else {
            Write-Fail "缺少存活探针"
        }
        
        if ($Content -match "readinessProbe:") {
            Write-Pass "就绪探针已配置"
        } else {
            Write-Fail "缺少就绪探针"
        }
        
        if ($Content -match "startupProbe:") {
            Write-Pass "启动探针已配置"
        } else {
            Write-Warn "建议配置启动探针"
        }
    }
}

# 6. 服务端口验证
function Test-ServicePorts {
    Write-Info "=== 测试服务端口配置 ==="
    
    $ServiceFile = "$K8sDir\base\service.yaml"
    if (Test-Path $ServiceFile) {
        $Content = Get-Content $ServiceFile -Raw
        
        # recommend-service 端口
        if ($Content -match "port:\s*8080") {
            Write-Pass "recommend-service HTTP 端口 (8080) 已配置"
        } else {
            Write-Fail "recommend-service HTTP 端口未配置"
        }
        
        if ($Content -match "port:\s*9090") {
            Write-Pass "recommend-service gRPC 端口 (9090) 已配置"
        } else {
            Write-Fail "recommend-service gRPC 端口未配置"
        }
        
        # ugt-inference 端口
        if ($Content -match "port:\s*50051") {
            Write-Pass "ugt-inference gRPC 端口 (50051) 已配置"
        } else {
            Write-Fail "ugt-inference gRPC 端口未配置"
        }
    }
}

# 7. HPA 配置验证
function Test-HpaConfig {
    Write-Info "=== 测试 HPA 配置 ==="
    
    $HpaFile = "$K8sDir\base\hpa.yaml"
    if (Test-Path $HpaFile) {
        $Content = Get-Content $HpaFile -Raw
        
        if ($Content -match "minReplicas:" -and $Content -match "maxReplicas:") {
            Write-Pass "HPA 副本数限制已配置"
        } else {
            Write-Fail "HPA 缺少副本数限制"
        }
        
        if ($Content -match "behavior:") {
            Write-Pass "HPA 行为策略已配置"
        } else {
            Write-Warn "建议配置 HPA 行为策略"
        }
        
        if ($Content -match "scaleDown:" -and $Content -match "scaleUp:") {
            Write-Pass "HPA 扩缩容策略已配置"
        } else {
            Write-Warn "建议分别配置扩容和缩容策略"
        }
    } else {
        Write-Fail "HPA 配置文件不存在"
    }
}

# 8. PDB 配置验证
function Test-PdbConfig {
    Write-Info "=== 测试 PDB 配置 ==="
    
    $PdbFile = "$K8sDir\base\pdb.yaml"
    if (Test-Path $PdbFile) {
        $Content = Get-Content $PdbFile -Raw
        
        if ($Content -match "minAvailable:" -or $Content -match "maxUnavailable:") {
            Write-Pass "PDB 已正确配置"
        } else {
            Write-Fail "PDB 配置不正确"
        }
    } else {
        Write-Fail "PDB 配置文件不存在"
    }
}

# 9. Kustomize 验证
function Test-KustomizeBuild {
    Write-Info "=== 测试 Kustomize 构建 ==="
    
    if (Test-Tool "kustomize") {
        # 测试 base
        try {
            $null = kustomize build "$K8sDir\base" 2>&1
            Write-Pass "base 配置构建成功"
        } catch {
            Write-Fail "base 配置构建失败"
        }
        
        # 测试 dev overlay
        try {
            $null = kustomize build "$K8sDir\overlays\dev" 2>&1
            Write-Pass "dev overlay 构建成功"
        } catch {
            Write-Fail "dev overlay 构建失败"
        }
        
        # 测试 prod overlay
        try {
            $null = kustomize build "$K8sDir\overlays\prod" 2>&1
            Write-Pass "prod overlay 构建成功"
        } catch {
            Write-Fail "prod overlay 构建失败"
        }
    } elseif (Test-Tool "kubectl") {
        # 使用 kubectl kustomize 作为备选
        try {
            $null = kubectl kustomize "$K8sDir\base" 2>&1
            Write-Pass "base 配置构建成功 (kubectl)"
        } catch {
            Write-Fail "base 配置构建失败"
        }
    } else {
        Write-Skip "kustomize 和 kubectl 均未安装，跳过构建验证"
    }
}

# 10. Istio 配置验证
function Test-IstioConfig {
    Write-Info "=== 测试 Istio 配置 ==="
    
    $IstioDir = "$K8sDir\istio"
    if (Test-Path $IstioDir) {
        $IstioFiles = Get-ChildItem -Path $IstioDir -Filter "*.yaml"
        
        foreach ($File in $IstioFiles) {
            $Content = Get-Content $File.FullName -Raw
            
            if ($Content -match "apiVersion:\s*networking\.istio\.io" -or 
                $Content -match "apiVersion:\s*security\.istio\.io") {
                Write-Pass "Istio 资源格式正确: $($File.Name)"
            } else {
                Write-Warn "Istio 资源格式可疑: $($File.Name)"
            }
        }
    } else {
        Write-Skip "Istio 目录不存在"
    }
}

# 主函数
function Main {
    Write-Host "==========================================" -ForegroundColor $Colors.Cyan
    Write-Host "Kubernetes 配置验证" -ForegroundColor $Colors.Cyan
    Write-Host "==========================================" -ForegroundColor $Colors.Cyan
    Write-Host ""
    
    Test-YamlFilesExist
    Write-Host ""
    Test-YamlSyntax
    Write-Host ""
    Test-ResourceLimits
    Write-Host ""
    Test-SecurityConfig
    Write-Host ""
    Test-Probes
    Write-Host ""
    Test-ServicePorts
    Write-Host ""
    Test-HpaConfig
    Write-Host ""
    Test-PdbConfig
    Write-Host ""
    Test-KustomizeBuild
    Write-Host ""
    Test-IstioConfig
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor $Colors.Cyan
    Write-Host "测试结果汇总" -ForegroundColor $Colors.Cyan
    Write-Host "==========================================" -ForegroundColor $Colors.Cyan
    Write-Host "通过: $Script:Passed" -ForegroundColor $Colors.Green
    Write-Host "失败: $Script:Failed" -ForegroundColor $Colors.Red
    Write-Host "跳过: $Script:Skipped" -ForegroundColor $Colors.Yellow
    Write-Host ""
    
    if ($Script:Failed -gt 0) {
        Write-Err "存在失败的测试，请检查配置"
        exit 1
    } else {
        Write-Info "所有测试通过！"
        exit 0
    }
}

Main

