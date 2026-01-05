# =============================================================================
# 性能测试运行脚本 (Windows PowerShell)
#
# 使用方法:
#   .\run-tests.ps1 -TestType baseline           # 运行基线测试
#   .\run-tests.ps1 -TestType stress             # 运行压力测试
#   .\run-tests.ps1 -TestType spike              # 运行峰值测试
#   .\run-tests.ps1 -TestType all                # 运行所有测试
#   .\run-tests.ps1 -Tool locust -TestType baseline  # 使用 Locust
#   $env:TEST_ENV="dev"; .\run-tests.ps1 baseline    # 指定环境
# =============================================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet("baseline", "stress", "spike", "all", "help")]
    [string]$TestType = "help",
    
    [Parameter()]
    [ValidateSet("k6", "locust")]
    [string]$Tool = "k6",
    
    [Parameter()]
    [string]$BaseUrl = $env:BASE_URL,
    
    [Parameter()]
    [string]$TestEnv = $env:TEST_ENV,
    
    [Parameter()]
    [string]$ApiKey = $env:API_KEY
)

# ============================================================================
# 配置
# ============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$K6Dir = Join-Path $ProjectRoot "load\k6"
$LocustDir = Join-Path $ProjectRoot "load\locust"
$ResultsDir = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { Join-Path $ProjectRoot "results" }
$ReportsDir = Join-Path $ProjectRoot "reports"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 默认值
if (-not $BaseUrl) { $BaseUrl = "http://localhost:8080" }
if (-not $TestEnv) { $TestEnv = "local" }
if (-not $ApiKey) { $ApiKey = "test-api-key" }

# ============================================================================
# 日志函数
# ============================================================================

function Write-LogInfo {
    param([string]$Message)
    Write-Host "[INFO] $(Get-Date -Format 'HH:mm:ss') $Message" -ForegroundColor Green
}

function Write-LogWarn {
    param([string]$Message)
    Write-Host "[WARN] $(Get-Date -Format 'HH:mm:ss') $Message" -ForegroundColor Yellow
}

function Write-LogError {
    param([string]$Message)
    Write-Host "[ERROR] $(Get-Date -Format 'HH:mm:ss') $Message" -ForegroundColor Red
}

function Write-LogStep {
    param([string]$Message)
    Write-Host "[STEP] $(Get-Date -Format 'HH:mm:ss') $Message" -ForegroundColor Cyan
}

function Write-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║           生成式推荐系统 - 性能测试套件                           ║" -ForegroundColor Blue
    Write-Host "║           Generative Recommendation System - Performance Test    ║" -ForegroundColor Blue
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
}

# ============================================================================
# 检查依赖
# ============================================================================

function Test-K6 {
    try {
        $version = k6 version 2>&1
        Write-LogInfo "K6 版本: $version"
        return $true
    }
    catch {
        Write-LogError "K6 未安装。请先安装 K6:"
        Write-Host "  choco install k6"
        Write-Host "  或者访问: https://k6.io/docs/getting-started/installation/"
        return $false
    }
}

function Test-Locust {
    try {
        $version = locust --version 2>&1
        Write-LogInfo "Locust 版本: $version"
        return $true
    }
    catch {
        Write-LogError "Locust 未安装。请先安装 Locust:"
        Write-Host "  pip install locust"
        return $false
    }
}

function Test-Python {
    try {
        $version = python --version 2>&1
        Write-LogInfo "Python 版本: $version"
        return $true
    }
    catch {
        Write-LogError "Python 未安装。"
        return $false
    }
}

# ============================================================================
# 健康检查
# ============================================================================

function Test-ServiceHealth {
    $url = "$BaseUrl/health"
    Write-LogStep "检查目标服务健康状态: $url"
    
    $maxRetries = 3
    $retry = 0
    
    while ($retry -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-LogInfo "✓ 目标服务健康"
                return $true
            }
        }
        catch {
            $retry++
            Write-LogWarn "健康检查失败，重试 $retry/$maxRetries..."
            Start-Sleep -Seconds 2
        }
    }
    
    Write-LogError "目标服务不可用: $url"
    return $false
}

# ============================================================================
# 创建目录
# ============================================================================

function Initialize-Directories {
    $directories = @(
        $ResultsDir,
        (Join-Path $ResultsDir "k6"),
        (Join-Path $ResultsDir "locust"),
        (Join-Path $ResultsDir "reports")
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-LogInfo "结果目录: $ResultsDir"
}

# ============================================================================
# K6 测试运行器
# ============================================================================

function Invoke-K6Baseline {
    Write-LogStep "运行 K6 基线测试..."
    
    $outputFile = Join-Path $ResultsDir "k6\baseline_$Timestamp.json"
    $logFile = Join-Path $ResultsDir "k6\baseline_$Timestamp.log"
    
    $env:BASE_URL = $BaseUrl
    $env:TEST_ENV = $TestEnv
    $env:API_KEY = $ApiKey
    
    $scenarioFile = Join-Path $K6Dir "scenarios\baseline.js"
    
    k6 run --out "json=$outputFile" $scenarioFile 2>&1 | Tee-Object -FilePath $logFile
    
    Write-LogInfo "基线测试完成"
    Write-LogInfo "结果文件: $outputFile"
}

function Invoke-K6Stress {
    Write-LogStep "运行 K6 压力测试..."
    
    $outputFile = Join-Path $ResultsDir "k6\stress_$Timestamp.json"
    $logFile = Join-Path $ResultsDir "k6\stress_$Timestamp.log"
    
    $env:BASE_URL = $BaseUrl
    $env:TEST_ENV = $TestEnv
    $env:API_KEY = $ApiKey
    
    $scenarioFile = Join-Path $K6Dir "scenarios\stress.js"
    
    k6 run --out "json=$outputFile" $scenarioFile 2>&1 | Tee-Object -FilePath $logFile
    
    Write-LogInfo "压力测试完成"
    Write-LogInfo "结果文件: $outputFile"
}

function Invoke-K6Spike {
    Write-LogStep "运行 K6 峰值测试..."
    
    $outputFile = Join-Path $ResultsDir "k6\spike_$Timestamp.json"
    $logFile = Join-Path $ResultsDir "k6\spike_$Timestamp.log"
    
    $env:BASE_URL = $BaseUrl
    $env:TEST_ENV = $TestEnv
    $env:API_KEY = $ApiKey
    
    $scenarioFile = Join-Path $K6Dir "scenarios\spike.js"
    
    k6 run --out "json=$outputFile" $scenarioFile 2>&1 | Tee-Object -FilePath $logFile
    
    Write-LogInfo "峰值测试完成"
    Write-LogInfo "结果文件: $outputFile"
}

# ============================================================================
# Locust 测试运行器
# ============================================================================

function Invoke-LocustBaseline {
    Write-LogStep "运行 Locust 基线测试..."
    
    $outputFile = Join-Path $ResultsDir "locust\baseline_$Timestamp"
    $logFile = "$outputFile.log"
    
    Push-Location $LocustDir
    
    try {
        $env:TEST_ENV = $TestEnv
        $env:API_KEY = $ApiKey
        
        locust -f locustfile.py --host $BaseUrl --headless -u 100 -r 10 -t 5m `
            --html "$outputFile.html" --csv $outputFile 2>&1 | Tee-Object -FilePath $logFile
    }
    finally {
        Pop-Location
    }
    
    Write-LogInfo "Locust 基线测试完成"
    Write-LogInfo "结果文件: $outputFile.html"
}

function Invoke-LocustStress {
    Write-LogStep "运行 Locust 压力测试..."
    
    $outputFile = Join-Path $ResultsDir "locust\stress_$Timestamp"
    $logFile = "$outputFile.log"
    
    Push-Location $LocustDir
    
    try {
        $env:TEST_ENV = $TestEnv
        $env:API_KEY = $ApiKey
        
        locust -f locustfile.py --host $BaseUrl --headless -u 1000 -r 50 -t 10m `
            --html "$outputFile.html" --csv $outputFile 2>&1 | Tee-Object -FilePath $logFile
    }
    finally {
        Pop-Location
    }
    
    Write-LogInfo "Locust 压力测试完成"
    Write-LogInfo "结果文件: $outputFile.html"
}

function Invoke-LocustSpike {
    Write-LogStep "运行 Locust 峰值测试..."
    
    $outputFile = Join-Path $ResultsDir "locust\spike_$Timestamp"
    $logFile = "$outputFile.log"
    
    Push-Location $LocustDir
    
    try {
        $env:TEST_ENV = $TestEnv
        $env:API_KEY = $ApiKey
        
        locust -f locustfile.py --host $BaseUrl --headless -u 5000 -r 500 -t 2m `
            --html "$outputFile.html" --csv $outputFile 2>&1 | Tee-Object -FilePath $logFile
    }
    finally {
        Pop-Location
    }
    
    Write-LogInfo "Locust 峰值测试完成"
    Write-LogInfo "结果文件: $outputFile.html"
}

# ============================================================================
# 报告生成
# ============================================================================

function New-Reports {
    Write-LogStep "生成测试报告..."
    
    if (-not (Test-Python)) {
        Write-LogWarn "跳过报告生成 (Python 不可用)"
        return
    }
    
    $jsonFiles = Get-ChildItem -Path (Join-Path $ResultsDir "k6") -Filter "*.json" -ErrorAction SilentlyContinue
    
    foreach ($jsonFile in $jsonFiles) {
        $htmlFile = $jsonFile.FullName -replace '\.json$', '.html'
        Write-LogInfo "生成报告: $htmlFile"
        
        try {
            $reportScript = Join-Path $ReportsDir "generate-report.py"
            python $reportScript $jsonFile.FullName -o $htmlFile
        }
        catch {
            Write-LogWarn "报告生成失败: $($jsonFile.Name)"
        }
    }
    
    Write-LogInfo "报告生成完成"
}

# ============================================================================
# 显示帮助
# ============================================================================

function Show-Help {
    Write-Host "使用方法: .\run-tests.ps1 -TestType <type> [-Tool <tool>]"
    Write-Host ""
    Write-Host "参数:"
    Write-Host "  -TestType    测试类型: baseline, stress, spike, all"
    Write-Host "  -Tool        测试工具: k6 (默认), locust"
    Write-Host "  -BaseUrl     目标服务 URL (默认: http://localhost:8080)"
    Write-Host "  -TestEnv     测试环境: local, dev, prod (默认: local)"
    Write-Host "  -ApiKey      API 密钥 (默认: test-api-key)"
    Write-Host ""
    Write-Host "测试类型说明:"
    Write-Host "  baseline     基线测试 (100 RPS, 5分钟)"
    Write-Host "  stress       压力测试 (逐步增加到 1000 RPS, 10分钟)"
    Write-Host "  spike        峰值测试 (突发 5000 RPS, 2分钟)"
    Write-Host "  all          运行所有测试"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\run-tests.ps1 -TestType baseline"
    Write-Host "  .\run-tests.ps1 -TestType stress -Tool locust"
    Write-Host "  .\run-tests.ps1 -TestType all -BaseUrl http://api.example.com"
    Write-Host ""
}

# ============================================================================
# 主函数
# ============================================================================

function Main {
    Write-Banner
    
    if ($TestType -eq "help") {
        Show-Help
        return
    }
    
    # 显示配置
    Write-LogInfo "配置信息:"
    Write-LogInfo "  工具: $Tool"
    Write-LogInfo "  测试类型: $TestType"
    Write-LogInfo "  目标 URL: $BaseUrl"
    Write-LogInfo "  环境: $TestEnv"
    Write-Host ""
    
    # 检查依赖
    if ($Tool -eq "k6") {
        if (-not (Test-K6)) { exit 1 }
    }
    elseif ($Tool -eq "locust") {
        if (-not (Test-Locust)) { exit 1 }
    }
    
    # 设置目录
    Initialize-Directories
    
    # 健康检查
    if (-not (Test-ServiceHealth)) { exit 1 }
    
    # 运行测试
    switch ($TestType) {
        "baseline" {
            if ($Tool -eq "k6") { Invoke-K6Baseline }
            else { Invoke-LocustBaseline }
        }
        "stress" {
            if ($Tool -eq "k6") { Invoke-K6Stress }
            else { Invoke-LocustStress }
        }
        "spike" {
            if ($Tool -eq "k6") { Invoke-K6Spike }
            else { Invoke-LocustSpike }
        }
        "all" {
            if ($Tool -eq "k6") {
                Invoke-K6Baseline
                Invoke-K6Stress
                Invoke-K6Spike
            }
            else {
                Invoke-LocustBaseline
                Invoke-LocustStress
                Invoke-LocustSpike
            }
        }
    }
    
    # 生成报告
    New-Reports
    
    Write-Host ""
    Write-LogInfo "=========================================="
    Write-LogInfo "  所有测试完成!"
    Write-LogInfo "  结果目录: $ResultsDir"
    Write-LogInfo "=========================================="
}

# 运行主函数
Main

