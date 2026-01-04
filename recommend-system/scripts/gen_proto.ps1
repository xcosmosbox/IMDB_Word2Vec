# gen_proto.ps1 - Proto 代码生成脚本 (Windows PowerShell)
#
# 该脚本用于从 .proto 文件生成 Go 代码。
#
# 依赖安装：
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
#
# 使用方法：
#   .\scripts\gen_proto.ps1
#   .\scripts\gen_proto.ps1 -Command all
#   .\scripts\gen_proto.ps1 -Command user

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "recommend", "user", "item", "clean", "help")]
    [string]$Command = "all"
)

# 错误处理
$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# 目录配置
$ProtoDir = Join-Path $ProjectRoot "proto"
$OutDir = $ProjectRoot

Write-Info "Project root: $ProjectRoot"
Write-Info "Proto directory: $ProtoDir"
Write-Info "Output directory: $OutDir"

# 检查 protoc 是否安装
function Test-Protoc {
    try {
        $version = & protoc --version 2>&1
        Write-Info "protoc version: $version"
        return $true
    }
    catch {
        Write-Error "protoc is not installed. Please install Protocol Buffers compiler."
        Write-Info "Installation guide:"
        Write-Info "  - Windows: choco install protoc"
        Write-Info "  - Or download from: https://github.com/protocolbuffers/protobuf/releases"
        return $false
    }
}

# 检查 Go 插件是否安装
function Test-GoPlugins {
    $goPath = & go env GOPATH
    $binPath = Join-Path $goPath "bin"
    
    $genGo = Join-Path $binPath "protoc-gen-go.exe"
    $genGoGrpc = Join-Path $binPath "protoc-gen-go-grpc.exe"
    
    if (-not (Test-Path $genGo)) {
        Write-Warning "protoc-gen-go not found. Installing..."
        & go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
    }
    
    if (-not (Test-Path $genGoGrpc)) {
        Write-Warning "protoc-gen-go-grpc not found. Installing..."
        & go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
    }
    
    # 确保 GOPATH/bin 在 PATH 中
    if ($env:PATH -notlike "*$binPath*") {
        $env:PATH = "$binPath;$env:PATH"
    }
    
    Write-Info "Go plugins ready"
}

# 确保输出目录存在
function New-OutputDirs {
    $dirs = @(
        "proto\recommend\v1",
        "proto\user\v1",
        "proto\item\v1"
    )
    
    foreach ($dir in $dirs) {
        $fullPath = Join-Path $OutDir $dir
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }
}

# 生成推荐服务 Proto
function Invoke-GenerateRecommendProto {
    Write-Info "Generating recommend service proto..."
    
    $protoFile = Join-Path $ProtoDir "recommend\v1\recommend.proto"
    
    & protoc `
        --proto_path="$ProtoDir" `
        --go_out="$OutDir" `
        --go_opt=paths=source_relative `
        --go-grpc_out="$OutDir" `
        --go-grpc_opt=paths=source_relative `
        "$protoFile"
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate recommend proto"
    }
    
    Write-Info "Recommend service proto generated"
}

# 生成用户服务 Proto
function Invoke-GenerateUserProto {
    Write-Info "Generating user service proto..."
    
    $protoFile = Join-Path $ProtoDir "user\v1\user.proto"
    
    & protoc `
        --proto_path="$ProtoDir" `
        --go_out="$OutDir" `
        --go_opt=paths=source_relative `
        --go-grpc_out="$OutDir" `
        --go-grpc_opt=paths=source_relative `
        "$protoFile"
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate user proto"
    }
    
    Write-Info "User service proto generated"
}

# 生成物品服务 Proto
function Invoke-GenerateItemProto {
    Write-Info "Generating item service proto..."
    
    $protoFile = Join-Path $ProtoDir "item\v1\item.proto"
    
    & protoc `
        --proto_path="$ProtoDir" `
        --go_out="$OutDir" `
        --go_opt=paths=source_relative `
        --go-grpc_out="$OutDir" `
        --go-grpc_opt=paths=source_relative `
        "$protoFile"
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate item proto"
    }
    
    Write-Info "Item service proto generated"
}

# 生成所有 Proto
function Invoke-GenerateAll {
    Invoke-GenerateRecommendProto
    Invoke-GenerateUserProto
    Invoke-GenerateItemProto
}

# 清理生成的文件
function Invoke-Clean {
    Write-Info "Cleaning generated files..."
    
    $protoOutDir = Join-Path $OutDir "proto"
    
    Get-ChildItem -Path $protoOutDir -Recurse -Filter "*.pb.go" | Remove-Item -Force
    Get-ChildItem -Path $protoOutDir -Recurse -Filter "*_grpc.pb.go" | Remove-Item -Force
    
    Write-Info "Clean completed"
}

# 显示帮助
function Show-Help {
    Write-Host @"
Usage: .\gen_proto.ps1 [-Command <command>]

Commands:
  all       Generate all proto files (default)
  recommend Generate recommend service proto
  user      Generate user service proto
  item      Generate item service proto
  clean     Clean generated files
  help      Show this help message

Examples:
  .\gen_proto.ps1                    # Generate all proto files
  .\gen_proto.ps1 -Command all       # Generate all proto files
  .\gen_proto.ps1 -Command user      # Generate only user service proto
  .\gen_proto.ps1 -Command clean     # Clean all generated files
"@
}

# 主函数
function Main {
    Write-Info "Starting proto generation..."
    
    if (-not (Test-Protoc)) {
        exit 1
    }
    
    Test-GoPlugins
    New-OutputDirs
    
    switch ($Command) {
        "all" {
            Invoke-GenerateAll
        }
        "recommend" {
            Invoke-GenerateRecommendProto
        }
        "user" {
            Invoke-GenerateUserProto
        }
        "item" {
            Invoke-GenerateItemProto
        }
        "clean" {
            Invoke-Clean
        }
        "help" {
            Show-Help
            exit 0
        }
    }
    
    Write-Info "Proto generation completed successfully!"
}

# 运行主函数
Main

