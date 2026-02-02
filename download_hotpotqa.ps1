param(
    [string]$OutDir = "data/hotpotqa",
    [switch]$Force
)

$Url = "https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
$OutPath = Join-Path $OutDir "hotpot_dev_distractor_v1.json"

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

if ((Test-Path $OutPath) -and -not $Force) {
    Write-Host "File already exists: $OutPath"
    Write-Host "Use -Force to re-download."
    exit 0
}

Write-Host "Downloading HotpotQA dev distractor set..."
Invoke-WebRequest -Uri $Url -OutFile $OutPath

Write-Host "Saved to: $OutPath"
