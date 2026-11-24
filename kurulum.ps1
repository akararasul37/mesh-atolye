param([switch]$Geometri)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_NO_CACHE_DIR = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program basarisiz oldu (cikis kodu: $LASTEXITCODE)."
    }
}

Start-Transcript -Path (Join-Path $PSScriptRoot 'kurulum.log') -Append | Out-Null
try {
    Invoke-Checked 'nvidia-smi.exe' @('--query-gpu=name,driver_version,memory.total,memory.free','--format=csv')
    $runtime = Join-Path $PSScriptRoot 'runtime'
    $null = New-Item -ItemType Directory -Path $runtime -Force
    $python = Join-Path $runtime 'python\python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        $archive = Join-Path $runtime 'python.tar.gz'
        $url = 'https://github.com/astral-sh/python-build-standalone/releases/download/20251031/cpython-3.11.14%2B20251031-x86_64-pc-windows-msvc-install_only.tar.gz'
        Invoke-Checked 'curl.exe' @('-fSL','--retry','3','--connect-timeout','20','-o',$archive,$url)
        $digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
        if ($digest -ne '5223b83ed9e2aa5e9e17d2ebcf767956e998876339b9cde1980a47e9d4655fb6') {
            throw 'Python arsivinin SHA-256 degeri uyusmuyor.'
        }
        Invoke-Checked 'tar.exe' @('-xzf',$archive,'-C',$runtime)
        Remove-Item -LiteralPath $archive
    }
    $venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    Invoke-Checked $python @('-m','venv',(Join-Path $PSScriptRoot 'venv'))
    Invoke-Checked $venvPython @('--version')
    Invoke-Checked $venvPython @('-m','pip','install','torch==2.7.1','torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu128')
    Invoke-Checked $venvPython @('-m','pip','install','-r','requirements.txt')
    $installArgs = @('-m','mesh_atolye.install')
    if ($Geometri) { $installArgs += '--geometry-only' }
    Invoke-Checked $venvPython $installArgs
    Write-Host 'Kurulum tamamlandi. baslat.bat ile uygulamayi acin.'
} catch {
    Write-Host $_
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
