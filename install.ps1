# HIVE Limited -- one-command installer for Windows (native, no WSL needed).
#
#   irm https://raw.githubusercontent.com/Questeria/HIVE-Limited/main/install.ps1 | iex
#
# Written for someone who has never used a terminal. Every failure explains, in plain
# language, what went wrong and exactly what to do next. Safe to run more than once:
# anything already installed is detected and skipped. Everything lands in ONE folder
# (%USERPROFILE%\hive-limited) plus the model cache (%USERPROFILE%\.hive-limited).
#
# ASCII-only on purpose: this script is piped through irm|iex, and fancy characters
# are exactly what gets mangled when encodings disagree.

$ErrorActionPreference = "Stop"

$InstallDir = if ($env:HIVE_LIMITED_DIR) { $env:HIVE_LIMITED_DIR } else { Join-Path $env:USERPROFILE "hive-limited" }
$Repo       = if ($env:HIVE_LIMITED_REPO) { $env:HIVE_LIMITED_REPO } else { "Questeria/HIVE-Limited" }
$Branch     = if ($env:HIVE_LIMITED_BRANCH) { $env:HIVE_LIMITED_BRANCH } else { "main" }
$ModelRepo  = if ($env:MODEL_REPO) { $env:MODEL_REPO } else { "Qwen/Qwen3-1.7B" }
$ModelsDir  = if ($env:HIVE_LIMITED_MODELS) { $env:HIVE_LIMITED_MODELS } else { Join-Path $env:USERPROFILE ".hive-limited\models" }
$ModelDir   = Join-Path $ModelsDir ($ModelRepo.Split("/")[-1])

function Step($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Green }
function Ok($m)   { Write-Host "    [OK] $m" -ForegroundColor Gray }
function Info($m) { Write-Host "    $m" -ForegroundColor Gray }
function Warn($m) { Write-Host "    [!] $m" -ForegroundColor Yellow }
function Die($title, $lines) {
  Write-Host ""
  Write-Host "+- Stopped ---------------------------------------------" -ForegroundColor Red
  Write-Host "|  $title" -ForegroundColor Red
  Write-Host "|" -ForegroundColor Red
  foreach ($l in $lines) { Write-Host "|  $l" -ForegroundColor Red }
  Write-Host "+-------------------------------------------------------" -ForegroundColor Red
  Write-Host ""
  exit 1
}

Write-Host ""
Write-Host "  HIVE Limited" -ForegroundColor White
Write-Host "  A small, fast AI text engine you can run on your own computer."
Write-Host "  This installer sets everything up for you. It takes about 10 minutes,"
Write-Host "  most of which is downloading. You can leave it running." -ForegroundColor Gray

# ---- 1. graphics card -------------------------------------------------------
Step "Checking your graphics card"

$nvcuda = Join-Path $env:SystemRoot "System32\nvcuda.dll"
if (-not (Test-Path $nvcuda)) {
  Die "No NVIDIA graphics driver found." @(
    "HIVE Limited needs an NVIDIA graphics card and its driver.",
    "It does NOT need the big 'CUDA Toolkit' -- just the ordinary driver.",
    "",
    "Install the normal NVIDIA driver from  nvidia.com/drivers ,",
    "restart Windows, then run this installer again.",
    "",
    "If you do not have an NVIDIA card, this software cannot run on this computer.")
}
Ok "NVIDIA driver found"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  try { $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        if ($gpu) { Ok "Graphics card: $gpu" } } catch {}
}

# ---- 2. python --------------------------------------------------------------
Step "Checking Python"

# Prefer the py launcher (ships with python.org installs); fall back to python on PATH.
# The Microsoft Store stub named python.exe opens the Store instead of running -- the
# version probe below returns nothing for it, so it fails the check rather than lying.
$Py = $null
foreach ($cand in @(@("py", "-3"), @("python", ""))) {
  $exe = Get-Command $cand[0] -ErrorAction SilentlyContinue
  if (-not $exe) { continue }
  try {
    $args = @(); if ($cand[1]) { $args += $cand[1] }
    $v = & $cand[0] @args -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
    if ($v -match "^3\.(9|1[0-9])$") { $Py = @($cand[0]) + $args; break }
  } catch {}
}
if (-not $Py) {
  $winget = Get-Command winget -ErrorAction SilentlyContinue
  if ($winget) {
    Warn "Python 3.9+ is not installed. It can be installed for you now."
    $r = Read-Host "    Install Python 3.12 from the official source with winget? [Y/n]"
    if ($r -eq "" -or $r -match "^[Yy]") {
      & winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
      Write-Host ""
      Info "Python installed. Close this window, open a NEW PowerShell window,"
      Info "and run the installer command again -- the new window picks up Python."
      exit 0
    }
  }
  Die "Python 3 is not installed." @(
    "Python is the language HIVE Limited is written in.",
    "",
    "Install it from  python.org/downloads  (choose 'Add python.exe to PATH'),",
    "then open a NEW PowerShell window and run this installer again.")
}
$pyver = & $Py[0] $Py[1..($Py.Count-1)] -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
Ok "Python $pyver"

function PyRun { & $Py[0] $Py[1..($Py.Count-1)] @args }

try { PyRun -c "import numpy" 2>$null | Out-Null; $hasNumpy = ($LASTEXITCODE -eq 0) } catch { $hasNumpy = $false }
if (-not $hasNumpy) {
  Info "Installing numpy (the one library this needs)..."
  PyRun -m pip install --user --quiet numpy
  if ($LASTEXITCODE -ne 0) {
    Die "Could not install numpy." @(
      "numpy is the only extra library HIVE Limited needs.",
      "Try it yourself in this same window:",
      "    $($Py -join ' ') -m pip install --user numpy",
      "Then run this installer again.")
  }
}
Ok "numpy ready"

# ---- 3. disk space ----------------------------------------------------------
Step "Checking free space"
try {
  $drive = (Get-Item $env:USERPROFILE).PSDrive
  $freeGB = [math]::Floor($drive.Free / 1GB)
  if ($freeGB -lt 8) {
    Die "Not enough free space: about $freeGB GB available on drive $($drive.Name):" @(
      "The AI model alone is roughly 4 GB, so you need about 8 GB free.",
      "Free some space and run this installer again.")
  }
  Ok "$freeGB GB free"
} catch { Warn "Could not check free space -- carrying on" }

# ---- 4. get the software ----------------------------------------------------
Step "Downloading HIVE Limited"

if (Test-Path (Join-Path $InstallDir "serve.py")) {
  Ok "Already downloaded -- using $InstallDir"
} else {
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  $zipUrl = "https://codeload.github.com/$Repo/zip/refs/heads/$Branch"
  $zip = Join-Path $env:TEMP "hive-limited.zip"
  try {
    Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing
  } catch {
    $code = try { [int]$_.Exception.Response.StatusCode } catch { 0 }
    if ($code -eq 404) {
      Die "Could not find the download." @(
        "The server replied 'not found' for:  $Repo (branch $Branch)",
        "",
        "This usually means the project is not published yet, or the link you",
        "were given is out of date.",
        "Please check for an updated link, or email  ajdemarco10@gmail.com")
    }
    Die "Could not reach the internet." @(
      "The download could not connect.",
      "- Check you are online.",
      "- If you are on a work or school network, it may be blocking GitHub.",
      "Then run this installer again.")
  }
  $tmp = Join-Path $env:TEMP ("hive-limited-unpack-" + [IO.Path]::GetRandomFileName())
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
  $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
  Get-ChildItem $inner.FullName -Force | Move-Item -Destination $InstallDir -Force
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $zip -Force -ErrorAction SilentlyContinue
  Ok "Downloaded to $InstallDir"
}

# ---- 5. the model -----------------------------------------------------------
Step "Downloading the AI model (about 4 GB -- this is the slow part)"

if (Test-Path (Join-Path $ModelDir "config.json")) {
  Ok "Model already downloaded"
} else {
  New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
  # The hf CLI lands in the per-user Scripts folder, which is often not on PATH.
  # Compute that folder from Python itself instead of guessing.
  PyRun -m pip install --user --quiet "huggingface_hub[cli]"
  if ($LASTEXITCODE -ne 0) {
    Die "Could not install the model downloader." @(
      "Try it yourself in this same window:",
      "    $($Py -join ' ') -m pip install --user huggingface_hub[cli]",
      "Then run this installer again.")
  }
  $scripts = PyRun -c "import sysconfig;print(sysconfig.get_path('scripts','nt_user'))" 2>$null
  $hf = $null
  foreach ($cand in @((Join-Path $scripts "hf.exe"), "hf", (Join-Path $scripts "huggingface-cli.exe"), "huggingface-cli")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) { $hf = $cand; break }
  }
  if (-not $hf) {
    Die "The model downloader installed but could not be found." @(
      "Close this window, open a NEW PowerShell window, and run the installer",
      "again -- the new window picks up the freshly installed tool.")
  }
  Info "Fetching $ModelRepo -- leave this running, it can take several minutes."
  & $hf download $ModelRepo --local-dir $ModelDir
  if ($LASTEXITCODE -ne 0) {
    Die "Model download failed." @(
      "This is usually a network problem. Run the installer again to resume --",
      "already-downloaded pieces are kept, so it picks up where it stopped.")
  }
  if (-not (Test-Path (Join-Path $ModelDir "config.json"))) {
    Die "The download finished but the model is incomplete." @(
      "Expected a config.json in:  $ModelDir",
      "Run the installer again -- it resumes rather than starting over.")
  }
  Ok "Model ready"
}

# ---- 6. a launcher they can re-use ------------------------------------------
Step "Setting up"

$startCmd = @"
@echo off
rem Start HIVE Limited. Created by the installer -- run it again any time.
cd /d "%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=$ModelDir"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"
echo.
echo   Starting HIVE Limited...
echo   The first start takes a minute while the AI model loads -- your web
echo   browser opens BY ITSELF the moment the engine is ready.
echo   If it does not, open this address yourself:  http://localhost:%PORT%
echo.
echo   To stop the engine, press Ctrl and C together, or close this window.
echo.
set "HIVE_OPEN_BROWSER=1"
$($Py -join ' ') serve.py "%MODEL%" %PORT%
"@
Set-Content -Path (Join-Path $InstallDir "start.bat") -Value $startCmd -Encoding ASCII
Ok "Created $InstallDir\start.bat"

Write-Host ""
Write-Host "+- Done ------------------------------------------------" -ForegroundColor Green
Write-Host "|  HIVE Limited is installed." -ForegroundColor Green
Write-Host "|" -ForegroundColor Green
Write-Host "|  To start it, double-click this file (or run it here):" -ForegroundColor Green
Write-Host "|"  -ForegroundColor Green
Write-Host "|      $InstallDir\start.bat" -ForegroundColor White
Write-Host "|" -ForegroundColor Green
Write-Host "|  Your browser opens on its own. Type a question, press" -ForegroundColor Green
Write-Host "|  LAUNCH, and watch it answer. To race another AI engine," -ForegroundColor Green
Write-Host "|  click ADD AN ENGINE on that page." -ForegroundColor Green
Write-Host "+-------------------------------------------------------" -ForegroundColor Green
Write-Host ""

$r = Read-Host "Start it now? [Y/n]"
if ($r -eq "" -or $r -match "^[Yy]") {
  & (Join-Path $InstallDir "start.bat")
} else {
  Write-Host "No problem -- run $InstallDir\start.bat whenever you are ready."
  Write-Host ""
}
