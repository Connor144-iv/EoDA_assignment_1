Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Test-Administrator {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Convert-WindowsPathToWsl {
    param([string]$WindowsPath)

    $resolved = (Resolve-Path $WindowsPath).Path
    if ($resolved -notmatch '^(?<drive>[A-Za-z]):\\(?<rest>.*)$') {
        throw "Cannot convert Windows path to WSL path: $resolved"
    }

    $drive = $Matches.drive.ToLowerInvariant()
    $rest = $Matches.rest -replace '\\', '/'
    return "/mnt/$drive/$rest"
}

function Get-WslDistroNames {
    $distros = cmd /c "wsl.exe --list --quiet 2>NUL"
    if ($LASTEXITCODE -ne 0) {
        return @()
    }

    return @($distros | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Test-WslInstalled {
    $statusOutput = cmd /c "wsl.exe --status 2>&1" | Out-String
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    return ($statusOutput -notmatch 'is not installed')
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$repoWindowsPath = $repoRoot.Path
$repoWslPath = Convert-WindowsPathToWsl -WindowsPath $repoWindowsPath
$distroName = "Ubuntu-24.04"

Write-Step "Checking WSL installation state"
$wslInstalled = Test-WslInstalled

if (-not $wslInstalled) {
    if (-not (Test-Administrator)) {
        Write-Warn "WSL is not installed and this shell is not elevated."
        Write-Host "Open PowerShell as Administrator and rerun:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
        exit 1
    }

    Write-Step "Installing WSL2 with Ubuntu 24.04"
    & wsl --install -d $distroName

    Write-Warn "WSL installation has started. Reboot Windows, launch Ubuntu once to create your Linux user, then rerun this script."
    exit 0
}

Write-Step "Updating WSL kernel"
try {
    & wsl --update
}
catch {
    Write-Warn "WSL update could not be completed from this shell. Continuing with the existing installation."
}

$distros = Get-WslDistroNames
if ($distros -notcontains $distroName) {
    Write-Step "Installing Ubuntu 24.04 distro"
    & wsl --install -d $distroName
    Write-Warn "Ubuntu 24.04 has been installed. Launch it once to create your Linux user, then rerun this script."
    exit 0
}

Write-Step "Checking Ubuntu user initialization"
$linuxUser = ""
try {
    $linuxUser = (& wsl -d $distroName -- bash -lc 'getent passwd 1000 | cut -d: -f1' 2>$null).Trim()
}
catch {
    $linuxUser = ""
}

if ([string]::IsNullOrWhiteSpace($linuxUser)) {
    Write-Warn "Ubuntu has not finished first-run setup yet."
    Write-Host "Run `wsl -d $distroName`, create your Linux username/password, then rerun this script."
    exit 1
}

Write-Step "Running the WSL bootstrap script"
$bootstrapCommand = "cd '$repoWslPath' && bash './scripts/setup_rapids_wsl.sh'"
& wsl -d $distroName -- bash -lc $bootstrapCommand

Write-Step "Setup finished"
Write-Host "Open the WSL copy of the repo with:"
Write-Host "  wsl -d $distroName -- bash -lc 'cd ~/projects/ml_lab/eng_of_data_analysis && code .'"
