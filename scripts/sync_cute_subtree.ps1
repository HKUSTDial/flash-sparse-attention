Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [switch]$Init,
    [string]$UpstreamRepo = "https://github.com/Dao-AILab/flash-attention.git",
    [string]$UpstreamPrefix = "flash_attn/cute",
    [string]$Prefix = "flash_sparse_attn/ops/cute",
    [string]$TempBranch = "sync/cute-upstream-temp",
    [string]$CacheDir = ".ref_repo/flash-attention",
    [switch]$SkipFetch,
    [switch]$KeepTempBranch
)

function Test-GitRemoteSpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return ($Value -match '^[A-Za-z][A-Za-z0-9+.-]*://') -or
        ($Value -match '^[^\s]+@[^\s:]+:[^\s]+$')
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$Repo
    )

    if ($Repo) {
        & git -C $Repo @Arguments
    }
    else {
        & git @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-GitOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$Repo
    )

    if ($Repo) {
        $output = & git -C $Repo @Arguments
    }
    else {
        $output = & git @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }

    return ($output | Out-String).Trim()
}

function Test-WorktreeClean {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [string]$Label
    )

    $status = Get-GitOutput -Repo $Repo -Arguments @("status", "--porcelain")
    if ($status) {
        throw "$Label has uncommitted changes. Commit or stash them before syncing.`n$status"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$cacheRepo = Join-Path $repoRoot $CacheDir
$upstreamRepoForSplit = $null

if (Test-GitRemoteSpec -Value $UpstreamRepo) {
    if (-not (Test-Path $cacheRepo)) {
        Write-Host "Cloning upstream repo into cache at $CacheDir ..."
        Invoke-Git -Arguments @("clone", "--origin", "origin", $UpstreamRepo, $cacheRepo)
    }

    $upstreamRepoForSplit = (Resolve-Path $cacheRepo).Path

    $currentOrigin = Get-GitOutput -Repo $upstreamRepoForSplit -Arguments @("remote", "get-url", "origin")
    if ($currentOrigin -ne $UpstreamRepo) {
        Write-Host "Updating cached upstream origin URL ..."
        Invoke-Git -Repo $upstreamRepoForSplit -Arguments @("remote", "set-url", "origin", $UpstreamRepo)
    }
}
else {
    $upstreamRepoForSplit = (Resolve-Path $UpstreamRepo).Path
}

$cutlassRepo = Join-Path $repoRoot "csrc/cutlass"
$targetPath = Join-Path $repoRoot $Prefix

Invoke-Git -Arguments @("rev-parse", "--show-toplevel") | Out-Null
Invoke-Git -Repo $upstreamRepoForSplit -Arguments @("rev-parse", "--show-toplevel") | Out-Null

Test-WorktreeClean -Repo $repoRoot -Label "Superproject"
if (Test-Path $cutlassRepo) {
    Test-WorktreeClean -Repo $cutlassRepo -Label "csrc/cutlass submodule"
}

if (-not $SkipFetch) {
    Write-Host "Fetching latest upstream changes from origin..."
    Invoke-Git -Repo $upstreamRepoForSplit -Arguments @("fetch", "origin")
}

Write-Host "Splitting upstream history for $UpstreamPrefix ..."
$splitCommit = Get-GitOutput -Repo $upstreamRepoForSplit -Arguments @("subtree", "split", "--prefix=$UpstreamPrefix", "HEAD")
Invoke-Git -Repo $upstreamRepoForSplit -Arguments @("branch", "-f", $TempBranch, $splitCommit)

try {
    if ($Init) {
        if (Test-Path $targetPath) {
            throw "$Prefix already exists. Remove -Init to do an update instead."
        }

        Write-Host "Adding subtree into $Prefix ..."
        Invoke-Git -Arguments @("subtree", "add", "--prefix=$Prefix", $upstreamRepoForSplit, $TempBranch)
    }
    else {
        if (-not (Test-Path $targetPath)) {
            throw "$Prefix does not exist yet. Run this script once with -Init first."
        }

        Write-Host "Pulling upstream updates into $Prefix ..."
        Invoke-Git -Arguments @("subtree", "pull", "--prefix=$Prefix", $upstreamRepoForSplit, $TempBranch)
    }
}
finally {
    if (-not $KeepTempBranch) {
        Invoke-Git -Repo $upstreamRepoForSplit -Arguments @("update-ref", "-d", "refs/heads/$TempBranch")
    }
}

Write-Host "Done."
Write-Host "Upstream source: $UpstreamRepo"
Write-Host "Upstream cache used for subtree split: $upstreamRepoForSplit"
Write-Host "Local edits inside $Prefix stay in this repo and future upstream changes can be merged by rerunning this script without -Init."
