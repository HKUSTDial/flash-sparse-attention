param(
    [switch]$Init,
    [string]$UpstreamRepo = "https://github.com/Dao-AILab/flash-attention.git",
    [string]$UpstreamPrefix = "flash_attn/cute",
    [string]$Prefix = "flash_sparse_attn/ops/cute",
    [string]$TempBranch = "sync/cute-upstream-temp",
    [string]$CacheDir = ".ref_repo/flash-attention",
    [switch]$SkipFetch,
    [switch]$KeepTempBranch,
    [switch]$NoTemporaryWorktree
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PrepareMergeCommitMessage = "Rewrite vendored CuTe namespace to flash_attn.cute before subtree merge"
$RewriteCommitMessage = "Rewrite vendored CuTe namespace to flash_sparse_attn.ops.cute"

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
        $output = & git -C $Repo @Arguments
    }
    else {
        $output = & git @Arguments
    }

    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Invoke-GitNoMergeEdit {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$Repo
    )

    $previousValue = $env:GIT_MERGE_AUTOEDIT
    $env:GIT_MERGE_AUTOEDIT = "no"

    try {
        Invoke-Git -Arguments $Arguments -Repo $Repo
    }
    finally {
        if ($null -eq $previousValue) {
            Remove-Item Env:GIT_MERGE_AUTOEDIT -ErrorAction SilentlyContinue
        }
        else {
            $env:GIT_MERGE_AUTOEDIT = $previousValue
        }
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

function Get-DirtyStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo
    )

    return Get-GitOutput -Repo $Repo -Arguments @("status", "--porcelain")
}

function Get-GitCommonDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo
    )

    $commonDir = (& git -C $Repo rev-parse --path-format=absolute --git-common-dir) 2>$null
    if ($LASTEXITCODE -eq 0 -and $commonDir) {
        return (($commonDir | Out-String).Trim())
    }

    $commonDir = Get-GitOutput -Repo $Repo -Arguments @("rev-parse", "--git-common-dir")
    if (-not $commonDir) {
        return $null
    }

    if (-not [System.IO.Path]::IsPathRooted($commonDir)) {
        return [System.IO.Path]::GetFullPath((Join-Path $Repo $commonDir))
    }

    return $commonDir
}

function Get-SubmoduleStatusLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$SubmodulePath
    )

    $output = (& git -C $Repo submodule status -- $SubmodulePath) 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        return $null
    }

    return (($output | Select-Object -First 1 | Out-String).Trim())
}

function Ensure-SubmoduleInitialized {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$SubmodulePath,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $statusLine = Get-SubmoduleStatusLine -Repo $Repo -SubmodulePath $SubmodulePath
    if (-not $statusLine) {
        return
    }

    if ($statusLine[0] -eq '-') {
        $commonDir = Get-GitCommonDir -Repo $Repo
        $referenceRepo = $null
        if ($commonDir) {
            $referenceRepo = Join-Path $commonDir "modules"
            foreach ($segment in $SubmodulePath -split '/') {
                $referenceRepo = Join-Path $referenceRepo $segment
            }
        }

        if ($referenceRepo -and (Test-Path $referenceRepo)) {
            Write-Host "Initializing $Label in $Repo from local git cache ..."
            Invoke-Git -Repo $Repo -Arguments @("submodule", "update", "--init", "--reference", $referenceRepo, "--", $SubmodulePath)
        }
        else {
            Write-Host "Initializing $Label in $Repo ..."
            Invoke-Git -Repo $Repo -Arguments @("submodule", "update", "--init", "--", $SubmodulePath)
        }
    }
}

function Test-IsGitRepo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo
    )

    & git -C $Repo rev-parse --show-toplevel *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-GitIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo
    )

    $currentName = (& git -C $Repo config --get user.name) 2>$null
    $currentEmail = (& git -C $Repo config --get user.email) 2>$null

    if ($currentName -and $currentEmail) {
        return
    }

    $fallbackName = Get-GitOutput -Repo $Repo -Arguments @("log", "-1", "--format=%an")
    $fallbackEmail = Get-GitOutput -Repo $Repo -Arguments @("log", "-1", "--format=%ae")

    if (-not $currentName) {
        Invoke-Git -Repo $Repo -Arguments @("config", "user.name", $fallbackName)
    }
    if (-not $currentEmail) {
        Invoke-Git -Repo $Repo -Arguments @("config", "user.email", $fallbackEmail)
    }
}

function Resolve-RemoteDefaultRef {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo
    )

    $remoteRef = (& git -C $Repo symbolic-ref --quiet --short refs/remotes/origin/HEAD) 2>$null
    if ($LASTEXITCODE -eq 0 -and $remoteRef) {
        return ($remoteRef | Out-String).Trim()
    }

    & git -C $Repo remote set-head origin --auto *> $null
    $remoteRef = (& git -C $Repo symbolic-ref --quiet --short refs/remotes/origin/HEAD) 2>$null
    if ($LASTEXITCODE -eq 0 -and $remoteRef) {
        return ($remoteRef | Out-String).Trim()
    }

    return "origin/main"
}

function Get-CommitSubject {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$Commit
    )

    return Get-GitOutput -Repo $Repo -Arguments @("log", "-1", "--format=%s", $Commit)
}

function Get-CommitParentCount {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$Commit
    )

    $revLine = Get-GitOutput -Repo $Repo -Arguments @("rev-list", "--parents", "-n", "1", $Commit)
    if (-not $revLine) {
        return 0
    }

    $fields = $revLine -split '\s+'
    return [Math]::Max(0, $fields.Count - 1)
}

function Get-PrefixSplitCommit {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$Prefix
    )

    return Get-GitOutput -Repo $Repo -Arguments @("subtree", "split", "--prefix=$Prefix", "HEAD")
}

function Test-CommitIsAncestor {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$OlderCommit,
        [Parameter(Mandatory = $true)]
        [string]$NewerCommit
    )

    & git -C $Repo merge-base --is-ancestor $OlderCommit $NewerCommit *> $null
    return $LASTEXITCODE -eq 0
}

function Assert-SyncContainsUpstream {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamSplitCommit,
        [Parameter(Mandatory = $true)]
        [string]$LocalSplitCommit,
        [string]$PreviousLocalSplitCommit
    )

    if (Test-CommitIsAncestor -Repo $Repo -OlderCommit $UpstreamSplitCommit -NewerCommit $LocalSplitCommit) {
        return
    }

    $message = @(
        "Subtree sync did not incorporate the upstream split commit.",
        "Upstream split commit: $UpstreamSplitCommit",
        "Local prefix split after sync: $LocalSplitCommit"
    )

    if ($PreviousLocalSplitCommit) {
        $message += "Local prefix split before sync: $PreviousLocalSplitCommit"
        if ($PreviousLocalSplitCommit -eq $LocalSplitCommit) {
            $message += "The local prefix split did not change even though upstream has newer CuTe commits."
        }
    }

    $message += "git subtree pull reported success, but the vendored prefix still does not contain the upstream split lineage."
    throw ($message -join [Environment]::NewLine)
}

function Get-SyncResultProperty {
    param(
        [Parameter(Mandatory = $false)]
        [object]$SyncResult,
        [Parameter(Mandatory = $true)]
        [string]$PropertyName
    )

    if ($null -eq $SyncResult) {
        return $null
    }

    if ($SyncResult -is [System.Array]) {
        foreach ($item in $SyncResult) {
            if ($null -ne $item -and $item.PSObject.Properties.Match($PropertyName).Count -gt 0) {
                return $item.$PropertyName
            }
        }

        return $null
    }

    if ($SyncResult.PSObject.Properties.Match($PropertyName).Count -gt 0) {
        return $SyncResult.$PropertyName
    }

    return $null
}

function Invoke-NamespaceRewrite {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RewriteScript,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [ValidateSet("local", "upstream")]
        [string]$Direction
    )

    $rewriteOutput = & python $RewriteScript --direction $Direction $TargetPath
    if ($rewriteOutput) {
        $rewriteOutput | ForEach-Object { Write-Host $_ }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "python $RewriteScript --direction $Direction $TargetPath failed with exit code $LASTEXITCODE"
    }
}

function Commit-PrefixIfChanged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repo,
        [Parameter(Mandatory = $true)]
        [string]$Prefix,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $prefixStatus = Get-GitOutput -Repo $Repo -Arguments @("status", "--porcelain", "--", $Prefix)
    if (-not $prefixStatus) {
        return $false
    }

    Ensure-GitIdentity -Repo $Repo
    Invoke-Git -Repo $Repo -Arguments @("add", "--", $Prefix)
    Invoke-Git -Repo $Repo -Arguments @("commit", "-m", $Message)
    return $true
}

function Invoke-CoreSync {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WorkRepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamRepoForSplit,
        [Parameter(Mandatory = $true)]
        [string]$Prefix,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamPrefix,
        [Parameter(Mandatory = $true)]
        [string]$TempBranch,
        [Parameter(Mandatory = $true)]
        [string]$RewriteScript,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamSplitRef,
        [switch]$Init,
        [switch]$SkipFetch,
        [switch]$KeepTempBranch
    )

    $cutlassRepo = Join-Path $WorkRepoRoot "csrc/cutlass"
    $cutlassSubmodulePath = "csrc/cutlass"
    $targetPath = Join-Path $WorkRepoRoot $Prefix
    $startHead = Get-GitOutput -Repo $WorkRepoRoot -Arguments @("rev-parse", "HEAD")
    $localSplitBefore = $null

    Get-GitOutput -Repo $WorkRepoRoot -Arguments @("rev-parse", "--show-toplevel") | Out-Null
    Get-GitOutput -Repo $UpstreamRepoForSplit -Arguments @("rev-parse", "--show-toplevel") | Out-Null

    Ensure-SubmoduleInitialized -Repo $WorkRepoRoot -SubmodulePath $cutlassSubmodulePath -Label "csrc/cutlass submodule"
    Test-WorktreeClean -Repo $WorkRepoRoot -Label "Superproject"
    if ((Test-Path $cutlassRepo) -and (Test-IsGitRepo -Repo $cutlassRepo)) {
        Test-WorktreeClean -Repo $cutlassRepo -Label "csrc/cutlass submodule"
    }

    if (-not $SkipFetch) {
        Write-Host "Fetching latest upstream changes from origin..."
        Invoke-Git -Repo $UpstreamRepoForSplit -Arguments @("fetch", "origin")
    }

    Write-Host "Splitting upstream history for $UpstreamPrefix from $UpstreamSplitRef ..."
    $splitCommit = Get-GitOutput -Repo $UpstreamRepoForSplit -Arguments @("subtree", "split", "--prefix=$UpstreamPrefix", $UpstreamSplitRef)
    Invoke-Git -Repo $UpstreamRepoForSplit -Arguments @("branch", "-f", $TempBranch, $splitCommit)

    if ((-not $Init) -and (Test-Path $targetPath)) {
        $localSplitBefore = Get-PrefixSplitCommit -Repo $WorkRepoRoot -Prefix $Prefix
    }

    try {
        if ($Init) {
            if (Test-Path $targetPath) {
                throw "$Prefix already exists. Remove -Init to do an update instead."
            }

            Write-Host "Adding subtree into $Prefix ..."
            Invoke-GitNoMergeEdit -Repo $WorkRepoRoot -Arguments @("subtree", "add", "--prefix=$Prefix", $UpstreamRepoForSplit, $TempBranch)
        }
        else {
            if (-not (Test-Path $targetPath)) {
                throw "$Prefix does not exist yet. Run this script once with -Init first."
            }

            Write-Host "Rewriting vendored CuTe imports to flash_attn.cute before subtree merge ..."
            Invoke-NamespaceRewrite -RewriteScript $RewriteScript -TargetPath $targetPath -Direction upstream
            [void](Commit-PrefixIfChanged -Repo $WorkRepoRoot -Prefix $Prefix -Message $PrepareMergeCommitMessage)

            Write-Host "Pulling upstream updates into $Prefix ..."
            Invoke-Git -Repo $WorkRepoRoot -Arguments @("fetch", $UpstreamRepoForSplit, $TempBranch)
            Invoke-GitNoMergeEdit -Repo $WorkRepoRoot -Arguments @("merge", "-X", "theirs", "-Xsubtree=$Prefix", "FETCH_HEAD")
        }
    }
    finally {
        if (-not $KeepTempBranch) {
            Invoke-Git -Repo $UpstreamRepoForSplit -Arguments @("update-ref", "-d", "refs/heads/$TempBranch")
        }
    }

    Write-Host "Rewriting vendored CuTe imports to flash_sparse_attn.ops.cute ..."
    Invoke-NamespaceRewrite -RewriteScript $RewriteScript -TargetPath $targetPath -Direction local
    [void](Commit-PrefixIfChanged -Repo $WorkRepoRoot -Prefix $Prefix -Message $RewriteCommitMessage)

    $localSplitAfter = Get-PrefixSplitCommit -Repo $WorkRepoRoot -Prefix $Prefix
    Assert-SyncContainsUpstream -Repo $WorkRepoRoot -UpstreamSplitCommit $splitCommit -LocalSplitCommit $localSplitAfter -PreviousLocalSplitCommit $localSplitBefore

    $endHead = Get-GitOutput -Repo $WorkRepoRoot -Arguments @("rev-parse", "HEAD")
    return [PSCustomObject]@{
        StartHead = $startHead
        EndHead = $endHead
    }
}

function Invoke-TemporaryWorktreeSync {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamRepoForSplit,
        [Parameter(Mandatory = $true)]
        [string]$Prefix,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamPrefix,
        [Parameter(Mandatory = $true)]
        [string]$TempBranch,
        [Parameter(Mandatory = $true)]
        [string]$RewriteScript,
        [Parameter(Mandatory = $true)]
        [string]$UpstreamSplitRef,
        [switch]$Init,
        [switch]$SkipFetch,
        [switch]$KeepTempBranch
    )

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $tempBranchName = "sync/cute-worktree-$timestamp"
    $tempWorktree = Join-Path (Split-Path $RepoRoot -Parent) ".cute-sync-worktree-$timestamp"
    $originalHead = Get-GitOutput -Repo $RepoRoot -Arguments @("rev-parse", "HEAD")

    Write-Host "Current worktree is dirty. Syncing in temporary worktree at $tempWorktree ..."
    Invoke-Git -Repo $RepoRoot -Arguments @("worktree", "add", "-b", $tempBranchName, $tempWorktree, $originalHead)

    $stashName = "sync-cute-autostash-$timestamp"
    $stashCreated = $false

    try {
        $result = Invoke-CoreSync -WorkRepoRoot $tempWorktree -UpstreamRepoForSplit $UpstreamRepoForSplit -Prefix $Prefix -UpstreamPrefix $UpstreamPrefix -TempBranch $TempBranch -RewriteScript $RewriteScript -UpstreamSplitRef $UpstreamSplitRef -Init:$Init -SkipFetch:$SkipFetch -KeepTempBranch:$KeepTempBranch

        $commitListOutput = Get-GitOutput -Repo $tempWorktree -Arguments @("rev-list", "--reverse", "--first-parent", "HEAD", "^$originalHead")
        $commits = @()
        if ($commitListOutput) {
            $commits = $commitListOutput -split "`r?`n" | Where-Object { $_ }
        }

        if ($commits.Count -eq 0) {
            Write-Host "No new subtree commits were created."
            return $result
        }

        $currentStatus = Get-DirtyStatus -Repo $RepoRoot
        if ($currentStatus) {
            Write-Host "Stashing current worktree before cherry-picking synced commits back ..."
            Invoke-Git -Repo $RepoRoot -Arguments @("stash", "push", "-u", "-m", $stashName)
            $stashCreated = $true
        }

        try {
            foreach ($commit in $commits) {
                Write-Host "Cherry-picking $commit back into current worktree ..."
                Ensure-GitIdentity -Repo $RepoRoot
                if ((Get-CommitParentCount -Repo $tempWorktree -Commit $commit) -gt 1) {
                    Invoke-Git -Repo $RepoRoot -Arguments @("cherry-pick", "-m", "1", $commit)
                }
                else {
                    Invoke-Git -Repo $RepoRoot -Arguments @("cherry-pick", $commit)
                }
            }
        }
        catch {
            throw "Cherry-pick failed. Resolve the cherry-pick in the current worktree manually."
        }

        if ($stashCreated) {
            try {
                Write-Host "Restoring stashed local changes ..."
                Invoke-Git -Repo $RepoRoot -Arguments @("stash", "pop")
            }
            catch {
                throw "Cherry-pick succeeded, but restoring stashed local changes failed. Resolve the stash pop manually with git stash list / git stash pop."
            }
        }

        return $result
    }
    finally {
        & git -C $RepoRoot worktree remove --force $tempWorktree *> $null
        & git -C $RepoRoot branch -D $tempBranchName *> $null
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRoot = $repoRoot.Path
Set-Location $repoRoot
$rewriteScript = Join-Path $repoRoot "scripts/rewrite_cute_namespace.py"

$cacheRepo = Join-Path $repoRoot $CacheDir
$upstreamRepoForSplit = $null
$upstreamSplitRef = "HEAD"

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

    $upstreamSplitRef = Resolve-RemoteDefaultRef -Repo $upstreamRepoForSplit
}
else {
    $upstreamRepoForSplit = (Resolve-Path $UpstreamRepo).Path
}

$dirtyStatus = Get-DirtyStatus -Repo $repoRoot
if ($dirtyStatus -and -not $NoTemporaryWorktree) {
    $syncResult = Invoke-TemporaryWorktreeSync -RepoRoot $repoRoot -UpstreamRepoForSplit $upstreamRepoForSplit -Prefix $Prefix -UpstreamPrefix $UpstreamPrefix -TempBranch $TempBranch -RewriteScript $rewriteScript -UpstreamSplitRef $upstreamSplitRef -Init:$Init -SkipFetch:$SkipFetch -KeepTempBranch:$KeepTempBranch
}
else {
    if ($dirtyStatus) {
        throw "Superproject has uncommitted changes and -NoTemporaryWorktree was set.`n$dirtyStatus"
    }
    $syncResult = Invoke-CoreSync -WorkRepoRoot $repoRoot -UpstreamRepoForSplit $upstreamRepoForSplit -Prefix $Prefix -UpstreamPrefix $UpstreamPrefix -TempBranch $TempBranch -RewriteScript $rewriteScript -UpstreamSplitRef $upstreamSplitRef -Init:$Init -SkipFetch:$SkipFetch -KeepTempBranch:$KeepTempBranch
}

Write-Host "Done."
Write-Host "Upstream source: $UpstreamRepo"
Write-Host "Upstream cache used for subtree split: $upstreamRepoForSplit"
$syncStartHead = Get-SyncResultProperty -SyncResult $syncResult -PropertyName "StartHead"
$syncEndHead = Get-SyncResultProperty -SyncResult $syncResult -PropertyName "EndHead"
if (-not $syncStartHead) {
    $syncStartHead = Get-GitOutput -Repo $repoRoot -Arguments @("rev-parse", "HEAD")
}
if (-not $syncEndHead) {
    $syncEndHead = Get-GitOutput -Repo $repoRoot -Arguments @("rev-parse", "HEAD")
}
Write-Host "Synced commit range: $syncStartHead -> $syncEndHead"
Write-Host "Local edits inside $Prefix stay in this repo and future upstream changes can be merged by rerunning this script without -Init."
