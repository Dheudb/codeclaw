param(
    [Parameter(Mandatory = $true)][string]$PackageDirectory,
    [Parameter(Mandatory = $true)][string]$StagedResources,
    [int]$WaitSeconds = 43200
)
$ErrorActionPreference = 'Stop'
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + ';' + $env:PSModulePath
$packageRoot = (Resolve-Path -LiteralPath $PackageDirectory).Path.TrimEnd('\')
$stageRoot = (Resolve-Path -LiteralPath $StagedResources).Path.TrimEnd('\')
$statusFile = Join-Path $stageRoot 'status.json'
function Write-Status([string]$state, [string]$message = '') {
    $json = @{ state = $state; message = $message; packageDirectory = $packageRoot; updatedAt = (Get-Date).ToString('o') } | ConvertTo-Json
    [System.IO.File]::WriteAllText($statusFile, $json, [System.Text.UTF8Encoding]::new($false))
}
function Assert-Child([string]$candidate, [string]$root) {
    $full = [System.IO.Path]::GetFullPath($candidate)
    if (-not $full.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Update path is outside the intended directory.' }
    $probe = $full
    while ($probe.Length -ge $root.Length) {
        if (Test-Path -LiteralPath $probe) {
            if ((Get-Item -LiteralPath $probe -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Refusing to update a linked directory.' }
        }
        if ($probe -eq $root) { break }
        $probe = Split-Path -Parent $probe
    }
}
$changes = [System.Collections.Generic.List[object]]::new()
try {
    if (-not (Test-Path -LiteralPath (Join-Path $packageRoot 'CodeClaw.exe') -PathType Leaf)) { throw 'The original CodeClaw.exe is missing.' }
    $manifest = Get-Content -LiteralPath (Join-Path $stageRoot 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.metadataDirectory -notmatch '^codeclaw-[0-9A-Za-z_.-]+\.dist-info$') { throw 'Invalid Python package metadata directory.' }
    $entries = @(
        @{ Source = (Join-Path $stageRoot 'app'); Target = (Join-Path $packageRoot 'resources\app'); Backup = (Join-Path $stageRoot 'backup\app') },
        @{ Source = (Join-Path $stageRoot 'codeclaw'); Target = (Join-Path $packageRoot 'resources\python\Lib\site-packages\codeclaw'); Backup = (Join-Path $stageRoot 'backup\codeclaw') },
        @{ Source = (Join-Path $stageRoot 'metadata'); Target = (Join-Path $packageRoot ('resources\python\Lib\site-packages\' + $manifest.metadataDirectory)); Backup = (Join-Path $stageRoot 'backup\metadata') }
    )
    foreach ($entry in $entries) {
        Assert-Child $entry.Source $stageRoot
        Assert-Child $entry.Target $packageRoot
        Assert-Child $entry.Backup $stageRoot
        if (-not (Test-Path -LiteralPath $entry.Source -PathType Container)) { throw 'The update payload is incomplete.' }
        if (Test-Path -LiteralPath $entry.Backup) { throw 'This update was already applied or needs recovery.' }
        $hashes = @{}
        foreach ($file in Get-ChildItem -LiteralPath $entry.Source -File -Recurse) {
            Assert-Child $file.FullName $stageRoot
            $hashes[$file.FullName.Substring($entry.Source.Length + 1)] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
        $entry.Hashes = $hashes
    }
    Write-Status 'waiting_for_exit'
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    do {
        $running = @(Get-CimInstance Win32_Process -Filter "Name = 'CodeClaw.exe' OR Name = 'python.exe'" | Where-Object {
            $_.ExecutablePath -and $_.ExecutablePath.StartsWith($packageRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)
        })
        if ($running.Count -eq 0) { break }
        if ((Get-Date) -gt $deadline) { throw 'CodeClaw is still open. The update payload is preserved for retry.' }
        Start-Sleep -Seconds 1
    } while ($true)
    Write-Status 'applying'
    foreach ($entry in $entries) {
        Assert-Child $entry.Target $packageRoot
        [void](New-Item -ItemType Directory -Path (Split-Path -Parent $entry.Backup) -Force)
        $change = @{ Entry = $entry; BackupMoved = $false; Installed = $false }
        $changes.Add($change)
        if (Test-Path -LiteralPath $entry.Target) {
            Move-Item -LiteralPath $entry.Target -Destination $entry.Backup
            $change.BackupMoved = $true
        }
        Move-Item -LiteralPath $entry.Source -Destination $entry.Target
        $change.Installed = $true
        foreach ($relative in $entry.Hashes.Keys) {
            $installedFile = Join-Path $entry.Target $relative
            Assert-Child $installedFile $packageRoot
            if ((Get-FileHash -LiteralPath $installedFile -Algorithm SHA256).Hash -ne $entry.Hashes[$relative]) { throw 'Updated file verification failed.' }
        }
    }
    Write-Status 'applied'
} catch {
    for ($index = $changes.Count - 1; $index -ge 0; $index--) {
        $change = $changes[$index]; $entry = $change.Entry
        if ($change.Installed) { Move-Item -LiteralPath $entry.Target -Destination $entry.Source }
        if ($change.BackupMoved) { Move-Item -LiteralPath $entry.Backup -Destination $entry.Target }
    }
    Write-Status 'failed' $_.Exception.Message
    exit 1
}
