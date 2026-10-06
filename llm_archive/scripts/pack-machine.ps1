<#
.SYNOPSIS
    Pack this machine's agent transcripts into one zip for `llma add` on the archive machine.

.DESCRIPTION
    For a machine llm-archive cannot reach over SSH. Run this there, carry the zip
    across, and on the archive machine:

        llma add <the zip>

    Nothing needs installing: Windows PowerShell 5.1 is enough. If running scripts is
    disabled on this machine, start it as:

        powershell -ExecutionPolicy Bypass -File pack-machine.ps1

    What goes in is exactly what the archive's adapters read -- transcripts, never
    settings or credentials. ~/.claude/.credentials.json, ~/.codex/auth.json and
    opencode's auth.json are not opened, let alone packed. The machine's hostname goes
    in too (llma-machine.json), because nothing in the transcripts records it.

    Transcripts are still private code and tool output, and may hold an API key that
    leaked into one. A USB stick is the safe way across; for mail or a cloud folder,
    encrypt the zip first (7-Zip: 7z a -p -mhe=on bundle.7z <the zip>).

    The path rules below re-state RULES in llm_archive/core/machines.py. A bundle is
    only right while the two agree, and tests/test_machines.py runs this script to check.

.PARAMETER OutDir
    Where to write the zip. Default: the Desktop.

.PARAMETER MachineName
    The name to file this machine under. Default: its hostname. Keep the default unless
    there is a reason not to, so that every bundle from here lands on the same machine.
#>
[CmdletBinding()]
param(
    [string]$OutDir = [Environment]::GetFolderPath('Desktop'),
    [string]$MachineName = [System.Net.Dns]::GetHostName()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

# source -> where its store lives here, and where under that to start walking
$stores = [ordered]@{
    claude_code = @{ Root = (Join-Path $env:USERPROFILE '.claude'); Walk = @('projects') }
    codex       = @{ Root = (Join-Path $env:USERPROFILE '.codex'); Walk = @('sessions', 'session_index.jsonl') }
    opencode    = @{ Root = (Join-Path $env:USERPROFILE '.local\share\opencode'); Walk = @('storage\session', 'storage\message', 'storage\part') }
    vscode_chat = @{ Root = (Join-Path $env:APPDATA 'Code\User'); Walk = @('workspaceStorage', 'globalStorage\emptyWindowChatSessions') }
}

# source -> the files its adapter reads, as '/'-separated paths relative to the store
$rules = @{
    claude_code = @('^projects/[^/]+/[^/]+\.jsonl$',
                    '^projects/[^/]+/[^/]+/(?:subagents|tool-results)/[^/]+$')
    codex       = @('^sessions/(?:[^/]+/)*[^/]+\.jsonl$',
                    '^session_index\.jsonl$')
    opencode    = @('^storage/(?:session|message|part)/(?:[^/]+/)*[^/]+\.json$')
    vscode_chat = @('^workspaceStorage/[^/]+/workspace\.json$',
                    '^workspaceStorage/[^/]+/chatSessions/[^/]+\.jsonl?$',
                    '^globalStorage/emptyWindowChatSessions/[^/]+\.jsonl?$')
}

function Test-Wanted([string]$Source, [string]$Rel) {
    foreach ($rule in $rules[$Source]) {
        if ($Rel -cmatch $rule) { return $true }
    }
    return $false
}

# -- what to pack ------------------------------------------------------------

$picked = New-Object System.Collections.Generic.List[object]
foreach ($source in $stores.Keys) {
    $root = $stores[$source].Root
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
    $rootFull = (Get-Item -LiteralPath $root -Force).FullName.TrimEnd('\')
    foreach ($sub in $stores[$source].Walk) {
        $start = Join-Path $rootFull $sub
        if (Test-Path -LiteralPath $start -PathType Leaf) {
            $files = @(Get-Item -LiteralPath $start -Force)
        } elseif (Test-Path -LiteralPath $start -PathType Container) {
            $files = @(Get-ChildItem -LiteralPath $start -Recurse -File -Force -ErrorAction SilentlyContinue)
        } else {
            continue
        }
        foreach ($file in $files) {
            $rel = $file.FullName.Substring($rootFull.Length).TrimStart('\').Replace('\', '/')
            if (Test-Wanted $source $rel) {
                $picked.Add([pscustomobject]@{ Source = $source; Rel = $rel; File = $file })
            }
        }
    }
}

# A workspace.json is only worth carrying for a workspace that has chats.
$chatDirs = @{}
foreach ($p in $picked) {
    if ($p.Source -eq 'vscode_chat' -and $p.Rel -cmatch '^workspaceStorage/([^/]+)/chatSessions/') {
        $chatDirs[$Matches[1]] = $true
    }
}
$picked = @($picked | Where-Object {
    -not ($_.Source -eq 'vscode_chat' -and $_.Rel -cmatch '^workspaceStorage/([^/]+)/workspace\.json$' -and
          -not $chatDirs.ContainsKey($Matches[1]))
})

if ($picked.Count -eq 0) {
    Write-Host "Nothing to pack: no transcripts found under $($env:USERPROFILE) or $($env:APPDATA)."
    exit 1
}

# Claude Code's retention, read for the archive's reminder. Only this one key.
$cleanup = $null
$settings = Join-Path $stores['claude_code'].Root 'settings.json'
if (Test-Path -LiteralPath $settings -PathType Leaf) {
    try {
        $parsed = Get-Content -LiteralPath $settings -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($parsed.PSObject.Properties.Name -contains 'cleanupPeriodDays') {
            $cleanup = [int]$parsed.cleanupPeriodDays
        }
    } catch { }
}

# -- writing it --------------------------------------------------------------

if (-not (Test-Path -LiteralPath $OutDir -PathType Container)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}
$safe = ($MachineName -replace '[^0-9A-Za-z._-]+', '-').Trim('.', '-', '_').ToLowerInvariant()
if (-not $safe) { $safe = 'machine' }
$zipPath = Join-Path (Get-Item -LiteralPath $OutDir).FullName ("llma-machine-{0}-{1}.zip" -f $safe, (Get-Date -Format 'yyyyMMdd-HHmm'))
$partPath = $zipPath + '.part'

$counts = [ordered]@{}
$skipped = 0
$minTime = New-Object DateTime(1980, 1, 2)
$stream = [System.IO.File]::Open($partPath, [System.IO.FileMode]::Create)
$zip = New-Object System.IO.Compression.ZipArchive($stream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($p in $picked) {
        # Opened sharing read/write: a session Claude Code is still appending to goes in
        # as it stands now, rather than failing the whole bundle.
        try {
            $in = [System.IO.File]::Open($p.File.FullName, [System.IO.FileMode]::Open,
                                         [System.IO.FileAccess]::Read, [System.IO.FileShare]'ReadWrite, Delete')
        } catch {
            Write-Warning "skipped (could not open): $($p.File.FullName)"
            $skipped++
            continue
        }
        try {
            $entry = $zip.CreateEntry("$($p.Source)/$($p.Rel)", [System.IO.Compression.CompressionLevel]::Optimal)
            $stamp = $p.File.LastWriteTime
            if ($stamp -lt $minTime) { $stamp = $minTime }
            $entry.LastWriteTime = New-Object DateTimeOffset($stamp)
            $out = $entry.Open()
            try { $in.CopyTo($out) } finally { $out.Dispose() }
        } finally {
            $in.Dispose()
        }
        if (-not $counts.Contains($p.Source)) {
            $counts[$p.Source] = [ordered]@{ files = 0; bytes = [long]0 }
        }
        $counts[$p.Source].files++
        $counts[$p.Source].bytes += $p.File.Length
    }
    if ($counts.Contains('claude_code')) {
        $counts['claude_code']['cleanup_period_days'] = $cleanup
    }

    $manifest = [ordered]@{
        format    = 'llma-machine-bundle'
        version   = 1
        host      = $MachineName
        packed_at = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', [Globalization.CultureInfo]::InvariantCulture)
        packed_by = 'pack-machine.ps1'
        os        = 'Windows'
        sources   = $counts
    }
    $entry = $zip.CreateEntry('llma-machine.json')
    $writer = New-Object System.IO.StreamWriter($entry.Open(), (New-Object System.Text.UTF8Encoding($false)))
    try { $writer.Write(($manifest | ConvertTo-Json -Depth 5)) } finally { $writer.Dispose() }
} finally {
    $zip.Dispose()
    $stream.Dispose()
}
Move-Item -LiteralPath $partPath -Destination $zipPath -Force

# -- what happened -----------------------------------------------------------

$total = 0
$bytes = [long]0
foreach ($source in $counts.Keys) {
    Write-Host ("  {0,-12} {1,6} file(s)" -f $source, $counts[$source].files)
    $total += $counts[$source].files
    $bytes += $counts[$source].bytes
}
Write-Host ("Packed {0} file(s), {1:N1} MB before compression, as {2}." -f $total, ($bytes / 1MB), $MachineName)
if ($skipped) { Write-Host "$skipped file(s) could not be opened and were left out; the next bundle will have them." }
if ($counts.Contains('claude_code') -and $null -eq $cleanup) {
    Write-Host ""
    Write-Host "Note: Claude Code here deletes a transcript 30 days after its last activity."
    Write-Host "To keep them until the next bundle, add this to $settings :"
    Write-Host '    "cleanupPeriodDays": 365'
}
Write-Host ""
Write-Host "-> $zipPath"
Write-Host "Then, on the archive machine:  llma add `"$zipPath`""
