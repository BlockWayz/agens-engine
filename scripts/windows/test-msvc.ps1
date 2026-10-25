# Agens engine validation suite (Windows, msvc): deployment, kernel, determinism and parity gates.
# Assumes the build already exists (see build-msvc.ps1); runs the binaries with NO extra environment
# to prove the side-by-side DLL deployment works.
#
#   powershell -File test-msvc.ps1 -Checkpoint D:\models\volundr-16gb -OutDir D:\tmp\agens-test
#   optional: -Exe path\to\agens.exe -NewTokens 64 -BudgetGiB 16 -Wctx 262148 -Port 30000 -Skip serve,session
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Exe = "$PSScriptRoot\..\..\target\release\agens.exe",
    [int]$NewTokens = 64,
    [double]$BudgetGiB = 16,
    [int]$Wctx = 262148,
    [int]$Port = 30000,
    [string]$Skip = ""
)
$ErrorActionPreference = 'Stop'
$SkipList = $Skip -split ',' | ForEach-Object { $_.Trim().ToLower() }
if (-not (Test-Path $Checkpoint)) { Write-Host "checkpoint not found: $Checkpoint" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $Exe)) { Write-Host "exe not found: $Exe" -ForegroundColor Red; exit 2 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Out = (Resolve-Path $OutDir).Path

# Engine default KV mode: clear any stale INT8 dense-KV env from the caller's shell (its prefill path is
# capped at MAXT=1024 rows/launch, kernels/m1/m1_dense8.cu, and the 1280-token prompt must run the plain path).
Remove-Item Env:AGENS_KV_INT8_DENSE -ErrorAction SilentlyContinue
$Results = New-Object System.Collections.Generic.List[object]
function Invoke-Step {
    param([string]$Id, [string]$Desc, [scriptblock]$Body)
    if ($SkipList -contains $Id.ToLower()) { $Results.Add([pscustomobject]@{ Id = $Id; Status = 'SKIP'; Note = 'skipped' }); return }
    Write-Host "== $Id : $Desc" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $note = ''
    try {
        $note = & $Body
        if ($null -eq $note) { $note = '' }
        $Results.Add([pscustomobject]@{ Id = $Id; Status = 'PASS'; Note = "$note $($sw.Elapsed.ToString('mm\:ss'))" })
    } catch {
        $Results.Add([pscustomobject]@{ Id = $Id; Status = 'FAIL'; Note = "$($_.Exception.Message)" })
        Write-Host "   FAIL: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Run-Exe {
    param([string[]]$Cmd, [string]$LogName)
    # PS 5.1: with $ErrorActionPreference=Stop, `2>&1` turns a native command's stderr line into a
    # terminating error whose message is that line (the engine's progress logs go to stderr). Run the
    # command under Continue and judge only by exit code + log scan.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $log = Join-Path $Out "log-$LogName.txt"
    $out = & $Exe @Cmd 2>&1 | Tee-Object -FilePath $log | ForEach-Object { "$_" }
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE (see $log)" }
    $tail = ($out | Select-Object -Last 2) -join ' | '
    if ($out -match '(?i)diverge|mismatch|FAILED') { throw "gate verdict in $log" }
    return $tail.Trim()
}

function Same-Hash {
    param([string]$A, [string]$B)
    foreach ($f in @($A, $B)) { if (-not (Test-Path $f)) { throw "no such file: $f" } }
    $ha = (Get-FileHash $A -Algorithm MD5).Hash; $hb = (Get-FileHash $B -Algorithm MD5).Hash
    if ($ha -ne $hb) { throw "hash mismatch $A ($ha) vs $B ($hb)" }
    if (-not $ha) { throw "empty hash for $A" }
    return "md5 $ha"
}

# ---- A. deployment layout (side-by-side DLLs) ----
Invoke-Step 'A' 'deployment layout' {
    $need = @('agens_m1.dll', 'agens_kernels_py.dll', 'agens_oracle_attn.dll', 'agens_oracle_prefill.dll',
              'agens_oracle_bridge.dll', 'torch.dll', 'torch_cuda.dll', 'c10.dll', 'c10_cuda.dll', 'cudart64_13.dll')
    $dir = Split-Path $Exe -Parent
    $missing = $need | Where-Object { -not (Test-Path (Join-Path $dir $_)) }
    if ($missing) { throw "missing beside exe: $($missing -join ', ')" }
    "dlls side-by-side in $dir"
}

# ---- B. startup + checkpoint tools (no GPU numerics) ----
Invoke-Step 'B' 'startup, validate, verify' {
    (Run-Exe @('--version') 'b0') | Out-Null
    (Run-Exe @('devices') 'b1') | Out-Null
    (Run-Exe @('validate', $Checkpoint) 'b2') | Out-Null
    (Run-Exe @('verify', $Checkpoint) 'b3')
}

# ---- C. GPU + kernel layer ----
Invoke-Step 'C' 'weight upload (cudart/cublas/kernels_py)' {
    Run-Exe @('load', $Checkpoint) 'c1'
}

# ---- D'. determinism: greedy repeat + teacher-forced logits repeat ----
Invoke-Step 'D' 'prompts + greedy determinism + logits determinism' {
    # tokenize with the checkpoint tokenizer if python+tokenizers available, else deterministic synthetic ids
    $py = @'
import json, sys, random
ckpt, out = sys.argv[1], sys.argv[2]
texts = ["Hello!", "Explain what a tensor-parallel all-reduce does in one short paragraph.",
         "Write a haiku about CUDA."]
rnd = random.Random(7)
long_synth = [{"name": f"s{i}", "ids": [rnd.randrange(100, 5000) for _ in range(n)], "new_tokens": 16}
              for i, n in enumerate((640, 1280))]  # length coverage incl. >MAXT prefill (plain KV path)
try:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(ckpt + "/tokenizer.json")
    prompts = [{"name": f"p{i}", "ids": tk.encode(t).ids} for i, t in enumerate(texts)] + long_synth
    mode = "tokenizer.json + 2 synthetic long prompts"
except Exception:
    prompts = [{"name": f"p{i}", "ids": [rnd.randrange(100, 5000) for _ in range(n)]}
               for i, n in enumerate((240, 640, 1280))]
    mode = "synthetic ids (tokenizers unavailable)"
json.dump(prompts, open(out, "w"), indent=1)
print(mode)
'@
    $pyFile = Join-Path $Out 'mkprompts.py'
    Set-Content -Path $pyFile -Value $py -Encoding UTF8
    $prompts = Join-Path $Out 'prompts.json'
    $pybin = if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") { "$PSScriptRoot\.venv\Scripts\python.exe" } else { 'python' }
    $mode = (& $pybin $pyFile $Checkpoint $prompts) 2>&1
    if ($LASTEXITCODE -ne 0) { throw "prompt generation failed: $mode" }

    (Run-Exe @('m1', $Checkpoint, $prompts, (Join-Path $Out 'run1.json'), '--new-tokens', "$NewTokens") 'd1') | Out-Null
    (Run-Exe @('m1', $Checkpoint, $prompts, (Join-Path $Out 'run2.json'), '--new-tokens', "$NewTokens") 'd2') | Out-Null
    $h = Same-Hash (Join-Path $Out 'run1.json') (Join-Path $Out 'run2.json')
    # m1-kv writes per-position logits only via --save-logits (the AGENS_M1_LOGITS_DUMP env is the `m1`
    # command's hook); save them on run 1, then --cmp-logits re-runs the forced sequence and the engine
    # compares the logits itself (top-1 / KL verdict lands in the log).
    $ld = Join-Path $Out 'logits1'
    Run-Exe @('m1-kv', $Checkpoint, $prompts, (Join-Path $Out 'kv1.json'), '--force', (Join-Path $Out 'run1.json'),
              '--save-logits', $ld) 'd3' | Out-Null
    if (-not (Get-ChildItem $ld -Filter *.bin -ErrorAction SilentlyContinue)) { throw "no logits bins saved to $ld" }
    $hl = Run-Exe @('m1-kv', $Checkpoint, $prompts, (Join-Path $Out 'kv2.json'), '--force', (Join-Path $Out 'run1.json'),
                    '--cmp-logits', $ld) 'd4'
    "$mode; greedy $h; logits $hl"
}

# ---- E'. parity gates ----
Invoke-Step 'spec' 'speculative decoding == greedy oracle' {
    Run-Exe @('m1-spec', $Checkpoint, (Join-Path $Out 'prompts.json'), (Join-Path $Out 'spec.json'),
              '--oracle', (Join-Path $Out 'run1.json'), '--corrupt', '3') 'e1'
}
Invoke-Step 'batch' 'batched rows == bs=1' {
    Run-Exe @('m1-batch', $Checkpoint, (Join-Path $Out 'prompts.json'), '--batch', '2,4,8') 'e2'
}
Invoke-Step 'session' 'session fork/restore exactness' {
    Run-Exe @('m1-session', $Checkpoint, (Join-Path $Out 'prompts.json'), '--sessions', '4', '--cuts', '2') 'e3'
}

# ---- F. serve: HTTP end to end, greedy reply identical across restarts ----
Invoke-Step 'serve' 'OpenAI endpoint + restart determinism' {
    # max_tokens 256: a reasoning model spends the first tokens in its think block, so content can be
    # empty at short budgets; compare the whole completion (message + usage), not just content.
    $body = '{"model":"agens","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'
    $reply = @(1, 2) | ForEach-Object {
        $i = $_
        $log = Join-Path $Out "serve$i.log"
        $proc = Start-Process -FilePath $Exe -NoNewWindow -PassThru `
            -ArgumentList @('serve', $Checkpoint, '--host', '127.0.0.1', '--port', "$Port",
                            '--budget-gib', "$BudgetGiB", '--wctx', "$Wctx", '--mode', 'personal',
                            '--no-fork', '1', '--vision', '0') `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        $deadline = (Get-Date).AddSeconds(300)
        while (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue)) {
            if ((Get-Date) -gt $deadline -or $proc.HasExited) { throw "serve did not bind :$Port (see $log)" }
            Start-Sleep -Seconds 3
        }
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))
        Stop-Process -Id $proc.Id -Force
        Start-Sleep -Seconds 2
        ($resp.choices[0].message | ConvertTo-Json -Compress) + "`n" + ($resp.usage | ConvertTo-Json -Compress)
    }
    if ($reply[0] -ne $reply[1]) { throw "greedy reply differs across restarts" }
    $ct = (($reply[0] -split "`n", 2)[1] | ConvertFrom-Json).completion_tokens
    if ($ct -le 0) { throw "completion produced 0 tokens" }
    "completion identical across restarts ($ct tokens, $($reply[0].Length) chars compared)"
}

# ---- summary ----
$Results | Format-Table -AutoSize Id, Status, Note
$failed = @($Results | Where-Object { $_.Status -eq 'FAIL' })
if ($failed.Count) { Write-Host "SUITE FAILED ($($failed.Count) of $($Results.Count) steps)" -ForegroundColor Red; exit 1 }
Write-Host "SUITE PASSED ($($Results.Count) steps; artifacts in $Out)" -ForegroundColor Green
