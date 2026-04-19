# Xeno-Canto v3 API batch fetch for BirdCLEF+ 2026 (234 species, filter Aves + Amphibia = 197)
# Usage: pwsh .\xc_batch_fetch.ps1
# Rate limit: 1 req/sec. Skips species with existing JSON cache.

$ErrorActionPreference = "Stop"

$taxFile = "C:\Users\fw_ya\Desktop\Claude_code\kaggle-competitions\birdclef-2026-embed\out\taxonomy.csv"
$outDir  = "C:\Users\fw_ya\Desktop\Claude_code\kaggle-competitions\birdclef-2026-work\xc_cache"
$keyFile = "$env:USERPROFILE\.xc_api_key.secure"

if (-not (Test-Path $taxFile)) {
    Write-Error "taxonomy.csv not found: $taxFile"
    exit 1
}
if (-not (Test-Path $keyFile)) {
    Write-Error "XC API key file not found: $keyFile. Create with: Read-Host 'XC API Key' -AsSecureString | ConvertFrom-SecureString | Out-File `"`$env:USERPROFILE\.xc_api_key.secure`""
    exit 1
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$secure = (Get-Content $keyFile -Raw).Trim() | ConvertTo-SecureString
$key = [System.Net.NetworkCredential]::new("", $secure).Password

$species = Import-Csv $taxFile | Where-Object { $_.class_name -in @('Aves','Amphibia') }
$total = $species.Count
Write-Host "Target species (Aves + Amphibia): $total"
Write-Host "Output dir: $outDir"
Write-Host ""

$i = 0
$okCount = 0
$skipCount = 0
$errCount = 0
$totalRecordings = 0

foreach ($sp in $species) {
    $i++
    $scientific = $sp.scientific_name
    $label = $sp.primary_label
    $class = $sp.class_name
    $outFile = Join-Path $outDir "$label.json"

    if (Test-Path $outFile) {
        $skipCount++
        continue
    }

    $encodedName = $scientific.Replace(" ", "+")
    $url = "https://xeno-canto.org/api/3/recordings?query=sp:%22$encodedName%22&key=$key"

    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile -ErrorAction Stop -TimeoutSec 30
        $json = Get-Content $outFile -Raw | ConvertFrom-Json
        $n = [int]$json.numRecordings
        $totalRecordings += $n
        $okCount++
        Write-Host ("[{0,3}/{1}] OK {2,-10} {3,-35} [{4,-8}] {5,5} recs" -f $i, $total, $label, $scientific, $class, $n)
    } catch {
        $errCount++
        Write-Host ("[{0,3}/{1}] ERR {2,-10} {3,-35} - {4}" -f $i, $total, $label, $scientific, $_.Exception.Message)
        if (Test-Path $outFile) { Remove-Item $outFile -Force }
    }

    Start-Sleep -Seconds 1
}

Remove-Variable key

Write-Host ""
Write-Host "===== Summary ====="
Write-Host "OK      : $okCount"
Write-Host "Skipped : $skipCount (already cached)"
Write-Host "Errors  : $errCount"
Write-Host "Total recordings fetched metadata for: $totalRecordings"
Write-Host ""
Write-Host "Cache dir: $outDir"
