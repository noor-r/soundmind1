param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [string]$OutputFile
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectPath = Join-Path $repoRoot "SoundMind.Downloader/SoundMind.Downloader.csproj"

if (-not (Test-Path $projectPath)) {
    Write-Error "Could not find project file at $projectPath"
    exit 1
}

$argsList = @("run", "--project", $projectPath, "--", $Url)
if ($OutputFile) {
    $argsList += $OutputFile
}

& dotnet @argsList
exit $LASTEXITCODE
