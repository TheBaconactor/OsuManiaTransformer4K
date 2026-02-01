param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$target = (Resolve-Path (Join-Path $PSScriptRoot "..\\training\\wsl\\run_wsl_ear_finetune.ps1")).Path
& $target @Args
exit $LASTEXITCODE

