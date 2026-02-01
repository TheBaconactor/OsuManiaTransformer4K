param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$target = (Resolve-Path (Join-Path $PSScriptRoot "..\\training\\wsl\\fix_wsl.ps1")).Path
& $target @Args
exit $LASTEXITCODE

