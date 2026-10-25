<#
.SYNOPSIS
  Build and package PhotoCraft: one entry point for every platform.

.DESCRIPTION
  Runs on PowerShell 7+ (pwsh) on Windows, macOS and Linux. Two modes:

  Release packages (default): calls the platform's packaging script, which builds, signs and
  writes the installers to $env:DIST (default: dist/release). See docs/releasing.md.
    windows  packaging/windows/package.ps1   .msi + portable .zip (needs WiX v5, Windows SDK)
    macos    packaging/macos/package.sh      .dmg + CLI .zip       (needs Xcode tools)
    linux    packaging/linux/package.sh      AppImage, .deb, .rpm, .tar.gz (needs nfpm for deb/rpm)
    freebsd  packaging/freebsd/package.sh    .tar.gz
    web      packaging/web/package.sh        static site .zip      (needs trunk + wasm target)

  Portable (-Portable): only cargo is needed. Builds photocraft and photocraft-cli in release
  and archives them with the licences:
    photocraft-<version>-<os>-<arch>-portable.zip     (Windows: with portable.txt, so settings
                                                       stay next to the exe)
    photocraft-<version>-<os>-<arch>-portable.tar.gz  (macOS, Linux, FreeBSD)
  Unsigned: for testing and for machines without the installer toolchains.

  Set CRAFT_FONTS_DIR (or pass -FontsDir) to build with storytold/craft-fonts, as official
  releases do; the font licences then ship in the package.

.PARAMETER Target
  auto (default: this machine's OS), windows, macos, linux, freebsd or web.

.PARAMETER Arch
  Windows: x64 (default), x86, arm64. macOS: universal (default), aarch64, x86_64.
  Portable mode: the Rust target's architecture; default this machine's.

.PARAMETER Portable
  Make the cargo-only portable archive instead of the release installers.

.PARAMETER SkipBuild
  Package binaries already built in target/ instead of building.

.PARAMETER Check
  Before packaging, run the gate for the release: cargo xtask layers, clippy and the tests of
  the app crates.

.PARAMETER FontsDir
  A craft-fonts checkout (sets CRAFT_FONTS_DIR for the build).

.PARAMETER Formats
  Linux release only: the formats to make, e.g. "appimage tar" (default: all).

.EXAMPLE
  pwsh shell/build.ps1                          # release packages for this OS
  pwsh shell/build.ps1 -Portable                # quick unsigned zip/tar.gz of the two binaries
  pwsh shell/build.ps1 -Target windows -Arch arm64
  pwsh shell/build.ps1 -Target linux -Formats "appimage deb"
  pwsh shell/build.ps1 -Target web
  pwsh shell/build.ps1 -Portable -Check -FontsDir ../craft-fonts
#>
[CmdletBinding()]
param(
  [ValidateSet('auto', 'windows', 'macos', 'linux', 'freebsd', 'web')] [string] $Target = 'auto',
  [string] $Arch = '',
  [switch] $Portable,
  [switch] $SkipBuild,
  [switch] $Check,
  [string] $FontsDir = '',
  [string] $Formats = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

if ($PSVersionTable.PSVersion.Major -lt 7) {
  throw "PowerShell 7 or later is needed (this is $($PSVersionTable.PSVersion)); run it with pwsh"
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Invoke-Native([string] $What, [scriptblock] $Block) {
  Write-Host "==> $What" -ForegroundColor Cyan
  & $Block
  if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

function Assert-Command([string] $Name, [string] $Hint) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "$Name not found: $Hint" }
}

# ---- what to build -----------------------------------------------------------------------------
if ($Target -eq 'auto') {
  $Target = if ($IsWindows) { 'windows' } elseif ($IsMacOS) { 'macos' }
            elseif ((uname -s) -eq 'FreeBSD') { 'freebsd' } else { 'linux' }
}
if ($Portable -and $Target -eq 'web') { throw '-Portable makes desktop archives; the web build is always a site zip (drop -Portable)' }
$hostOs = if ($IsWindows) { 'windows' } elseif ($IsMacOS) { 'macos' } elseif ((uname -s) -eq 'FreeBSD') { 'freebsd' } else { 'linux' }
if ($Target -ne 'web' -and $Target -ne $hostOs) {
  throw "packaging for $Target has to run on $Target (this machine is $hostOs); CI builds each platform on its own runner"
}

# The version lives in one place: [workspace.package] version in the root Cargo.toml.
$Version = $env:PHOTOCRAFT_VERSION
if (-not $Version) {
  $inPkg = $false
  foreach ($line in Get-Content (Join-Path $Root 'Cargo.toml')) {
    if ($line -match '^\s*\[') { $inPkg = ($line.Trim() -eq '[workspace.package]'); continue }
    if ($inPkg -and $line -match '^\s*version\s*=\s*"([^"]+)"') { $Version = $Matches[1]; break }
  }
}
if (-not $Version) { throw 'could not read [workspace.package] version from Cargo.toml' }

if (-not $env:DIST) { $env:DIST = Join-Path $Root (Join-Path 'dist' 'release') }
New-Item -ItemType Directory -Force -Path $env:DIST | Out-Null
if (-not $env:PHOTOCRAFT_BUILD_SHA) { $env:PHOTOCRAFT_BUILD_SHA = (git -C $Root rev-parse HEAD 2>$null) }
if (-not $env:PHOTOCRAFT_BUILD_DATE) { $env:PHOTOCRAFT_BUILD_DATE = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd') }
if ($FontsDir) {
  if (-not (Test-Path (Join-Path $FontsDir 'fonts'))) { throw "-FontsDir $FontsDir is not a craft-fonts checkout (no fonts/ folder)" }
  $env:CRAFT_FONTS_DIR = (Resolve-Path $FontsDir).Path
}

$mode = if ($Portable) { 'portable' } else { 'release' }
Write-Host "PhotoCraft ${Version}: $mode package for $Target -> $($env:DIST)" -ForegroundColor Green
if ($env:CRAFT_FONTS_DIR) { Write-Host "    fonts: $($env:CRAFT_FONTS_DIR)" } else { Write-Host '    fonts: none (CRAFT_FONTS_DIR unset; official releases set it)' }

Push-Location $Root
try {
  Assert-Command 'cargo' 'install Rust from https://rustup.rs'

  # ---- optional gate ---------------------------------------------------------------------------
  if ($Check) {
    Invoke-Native 'cargo xtask layers' { cargo xtask layers }
    Invoke-Native 'clippy' { cargo clippy --locked -p photocraft -p photocraft-cli -p photocraft-ui-egui -p photocraft-automation -p photocraft-engine --all-targets -- -D warnings }
    Invoke-Native 'tests' { cargo test --locked -p photocraft -p photocraft-cli -p photocraft-automation --lib --tests }
    if ($Target -eq 'web') { Invoke-Native 'cargo xtask wasm' { cargo xtask wasm } }
  }

  # ---- release packages: the platform scripts ----------------------------------------------------
  if (-not $Portable) {
    $skip = if ($SkipBuild) { @('--skip-build') } else { @() }
    switch ($Target) {
      'windows' {
        $a = if ($Arch) { $Arch } else { 'x64' }
        if ($a -notin 'x64', 'x86', 'arm64') { throw "-Arch for Windows is x64, x86 or arm64 (got $a)" }
        & (Join-Path $Root 'packaging/windows/package.ps1') -Arch $a -SkipBuild:$SkipBuild
      }
      'macos' {
        $a = if ($Arch) { $Arch } else { 'universal' }
        if ($a -notin 'universal', 'aarch64', 'x86_64') { throw "-Arch for macOS is universal, aarch64 or x86_64 (got $a)" }
        Invoke-Native 'packaging/macos/package.sh' { bash (Join-Path $Root 'packaging/macos/package.sh') --arch $a @skip }
      }
      'linux' {
        if ($Arch) { Write-Warning "-Arch is ignored for Linux release packages: they are built for this machine ($(uname -m))" }
        $fmt = if ($Formats) { @('--formats', $Formats) } else { @() }
        Invoke-Native 'packaging/linux/package.sh' { bash (Join-Path $Root 'packaging/linux/package.sh') @skip @fmt }
      }
      'freebsd' {
        Invoke-Native 'packaging/freebsd/package.sh' { bash (Join-Path $Root 'packaging/freebsd/package.sh') @skip }
      }
      'web' {
        Invoke-Native 'packaging/web/package.sh' { bash (Join-Path $Root 'packaging/web/package.sh') @skip }
      }
    }
    Write-Host "Done. Artifacts in $($env:DIST):" -ForegroundColor Green
    Get-ChildItem $env:DIST -File | Where-Object { $_.Name -like "photocraft*$Version*" } | Format-Table Name, Length
    return
  }

  # ---- portable archive: cargo only --------------------------------------------------------------
  $hostArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
  $a = if ($Arch) { $Arch.ToLowerInvariant() } else { $hostArch }
  $cpu = switch ($a) {
    { $_ -in 'x64', 'x86_64', 'amd64' } { 'x86_64' }
    { $_ -in 'arm64', 'aarch64' } { 'aarch64' }
    { $_ -in 'x86', 'i686' } { 'i686' }
    'riscv64' { 'riscv64gc' }
    default { throw "unknown -Arch $a (x64, arm64, x86, riscv64)" }
  }
  $triple = switch ($Target) {
    'windows' { "$cpu-pc-windows-msvc" }
    'macos' { if ($cpu -notin 'x86_64', 'aarch64') { throw 'macOS is x86_64 or aarch64' }; "$cpu-apple-darwin" }
    'linux' { "$cpu-unknown-linux-gnu" }
    'freebsd' { "$cpu-unknown-freebsd" }
  }
  # Release file names: x64/x86/arm64 on Windows (as the MSI), the uname spelling elsewhere.
  $label = if ($Target -eq 'windows') { switch ($cpu) { 'x86_64' { 'x64' } 'aarch64' { 'arm64' } 'i686' { 'x86' } } }
           else { $cpu -replace 'riscv64gc', 'riscv64' }
  $exe = if ($Target -eq 'windows') { '.exe' } else { '' }
  $targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'target' }
  $bin = Join-Path $targetDir (Join-Path $triple 'release')

  if (-not $SkipBuild) {
    $installed = (rustup target list --installed 2>$null)
    if ($LASTEXITCODE -eq 0 -and $installed -notcontains $triple) {
      Invoke-Native "rustup target add $triple" { rustup target add $triple }
    }
    if ($Target -eq 'windows') {
      # Static C runtime, as the MSI build: no Visual C++ redistributable needed.
      $flagVar = 'CARGO_TARGET_' + ($triple.ToUpper() -replace '-', '_') + '_RUSTFLAGS'
      [Environment]::SetEnvironmentVariable($flagVar, '-C target-feature=+crt-static')
    }
    Invoke-Native "cargo build --release ($triple)" { cargo build --release --locked -p photocraft -p photocraft-cli --features heif --target $triple }
  }
  foreach ($b in "photocraft$exe", "photocraft-cli$exe") {
    if (-not (Test-Path (Join-Path $bin $b))) { throw "$b not found in $bin; run without -SkipBuild" }
  }

  $name = "photocraft-$Version-$Target-$label-portable"
  $stage = Join-Path $targetDir (Join-Path 'portable-package' $name)
  Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $stage | Out-Null
  Copy-Item (Join-Path $bin "photocraft$exe"), (Join-Path $bin "photocraft-cli$exe") $stage
  foreach ($f in 'README.md', 'LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE', 'ATTRIBUTION.md') {
    $p = Join-Path $Root $f
    if (Test-Path $p) { Copy-Item $p $stage }
  }
  # Builds with craft-fonts embed its OFL-1.1 fonts: ship each font's licence.
  if ($env:CRAFT_FONTS_DIR) {
    Get-ChildItem -Path (Join-Path $env:CRAFT_FONTS_DIR 'fonts') -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $lic = Join-Path $_.FullName 'OFL.txt'
      if (Test-Path $lic) { Copy-Item $lic (Join-Path $stage "OFL-$($_.Name).txt") }
    }
  }
  # The portable marker: settings, presets and recovery files go to PhotoCraftData/ beside the
  # binary (apps/photocraft/src/app_dirs.rs; it works on every desktop OS).
  Copy-Item (Join-Path $Root 'packaging/windows/portable.txt') $stage

  if ($Target -eq 'windows') {
    $archive = Join-Path $env:DIST "$name.zip"
    Remove-Item -Force $archive -ErrorAction SilentlyContinue
    Compress-Archive -Path $stage -DestinationPath $archive
  } else {
    $archive = Join-Path $env:DIST "$name.tar.gz"
    Remove-Item -Force $archive -ErrorAction SilentlyContinue
    Invoke-Native 'tar' { tar -czf $archive -C (Split-Path $stage -Parent) $name }
  }

  # Smoke test when this machine can run the build.
  $canRun = ($cpu -eq 'x86_64' -and $hostArch -eq 'x64') -or ($cpu -eq 'aarch64' -and $hostArch -eq 'arm64') -or ($cpu -eq 'i686' -and $IsWindows -and $hostArch -eq 'x64')
  if ($canRun) {
    Invoke-Native 'photocraft-cli --version' { & (Join-Path $stage "photocraft-cli$exe") --version }
  } else {
    Write-Host "skipping photocraft-cli --version: a $cpu build doesn't run on this $hostArch machine"
  }
  Write-Host 'Done (unsigned portable build):' -ForegroundColor Green
  Get-Item $archive | Format-Table Name, Length
} finally {
  Pop-Location
}
