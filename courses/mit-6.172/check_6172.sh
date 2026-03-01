#!/usr/bin/env bash
# =============================================================================
# check_6172.sh — MIT 6.172 (Fall 2018) Environment Checker for WSL2
# Idempotent. Safe to rerun. Run with: bash check_6172.sh
# =============================================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

BLOCKERS=()
WARNINGS=()
MISSING_PKGS=()

ok()   { printf "${GREEN}[OK]${NC}    %s\n" "$1"; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; WARNINGS+=("$1"); }
fail() { printf "${RED}[FAIL]${NC}  %s\n" "$1"; BLOCKERS+=("$1"); }
info() { printf "${CYAN}[INFO]${NC}  %s\n" "$1"; }
header() { printf "\n${BOLD}=== %s ===${NC}\n" "$1"; }

# ─── Section 1: System Info ──────────────────────────────────────────────────
header "SYSTEM INFORMATION"

# WSL version detection
KERNEL=$(uname -r)
if [[ "$KERNEL" == *microsoft-standard-WSL2* ]]; then
    ok "WSL2 confirmed (kernel: $KERNEL)"
elif [[ "$KERNEL" == *Microsoft* ]]; then
    fail "WSL1 detected — upgrade to WSL2: wsl --set-version <distro> 2"
else
    warn "Not running in WSL (kernel: $KERNEL)"
fi

# Architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    ok "Architecture: $ARCH"
else
    fail "Expected x86_64, got $ARCH — course assumes x86_64"
fi

# Ubuntu version
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    info "OS: $PRETTY_NAME"
    case "$VERSION_ID" in
        22.04|24.04) ok "Ubuntu $VERSION_ID is supported" ;;
        *)           warn "Ubuntu $VERSION_ID — script targets 22.04/24.04, packages may differ" ;;
    esac
else
    warn "Cannot determine OS version"
fi

# CPU
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
CPU_CORES=$(nproc 2>/dev/null || echo "?")
info "CPU: $CPU_MODEL ($CPU_CORES cores)"

# ─── Section 2: Resources ────────────────────────────────────────────────────
header "RESOURCE CHECK"

# RAM (minimum 2 GB recommended, 4+ preferred)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_MB=$((TOTAL_RAM_KB / 1024))
TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_MB/1024}")
if (( TOTAL_RAM_MB >= 4000 )); then
    ok "RAM: ${TOTAL_RAM_GB} GB (recommended: >= 4 GB)"
elif (( TOTAL_RAM_MB >= 2000 )); then
    warn "RAM: ${TOTAL_RAM_GB} GB — workable but tight (recommended: >= 4 GB)"
else
    fail "RAM: ${TOTAL_RAM_GB} GB — too low (minimum 2 GB, recommended 4 GB)"
fi

# Disk
AVAIL_GB=$(df / --output=avail -BG 2>/dev/null | tail -1 | tr -d ' G')
if (( AVAIL_GB >= 10 )); then
    ok "Disk: ${AVAIL_GB} GB free (need >= 5 GB for toolchain + coursework)"
elif (( AVAIL_GB >= 5 )); then
    warn "Disk: ${AVAIL_GB} GB free — cutting it close"
else
    fail "Disk: ${AVAIL_GB} GB free — need at least 5 GB"
fi

# ─── Section 3: Required Tools ───────────────────────────────────────────────
header "TOOL CHECK"

check_tool() {
    local cmd="$1"
    local pkg="${2:-$1}"        # apt package name (defaults to cmd name)
    local ver_flag="${3:---version}"

    if command -v "$cmd" &>/dev/null; then
        local ver
        ver=$("$cmd" $ver_flag 2>&1 | head -1) || ver="(installed but version check failed)"
        ok "$cmd: $ver"
        return 0
    else
        fail "$cmd not found"
        MISSING_PKGS+=("$pkg")
        return 1
    fi
}

check_tool git          git
check_tool make         make
check_tool gcc          gcc
check_tool clang        clang
check_tool gdb          gdb
check_tool valgrind     valgrind
check_tool python3      python3
check_tool cmake        cmake

# llvm-cov: try versioned names too
if command -v llvm-cov &>/dev/null; then
    VER=$(llvm-cov --version 2>&1 | head -1)
    ok "llvm-cov: $VER"
elif command -v llvm-cov-18 &>/dev/null; then
    ok "llvm-cov-18 found (symlink recommended: sudo ln -s \$(which llvm-cov-18) /usr/local/bin/llvm-cov)"
elif command -v llvm-cov-14 &>/dev/null; then
    ok "llvm-cov-14 found (symlink recommended: sudo ln -s \$(which llvm-cov-14) /usr/local/bin/llvm-cov)"
else
    fail "llvm-cov not found"
    MISSING_PKGS+=("llvm")
fi

# perf (optional but important)
header "PERF CHECK (optional)"
if command -v perf &>/dev/null; then
    PERF_VER=$(perf --version 2>&1 || true)
    # Try actually running perf
    if perf stat -- true 2>/dev/null; then
        ok "perf works: $PERF_VER"
    else
        warn "perf installed ($PERF_VER) but cannot collect events"
        info "Fix: edit /etc/sysctl.conf, add kernel.perf_event_paranoid=-1, then: sudo sysctl -p"
        info "Or run as root: sudo perf stat ..."
    fi
else
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
    warn "perf not installed (perf_event_paranoid=$PARANOID)"
    info "Install: sudo apt install linux-tools-generic"
    info "On WSL2, perf often requires a custom kernel or linux-tools matching your kernel."
    info "Workaround: use 'sudo perf' or build perf from WSL2 kernel source."
    info "Course HW1 does NOT require perf — later assignments may."
fi

# ─── Section 4: Functional Verification ──────────────────────────────────────
header "FUNCTIONAL VERIFICATION"

# gcc compile test
if command -v gcc &>/dev/null; then
    TMPF=$(mktemp /tmp/test_XXXXXX.c)
    echo 'int main(){return 0;}' > "$TMPF"
    if gcc -o "${TMPF%.c}" "$TMPF" 2>/dev/null && [ -x "${TMPF%.c}" ]; then
        ok "gcc can compile and link"
    else
        fail "gcc compile test failed"
    fi
    rm -f "$TMPF" "${TMPF%.c}"
fi

# clang compile test
if command -v clang &>/dev/null; then
    TMPF=$(mktemp /tmp/test_XXXXXX.c)
    echo 'int main(){return 0;}' > "$TMPF"
    if clang -o "${TMPF%.c}" "$TMPF" 2>/dev/null && [ -x "${TMPF%.c}" ]; then
        ok "clang can compile and link"
    else
        fail "clang compile test failed"
    fi
    rm -f "$TMPF" "${TMPF%.c}"
fi

# clang AddressSanitizer test
if command -v clang &>/dev/null; then
    TMPF=$(mktemp /tmp/test_XXXXXX.c)
    echo 'int main(){return 0;}' > "$TMPF"
    if clang -fsanitize=address -o "${TMPF%.c}" "$TMPF" 2>/dev/null && [ -x "${TMPF%.c}" ]; then
        ok "clang -fsanitize=address works"
    else
        fail "ASAN build failed — may need: sudo apt install libasan8 (or similar)"
    fi
    rm -f "$TMPF" "${TMPF%.c}"
fi

# valgrind smoke test
if command -v valgrind &>/dev/null; then
    TMPF=$(mktemp /tmp/test_XXXXXX.c)
    echo 'int main(){return 0;}' > "$TMPF"
    if gcc -g -o "${TMPF%.c}" "$TMPF" 2>/dev/null; then
        if valgrind --error-exitcode=1 "${TMPF%.c}" &>/dev/null; then
            ok "valgrind works"
        else
            warn "valgrind installed but failed smoke test"
        fi
    fi
    rm -f "$TMPF" "${TMPF%.c}"
fi

# gdb smoke test
if command -v gdb &>/dev/null; then
    PTRACE=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo "?")
    if [[ "$PTRACE" == "0" ]]; then
        ok "ptrace_scope=0 — gdb attach works without restrictions"
    else
        warn "ptrace_scope=$PTRACE — gdb may fail to attach to running processes"
        info "Fix: echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope"
        info "Persist: add kernel.yama.ptrace_scope=0 to /etc/sysctl.d/10-ptrace.conf"
    fi
fi

# ─── Section 5: Missing Package Install Offer ────────────────────────────────
if (( ${#MISSING_PKGS[@]} > 0 )); then
    header "MISSING PACKAGES"
    # Deduplicate
    UNIQUE_PKGS=($(echo "${MISSING_PKGS[@]}" | tr ' ' '\n' | sort -u))
    APT_CMD="sudo apt update && sudo apt install -y ${UNIQUE_PKGS[*]}"
    info "Missing packages: ${UNIQUE_PKGS[*]}"
    info "Install command:"
    echo ""
    echo "  $APT_CMD"
    echo ""
    read -rp "Install now? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        eval "$APT_CMD"
        echo ""
        info "Packages installed. Re-run this script to verify."
    else
        info "Skipped. Run the command above manually when ready."
    fi
fi

# ─── Section 6: Final Verdict ────────────────────────────────────────────────
header "RESULT"

if (( ${#WARNINGS[@]} > 0 )); then
    echo ""
    printf "${YELLOW}Warnings:${NC}\n"
    for w in "${WARNINGS[@]}"; do
        echo "  - $w"
    done
fi

if (( ${#BLOCKERS[@]} > 0 )); then
    echo ""
    printf "${RED}${BOLD}BLOCKERS (must fix):${NC}\n"
    for b in "${BLOCKERS[@]}"; do
        echo "  - $b"
    done
    echo ""
    printf "${RED}${BOLD}NOT READY — fix the blockers above and rerun.${NC}\n"
    exit 1
else
    echo ""
    printf "${GREEN}${BOLD}========================================${NC}\n"
    printf "${GREEN}${BOLD}  READY FOR HW1  ${NC}\n"
    printf "${GREEN}${BOLD}========================================${NC}\n"
    exit 0
fi
