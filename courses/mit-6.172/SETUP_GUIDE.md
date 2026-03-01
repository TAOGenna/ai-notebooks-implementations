# MIT 6.172 — WSL2 Setup Guide

## Current Status (All Green)

| Check            | Result                                           |
|------------------|--------------------------------------------------|
| WSL version      | WSL2 (kernel 5.15.167.4-microsoft-standard-WSL2) |
| OS               | Ubuntu 24.04.3 LTS (Noble)                       |
| Arch             | x86_64                                            |
| CPU              | i5-1135G7 @ 2.40GHz, 8 cores                     |
| RAM              | 4.8 GB                                            |
| Disk             | 256 GB SSD (WSL2 vhdx reports ~919 GB virtual)    |
| git              | 2.43.0                                            |
| make             | 4.3                                               |
| gcc              | 13.3.0                                            |
| clang            | 18.1.3                                            |
| llvm-cov         | 18.1.3                                            |
| python3          | 3.12.3                                            |
| gdb              | 15.0.50                                           |
| valgrind         | 3.22.0                                            |
| cmake            | 3.28.3                                            |
| perf             | 6.6.114 (built from WSL2 kernel source)           |

> **Note on ptrace_scope:** The checker warns about `ptrace_scope=?` — this is a false positive.
> The WSL2 kernel does not have Yama LSM enabled, so there are no ptrace restrictions. gdb works without issues.

---

## How This Was Set Up

### Step 1: Core packages

```bash
sudo apt update && sudo apt install -y \
  git make gcc clang llvm \
  gdb valgrind cmake \
  python3 python3-pip \
  build-essential
```

### Step 2: perf (built from WSL2 kernel source)

The Ubuntu `linux-tools-generic` package ships perf for the Ubuntu kernel, not the WSL2 kernel. To get a working perf on WSL2:

```bash
# Build dependencies
sudo apt install -y flex bison libelf-dev libdw-dev libunwind-dev libslang2-dev \
  libtraceevent-dev libdebuginfod-dev libcap-dev libnuma-dev libperl-dev \
  python3-dev systemtap-sdt-dev libbabeltrace-dev libpfm4-dev

# Build from WSL2 kernel source
cd /tmp
git clone --depth 1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
make -j$(nproc)
sudo cp perf /usr/local/bin/perf
rm -rf /tmp/WSL2-Linux-Kernel

# Allow non-root usage
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### Step 3: Increase WSL2 RAM (Recommended)

Edit (or create) `C:\Users\<YourWindowsUsername>\.wslconfig` in Windows:
```ini
[wsl2]
memory=8GB
swap=4GB
processors=8
```
Then restart WSL from PowerShell:
```powershell
wsl --shutdown
```

---

## HW Smoke Test

Assumes you downloaded and unzipped a homework into `~/mit-6.172/hwN`.

### Build
```bash
cd ~/mit-6.172/hw1
make
```

### Run the main binary
```bash
# Check what the Makefile builds (common names: isort, matrix_multiply, etc.)
ls -la *.out 2>/dev/null || ls -la isort 2>/dev/null || find . -maxdepth 1 -executable -type f
# Then run it:
./isort          # or whatever the binary is called
```

### GDB
```bash
gdb --args ./isort
# Inside gdb:
#   (gdb) break main
#   (gdb) run
#   (gdb) next
#   (gdb) print variable_name
#   (gdb) quit
```

### AddressSanitizer (ASAN)
```bash
make clean
make ASAN=1
./isort
# If the Makefile doesn't support ASAN=1, build manually:
# clang -fsanitize=address -g -O1 -fno-omit-frame-pointer -o isort isort.c
```

### Valgrind
```bash
make clean && make    # rebuild without ASAN (they conflict)
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./isort
```

### perf
```bash
perf stat ./isort
perf record -g ./isort && perf report
```

### Coverage (llvm-cov / gcov)
```bash
# Option A: gcc + gcov
make clean
gcc -fprofile-arcs -ftest-coverage -g -o isort isort.c
./isort
gcov isort.c
cat isort.c.gcov

# Option B: clang + llvm-cov (source-based coverage)
make clean
clang -fprofile-instr-generate -fcoverage-mapping -g -o isort isort.c
LLVM_PROFILE_FILE="isort.profraw" ./isort
llvm-profdata merge -sparse isort.profraw -o isort.profdata
llvm-cov show ./isort -instr-profile=isort.profdata
llvm-cov report ./isort -instr-profile=isort.profdata
```

---

## Tapir / OpenCilk (needed for HW2+)

MIT 6.172 (Fall 2018) used **Tapir/LLVM** — a modified clang with Cilk support (`cilk_spawn`, `cilk_sync`, `cilk_for`). This is NOT in standard clang.

### How to detect if an assignment needs Cilk:
```bash
# Check for Cilk keywords in source:
grep -rn 'cilk_spawn\|cilk_sync\|cilk_for\|#include.*cilk' *.c *.h 2>/dev/null

# Check Makefile for Tapir-specific flags:
grep -n 'fcilkplus\|ftapir\|cilk\|TAPIR\|opencilk' Makefile 2>/dev/null
```

### Install OpenCilk (recommended modern replacement)

```bash
# Download OpenCilk 2.1 (based on LLVM 16)
cd /tmp
wget https://github.com/OpenCilk/opencilk-project/releases/download/opencilk%2Fv2.1/OpenCilk-2.1.0-x86_64-Linux-Ubuntu-22.04.sh
chmod +x OpenCilk-2.1.0-x86_64-Linux-Ubuntu-22.04.sh
sudo ./OpenCilk-2.1.0-x86_64-Linux-Ubuntu-22.04.sh --prefix=/opt/opencilk --skip-license

# Add to PATH
echo 'export PATH="/opt/opencilk/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
/opt/opencilk/bin/clang --version

# Point HW Makefile at it:
# Edit Makefile: CC = /opt/opencilk/bin/clang
# Or: make CC=/opt/opencilk/bin/clang
```

---

## Troubleshooting

| # | Problem | Symptom | Fix |
|---|---------|---------|-----|
| 1 | **valgrind unsupported instruction** | `vex amd64->IR: unhandled instruction` on newer CPUs | Upgrade valgrind: `sudo apt install valgrind` (3.22+ handles most AVX-512). If still failing: build valgrind from git HEAD |
| 2 | **perf kernel mismatch** | `WARNING: perf not found for kernel` | Don't use `linux-tools-generic`. Build perf from WSL2 kernel source (see Step 2 above) |
| 3 | **perf permission denied** | `perf_event_open failed: Permission denied` | `echo -1 \| sudo tee /proc/sys/kernel/perf_event_paranoid` or run `sudo perf ...` |
| 4 | **llvm-cov not found** | `llvm-cov: command not found` after installing llvm | Versioned binary: `sudo ln -s /usr/bin/llvm-cov-18 /usr/local/bin/llvm-cov` |
| 5 | **Makefile expects Tapir clang** | `unknown argument: '-fcilkplus'` or `cilk_spawn undeclared` | Install OpenCilk (see above). Point `CC=/opt/opencilk/bin/clang` |
| 6 | **ASAN + valgrind conflict** | Crash or false positives running valgrind on ASAN binary | Never combine them. `make clean` between ASAN and valgrind runs |
| 7 | **clock_gettime link error** | `undefined reference to clock_gettime` | Add `-lrt` to LDFLAGS in Makefile |
| 8 | **WSL2 RAM exhaustion** | OOM killer terminates builds | Create `.wslconfig` (see Step 3). Use `make -j4` instead of `-j$(nproc)` |
| 9 | **File permission issues** | `Permission denied` on executables | Work in `~/`, NOT `/mnt/c/`. Windows filesystem has permission mismatches |
| 10 | **Missing llvm-profdata** | `llvm-profdata: command not found` | `sudo apt install llvm` or symlink: `sudo ln -s /usr/bin/llvm-profdata-18 /usr/local/bin/llvm-profdata` |

---

## Quick Reference: Full Install From Scratch

```bash
# Core tools
sudo apt update && sudo apt install -y \
  git make gcc clang llvm \
  gdb valgrind cmake \
  python3 python3-pip \
  build-essential

# perf build dependencies
sudo apt install -y flex bison libelf-dev libdw-dev libunwind-dev libslang2-dev \
  libtraceevent-dev libdebuginfod-dev libcap-dev libnuma-dev libperl-dev \
  python3-dev systemtap-sdt-dev libbabeltrace-dev libpfm4-dev

# Build perf from WSL2 kernel source
cd /tmp && git clone --depth 1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf && make -j$(nproc)
sudo cp perf /usr/local/bin/perf
rm -rf /tmp/WSL2-Linux-Kernel

# Permissions
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Verify everything:
bash ~/mit-6.172/check_6172.sh
```
