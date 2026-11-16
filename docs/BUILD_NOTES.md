# Build notes

How to get each engine onto a platform the suite does not have a prebuilt
artifact for. The default environment (`pixi.toml` + `pixi.lock`) covers
**linux-64** and **osx-arm64** out of the box: every dependency resolves to a
conda package or a PyPI wheel, and `pixi install` is all you need. These notes
are for the cases that fall outside that — chiefly **linux-aarch64 (ARM)**,
which the cross-platform timing gate of `docs/DESIGN.md` section 9.1 needs and
which PyPI/conda do not fully cover.

Nothing here changes the x86-64 path. The wheels the suite runs on today are
untouched; this is additive.

> **The one rule that governs every native build here — ABI consistency.** Build
> each engine's compiled core with the **same conda-forge toolchain that provides
> the `libstdc++` it will run against** — never the cluster's system `gcc`/`g++`.
> A C++ object built against one `libstdc++` and run against another is undefined
> behaviour, and the usual symptom is a **segfault at runtime, not a link error at
> build time** — so it passes every build step and then crashes in use. The sharp
> edge is `std::filesystem::path`, whose internal layout changed at the **GCC 9
> ABI break**: a core built with a system GCC 8 but run against a conda-forge
> GCC ≥ 9 `libstdc++` corrupts the heap the first time it handles a path (e.g. a
> `.net` filename) and crashes with no traceback. The suite's run environments are
> conda-forge (GCC 13-era `libstdc++`), so **every engine core here is built with
> the conda-forge compiler**, matching the x86 wheels' own toolchain. `bench
> doctor` now actively guards this: it loads a trivial model in a child process
> and reports the ABI mismatch by name if the core crashes, rather than letting a
> run reach a compute node and die silently.

> **Cluster specifics.** The commands below use placeholders — `<arm-partition>`,
> `<account>`, `<qos>` — for Slurm identifiers. Fill them from your site's
> allocation (on Darwin these live in the git-ignored `slurm.yaml` and `dev/`,
> never in a tracked file). Replace `$USER` as needed.

## Per-engine build matrix (aarch64)

| Engine | aarch64 availability | Build effort | Needs LLVM? |
| --- | --- | --- | --- |
| **BNGsim** (system under test) | x86-64 PyPI wheel only; **aarch64 builds from the `v0.17.2` git checkout** | moderate CMake (SUNDIALS vendored) | no |
| **run_network** (BioNetGen) | no aarch64 prebuilt; **build from 2.9.3 source** | moderate CMake | no |
| **AMICI** | PyPI **sdist** → `uv pip install amici` compiles it | moderate (SUNDIALS vendored) | no |
| **COPASI CLI** | plain CMake C++ build if no aarch64 prebuilt | moderate CMake | no |
| **libRoadRunner** | **no aarch64 wheel and no sdist on PyPI**; **built from source (done)** | moderate CMake | yes — satisfied by a prebuilt conda-forge LLVM |

libRoadRunner is the only engine that needs LLVM, but LLVM does **not** have to
be built from source: conda-forge ships a prebuilt `llvmdev` for linux-aarch64
at the exact version libRoadRunner wants (13.0.x), and RoadRunner's
`FindLLVM.cmake` discovers it through `llvm-config`. Installing that prebuilt
LLVM turns what used to be a multi-hour build into an ordinary ~6-minute CMake
build. run_network, AMICI and COPASI CLI are ordinary native builds with no LLVM
dependency.

## aarch64 cluster tooling

Verified on a ThunderX2 node (2026-11-13): aarch64, RHEL 8.10, glibc 2.28 (the
same floor as the x86 side), system gcc 8.5 with gcc up to 13.1 via Lmod; cmake,
ninja and git present; **no LLVM module** (only `llvm-compat-libs` runtime).

Grab an interactive ARM node:

```sh
srun -p <arm-partition> -A <account> -q <qos> -N1 -n1 -c 16 -t 00:25:00 bash -lc '...'
```

Two gotchas that will waste an afternoon if you hit them cold:

- **The `pixi`/`uv` on `PATH` are x86-64 binaries.** `~/.pixi/bin/pixi` and
  `~/.local/bin/uv` are x86-64 ELF and die with `Exec format error` on aarch64.
  Install a **native aarch64 pixi** to a separate prefix — e.g.
  `$HOME/.pixi-aarch64/bin/pixi` — and call it by full path on ARM. (Verified
  with pixi 0.84.2, static build.)

- **x86 user-site packages leak into the ARM env.** `~/.local/lib/python3.12`
  holds x86-64 numpy that Python picks up via user-site, so an ARM interpreter
  tries to load `_multiarray_umath.cpython-312-x86_64-linux-gnu.so` and fails
  with an `ImportError` about an incompatible C-extension. Always neutralize
  user-site on ARM:

  ```sh
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  ```

- **`/tmp` is node-local tmpfs and does not persist across allocations.** Put
  scratch builds and caches under `$HOME/.cache` (NFS, shared across
  architectures and allocations) so a second `srun` can still see them.

## BNGsim on aarch64 (from the git checkout, conda-forge toolchain)

PyPI ships only an x86-64 wheel for BNGsim, so on aarch64 the native core is
built locally. Two non-obvious requirements — miss either and the build
"succeeds" but the result is unusable:

1. **Build from the `v0.17.2` git checkout, not the PyPI sdist.** BNGsim's CMake
   bakes the build commit from `git rev-parse` into the compiled core, and that
   commit gates `bench doctor` READY, gates `bench submit`, and is part of the
   result freshness key. A PyPI sdist has no `.git`, so an sdist build reports a
   null commit — which both fails `doctor` and gives the ARM result a different
   fingerprint than the x86 wheel, defeating the "same build, two platforms"
   premise of the `docs/DESIGN.md` section 9.1 timing gate. Clone with
   `--recurse-submodules` at the tag so the commit matches the x86 wheel exactly.

2. **Build with the conda-forge compiler, never the node's system `g++`** (the
   ABI rule at the top of this file). A `uv pip install <path>` / `pip install
   <path>` picks up whatever `g++` is first on `PATH`, which on a RHEL 8 ARM node
   is the system **GCC 8.5** — a `std::filesystem` ABI that predates the GCC 9
   break. Run against the env's conda-forge `libstdc++` (GCC 13), that core
   segfaults the first time `Model.from_net` touches the `.net` path, with no
   traceback. Build inside the **same conda-forge toolchain** as the other
   engines and the ABI matches.

Pinned toolchain (native aarch64 pixi; mirrors the roadrunner one). The
`c-compiler`/`cxx-compiler` metapackages install the activation scripts that
export `$CC`/`$CXX`; `pip` must be present (scikit-build-core drives the build
through it):

```
channels = ["conda-forge"], platforms = ["linux-aarch64"]
python=3.12  pip  numpy  c-compiler  cxx-compiler  gcc=13.*  gxx=13.*
make  cmake>=3.18  ninja  pkg-config  scikit-build-core>=0.10  pybind11>=2.13
```

Build a wheel, then install **that wheel** (not a `{ path = <src> }` entry,
which would rebuild with the wrong compiler at install time):

```bash
git clone --branch v0.17.2 --recurse-submodules \
  https://github.com/lanl/bngsim.git <bngsim-src>

# Inside the conda-forge toolchain env, so $CC/$CXX point at the conda gcc 13.
# The build FetchContents SUNDIALS 7.2.1; pre-download it and name it with an
# absolute path so the build needs no network (its sha256 is checked against
# third_party/sundials/VENDOR.json):
export BNGSIM_SUNDIALS_ARCHIVE=<abs>/sundials-7.2.1.tar.gz
export BNGSIM_ENABLE_KLU=ON BNGSIM_REQUIRE_KLU=ON
export CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"
python -m pip wheel <bngsim-src> --no-deps --no-build-isolation -w <wheel-out>
```

Confirm the result before trusting it:

- `readelf -p .comment _bngsim_core*.so` names the **conda-forge gcc**, not the
  system GCC 8.
- `bench doctor` prints `OK bngsim core loads a model (from_net)` (the native
  load self-test) and the baked `build_commit` matches the x86 wheel's.

## run_network on aarch64 (BioNetGen 2.9.3 from source)

The BioNetGen 2.9.3 **release tarball is x86-only and ships no source**, so the
source has to come from the tagged commit in the repository. The build vendors
its own solvers (CVODE 2.6.0 + muparser via `bng2/libsource`) — there is no
external SUNDIALS, no network access at build time, and no LLVM.

```sh
# 1. Source from the tag (NOT the release tarball).
curl -fsSL https://github.com/RuleWorld/bionetgen/archive/refs/tags/BioNetGen-2.9.3.tar.gz \
  | tar xz
cd bionetgen-BioNetGen-2.9.3/bng2/Network3

# 2. Unpack the bundled solver sources in place.
tar xzf ../libsource/cvode-2.6.0.tar.gz -C .
tar xzf ../libsource/muparser_v2_2_4.tar.gz -C .
```

**The one real gotcha.** CVODE 2.6.0 and muparser ship an autotools
`config.guess` too old to recognize aarch64, so `configure` aborts with
`configure: error: cannot guess build type`. The fix is to pass an explicit
`--build` triple (configure skips `config.guess` when `--build` is given). The
`Network3` Makefile **re-extracts cvode on every run** (`rm -rf … && tar -xzf …`),
so patch the two `./configure` invocations in the **Makefile**, not in the
extracted tree:

```
./configure  →  ./configure --build=aarch64-unknown-linux-gnu
```

Then build:

```sh
cp Makefile.cmake Makefile    # the aarch64-ready driver Makefile
make -j16                      # binary lands at bng2/bin/run_network
```

Notes:

- The `auto_ptr` line in the output is a benign C++17 deprecation warning, not a
  build error.
- **Verified numerically** on aarch64: a decay model integrates to
  `100·e⁻¹ = 36.78794` to seven digits, the solver reports
  `cvode using dense LU`, and the binary identifies as `run_network 3.0`.
- Unlike the x86-64 prebuilt binary, this native build **needs no glibc loader
  swap** — it links against the node's own glibc 2.28. (The loader swap in
  `pixi run fetch-glibc` exists only because the *prebuilt* 2.9.3 binary wants
  glibc ≥ 2.34; see the freshness note in `docs/DESIGN.md` section 6.)

## libRoadRunner on aarch64 (built from source)

No `linux-aarch64` wheel and no sdist exist on PyPI for 2.11.0, so it is built
from `sys-bio/roadrunner` at tag `v2.11.0`. This does not block the x86-64 suite,
which uses the published wheel.

The one piece that looks heavy — LLVM — does **not** need to be built from
source. conda-forge publishes a prebuilt `llvmdev` for linux-aarch64 at the
version RoadRunner wants, and RoadRunner finds it via `llvm-config`. With that,
the whole thing is two ordinary CMake builds and finishes in minutes, not hours.

### Pins (verified against roadrunner v2.11.0's own CI, `.github/workflows/main.yml`)

- **roadrunner** `v2.11.0`.
- **libroadrunner-deps** `v2.3.7` — must match the roadrunner tag. RoadRunner
  checks the SUNDIALS version it finds, so a mismatched deps revision fails the
  configure (upstream issue #1403). The deps repo vendors everything it needs as
  git submodules (libSBML 5.21.2, SUNDIALS, POCO, CLAPACK, nleq1/2, rr-libstruct,
  expat, zlib, bzip2, NuML, libSEDML, SBMLNetwork), so a `--recurse-submodules`
  clone is self-contained — no per-dependency hunting.
- **LLVM 13.0.x** — RoadRunner 2.11.0 is an LLVM-13 codebase (MCJIT/OrcJIT). Use
  conda-forge `llvmdev=13.0.1` for linux-aarch64 (prebuilt, PIC- and
  RTTI-enabled, so its static archives link into the Python extension). This is
  the step that would otherwise be an LLVM-from-source compile.
- **SWIG 4.x** for the Python bindings (the docs say "3.0.0" but that is stale;
  CI uses 4.x). **C++17.** **No Fortran compiler** anywhere — nleq1/2 and CLAPACK
  are f2c-translated C, and SuiteSparse/KLU is not part of deps `v2.3.7`.

### Toolchain (one env, for ABI consistency)

Build libroadrunner-deps *and* roadrunner with a single conda-forge aarch64
toolchain — the same family that built `llvmdev` — so libstdc++ ABI is
consistent across LLVM, the deps, and the final `.so`. Use the
**`c-compiler` / `cxx-compiler` metapackages**, not bare `gcc`/`gxx`: only the
metapackages install the conda activation scripts that export `$CC`/`$CXX`.
Without them CMake fails immediately with "No CMAKE_C_COMPILER could be found".
A pinned env (native aarch64 pixi) that works:

```
channels   = ["conda-forge"], platforms = ["linux-aarch64"]
python=3.12  c-compiler  cxx-compiler  gcc=13.*  make  cmake>=3.18  ninja
pkg-config  swig>=4.0.2  llvmdev=13.0.1.*  zlib  ncurses  libxml2  numpy
```

### Build

```bash
# 1. deps (self-contained; vendors libSBML/SUNDIALS/POCO/CLAPACK/nleq/... as submodules)
git clone --branch v2.3.7 --recurse-submodules \
  https://github.com/sys-bio/libroadrunner-deps.git
cmake -S libroadrunner-deps -B libroadrunner-deps/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<deps-install> \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build libroadrunner-deps/build --target install

# 2. roadrunner against the deps tree + the conda-forge LLVM
git clone --branch v2.11.0 https://github.com/sys-bio/roadrunner.git
cmake -S roadrunner -B roadrunner/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<rr-install> \
  -DRR_DEPENDENCIES_INSTALL_PREFIX=<deps-install> \
  -DLLVM_INSTALL_PREFIX=<conda-env-prefix> \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_PYTHON=ON -DPython_ROOT_DIR=<conda-env-prefix> \
  -DSWIG_EXECUTABLE=<conda-env-prefix>/bin/swig \
  -DBUILD_TESTS=OFF -DBUILD_RR_PLUGINS=OFF -DBUILD_EXAMPLES=OFF
cmake --build roadrunner/build --target install
```

The importable package lands at `<rr-install>/site-packages/roadrunner`; add
that `site-packages` to `PYTHONPATH`.

### Two things that trip up downstream use

- **dist-info.** The CMake install drops the `roadrunner` package but no
  `*.dist-info`, so `importlib.metadata.version("libroadrunner")` — how the suite
  reads the engine version — returns nothing. Write a minimal
  `libroadrunner-<ver>.dist-info/METADATA` (`Name: libroadrunner`,
  `Version: 2.11.0`) next to the package.
- **libpython.** The `_roadrunner.so` needs the env's `libpython3.12` at import;
  the suite already preloads it with `RTLD_GLOBAL` (`src/bench/env.py`). Outside
  the suite, ensure the conda env's `lib/` is reachable.

### Verified on aarch64 (ThunderX2, 2026-11-16)

`roadrunner.__version__ == 2.11.0`; `getVersionStr` reports
`JIT Compiler: LLVM-13.0`, `gcc 13.4.0`, `LibSBML Version: 5.21.2`; a first-order
decay integrates to `100·e⁻¹ = 36.78794` under CVODE. End-to-end build was
~6 minutes on 48 cores.

### Wiring the source build into the suite (`$BENCH_ROADRUNNER_PATH`)

The tracked environment pins `libroadrunner` as a wheel, which does not exist
for linux-aarch64, so on ARM the suite is pointed at the source build instead of
the (absent) wheel. Set **`BENCH_ROADRUNNER_PATH`** to the build's
`site-packages` directory — the one holding the `roadrunner` package and the
synthesized `libroadrunner-<ver>.dist-info`:

```sh
export BENCH_ROADRUNNER_PATH=<rr-install>/site-packages
```

The suite prepends it to `sys.path` in-process (`src/bench/env.py`), so it does
**not** go through `$PYTHONPATH` — on ARM that variable must stay empty (x86
user-site numpy leaks onto the ARM interpreter through it; see the tooling
gotchas above). `importlib.metadata.version("libroadrunner")` then resolves from
the synthesized dist-info, and `env.roadrunner_identity()` records a `source`
field so an ARM result is never confused with a wheel one. Two things make this
work without any `LD_LIBRARY_PATH` plumbing:

- The built `_roadrunner.so` carries an **absolute RPATH** into the build
  toolchain's `lib/`, so `libLLVM-13.so`, `libstdc++.so.6`, and
  `libpython3.12.so.1.0` all resolve from there automatically.
- Its `NEEDED libpython3.12.so.1.0` is satisfied by the run env's own
  Python 3.12 (the suite preloads libpython with `RTLD_GLOBAL`), so the run env
  and the build env share a Python minor version.

The run environment itself is an ordinary aarch64 conda/pixi env carrying what
the ODE capability needs — `python=3.12`, `perl`, `numpy`, `pyyaml`, and
`python-libsbml` (the SBML reader; it is on conda-forge for aarch64), plus the
locally built **BNGsim wheel** (see "BNGsim on aarch64" above — install the
prebuilt wheel, **not** a `{ path = <src> }` entry or the PyPI sdist, either of
which rebuilds the core with the wrong compiler and segfaults). Confirm it with
`bench doctor`, which now both imports roadrunner and runs the native load
self-test on BNGsim's core.

## AMICI / COPASI CLI (future reference engines)

Neither is wired into the suite yet (they are drop-in `REFERENCE_ENGINES`
entries per `docs/DESIGN.md` section 5.2). For aarch64 when the time comes:

- **AMICI** publishes a PyPI **sdist**, so `uv pip install amici` compiles it
  locally (SUNDIALS is vendored; no LLVM). A native toolchain is enough.
- **COPASI CLI** is a plain CMake C++ build from source if no aarch64 prebuilt
  is available (no LLVM).
