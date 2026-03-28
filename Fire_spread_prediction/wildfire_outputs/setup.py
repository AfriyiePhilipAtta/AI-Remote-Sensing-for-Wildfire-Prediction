"""
╔══════════════════════════════════════════════════════════════════════╗
║  WILDFIRE PIPELINE — ENVIRONMENT SETUP                              ║
║  Upper West Ghana · ConvLSTM · Gradient Boosting · Random Forest    ║
║                                                                      ║
║  USAGE:                                                              ║
║    conda activate ghana_fire_env                                     ║
║    python setup_environment.py                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import subprocess
import sys
import importlib

# ── Colour helpers ────────────────────────────────────────────────────────────
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"


def pip_install(pip_name, no_deps=False):
    """
    Install a package using THE SAME Python running this script.
    sys.executable = /opt/anaconda3/envs/ghana_fire_env/bin/python
    so packages always land in the active conda environment.

    skip_upgrade=True  →  only install if missing; never attempt an upgrade
                          that could trigger unwanted dependency resolution.
    no_deps=True       →  pass --no-deps to pip (safe for already-satisfied pkgs).
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.append(pip_name)
    subprocess.run(cmd, check=True)


def pip_install_if_missing(import_name, pip_name, no_deps=False):
    """
    Only call pip if the package cannot be imported.
    This avoids triggering pip's dependency resolver for packages that are
    already correctly installed — which was the root cause of the geemap/pandas
    downgrade attempt warning in the previous run.
    """
    try:
        importlib.import_module(import_name)
        return True   # already importable — nothing to do
    except ImportError:
        pass

    try:
        pip_install(pip_name, no_deps=no_deps)
        return True
    except subprocess.CalledProcessError:
        return False


# ── Package manifest ──────────────────────────────────────────────────────────
#
#  (display_name, import_name, pip_name, no_deps, why_needed)
#
#  no_deps=True is set for packages that already have all their deps satisfied
#  in the environment and whose upgrade path would try to downgrade other pkgs.
#
PACKAGES = [
    # display               import           pip name            no_deps  reason
    ("numpy",               "numpy",         "numpy",            False,  "array & tensor math, ConvLSTM"),
    ("pandas",              "pandas",        "pandas",           False,  "CSV loading, groupby, time series"),
    ("scipy",               "scipy",         "scipy",            False,  "gaussian_filter, KDE, pearsonr"),
    ("scikit-learn",        "sklearn",       "scikit-learn",     False,  "GradientBoosting, RandomForest, metrics"),
    ("imbalanced-learn",    "imblearn",      "imbalanced-learn", False,  "SMOTE oversampling for class imbalance (26:1 ratio)"),
    ("matplotlib",          "matplotlib",    "matplotlib",       False,  "all 30 spatial maps & figures"),
    ("seaborn",             "seaborn",       "seaborn",          False,  "heatmaps, calibration plots"),
    ("tqdm",                "tqdm",          "tqdm",             False,  "progress bars"),
    ("earthengine-api",     "ee",            "earthengine-api",  False,  "Google Earth Engine extraction"),
    # geemap: install with --no-deps when already present to prevent pip from
    # attempting to downgrade pandas (3.x → 1.5.x) which breaks the build.
    ("geemap",              "geemap",        "geemap",           True,   "interactive GEE visualisation"),

    # ── PyTorch (added for ConvLSTM backend) ─────────────────────────────────
    # torch     : core deep learning framework — replaces manual NumPy ConvLSTM
    # torchvision is optional but included for future image-based feature work
    # NOTE: pip installs the CPU-only build by default.
    #       For GPU support, replace "torch" with:
    #       "torch --index-url https://download.pytorch.org/whl/cu121"
    ("torch (PyTorch)",     "torch",         "torch",            False,  "ConvLSTM model, autograd, Adam optimiser"),
    ("torchvision",         "torchvision",   "torchvision",      False,  "optional — image transforms & future CNN features"),
]

PROTOBUF_PIN = "protobuf>=3.20.0,<5"


# ══════════════════════════════════════════════════════════════════════════════
print(bold("\n🔥  Wildfire Pipeline — Environment Setup"))
print(f"    Python  : {sys.executable}")
print(f"    Version : {sys.version.split()[0]}\n")
print("=" * 66)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — PyTorch GPU check (informational only)
# ══════════════════════════════════════════════════════════════════════════════
print(bold("\n0️⃣   PyTorch GPU check (informational)..."))
try:
    import torch as _torch
    if _torch.cuda.is_available():
        gpu = _torch.cuda.get_device_name(0)
        print(f"  {green(f'✅  CUDA GPU detected: {gpu}')}")
        print(f"     ConvLSTM will train on GPU automatically.\n")
    else:
        print(f"  {yellow('⚠️   No CUDA GPU detected — PyTorch will use CPU.')}")
        print(f"     Training will still work but will be slower.\n")
        print(f"     For GPU support install the CUDA build manually:")
        print(f"     pip install torch --index-url https://download.pytorch.org/whl/cu121\n")
except ImportError:
    print(f"  {yellow('⚠️   PyTorch not yet installed — will be installed below.')}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Install only what is missing (no unnecessary upgrades)
# ══════════════════════════════════════════════════════════════════════════════
print(bold("\n1️⃣   Installing / verifying packages...\n"))

install_failed = []

for display, import_name, pip_name, no_deps, reason in PACKAGES:
    print(f"  ▸ {display:<30s}  ({reason})")

    # Check if already importable first
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"    {green(f'✅  already installed  v{ver}')}\n")
        continue
    except ImportError:
        pass

    # Not present — install it
    try:
        pip_install(pip_name, no_deps=no_deps)
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"    {green(f'✅  installed  v{ver}')}\n")
    except Exception as e:
        print(f"    {red(f'❌  failed: {e}')}\n")
        install_failed.append((display, pip_name))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Pin protobuf only if it needs updating
# ══════════════════════════════════════════════════════════════════════════════
print(bold("2️⃣   Checking protobuf pin (earthengine-api stability)..."))
try:
    import google.protobuf as _pb
    ver_parts = tuple(int(x) for x in _pb.__version__.split(".")[:2])
    if (3, 20) <= ver_parts < (5, 0):
        print(f"  {green(f'✅  protobuf v{_pb.__version__} already in safe range')}\n")
    else:
        pip_install(PROTOBUF_PIN)
        print(f"  {green('✅  protobuf re-pinned to >=3.20.0,<5')}\n")
except Exception as e:
    print(f"  {yellow(f'⚠️   protobuf check failed: {e}')}\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Final import verification
# ══════════════════════════════════════════════════════════════════════════════
print(bold("3️⃣   Final import verification...\n"))

all_ok         = True
failed_imports = []

importlib.invalidate_caches()   # flush after any installs above

for display, import_name, pip_name, _, _ in PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {green('✅')}  {display:<30s}  v{ver}")
    except ImportError:
        print(f"  {red('❌')}  {display:<30s}  NOT importable")
        all_ok = False
        failed_imports.append((display, pip_name))

# ── Extra PyTorch info after verification ─────────────────────────────────────
print()
try:
    import torch as _torch
    cuda_status = (f"CUDA available — GPU: {_torch.cuda.get_device_name(0)}"
                   if _torch.cuda.is_available() else "CPU only")
    print(f"  {yellow('ℹ️')}   PyTorch device status: {cuda_status}")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Auto-retry failures with --no-deps as last resort
# ══════════════════════════════════════════════════════════════════════════════
if failed_imports:
    print(bold(f"\n4️⃣   Retrying {len(failed_imports)} failed package(s) "
               "with --no-deps...\n"))
    for display, pip_name in failed_imports:
        print(f"  ▸ {display} …")
        try:
            pip_install(pip_name, no_deps=True)
            importlib.invalidate_caches()
            importlib.import_module(pip_name.replace("-", "_"))
            print(f"    {green('✅  success')}")
        except Exception as e:
            print(f"    {red(f'❌  still failing: {e}')}")
            print(f"       👉  Run manually:  pip install {pip_name}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print(bold("🎉  SETUP COMPLETE"))
print("=" * 66)

if all_ok and not install_failed:
    print(green("✅  All packages installed and verified.\n"))
else:
    print(yellow("⚠️   Some packages had issues — check the log above.\n"))

print("Next steps:")
print("   conda activate ghana_fire_env:")
print("  1️⃣   Authenticate Google Earth Engine (run once):")
print("        earthengine authenticate\n")
print("  2️⃣   Extract data from GEE:")
print("        python gee_extract.py\n")
print("  3️⃣   Run the full modeling pipeline:")
print("        python wildfire_pipeline_pytorch.py\n")
print("🔥  Environment ready for Upper West Fire Spread modeling.")