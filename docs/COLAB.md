# Running Propwash on Google Colab

The repository is public, so getting it onto a runtime is one cell.

> Jump to [Troubleshooting](#troubleshooting) if something has already gone
> wrong — in particular if you saw
> `could not read Username for 'https://github.com'`.

---

## Step 1 — pick a GPU runtime, *first*

*Runtime → Change runtime type → T4 GPU → Save.*

Do this **before** cloning. Changing the runtime type restarts the VM and wipes
`/content`, so doing it afterwards means cloning again.

Everything works on a CPU runtime too — you just get the NumPy and Numba
backends instead of the CUDA ones.

---

## Step 2 — clone

```python
!git clone --depth 1 https://github.com/abu-infidel/Python-based-BEMT-visualizer-calculator.git /content/propwash
%cd /content/propwash
```

`--depth 1` skips the history; you only want the files.

---

## Step 3 — install and see what this host can do

```python
import propwash.colab as pc
env = pc.setup()
```

`setup()` installs only what is missing and is safe to re-run — a second call
costs milliseconds, not a re-download. It installs CuPy only when it actually
sees a GPU.

On a T4 you should see roughly:

```
Propwash environment
  Python        3.11.x
  Host          Google Colab (notebook)
  CPU / RAM     2 cores, 12.7 GiB
  GPU           Tesla T4, 15360 MiB, 7.5, 550.54.15
  Compute backends
    cupy         [ok] Tesla T4, CuPy 13.x, SM 7.5, 40 SMs
    numba-cuda   [ok] Tesla T4, SM 7.5, 14.7 GiB (14.6 GiB free), 40 SMs
    numba-cpu    [ok] Numba 0.60.x, 2 threads
    numpy        [ok] NumPy 1.26.x, vectorised reference path
  -> selected   cupy
```

`GPU  none detected` means you are on a CPU runtime — go back to Step 1.

An `[unavailable]` backend always says *why*, and the next one down is used;
nothing here fails because a GPU is absent.

---

## Step 4 — the capacity test

This is the short version of what the whole project is for:

```python
from propwash.accel.bench import benchmark
print(benchmark(n_cases=65536, n_elements=64, verbose=False).format())
```

Every available backend solves the same grid and is diffed against the NumPy
reference, so correctness is printed next to speed — a GPU path that is a
hundred times faster and half a percent wrong is a bug, not a win.

To find the actual ceiling, walk it up until something breaks:

```python
for n in (2**14, 2**16, 2**18, 2**20):
    print(benchmark(n_cases=n, n_elements=64, verbose=False).format(), "\n")
```

Spanwise memory is `n_cases × n_elements × 8 bytes` **per field**, so 2²⁰ cases
at 64 elements asks for ~0.5 GB each. The out-of-memory error from CuPy or
Numba is the limit you were looking for. To push case count without the memory,
pass `want_spanwise=False` through the backend directly — only thrust and
torque come back, and the per-case cost collapses.

---

## Step 5 — the GUI

```python
lab = pc.launch()
```

Controls on the left, tabs on the right. Worth touching first:

- **Flight condition → Motor-matched** is on by default, which makes *Shaft
  speed* read-only: it shows where the propeller actually settles, at the RPM
  where its torque demand meets the motor's torque supply.
- **Propeller → Collective** is the instructive slider. Add pitch and thrust
  rises while RPM comes *down* and current climbs.
- **View → Colour by** maps any solver field onto the blade.
- **View → Spin** animates at a rate proportional to the solved RPM.

Or open `notebooks/Propwash_Propeller_Lab.ipynb` straight from GitHub:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abu-infidel/Python-based-BEMT-visualizer-calculator/blob/main/notebooks/Propwash_Propeller_Lab.ipynb)

---

## Troubleshooting

### `fatal: could not read Username for 'https://github.com'`

```
Cloning into 'propwash'...
fatal: could not read Username for 'https://github.com': No such device or address
[Errno 2] No such file or directory: 'propwash'
ModuleNotFoundError: No module named 'propwash'
```

The repository was private when you ran that. `git clone` over HTTPS tried to
prompt for credentials, a Colab cell has no terminal to type into, and git gave
up. Nothing downloaded, so `%cd` had nothing to enter and the import failed for
the obvious reason — three errors, one cause.

This repository is public now, so a plain clone works. If you make it private
again, see [Appendix: cloning a private repo](#appendix--cloning-a-private-repo).

### The GUI cell runs but the output is blank

Colab sandboxes each output cell and blocks any widget beyond the built-in
ipywidgets set — which includes Plotly's `FigureWidget`. `pc.launch()` enables
the custom widget manager for you, but if you constructed `PropwashLab`
directly:

```python
from google.colab import output
output.enable_custom_widget_manager()
```

Then re-run the cell.

### The 3-D view is a static image with a "spin" button

That is the documented fallback for when `FigureWidget` is unavailable, usually
a missing `anywidget` on Plotly 6+. `pc.setup()` installs it. The fallback is
fully functional — the animation just runs client-side instead of being driven
from Python.

### `ModuleNotFoundError: No module named 'propwash'`

The clone did not land where Python is looking. Check:

```python
import pathlib
print(list(pathlib.Path("/content").rglob("propwash/__init__.py")))
```

Empty means Step 2 failed — scroll up to its output for the real error. If it
is non-empty, add that file's grandparent to `sys.path`:

```python
import sys, pathlib
root = next(p.parent for p in pathlib.Path("/content").rglob("propwash/__init__.py"))
sys.path.insert(0, str(root))
```

### CuPy fails to import with a CUDA version mismatch

Colab's driver moves occasionally. `pc.setup()` installs `cupy-cuda12x`; if the
runtime is on CUDA 11, install `cupy-cuda11x` instead. Either way `numba-cuda`
still works, so this costs one backend, not the GPU.

### Everything is slow and the CUDA backends are missing

You are on a CPU runtime, or you have hit Colab's GPU quota. `pc.probe()` says
which.

### The session disconnects mid-benchmark

Colab reclaims idle runtimes and caps session length on the free tier. Use a
smaller `n_cases` or keep the tab focused.

---

## Appendix — cloning a private repo

Kept because this repository was private until recently, and a fork of it might
be.

**Never paste a token into a cell.** Notebook source and output are saved to
Drive and travel with a share link. Use Colab's secret store instead.

1. Make a **fine-grained** personal access token: GitHub → Settings →
   Developer settings → Personal access tokens → Fine-grained tokens.
   - *Repository access*: only the one repository
   - *Permissions*: Repository permissions → **Contents: Read-only**

   That is all this needs. Do not use a classic token with `repo` scope — it
   grants write access to everything you own.

2. Colab → **key icon** in the left sidebar → **Add new secret** → name it
   `GITHUB_TOKEN`, paste the value, enable *Notebook access*.

3. Run:

```python
import subprocess, sys
from google.colab import userdata

TOKEN = userdata.get("GITHUB_TOKEN")
REPO  = "abu-infidel/Python-based-BEMT-visualizer-calculator"
DEST  = "/content/propwash"

# capture_output keeps the tokenised URL out of the saved notebook output
result = subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://x-access-token:{TOKEN}@github.com/{REPO}.git", DEST],
    capture_output=True, text=True)
if result.returncode != 0:
    raise SystemExit("clone failed -- check the secret name, the token's expiry, "
                     "and that it can read this repository")

# Strip the token back out of .git/config so it is not left on disk.
subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin",
                f"https://github.com/{REPO}.git"], check=True)

sys.path.insert(0, DEST)
print("cloned to", DEST)
```

---

## Without a notebook

```bash
git clone https://github.com/abu-infidel/Python-based-BEMT-visualizer-calculator.git
cd Python-based-BEMT-visualizer-calculator
pip install -e ".[notebook]"

python -m propwash env                  # what can this machine do?
python -m propwash bench --cases 65536  # benchmark and cross-validate
python -m propwash cuda-check           # compile the CUDA C via NVRTC, no GPU needed
```
