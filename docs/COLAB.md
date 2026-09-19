# Running Propwash on Google Colab

Start here if the quick start in the README failed. The common failure has
nothing to do with the code.

---

## If you saw `could not read Username for 'https://github.com'`

```
Cloning into 'propwash-repo'...
fatal: could not read Username for 'https://github.com': No such device or address
[Errno 2] No such file or directory: 'propwash-repo'
ModuleNotFoundError: No module named 'propwash'
```

**This repository is private.** `git clone` over HTTPS tried to prompt for a
username and password, and a Colab cell has no terminal to type them into, so
git gave up. Nothing was downloaded, `%cd` had no directory to enter, and the
import then failed for the obvious reason. The three errors are one problem.

Pick one of the three fixes below. They are ordered by how little can go wrong.

---

## Option A — upload a ZIP (no tokens, works immediately)

Best for a one-off run, and the only option that involves no credentials at all.

1. On GitHub open the repo → green **Code** button → **Download ZIP**.
2. In Colab, run this cell and pick the ZIP when the file dialog appears:

```python
import zipfile, pathlib, sys
from google.colab import files

uploaded = files.upload()                      # choose the downloaded .zip
name = next(iter(uploaded))
with zipfile.ZipFile(name) as z:
    z.extractall("/content")

# GitHub's ZIP wraps everything in <repo>-<branch>/, so find the real root.
root = next(p.parent for p in pathlib.Path("/content").rglob("propwash/__init__.py"))
sys.path.insert(0, str(root))
print("propwash root:", root)
```

Then go to [Step 2](#step-2--pick-a-gpu-runtime).

---

## Option B — clone with a token from Colab Secrets (best for repeat use)

Do this if you will open the notebook more than once. The token lives in
Colab's secret store, not in the notebook, so it is not saved into the `.ipynb`
and does not travel with a share link.

**Never paste a token directly into a cell.** Notebook outputs and cell source
are saved to your Drive and included when you share.

1. Make a **fine-grained personal access token**:
   GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → **Generate new token**.
   - *Repository access*: Only select repositories → `collab-test-`
   - *Permissions*: Repository permissions → **Contents: Read-only**
   - Set the shortest expiry you can live with.

   Read-only on one repository is all this needs. Do not use a classic token
   with `repo` scope — that grants write access to everything you own.

2. In Colab, open the **key icon** in the left sidebar (Secrets) →
   **Add new secret** → name it `GITHUB_TOKEN`, paste the token as the value,
   and enable *Notebook access*.

3. Run:

```python
import subprocess, sys, pathlib
from google.colab import userdata

TOKEN = userdata.get("GITHUB_TOKEN")
REPO  = "abu-infidel/collab-test-"
DEST  = "/content/propwash-repo"

# capture_output keeps the tokenised URL out of the notebook's saved output
result = subprocess.run(
    ["git", "clone", "--depth", "1",
     f"https://x-access-token:{TOKEN}@github.com/{REPO}.git", DEST],
    capture_output=True, text=True)

if result.returncode != 0:
    raise SystemExit("clone failed -- check the secret name, the token's "
                     "expiry, and that it can read this repository")

# Strip the token back out of .git/config so it is not left on disk.
subprocess.run(["git", "-C", DEST, "remote", "set-url", "origin",
                f"https://github.com/{REPO}.git"], check=True)

sys.path.insert(0, DEST)
print("cloned to", DEST)
```

---

## Option C — make the repository public

Then the README's one-liner works as written and nobody needs a token:

```python
!git clone --depth 1 https://github.com/abu-infidel/collab-test- /content/propwash-repo
import sys; sys.path.insert(0, "/content/propwash-repo")
```

GitHub → repo **Settings** → scroll to **Danger Zone** → **Change visibility**.
Only do this if you are happy for the code to be world-readable; it cannot be
undone for anything already cloned or indexed.

---

## Step 2 — pick a GPU runtime

*Runtime → Change runtime type → T4 GPU → Save.* This restarts the runtime and
**wipes `/content`**, so do it **before** cloning or uploading, or you will have
to repeat Step 1.

Everything works on a CPU runtime too; you just see the NumPy and Numba
backends instead of the CUDA ones.

---

## Step 3 — install and check the host

```python
import propwash.colab as pc
env = pc.setup()          # installs only what is missing
```

`setup()` is safe to re-run: anything already importable is left alone. It
installs CuPy only when it actually sees a GPU.

Expected on a T4:

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

If `GPU` says `none detected`, you are on a CPU runtime — go back to Step 2.

---

## Step 4 — launch the GUI

```python
lab = pc.launch()
```

Controls on the left, tabs on the right. The things worth touching first:

- **Flight condition → Motor-matched** is on by default, so *Shaft speed* is
  read-only: it shows the speed the propeller actually settles at, where its
  torque demand meets the motor's torque supply.
- **Propeller → Collective** is the instructive one. Add pitch and thrust goes
  up while RPM comes *down* and current climbs.
- **View → Colour by** maps any solver field onto the blade.
- **View → Spin** animates at a rate proportional to the solved RPM.
- **Benchmark** tab is the capacity test.

---

## Step 5 — the capacity test

What you originally wanted, in one cell:

```python
from propwash.accel.bench import benchmark
print(benchmark(n_cases=65536, n_elements=64, verbose=False).format())
```

Every available backend solves the same grid and is diffed against the NumPy
reference, so correctness is reported next to speed — a GPU path that is a
hundred times faster and half a percent wrong is a bug, not a win.

Push it until something breaks:

```python
for n in (2**14, 2**16, 2**18, 2**20):
    print(benchmark(n_cases=n, n_elements=64, verbose=False).format(), "\n")
```

`n_cases x n_elements x 8 bytes` is the spanwise memory per field, so at
2²⁰ cases and 64 elements you are asking for ~0.5 GB per field. When you run
out of GPU memory you will get an out-of-memory error from CuPy or Numba, which
is the limit you were looking for.

---

## Troubleshooting

**The GUI cell runs but shows nothing.** Colab sandboxes output cells and
blocks third-party widgets until the custom widget manager is enabled.
`pc.launch()` does this for you, but if you built `PropwashLab` directly:

```python
from google.colab import output
output.enable_custom_widget_manager()
```

Then re-run the cell.

**The 3-D view is a static image with a "spin" button instead of a live
widget.** That is the documented fallback for when `FigureWidget` is
unavailable — usually a missing `anywidget` on Plotly 6+. `pc.setup()` installs
it. The fallback is fully functional; the animation just runs client-side.

**`ModuleNotFoundError: No module named 'propwash'`.** The clone or upload did
not land where Python is looking. Check:

```python
import pathlib
print(list(pathlib.Path("/content").glob("*")))
print(list(pathlib.Path("/content").rglob("propwash/__init__.py")))
```

If the second list is empty, Step 1 did not succeed — scroll up for its error.
If it is non-empty, add that file's grandparent to `sys.path`.

**`cupy` fails to import with a CUDA version mismatch.** Colab's driver moves
occasionally. `pc.setup()` installs `cupy-cuda12x`; if the runtime is on CUDA 11
install `cupy-cuda11x` instead. Either way `numba-cuda` still works, so this
costs you one backend, not the GPU.

**Everything is slow and the GPU backends are missing.** You are on a CPU
runtime, or you hit Colab's GPU quota. `pc.probe()` tells you which.

**The session disconnects mid-benchmark.** Colab reclaims idle runtimes and
caps session length on the free tier. Run smaller `n_cases`, or keep the tab
focused.

---

## Without a notebook

The same benchmark from a terminal:

```bash
python -m propwash env
python -m propwash bench --cases 65536
python -m propwash cuda-check     # compiles the CUDA C via NVRTC, no GPU needed
```
