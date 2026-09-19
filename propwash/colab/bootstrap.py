"""Environment setup and probing for Google Colab (and any other notebook).

Colab runtimes differ in ways that matter here -- CPU vs T4 vs A100, CUDA 11 vs
12, whether CuPy is preinstalled -- so rather than pinning a wheel and hoping,
:func:`setup` looks at what is actually present and installs only the gaps.
:func:`probe` prints the resulting inventory, which is also the honest answer to
"what are the limits of this host?".
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable

#: Packages the notebook GUI needs, and the import name to test for each.
NOTEBOOK_REQUIREMENTS = {
    "numpy": "numpy",
    "numba": "numba",
    "plotly": "plotly",
    "ipywidgets": "ipywidgets",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    # Plotly 6+ moved FigureWidget onto anywidget.  Without it the 3-D view
    # still works, but it falls back to redrawing a static figure instead of
    # updating vertices in place.
    "anywidget": "anywidget",
}

#: Only attempted when a CUDA device is present.
CUDA_EXTRAS = {"cupy": "cupy-cuda12x"}


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__       # type: ignore[name-defined]
    except NameError:
        return False
    return shell in ("ZMQInteractiveShell", "Shell", "google.colab._shell.Shell")


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def nvidia_smi() -> str | None:
    """Raw ``nvidia-smi`` header, or None when there is no GPU."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def pip_install(packages: Iterable[str], quiet: bool = True) -> bool:
    pkgs = [p for p in packages if p]
    if not pkgs:
        return True
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += list(pkgs)
    try:
        return subprocess.run(cmd, check=False).returncode == 0
    except Exception:
        return False


@dataclass
class Environment:
    """What this runtime can actually do."""

    python: str = ""
    platform: str = ""
    colab: bool = False
    notebook: bool = False
    gpu: str | None = None
    cpu_count: int = 0
    total_ram_gb: float = 0.0
    packages: dict[str, str | None] = field(default_factory=dict)
    backends: list[str] = field(default_factory=list)
    selected_backend: str = ""

    def report(self) -> str:
        lines = [
            "Propwash environment",
            f"  Python        {self.python}",
            f"  Platform      {self.platform}",
            f"  Host          {'Google Colab' if self.colab else 'local'}"
            f"{' (notebook)' if self.notebook else ''}",
            f"  CPU / RAM     {self.cpu_count} cores, {self.total_ram_gb:.1f} GiB",
            f"  GPU           {self.gpu or 'none detected'}",
            "  Packages",
        ]
        for name, version in sorted(self.packages.items()):
            lines.append(f"    {name:<12} {version or '-- not installed'}")
        lines.append("  Compute backends")
        for row in self.backends:
            lines.append(f"    {row}")
        lines.append(f"  -> selected   {self.selected_backend}")
        return "\n".join(lines)


def probe(include_backends: bool = True) -> Environment:
    """Inspect the runtime without changing anything."""
    env = Environment(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        colab=in_colab(), notebook=in_notebook(),
        gpu=nvidia_smi(), cpu_count=os.cpu_count() or 0,
    )
    try:
        env.total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, AttributeError, OSError):
        env.total_ram_gb = 0.0

    for mod in list(NOTEBOOK_REQUIREMENTS) + list(CUDA_EXTRAS) + ["torch", "PySide6"]:
        try:
            env.packages[mod] = getattr(importlib.import_module(mod), "__version__", "present")
        except Exception:
            env.packages[mod] = None

    if include_backends:
        try:
            from ..accel import get_backend, list_backends
            env.backends = [str(i) for i in list_backends()]
            env.selected_backend = get_backend().name
        except Exception as exc:  # pragma: no cover - defensive
            env.backends = [f"probe failed: {exc}"]
            env.selected_backend = "numpy"

    return env


def setup(cuda: bool = True, desktop: bool = False, verbose: bool = True) -> Environment:
    """Install whatever is missing, then report.

    Safe to re-run: anything already importable is left alone, so a second cell
    execution costs a few milliseconds rather than a re-download.
    """
    missing = [pkg for mod, pkg in NOTEBOOK_REQUIREMENTS.items() if not has_module(mod)]

    if cuda and nvidia_smi() is not None:
        missing += [pkg for mod, pkg in CUDA_EXTRAS.items() if not has_module(mod)]
    if desktop:
        for mod, pkg in (("PySide6", "PySide6"), ("pyqtgraph", "pyqtgraph"),
                         ("OpenGL", "PyOpenGL")):
            if not has_module(mod):
                missing.append(pkg)

    if missing:
        if verbose:
            print(f"Installing: {', '.join(sorted(set(missing)))}")
        pip_install(sorted(set(missing)), quiet=verbose)
    elif verbose:
        print("All notebook requirements already present.")

    env = probe()
    if verbose:
        print()
        print(env.report())
    return env


def enable_plotly_in_colab() -> None:
    """Make ipywidgets and Plotly render inside Colab output cells.

    Two separate things are needed and both fail silently when missing, which
    is why the GUI otherwise appears as a blank cell:

    * Colab sandboxes each output cell, so any widget beyond the built-in
      ipywidgets set -- which includes Plotly's ``FigureWidget`` -- needs the
      *custom widget manager* switched on for the session.
    * Older Colab runtimes also need Plotly's renderer pointed at ``colab``.

    Both are harmless outside Colab, so this is called unconditionally from the
    app launcher.
    """
    try:
        if in_colab():
            from google.colab import output as _colab_output
            _colab_output.enable_custom_widget_manager()
    except Exception:
        pass

    try:
        import plotly.io as pio
        if in_colab() and "colab" in getattr(pio.renderers, "_renderers", {}):
            pio.renderers.default = "colab"
    except Exception:
        pass


__all__ = ["Environment", "probe", "setup", "in_colab", "in_notebook",
           "nvidia_smi", "has_module", "pip_install", "enable_plotly_in_colab",
           "NOTEBOOK_REQUIREMENTS", "CUDA_EXTRAS"]
