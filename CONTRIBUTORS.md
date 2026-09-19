# Contributors

## Project owner

**[@abu-infidel](https://github.com/abu-infidel)** — commissioned the project,
set the requirements (a BEMT propeller visualiser with CUDA acceleration,
targeting Google Colab), and directed its scope.

## Authorship

**Claude (Opus 5), via [Claude Code](https://claude.ai/code)** — wrote the
initial implementation in a single session: the BEMT solver and its residual
formulation, the aerofoil and atmosphere models, the Numba CPU / Numba CUDA /
raw CUDA C / PyTorch backends, the 3-D blade lofting, the Qt and ipywidgets
GUIs, the CLI, the test suite and the documentation.

To be clear about what that means: Claude is a language model, not a person. It
does not hold copyright and cannot take responsibility for the code. Every
commit it authored carries a `Co-Authored-By: Claude Opus 5` trailer and a link
to the session that produced it, so the provenance of any line can be traced.
The project owner is responsible for the repository.

## What was and was not verified

Worth stating plainly, because AI-written code invites the question:

**Checked by running it.** The ISA implementation against the published
standard-atmosphere table; every compute backend diffed against the NumPy
reference (agreement ~1e-14 relative); the CUDA kernel executed under
`NUMBA_ENABLE_CUDASIM`; the raw CUDA C compiled to PTX by NVRTC for compute_75
through compute_90; the PyTorch path forced onto CPU; both GUIs driven
headlessly and screenshotted; every notebook cell executed end to end. 166
tests pass.

**Not checked.** No GPU was available in the environment where this was
written, so the CUDA backends have never executed on real hardware — only in
simulation and compilation. Predictions were compared against *published*
propeller data, not against anything measured for this project; they land
roughly 10% low, which is the expected direction and size of BEMT error.

`docs/PHYSICS.md` lists the model's limits in full.

## Contributing

Bug reports and pull requests are welcome. If you change the solver, run
`python -m pytest` first — the backend cross-validation tests exist to catch
exactly the kind of silent divergence that is easy to introduce between the CPU
and GPU paths.
