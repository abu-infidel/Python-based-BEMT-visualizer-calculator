# Contributors

## Project owner

**[@abu-infidel](https://github.com/abu-infidel)** commissioned the project,
set the requirements and directed its scope. The requirements so far: a BEMT
propeller visualiser and calculator for Google Colab; then full-size aircraft
only, with no defaults, no input caps, structured report export and
agreement with real-world data.

## Authorship

**Claude, via [Claude Code](https://claude.ai/code)**, wrote the implementation
under the project owner's direction. That covers the BEMT solver and its
compute backends (NumPy, Numba CPU/CUDA, CUDA C/OpenCL C, PyTorch), the
full-size section and engine models, the matching and sizing methods, the input
validation and report format, both GUIs, the CLI, the tests and the
documentation.

To be clear about what that means: Claude is a language model, not a person. It
does not hold copyright and cannot take responsibility for the code. Every
commit it authored carries a `Co-Authored-By` trailer and a link to the session
that produced it, so the provenance of any line can be traced. The project
owner is responsible for the repository.

## What was and was not verified

Worth stating plainly, because AI-written code invites the question.

**Checked by running it.**

- The model against published Cessna 172N data, with the calibrated quantities
  named ([docs/VALIDATION.md](docs/VALIDATION.md)).
- Every compute backend diffed against the NumPy reference. The OpenCL kernel
  was executed on a POCL CPU device, the CUDA kernel under
  `NUMBA_ENABLE_CUDASIM`, and the raw CUDA C compiled to PTX by NVRTC.
- Extreme and invalid inputs swept, to check that no report contains a
  non-finite number or a silent zero.
- Both apps driven headlessly, and every notebook cell executed. The test
  suite passes (`python -m pytest`).

**Not checked.**

- No GPU was available where this was written. The CUDA backends have run only
  in simulation and compilation, and OpenCL only on a CPU device.
- The validation is against one aircraft's published figures, not against
  measurements made for this project.
- Climb and ceiling are over-predicted by 16–22% for that aircraft.

## Contributing

Bug reports and pull requests are welcome. If you change the solver, run
`python -m pytest` first. The backend cross-validation and
`propwash validate` exist to catch exactly the kind of silent divergence that is
easy to introduce.
