# The physics, and where it comes from

This document derives what Propwash actually solves, states the assumptions
plainly, and says where the model stops being trustworthy. Notation is SI
throughout; the code is too.

---

## 1. The two descriptions

Blade-element momentum theory is a marriage of two models of the same
propeller, each of which is useless on its own.

**Momentum theory** treats the propeller as an infinitely-bladed actuator disk.
It knows how much momentum the air gains, but nothing about blades.

**Blade-element theory** treats each radial slice as an isolated 2-D aerofoil.
It knows lift and drag, but not what the air ahead of it is doing.

BEMT closes the loop: the induced velocity from momentum theory sets the angle
of attack that blade-element theory needs, and the forces from blade-element
theory set the momentum change that momentum theory needs. They agree at exactly
one inflow angle.

### Velocity triangle

For an element at radius `r` with the disk turning at `Ω`:

| | |
|---|---|
| axial velocity at the disk | `U_a = V + v_a` |
| tangential velocity at the disk | `U_t = Ω r − v_t` |
| resultant | `W = sqrt(U_a² + U_t²)` |
| inflow angle | `φ = atan2(U_a, U_t)` |
| angle of attack | `α = θ − φ` |

`θ` is the local blade angle (twist plus collective), `v_a` the axial induced
velocity and `v_t` the swirl induced velocity. Everything else follows.

### Momentum side

Mass flow through the annulus is `dṁ = 2π r ρ U_a dr`. Far downstream the axial
velocity increment is `2 v_a` and the swirl is `2 v_t`, giving

```
dT = 4π r ρ (V + v_a) v_a F dr
dQ = 4π r² ρ (V + v_a) v_t F dr
```

`F` is the Prandtl tip/hub loss factor (§3), which is what stands in for the
disk having a finite number of blades.

### Blade-element side

With `Cn = Cl cos φ − Cd sin φ` and `Ct = Cl sin φ + Cd cos φ`, over `B` blades:

```
dT = ½ ρ B c W² Cn dr
dQ = ½ ρ B c W² Ct r dr
```

---

## 2. The residual, and why this one

Substituting `v_a = W sin φ − V` and `v_t = Ω r − W cos φ` and equating the two
descriptions gives two expressions for `W`:

```
W_T = 4 F V sin φ / (4 F sin²φ − σ Cn)          (from thrust)
W_Q = 4 F Ω r sin φ / (σ Ct + 4 F sin φ cos φ)   (from torque)
```

with `σ = B c / 2π r`. Setting `W_T = W_Q` and cancelling the common factor
`4 F sin φ` — which is non-zero on the open interval `(0, π/2)` — leaves

```
R(φ) = Ω r (4 F sin²φ − σ Cn) − V (σ Ct + 4 F sin φ cos φ) = 0
```

This is the equation `propwash.bemt.core.residual_phi` evaluates and
`solve_phi` brackets. Four properties earn it its place:

**No division.** The textbook formulation solves for induction factors,
`a = k/(1−k)` with `k = σ Cn / (4 F sin²φ)`. That expression is singular at
`k = 1` and changes sign through it, which is exactly the heavily-loaded regime a
static propeller lives in. `R(φ)` is a smooth bounded function of `φ`
everywhere.

**Finite at V = 0.** Static thrust is the first case anybody checks. There the
induction-factor form is undefined — `a = v_a/V` with `V = 0` — while `R(φ)`
collapses to `4 F sin²φ = σ Cn`, which is simply the correct static relation.
The code needs no special case for hover.

**Guaranteed bracket.** As `φ → 0⁺`, `sin φ → 0` and `Cn → Cl(θ) > 0` for any
sensible blade angle, so the first term is negative; the second is negative too.
As `φ → π/2⁻`, `4 F sin²φ → 4F` dominates and `Cn → −Cd < 0`, so both terms are
positive. `R` changes sign once on `(0, π/2)`, so bisection converges
unconditionally — no initial guess, no relaxation factor, no divergence, ever.

**It suits a GPU.** Bisection is normally the slow choice. Here it is the right
one: 60 halvings reach double precision, the trip count is fixed, there is no
early exit and no data-dependent branch, so every thread in a warp executes an
identical instruction stream and retires together. Newton–Raphson would win on
one CPU core and lose badly on 3,000 CUDA cores.

`W` is recovered from the torque branch `W_Q`, whose denominator is strictly
positive on `(0, π/2)` and which stays finite at `V = 0`.

---

## 3. Corrections

### Prandtl tip and hub loss

A real blade sheds a vortex at each end, so circulation — and with it induced
velocity — collapses there. Prandtl's correction:

```
F_tip = (2/π) arccos( exp(−f_tip) ),   f_tip = (B/2) (R − r) / (r |sin φ|)
F_hub = (2/π) arccos( exp(−f_hub) ),   f_hub = (B/2) (r − r_hub) / (r_hub |sin φ|)
F = F_tip · F_hub
```

Without it, predicted thrust is optimistic by roughly 5–15%, concentrated
entirely near the tip. `F` depends on `φ`, so it is evaluated inside the
residual, not once beforehand.

### Reynolds number

A model propeller blade spans a factor of ~5 in chord Reynolds number from root
to tip, and the root of a small propeller can drop to a few thousand. Turbulent
skin friction goes as `Re^−0.2`; below `Re ≈ 2×10⁴` the boundary layer is
laminar and separation-prone, so drag is inflated further. The two factors
compound, so their product is clamped to `[0.5, 4.0]` — measured low-Re polars
show profile drag rising three- to four-fold, not ten-fold. Maximum lift is
scaled down too.

### Compressibility

Prandtl–Glauert amplifies lift as `1/sqrt(1 − M²)`, clipped at `M = 0.92`. Wave
drag follows Lock's fourth-power rise past the drag-divergence Mach number,
`ΔCd = 20 (M − M_dd)⁴`, with `M_dd` falling with both lift and thickness. This
is why tip speed, not power, is what limits propeller RPM.

### Deep stall

During iteration the solver probes angles of attack no propeller ever sees. A
polar that returns garbage outside ±15° destroys convergence, so every section
is extended to the full ±180° with the Viterna–Corrigan model, blended into the
attached-flow polar with a finite-support smoothstep so the linear range stays
exactly linear. The `a₂ cos²α / sin α` term is singular at `α = 0`, where the
model has no meaning; it is floored at the stall angle's sine.

This extrapolation exists to keep the solver convergent, not to be accurate at
60° angle of attack.

---

## 4. Outputs

| symbol | definition |
|---|---|
| `C_T` | `T / (ρ n² D⁴)` |
| `C_Q` | `Q / (ρ n² D⁵)` |
| `C_P` | `P / (ρ n³ D⁵) = 2π C_Q` |
| `J` | `V / (n D)` |
| `η` | `T V / P = C_T J / C_P` |
| `FoM` | `T^1.5 / (sqrt(2 ρ A) P)` |

`n` is revolutions per second. `η` is a forward-flight metric and `FoM` a hover
one; each is reported as zero where it stops meaning anything, rather than as a
spike. Past the zero-thrust advance ratio `T V / P` goes negative and then
diverges as power crosses zero too — that is a coordinate artefact, not
performance.

---

## 5. Propeller/motor matching

A propeller has no RPM of its own. The first-order brushless DC model,

```
ω = Kv (V_applied − I R_m),    Q_shaft = (I − I₀) / Kv
```

gives torque falling linearly with speed to zero at no-load (flat at the top
where the ESC current limit bites), while propeller torque rises roughly as
`n²`. Monotone in opposite directions means exactly one crossing, so bisection
is again unconditionally safe — which matters when a GUI slider is calling it
sixty times a second.

That crossing is the operating point, and moving it is what every control in
the program is really doing.

---

## 6. What this model does not do

- **Only the thrusting branch is bracketed.** Windmilling and propeller-brake
  states have their root outside `(0, π/2)`. Those elements are reported as
  non-converged — `converged_fraction < 1` — rather than returned as
  plausible-looking nonsense.
- **Strip theory.** Each station is independent. No radial flow, no
  hub/spinner blockage, no blade flexibility, no unsteady aerodynamics, no
  non-axial inflow, no ground effect.
- **Rotational stall delay is off by default.** The Du-Selig/Eggers model is
  implemented (`SolverOptions.stall_delay`) but not enabled, because it mostly
  affects the near-root region that contributes little thrust.
- **The bundled polars are parametrised models**, fitted to published section
  characteristics, not digitised wind-tunnel data. Load real polars with
  `propwash.airfoil.load_polar_file` for anything that matters.

Against published static data for an APC 10×5, this model lands within roughly
10% and under-predicts. That is the normal direction and magnitude for BEMT, and
it is the number to keep in mind before trusting any third significant figure.

---

## References

The formulation here follows the standard treatment; these are the sources worth
reading if you want the full arguments.

- Glauert, H. (1935). *Airplane Propellers*, in Durand (ed.), Aerodynamic Theory
  Vol. IV. The original.
- Prandtl, L. & Betz, A. (1919). *Schraubenpropeller mit geringstem
  Energieverlust*. Where the tip-loss factor comes from.
- Viterna, L. A. & Corrigan, R. D. (1981). *Fixed-pitch rotor performance of
  large horizontal-axis wind turbines*. The deep-stall extrapolation.
- Du, Z. & Selig, M. S. (1998). *A 3-D stall-delay model for horizontal axis wind
  turbine performance prediction*. AIAA 98-0021.
- Ning, A. (2014). *A simple solution method for the blade element momentum
  equations with guaranteed convergence*. Wind Energy 17(9). The
  single-residual-in-φ idea, here adapted to the propeller convention.
- McCormick, B. W. (1995). *Aerodynamics, Aeronautics and Flight Mechanics*.
  Good on the propeller coefficients and matching.
