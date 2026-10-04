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

### Section data for full-size blades

Propwash models full-size propellers only, so every section in the library is
fitted at a chord Reynolds number of 3 million, typical of a light-aircraft
blade at cruise. The library holds Clark Y, R.A.F. 6, NACA 16-series, NACA
64-series, ARA-D and NACA 0012 (`propwash.airfoil.AIRFOIL_LIBRARY`). Each blade
station uses its own thickness ratio:

- **Profile drag** scales with Hoerner's form factor, `1 + 2t + 60t⁴`, relative
  to the section's nominal thickness.
- **Maximum lift** scales as `(t/0.12)^0.85` below 12% thickness, holds flat to
  15%, and falls as `1 − 1.5(t − 0.15)` above that, floored at half. Thin tips
  stall early; thick roots lose lift.

**Blade angle reference.** Propeller pitch and blade angles are measured off
the blade's flat face. On a flat-bottomed Clark Y that face sits about 2° nose
down from the leading-edge-to-trailing-edge chord, so the Clark Y zero-lift
angle is referenced to the face: −4.2°. That value lies inside the published
range (−3.6° chord-referenced to about −5.5° face-referenced). It was chosen
against the Cessna 172N cruise data, as described in [VALIDATION.md](VALIDATION.md).

### Reynolds number

A full-size blade runs at chord Reynolds numbers from a few hundred thousand at
the root to a few million outboard, so the corrections are mild and bounded:

```
Cd  ×= clip((Re_ref / Re)^0.2, 0.75, 1.6)            turbulent skin friction
Cl_max ×= clip(1 + 0.06 log10(Re / Re_ref), 0.85, 1.05)
```

with `Re` floored at 10⁴. Only the lift *ceiling* moves with Reynolds number.
The lift-curve slope does not, which is what wind-tunnel data in this range
show.

### Compressibility

Prandtl–Glauert amplifies lift as `1/sqrt(1 − M²)`, with `M` clipped at 0.92 for
the amplification. Wave drag follows Korn's drag-divergence estimate and Lock's
fourth-power rise:

```
M_dd  = κ − t/c − |Cl|/10                (Korn; κ per section, 0.83–0.90)
M_cr  = M_dd − (0.1/80)^(1/3)            (= M_dd − 0.1077)
ΔCd   = 20 · max(min(M, 1.2) − M_cr, 0)⁴  (Lock)
```

This is why tip speed, not power, limits propeller RPM, and why thin high-κ
sections (NACA 16) belong at fast tips.

**The amplified lift is then capped.** Prandtl–Glauert raises the lift-curve
*slope*; maximum lift *falls* with Mach. Without a ceiling, a transonic tip is
handed a lift coefficient it could never reach. The ceiling used is

```
Cl_max(M) = Cl_max · Re_factor · max(1 − 0.9 (M − 0.35)^1.5, 0.35)
```

which is a smooth fit to the usual shape of measured high-subsonic `Cl_max`
data, not a first-principles result. Without the Mach correction, lift is still
capped at the station's `Cl_max`.

All four compute paths (NumPy, Numba CPU/CUDA, the CUDA/OpenCL C kernel and
PyTorch) implement these corrections identically. The test suite holds them to
agreement within 1e-12.

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

`n` is revolutions per second. `η` is a forward-flight metric and `FoM` a static
one. In a report, each is `null` with a reason code where it stops meaning
anything (`NA_STATIC`, `NA_IN_FORWARD_FLIGHT`, `NOT_PRODUCING_THRUST`). It is
never set to zero, because a zero would read as a real, terrible result.

---

## 5. Sizing and design

Analysis answers "what does this propeller do?". Sizing answers the two
questions that come first.

### How big? — the actuator disk

Momentum theory alone bounds what any propeller of a given diameter can do. A
disk of area `A` producing thrust `T` at speed `V` accelerates the air by an
induced velocity `w`:

```
T = 2 ρ A w (V + w)   ⟹   w = −V/2 + sqrt(V²/4 + T/(2ρA))
```

with ideal power `P = T (V + w)` and shaft power `P/η_p`. The Froude efficiency
`V/(V + w)` is the ceiling — no blade design beats it.

Power falls monotonically with diameter, so **there is no interior optimum**.
The diameter is chosen by the constraints: ground clearance (and gyroscopic and
structural loads) at the big end, tip Mach number at the small end. Both are
plotted by `diameter_sweep`.

### What shape? — minimum induced loss

Betz's condition: induced losses are least when the shed vortex sheet is a rigid
helicoid translating aft at constant speed. That fixes the circulation
distribution and with it the blade.

Propwash implements the Adkins & Liebeck (1994) formulation, which iterates on
the *displacement velocity ratio* `ζ`. For a trial `ζ`:

```
tan φ = (1 + ζ/2) / x,          x = Ω r / V
F     = (2/π) arccos(exp(−f)),  f = (B/2)(1 − ξ)/sin φ_t
G     = F x cos φ sin φ
Wc    = 4π λ G V R ζ / (Cl B)
```

`Wc` is the product of resultant velocity and chord — Betz's condition in the
form you can actually use. Dividing by `W` gives the chord; `α + φ` gives the
twist. Integrating the loading gives thrust and power coefficients, which invert
for a new `ζ`. It converges in about five iterations.

Prandtl's tip loss appears here as a *design* constraint, in the circulation
itself, rather than as an after-the-fact correction.

Two limits worth knowing. The method needs `V > 0` — it is built on the advance
ratio, and a static design point has no minimum-induced-loss solution in this
form; size a static propeller with the disk method instead. And the classical
formulation is incompressible: on a fast propeller that matters, because the
blade it designs then over-produces by about 17% once compressibility is
accounted for. Propwash therefore recomputes the section's angle of attack and
drag at each station's local Mach number, so the blade is designed to *achieve*
the target `Cl` rather than to achieve it only in incompressible flow. With that
correction the design agrees with the independent BEMT solver to 1.4% on thrust.

### Which diameter? — a designed blade at every candidate

Momentum theory always prefers a bigger disk, so on its own it cannot choose a
propeller. Sizing mode (`size_propeller`) therefore designs the
minimum-induced-loss blade for every candidate diameter. The candidates run
from half the largest allowed diameter up to the smaller of the installation
limit and the diameter at which the helical tip Mach reaches its limit. The
diameters are then compared on the efficiency of their *best possible* blade.
Profile drag and compressible wave drag both grow with tip speed, so the best
diameter can be inside the range rather than at its edge. The report says
which it is. The winner is analysed again by the independent BEMT solver and
both answers are reported. For the Cessna 172N cruise point (8,000 ft, 122 KTAS,
75%, 2,650 rpm) sizing picks a 72-inch, 60-inch-pitch, activity factor 90
two-blade propeller. The real one is 75 × 57.

## 6. The engine, and where the propeller settles

### Piston engine

Every engine number is an input: rated power, rated RPM (also the RPM limit),
gear ratio, aspiration and, for a turbocharged engine, its critical altitude.
The model constants below are printed in every report's `[MODEL]` section.

- **Torque curve.** Full-throttle crankshaft torque is a parabola in RPM. It
  peaks at 0.78 of rated speed and is 8% lower again at rated speed, anchored
  so that power at rated RPM equals rated power.
- **Altitude.** Normally aspirated: Gagg–Farrar, `P/P_SL = σ − (1 − σ)/7.55`.
  Turbocharged: flat to the critical altitude, then the same lapse relative
  to the density there.
- **Fuel.** Fuel flow is BSFC × shaft power. Without a BSFC input it is
  reported as not available, never assumed.
- **Gearing** divides propeller speed from crankshaft speed.

### Matching (`propwash/matching.py`)

A propeller has no RPM of its own. Each solve below is vectorised over flight
speed: a scan grid, then three rounds of sub-gridding, each a single batched
BEMT call for all speeds together.

| power setting | fixed pitch | constant speed |
|---|---|---|
| full throttle | RPM where engine torque = propeller torque | blade angle that absorbs full-throttle power at the governor RPM |
| set RPM | power absorbed at that RPM (and the throttle needed) | (not meaningful: RPM is the governor's) |
| set power | RPM where the propeller absorbs it | blade angle that absorbs it |
| level-flight trim | RPM where thrust = drag | blade angle where thrust = drag |

Every solve returns a status per speed instead of a number that might be a
fallback. The statuses are `ok`, `rpm_limited`, `on_fine_stop`,
`on_coarse_stop`, `windmilling`, `too_coarse`, `power_not_available`,
`thrust_not_reachable` and `below_running_range`; the
[report format](report-format/README.md#52-operating-point-statuses) defines
them. In particular:

- **A fixed-pitch propeller that would over-rev** at full throttle is reported
  at the RPM limit, with the reduced throttle that holds it (`rpm_limited`). It
  is not clipped silently and not reported as zero.
- **A constant-speed propeller on a stop** has lost its governor. It is solved
  as a fixed-pitch propeller at that stop and throttle, so its RPM falls below
  (fine stop) or rises above (coarse stop) the setting.
- **Spurious roots are rejected.** At high airspeed and very low RPM, momentum
  theory has nonsense solutions deep in the windmill-brake state, with drags
  several times that of a solid disk. Those points are masked as what they
  physically are, a propeller being driven by the air. The RPM searches also
  take the highest stable crossing, the one a propeller reaches by spinning
  up.

### Airframe performance (optional)

With weight, wing area, span, CD0, Oswald *e* and CLmax, the drag polar
`CD = CD0 + CL²/(π e AR)` gives:

- drag and excess thrust at the operating point;
- the level-flight trim at the operating airspeed;
- the top level-flight speed at the power setting, searched up to a bound no
  aircraft can exceed: `(2P/(ρ S CD0))^(1/3)`;
- the best climb and its speed;
- the service ceiling (100 ft/min) and absolute ceiling at full throttle, by
  false position on altitude.

---

## 7. What this model does not do

- **Full-size aircraft only.** The section data, Reynolds-number corrections
  and engine model are for light-aircraft and larger propellers. The drone and
  electric-motor version is preserved on the `legacy/drone-and-aircraft` branch.
- **Climb and ceiling are over-predicted.** For the Cessna 172N they come out
  16% (climb) and 22% (ceiling) high (see [VALIDATION.md](VALIDATION.md)). The
  model leaves out installation losses, namely slipstream scrubbing over the
  cowling and fuselage and cooling drag, and these matter most at the low
  speeds where climb happens. The airframe's drag polar is an input you supply.
- **Static thrust** relies on post-stall section data over the inner blade,
  where 2-D strip theory is least reliable. The static RPM of the C172N
  reference case is right, but only because the unpublished blade activity
  factor was calibrated to it.
- **Strip theory.** Each station is independent. There is no radial flow, no
  hub or spinner blockage, no blade flexibility, no unsteady aerodynamics, no
  non-axial inflow and no ground effect. There is also no rotational stall
  delay.
- **Helical twist.** Blades built from a pitch are constant-pitch screws.
  Real blades often deviate near the root and tip.
- **The bundled polars are parametrised models** fitted to published section
  characteristics, not digitised wind-tunnel data. For anything that matters,
  load real polars with `propwash.airfoil.load_polar_file`.
- **The engine torque curve is generic.** Near rated RPM it matters little;
  far below it, it is only a rough guide.

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
- Ning, A. (2014). *A simple solution method for the blade element momentum
  equations with guaranteed convergence*. Wind Energy 17(9). The
  single-residual-in-φ idea, here adapted to the propeller convention.
- McCormick, B. W. (1995). *Aerodynamics, Aeronautics and Flight Mechanics*.
  Good on the propeller coefficients and matching.
- Adkins, C. N. & Liebeck, R. H. (1994). *Design of optimum propellers*. Journal
  of Propulsion and Power 10(5). The minimum-induced-loss design method, and the
  corrections that make Larrabee's version converge.
- Larrabee, E. E. (1979). *Practical design of minimum induced loss propellers*.
  SAE 790585. The engineering form of Betz's condition.
- Gudmundsson, S. (2014). *General Aviation Aircraft Design*, ch. 15. The
  actuator-disk sizing procedure, and the Gagg–Farrar piston power lapse.
- Korn, D. (1979) and Lock, R. C. (1986). The drag-divergence estimate and the
  fourth-power wave-drag rise used for compressibility.
- Hoerner, S. F. (1965). *Fluid-Dynamic Drag*. The thickness form factor.
