"""Where does the propeller actually run?

A propeller on its own has no operating point: it needs something to say how
fast it turns (fixed pitch) or how coarse its blades are (constant speed).
This module closes that loop against a :class:`~propwash.engine.PistonEngine`,
vectorised over flight speed so a whole performance table is a handful of
batched solves.

Every solve returns a status per flight speed instead of a number that might be
a fallback.  The statuses are:

``ok``
    A genuine operating point inside every limit.
``rpm_limited``
    Fixed pitch: at full throttle the propeller would drive the engine past its
    RPM limit.  The point is reported *at* the limit, with the reduced throttle
    needed to hold it.
``windmilling``
    The airflow drives the propeller even with the engine at the RPM limit; it
    produces drag, not thrust.
``too_coarse``
    The propeller holds the engine below a running speed; no operating point.
``power_not_available``
    The requested power or RPM needs more than the engine can give here.
``on_fine_stop`` / ``on_coarse_stop``
    Constant speed: the governor cannot hold the set RPM because the blades
    are on a pitch stop; the point is solved as a fixed-pitch propeller at the
    stop.
``thrust_not_reachable``
    Level-flight trim only: no setting inside the limits makes enough thrust.
``below_running_range``
    Less is asked for than the propeller absorbs (or produces) at the lowest
    running speed of the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .atmosphere import AirState
from .bemt.core import SolverOptions, solve_batch
from .engine import PistonEngine
from .geometry import BladeGeometry, BladeStations

OK = "ok"
RPM_LIMITED = "rpm_limited"
WINDMILLING = "windmilling"
TOO_COARSE = "too_coarse"
POWER_NOT_AVAILABLE = "power_not_available"
ON_FINE_STOP = "on_fine_stop"
ON_COARSE_STOP = "on_coarse_stop"
THRUST_NOT_REACHABLE = "thrust_not_reachable"
BELOW_RUNNING_RANGE = "below_running_range"

#: Statuses whose numbers describe a real, reachable operating point.
VALID_STATUSES = (OK, RPM_LIMITED, ON_FINE_STOP, ON_COARSE_STOP)

_GRID = 9           # first scan, points per speed
_ROUNDS = 3         # refinement rounds (each shrinks the bracket 8x)
_SUB = 7            # interior points per refinement round


class Propeller:
    """A blade discretised once, solved many times on the NumPy path.

    The NumPy reference path is used for operating-point work because it has
    no compile step, and a single matching solve is a few hundred points --
    nothing a GPU would notice.
    """

    def __init__(self, geometry: BladeGeometry, options: SolverOptions) -> None:
        self.geometry = geometry
        self.options = options
        self.stations: BladeStations = geometry.discretize(options.n_elements,
                                                           options.spacing)
        self.tables = self.stations.polar_tables(options.n_alpha_table)

    def solve(self, rpm, v_inf, air: AirState, delta_pitch=None) -> dict[str, np.ndarray]:
        """Batched solve, plus a ``physical`` mask.

        Deep in the windmill-brake state (high airspeed, very low RPM) the
        momentum equations have spurious roots that report drags several times
        that of a solid disk.  Those points are flagged so the searches treat
        them as what they physically are -- a propeller being driven by the
        air -- instead of mistaking them for an operating point.
        """
        sol = solve_batch(self.stations, rpm, v_inf, air, self.options, self.tables,
                          delta_pitch)
        v = np.broadcast_to(np.asarray(v_inf, dtype=float), np.shape(sol["thrust"]))
        disk_drag = 2.0 * 0.5 * air.density * v * v * self.geometry.disk_area
        sol["physical"] = (v <= 0.0) | (sol["thrust"] >= -disk_drag)
        return sol


@dataclass(slots=True)
class Match:
    """Operating points for a vector of flight speeds."""

    v_inf: np.ndarray
    status: np.ndarray                 # str per speed
    prop_rpm: np.ndarray               # nan where there is no operating point
    delta_pitch: np.ndarray            # rad, collective added to the blade
    throttle: np.ndarray               # fraction of full-throttle torque in use
    solution: dict[str, np.ndarray]    # solver outputs at the operating points
    notes: list[str] = field(default_factory=list)

    def valid(self) -> np.ndarray:
        return np.isin(self.status, VALID_STATUSES)

    def at(self, i: int) -> dict:
        """Everything about speed ``i`` as plain Python values."""
        out = {"v_inf": float(self.v_inf[i]), "status": str(self.status[i]),
               "prop_rpm": float(self.prop_rpm[i]),
               "delta_pitch": float(self.delta_pitch[i]),
               "throttle": float(self.throttle[i])}
        for key, val in self.solution.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == self.v_inf.size:
                out[key] = val[i]
        return out


# ---------------------------------------------------------------------------
# A vectorised monotone root finder
# ---------------------------------------------------------------------------

FOUND, NEVER_CROSSES, NONPOSITIVE_AT_START = 0, 1, -1


def solve_decreasing(evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     lo: np.ndarray, hi: np.ndarray, which: str = "first"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Find, per row, where ``f`` crosses from positive to non-positive.

    ``evaluate(rows, x)`` returns ``f`` for ``x`` of shape ``(len(rows), k)``.
    Each row is scanned on a grid, bracketed at a downward crossing -- a stable
    operating point when ``f`` is supply minus demand -- and the bracket is
    shrunk by repeated sub-gridding, one batched solve per round for all rows.

    ``which`` picks the ``"first"`` or ``"last"`` crossing when there are
    several.  Searches over RPM use ``"last"``: at high airspeed and very low
    RPM the blade is deep in the windmill-brake state, where momentum theory
    has spurious roots, and the physical operating point is the one the
    propeller reaches by spinning up -- the highest.

    Returns ``(x, kind)``.  ``kind`` is :data:`FOUND`, :data:`NEVER_CROSSES`
    (``f`` positive over the whole range) or :data:`NONPOSITIVE_AT_START`
    (``f`` never positive before a crossing); ``x`` is NaN unless found.
    """
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    m = lo.size
    rows = np.arange(m)
    t = np.linspace(0.0, 1.0, _GRID)
    x = lo[:, None] + (hi - lo)[:, None] * t[None, :]
    f = evaluate(rows, x)

    down = (f[:, :-1] > 0.0) & (f[:, 1:] <= 0.0)
    found = np.any(down, axis=1)
    kind = np.where(found, FOUND, np.where(np.all(f > 0.0, axis=1),
                                           NEVER_CROSSES, NONPOSITIVE_AT_START))
    root = np.full(m, np.nan)
    if not np.any(found):
        return root, kind

    def bracket(xs, fs, pick):
        dn = (fs[:, :-1] > 0.0) & (fs[:, 1:] <= 0.0)
        if pick == "last":
            j = dn.shape[1] - 1 - np.argmax(dn[:, ::-1], axis=1)
        else:
            j = np.argmax(dn, axis=1)
        k = np.arange(xs.shape[0])
        return xs[k, j], xs[k, j + 1], fs[k, j], fs[k, j + 1]

    idx = np.nonzero(found)[0]
    a, b, fa, fb = bracket(x[idx], f[idx], which)
    for _ in range(_ROUNDS):
        s = np.linspace(0.0, 1.0, _SUB + 2)[1:-1]
        xi = a[:, None] + (b - a)[:, None] * s[None, :]
        fi = evaluate(idx, xi)
        xs = np.concatenate([a[:, None], xi, b[:, None]], axis=1)
        fs = np.concatenate([fa[:, None], fi, fb[:, None]], axis=1)
        a, b, fa, fb = bracket(xs, fs, which)
    # Linear interpolation inside the final bracket.
    gap = fa - fb
    root[idx] = a + (b - a) * fa / np.where(gap == 0.0, 1.0, gap)
    return root, kind


# ---------------------------------------------------------------------------
# Fixed pitch
# ---------------------------------------------------------------------------

def _fixed_pitch_rpm(prop: Propeller, engine: PistonEngine, v: np.ndarray,
                     air: AirState, throttle: np.ndarray, delta: np.ndarray):
    """Torque balance at a given throttle, per speed.  Returns rpm and status."""
    def f(rows, rpm):
        sol = prop.solve(rpm, v[rows][:, None], air, delta[rows][:, None])
        supply = throttle[rows][:, None] * engine.full_throttle_torque(rpm, air)
        f = supply - sol["torque"]
        return np.where(sol["physical"], f, np.inf)

    rpm, kind = solve_decreasing(f, np.full(v.size, engine.min_prop_rpm),
                                 np.full(v.size, engine.max_prop_rpm), "last")
    status = np.full(v.size, OK, dtype=object)
    status[kind == NEVER_CROSSES] = RPM_LIMITED
    rpm[kind == NEVER_CROSSES] = engine.max_prop_rpm
    status[kind == NONPOSITIVE_AT_START] = TOO_COARSE
    return rpm, status


def fixed_pitch(prop: Propeller, engine: PistonEngine, v_inf, air: AirState,
                throttle: float = 1.0) -> Match:
    """Fixed-pitch propeller at a throttle setting (1 = full throttle)."""
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    thr = np.full(v.size, float(throttle))
    delta = np.zeros(v.size)
    rpm, status = _fixed_pitch_rpm(prop, engine, v, air, thr, delta)
    return _finish(prop, engine, v, air, rpm, delta, status, thr)


def fixed_pitch_at_rpm(prop: Propeller, engine: PistonEngine, prop_rpm: float,
                       v_inf, air: AirState) -> Match:
    """Fixed pitch held at a set RPM (the way a POH cruise table is flown)."""
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    rpm = np.full(v.size, float(prop_rpm))
    delta = np.zeros(v.size)
    return _finish(prop, engine, v, air, rpm, delta,
                   np.full(v.size, OK, dtype=object), None)


def fixed_pitch_at_power(prop: Propeller, engine: PistonEngine, power: float,
                         v_inf, air: AirState) -> Match:
    """Fixed pitch absorbing a set shaft power (W): find the RPM.

    If even the RPM limit does not absorb that much the point is reported at
    the limit (``rpm_limited``) with the power it does absorb there.
    """
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    delta = np.zeros(v.size)

    def f(rows, rpm):
        sol = prop.solve(rpm, v[rows][:, None], air)
        f = power - sol["power"]
        return np.where(sol["physical"], f, np.inf)

    rpm, kind = solve_decreasing(f, np.full(v.size, engine.min_prop_rpm),
                                 np.full(v.size, engine.max_prop_rpm), "last")
    status = np.full(v.size, OK, dtype=object)
    status[kind == NEVER_CROSSES] = RPM_LIMITED
    rpm[kind == NEVER_CROSSES] = engine.max_prop_rpm
    status[kind == NONPOSITIVE_AT_START] = BELOW_RUNNING_RANGE
    return _finish(prop, engine, v, air, rpm, delta, status, None)


# ---------------------------------------------------------------------------
# Constant speed
# ---------------------------------------------------------------------------

def constant_speed(prop: Propeller, engine: PistonEngine, prop_rpm: float,
                   power: np.ndarray | float, v_inf, air: AirState,
                   delta_max: float) -> Match:
    """Constant-speed propeller: find the collective that absorbs ``power``.

    The blade's built-in twist is its fine-stop setting (``delta = 0``) and
    ``delta_max`` is the coarse stop.  When the required collective falls
    outside the stops the governor has lost control and the point is solved
    as a fixed-pitch propeller on that stop, at the throttle that would have
    given ``power`` at the set RPM.
    """
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    p_target = np.broadcast_to(np.asarray(power, dtype=float), v.shape).copy()
    rpm_set = float(prop_rpm)

    def f(rows, delta):
        sol = prop.solve(np.full(delta.shape, rpm_set), v[rows][:, None], air, delta)
        f = p_target[rows][:, None] - sol["power"]
        return np.where(sol["physical"], f, np.inf)

    delta, kind = solve_decreasing(f, np.zeros(v.size), np.full(v.size, float(delta_max)))
    status = np.full(v.size, OK, dtype=object)
    rpm = np.full(v.size, rpm_set)

    available = float(engine.full_throttle_power(rpm_set, air))
    throttle = p_target / max(available, 1e-9)
    short = throttle > 1.0 + 1e-9
    status[short] = POWER_NOT_AVAILABLE
    rpm[short] = np.nan
    delta[short] = 0.0

    for kind_code, stop_delta, label in ((NONPOSITIVE_AT_START, 0.0, ON_FINE_STOP),
                                         (NEVER_CROSSES, float(delta_max), ON_COARSE_STOP)):
        sel = np.nonzero((kind == kind_code) & ~short)[0]
        if sel.size == 0:
            continue
        d = np.full(sel.size, stop_delta)
        r, st = _fixed_pitch_rpm(prop, engine, v[sel], air, throttle[sel], d)
        delta[sel] = stop_delta
        rpm[sel] = r
        status[sel] = np.where(st == OK, label, st)

    return _finish(prop, engine, v, air, rpm, delta, status, throttle)


# ---------------------------------------------------------------------------
# Level-flight trim: what does it take to make a given thrust?
# ---------------------------------------------------------------------------

def _trim_rpm(prop: Propeller, engine: PistonEngine, v: np.ndarray, air: AirState,
              t_req: np.ndarray, delta: np.ndarray, n_hi: np.ndarray):
    def f(rows, rpm):
        sol = prop.solve(rpm, v[rows][:, None], air, delta[rows][:, None])
        f = t_req[rows][:, None] - sol["thrust"]
        return np.where(sol["physical"], f, np.inf)

    rpm, kind = solve_decreasing(f, np.full(v.size, engine.min_prop_rpm), n_hi, "last")
    # f = required - produced falls with RPM; "never crosses" means not enough
    # thrust even at the top of the range.
    status = np.full(v.size, OK, dtype=object)
    status[kind == NEVER_CROSSES] = THRUST_NOT_REACHABLE
    status[kind == NONPOSITIVE_AT_START] = BELOW_RUNNING_RANGE
    return rpm, status


def thrust_trim_fixed(prop: Propeller, engine: PistonEngine, thrust: np.ndarray,
                      v_inf, air: AirState) -> Match:
    """Fixed pitch: the RPM (and so power) that makes ``thrust`` at each speed."""
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    t_req = np.broadcast_to(np.asarray(thrust, dtype=float), v.shape).copy()
    delta = np.zeros(v.size)
    rpm, status = _trim_rpm(prop, engine, v, air, t_req, delta,
                            np.full(v.size, engine.max_prop_rpm))
    match = _finish(prop, engine, v, air, rpm, delta, status, None)
    return match


def thrust_trim_constant_speed(prop: Propeller, engine: PistonEngine, prop_rpm: float,
                               thrust: np.ndarray, v_inf, air: AirState,
                               delta_max: float) -> Match:
    """Constant speed at a set RPM: the collective that makes ``thrust``.

    If the fine stop already makes more thrust than needed at the set RPM, the
    governor bottoms out and RPM falls below the set value; that point is
    solved as a fixed-pitch propeller on the fine stop.
    """
    v = np.atleast_1d(np.asarray(v_inf, dtype=float))
    t_req = np.broadcast_to(np.asarray(thrust, dtype=float), v.shape).copy()
    rpm_set = float(prop_rpm)

    def f(rows, delta):
        sol = prop.solve(np.full(delta.shape, rpm_set), v[rows][:, None], air, delta)
        f = t_req[rows][:, None] - sol["thrust"]
        return np.where(sol["physical"], f, np.inf)

    delta, kind = solve_decreasing(f, np.zeros(v.size), np.full(v.size, float(delta_max)))
    status = np.full(v.size, OK, dtype=object)
    rpm = np.full(v.size, rpm_set)
    status[kind == NEVER_CROSSES] = THRUST_NOT_REACHABLE
    rpm[kind == NEVER_CROSSES] = np.nan

    fine = np.nonzero(kind == NONPOSITIVE_AT_START)[0]
    if fine.size:
        d0 = np.zeros(fine.size)
        r, st = _trim_rpm(prop, engine, v[fine], air, t_req[fine], d0,
                          np.full(fine.size, rpm_set))
        rpm[fine] = r
        status[fine] = np.where(st == OK, ON_FINE_STOP, st)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    return _finish(prop, engine, v, air, rpm, delta, status, None)


# ---------------------------------------------------------------------------
# Final solve
# ---------------------------------------------------------------------------

def _finish(prop: Propeller, engine: PistonEngine, v: np.ndarray, air: AirState,
            rpm: np.ndarray, delta: np.ndarray, status: np.ndarray,
            throttle: np.ndarray | None) -> Match:
    """Solve once at the matched points and attach engine quantities."""
    has_point = np.isfinite(rpm)
    rpm_eval = np.where(has_point, rpm, engine.max_prop_rpm)
    sol = prop.solve(rpm_eval, v, air, delta)

    avail = engine.full_throttle_power(rpm_eval, air)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = sol["power"] / np.where(avail > 0.0, avail, np.nan)
    thr = needed if throttle is None else np.where(status == RPM_LIMITED, needed, throttle)

    status = status.copy()
    # Absorbing no power means the airflow is turning the engine.
    live = np.isin(status, (OK, RPM_LIMITED, ON_FINE_STOP, ON_COARSE_STOP))
    status[live & has_point & (sol["power"] <= 0.0)] = WINDMILLING
    # A set RPM, set power or trim point that needs more than full throttle.
    live = np.isin(status, (OK, ON_FINE_STOP, ON_COARSE_STOP))
    over = live & has_point & (np.nan_to_num(np.asarray(thr, dtype=float)) > 1.0 + 1e-9)
    status[over] = POWER_NOT_AVAILABLE

    for key, val in list(sol.items()):
        if isinstance(val, np.ndarray) and val.dtype.kind == "f":
            mask = has_point.reshape((-1,) + (1,) * (val.ndim - 1))
            sol[key] = np.where(mask, val, np.nan)
    sol["available_power"] = np.where(has_point, avail, np.nan)
    sol["engine_rpm"] = np.where(has_point, engine.engine_rpm(rpm_eval), np.nan)

    return Match(v_inf=v, status=status, prop_rpm=np.where(has_point, rpm, np.nan),
                 delta_pitch=delta, throttle=np.where(has_point, thr, np.nan),
                 solution=sol)


def beta75(geometry: BladeGeometry, delta_pitch: float) -> float:
    """Blade angle at 0.75 R (degrees) with a collective applied."""
    return math.degrees(geometry.beta75() + delta_pitch)


__all__ = ["Propeller", "Match", "solve_decreasing", "fixed_pitch", "fixed_pitch_at_rpm",
           "fixed_pitch_at_power", "constant_speed", "thrust_trim_fixed",
           "thrust_trim_constant_speed", "beta75", "OK", "RPM_LIMITED", "WINDMILLING",
           "TOO_COARSE", "POWER_NOT_AVAILABLE", "ON_FINE_STOP", "ON_COARSE_STOP",
           "THRUST_NOT_REACHABLE", "BELOW_RUNNING_RANGE", "VALID_STATUSES",
           "FOUND", "NEVER_CROSSES", "NONPOSITIVE_AT_START"]
