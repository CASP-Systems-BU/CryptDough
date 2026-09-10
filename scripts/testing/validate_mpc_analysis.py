#!/usr/bin/env python3
"""Independent plaintext oracle for ``playground/mpc-analysis.cpp``.

The MPC program can dump the cohort it ran on::

    ./mpc-analysis -S describe -r 200 -O /tmp/dump

This script reads that dump and recomputes every node of the pipeline in double
precision, so the secure results can be checked against an implementation that
shares no code with them.

The descriptive and aggregate nodes are exact counting problems and must match
to the last digit. The regression coefficients are compared with a tolerance,
because the secure fits run in 16-bit fixed point.

Usage::

    python3 validate_mpc_analysis.py /tmp/dump
    python3 validate_mpc_analysis.py /tmp/dump --model 2a --scope any
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

# Follow-up windows the pipeline reports on.
FU_WINDOWS: tuple[int, ...] = (1, 3, 6)

# PROC GLIMMIX CLASS makes the LAST sorted level the reference; the dummies
# below omit it, matching the secure implementation.
GENDER_LEVELS: tuple[int, ...] = (1, 2, 3)
HISPANIC_LEVELS: tuple[int, ...] = (0, 1)

# Age is centred for conditioning in the secure code; mirror it exactly.
AGE_CENTRE: float = 40.0

# Damping and clamping constants copied from the secure implementation, so the
# two are the same estimator rather than merely similar ones.
ETA_CLAMP: float = 10.0
NEWTON_STEP_CLAMP: float = 4.0
NEWTON_ITERATIONS: int = 5
IRLS_ITERATIONS: int = 8
IRLS_RIDGE: float = 1e-3

# The secure code clamps sigma^2 into a public band so Log keeps a positive
# argument; mirror the lower bound when scoring its parameters.
SIGMA2_FLOOR: float = 1e-3

# Finite-difference step for the observed-information Hessian. Matches
# kHessianStep in the secure implementation so the two agree by construction
# rather than by luck.
HESSIAN_STEP: float = 0.1


@dataclass
class Cohort:
    """One input table, as dumped by the MPC program."""

    subject_id: np.ndarray
    newage: np.ndarray
    gender: np.ndarray
    hispanic: np.ndarray
    index_visit: np.ndarray
    visit_num: np.ndarray
    final_visit: np.ndarray
    fu_month: np.ndarray
    fu_month_present: np.ndarray
    data_source: np.ndarray
    sisa: np.ndarray
    sa: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.subject_id.size)

    @property
    def patients(self) -> int:
        return int(np.unique(self.subject_id).size)


_COLUMNS: dict[str, str] = {
    "[subject_id]": "subject_id",
    "newage": "newage",
    "[gender]": "gender",
    "[hispanic]": "hispanic",
    "[index_visit]": "index_visit",
    "visit_num": "visit_num",
    "[final_visit]": "final_visit",
    "[fu_month]": "fu_month",
    "[fu_month_present]": "fu_month_present",
    "[data_source]": "data_source",
    "[sisa]": "sisa",
    "[sa]": "sa",
}


def read_cohort(path: Path) -> Cohort:
    """Read one dumped cohort CSV."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [name for name in _COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    columns = {
        field_name: np.array([int(row[header]) for row in rows], dtype=np.int64)
        for header, field_name in _COLUMNS.items()
    }
    return Cohort(**columns)


# --------------------------------------------------------------------------
# Descriptive and aggregate nodes
# --------------------------------------------------------------------------


def univariate(values: np.ndarray) -> dict[str, float]:
    """Moments and PCTLDEF=4 quantiles, matching the secure implementation."""
    ordered = np.sort(values.astype(float))
    n = ordered.size

    def percentile(q: float) -> float:
        position = (n - 1) * q
        low = int(np.floor(position))
        frac = position - low
        if frac <= 0.0 or low + 1 >= n:
            return float(ordered[low])
        return float(ordered[low] + frac * (ordered[low + 1] - ordered[low]))

    return {
        "n": float(n),
        "mean": float(np.mean(ordered)),
        "sd": float(statistics.stdev(ordered.tolist())) if n > 1 else 0.0,
        "min": float(ordered[0]),
        "q1": percentile(0.25),
        "median": percentile(0.50),
        "q3": percentile(0.75),
        "max": float(ordered[-1]),
    }


def frequency(values: np.ndarray, levels: Iterable[int]) -> list[tuple[int, int, float]]:
    """One-way frequency table over a public level set."""
    counts = [(level, int(np.count_nonzero(values == level))) for level in levels]
    total = sum(count for _, count in counts)
    return [
        (level, count, 100.0 * count / total if total else 0.0) for level, count in counts
    ]


def value_histogram(values: np.ndarray) -> list[tuple[int, int, float]]:
    """Full distribution of a small-integer column."""
    unique, counts = np.unique(values, return_counts=True)
    total = int(counts.sum())
    return [
        (int(value), int(count), 100.0 * int(count) / total if total else 0.0)
        for value, count in zip(unique, counts)
    ]


def sisa_counts(cohort: Cohort) -> list[dict[str, float]]:
    """The sisa_perct_cnt table: distinct-patient rates plus encounter counts."""
    out: list[dict[str, float]] = []
    for window in FU_WINDOWS:
        in_window = (cohort.fu_month_present == 1) & (cohort.fu_month == window)
        total_pts = np.unique(cohort.subject_id[in_window]).size
        with_sisa = np.unique(cohort.subject_id[in_window & (cohort.sisa == 1)]).size
        # No DISTINCT here, deliberately: the source query counts encounters.
        total_sa = int(np.count_nonzero(in_window & (cohort.sa == 1)))
        out.append(
            {
                "window": float(window),
                "pts_with_sisa": float(with_sisa),
                "total_pts": float(total_pts),
                "sisa_pct": 100.0 * with_sisa / total_pts if total_pts else 0.0,
                "total_sa": float(total_sa),
            }
        )
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """One of the six specifications, fitted to one population."""

    step: str
    time: str  # "visit_num" or "fu_month"
    covars: bool = False
    interaction: bool = False
    random_intercept: bool = True
    drop_hispanic: bool = False
    terms: list[str] = field(default_factory=list)


def all_specs(scope: str) -> list[ModelSpec]:
    """The specifications that can run on a given population."""
    bases = [
        ModelSpec("2a", "visit_num", False, False, True),
        ModelSpec("2b", "fu_month", False, False, True),
        ModelSpec("5a", "visit_num", True, False, True),
        ModelSpec("5b", "fu_month", True, False, True),
        ModelSpec("6a", "visit_num", True, True, False),
        ModelSpec("6b", "fu_month", True, True, False),
    ]
    out = []
    for spec in bases:
        if spec.interaction and scope != "any":
            continue
        spec.drop_hispanic = spec.step == "5b" and scope == "umass"
        out.append(spec)
    return out


def design(cohort: Cohort, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the design matrix, outcome and row mask for one model."""
    time = (cohort.visit_num if spec.time == "visit_num" else cohort.fu_month).astype(float)
    mask = np.ones(cohort.rows, dtype=float)
    if spec.time == "fu_month":
        mask[cohort.fu_month_present == 0] = 0.0

    columns: list[np.ndarray] = [np.ones(cohort.rows), time]
    names: list[str] = ["Intercept", spec.time]

    if spec.covars:
        columns.append(cohort.newage.astype(float) - AGE_CENTRE)
        names.append("newage")
        for level in GENDER_LEVELS[:-1]:
            columns.append((cohort.gender == level).astype(float))
            names.append(f"gender={level}")
        if not spec.drop_hispanic:
            for level in HISPANIC_LEVELS[:-1]:
                columns.append((cohort.hispanic == level).astype(float))
                names.append(f"hispanic={level}")
    if spec.interaction:
        umass = (cohort.data_source == 1).astype(float)
        columns.append(umass)
        names.append("data_source=UMass")
        columns.append(umass * time)
        names.append(f"data_source=UMass * {spec.time}")

    matrix = np.column_stack(columns) * mask[:, None]
    outcome = cohort.sisa.astype(float) * mask
    return matrix, outcome, mask, names


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta)
    positive = eta >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    out[~positive] = exp_eta / (1.0 + exp_eta)
    return out


def fit_irls(
    matrix: np.ndarray, outcome: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-effects logistic regression by IRLS, as steps 6a and 6b are fitted."""
    width = matrix.shape[1]
    beta = np.zeros(width)
    inverse = np.eye(width)
    for _ in range(IRLS_ITERATIONS):
        eta = np.clip(matrix @ beta, -ETA_CLAMP, ETA_CLAMP)
        prob = _sigmoid(eta)
        weights = mask * prob * (1.0 - prob)
        residual = mask * (outcome - prob)
        gram = matrix.T @ (matrix * weights[:, None]) + IRLS_RIDGE * np.eye(width)
        inverse = np.linalg.inv(gram)
        beta = beta + np.clip(inverse @ (matrix.T @ residual), -NEWTON_STEP_CLAMP, NEWTON_STEP_CLAMP)
    return beta, np.sqrt(np.maximum(0.0, np.diag(inverse)))


def laplace_objective(
    matrix: np.ndarray,
    outcome: np.ndarray,
    mask: np.ndarray,
    subject_id: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Build the negative Laplace-approximated marginal log-likelihood.

    Deliberately mirrors ``FlatNegMarginalLogLik`` in the secure implementation
    term for term, including the fixed five-step damped Newton search for each
    conditional mode and the same clamps, so that a disagreement points at the
    fixed-point arithmetic rather than at a different estimator.

    Parameters are ``[beta..., log_sigma]``, with the variance parameterised as
    ``sigma^2 = exp(2 * log_sigma)`` to keep it strictly positive.
    """
    groups = [np.where(subject_id == value)[0] for value in np.unique(subject_id)]
    width = matrix.shape[1]

    def negative_log_likelihood(params: np.ndarray) -> float:
        beta, log_sigma = params[:width], params[width]
        sigma2 = float(np.exp(2.0 * log_sigma))
        inv_sigma2 = 1.0 / sigma2
        linear = matrix @ beta
        total = 0.0
        for index in groups:
            rows_mask = mask[index]
            if not np.any(rows_mask):
                continue
            base, response = linear[index], outcome[index]
            mode = 0.0
            for _ in range(NEWTON_ITERATIONS):
                eta = np.clip(base + mode, -ETA_CLAMP, ETA_CLAMP)
                prob = _sigmoid(eta)
                gradient = -mode * inv_sigma2 + float(np.sum(rows_mask * (response - prob)))
                curvature = inv_sigma2 + float(np.sum(rows_mask * prob * (1.0 - prob)))
                mode += float(
                    np.clip(gradient / curvature, -NEWTON_STEP_CLAMP, NEWTON_STEP_CLAMP)
                )
            eta = np.clip(base + mode, -ETA_CLAMP, ETA_CLAMP)
            prob = _sigmoid(eta)
            conditional = float(np.sum(rows_mask * (response * eta - np.logaddexp(0.0, eta))))
            curvature = inv_sigma2 + float(np.sum(rows_mask * prob * (1.0 - prob)))
            total += (
                conditional
                - 0.5 * mode * mode * inv_sigma2
                - 0.5 * np.log(sigma2)
                - 0.5 * np.log(curvature)
            )
        return -total

    return negative_log_likelihood


def fit_laplace_glmm(
    matrix: np.ndarray,
    outcome: np.ndarray,
    mask: np.ndarray,
    subject_id: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the mixed model; return (beta, sigma^2, attained objective)."""
    from scipy.optimize import minimize  # imported lazily: only the models need it

    objective = laplace_objective(matrix, outcome, mask, subject_id)
    width = matrix.shape[1]
    result = minimize(objective, np.zeros(width + 1), method="BFGS")
    return result.x[:width], float(np.exp(2.0 * result.x[width])), float(result.fun)


def observed_information_se(
    objective: Callable[[np.ndarray], float], point: np.ndarray, reported: int
) -> np.ndarray:
    """Standard errors from a finite-difference Hessian at ``point``.

    The same construction the secure code uses -- central second differences on
    the diagonal, forward mixed differences off it, at the same step size -- so
    the two are comparable term by term.
    """
    dim = point.size
    step = HESSIAN_STEP
    base = objective(point)

    plus = np.empty(dim)
    minus = np.empty(dim)
    for k in range(dim):
        delta = np.zeros(dim)
        delta[k] = step
        plus[k] = objective(point + delta)
        minus[k] = objective(point - delta)

    hessian = np.zeros((dim, dim))
    for k in range(dim):
        hessian[k, k] = (plus[k] - 2.0 * base + minus[k]) / step**2
    for j in range(dim):
        for k in range(j + 1, dim):
            delta = np.zeros(dim)
            delta[j] = step
            delta[k] = step
            value = (objective(point + delta) - plus[j] - plus[k] + base) / step**2
            hessian[j, k] = hessian[k, j] = value

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.full(reported, np.nan)
    variances = np.diag(covariance)[:reported]
    return np.where(variances > 0.0, np.sqrt(np.abs(variances)), np.nan)


def report_descriptive(cohort: Cohort) -> None:
    """Print the d1a and d1b nodes."""
    print("\n=== d1a  visits per patient  [any_system, final_visit = 1] ===")
    stats = univariate(cohort.visit_num[cohort.final_visit == 1])
    for key, value in stats.items():
        print(f"    {key:8s}{value:.4f}")
    print("\n    distribution")
    for value, count, percent in value_histogram(cohort.visit_num[cohort.final_visit == 1]):
        print(f"    {value:<22}{count:<12}{percent:.2f}")

    print("\n=== d1b  demographics  [any_system, index_visit = 1] ===")
    at_index = cohort.index_visit == 1
    for key, value in univariate(cohort.newage[at_index]).items():
        print(f"    {key:8s}{value:.4f}")
    for label, values, levels in (
        ("gender", cohort.gender[at_index], GENDER_LEVELS),
        ("hispanic", cohort.hispanic[at_index], HISPANIC_LEVELS),
    ):
        print(f"\n    {label}")
        for level, count, percent in frequency(values, levels):
            print(f"    {level:<22}{count:<12}{percent:.2f}")


def report_counts(cohort: Cohort, label: str) -> None:
    """Print one sisa_perct_cnt table."""
    print(f"\n=== sisa_perct_cnt  [{label}] ===")
    print(f"  {'window':<10}{'pts_with_sisa':<18}{'total_pts':<14}{'sisa_pct':<12}{'total_sa':<12}")
    for row in sisa_counts(cohort):
        print(
            f"  {int(row['window'])}m{'':<8}{int(row['pts_with_sisa']):<18}"
            f"{int(row['total_pts']):<14}{row['sisa_pct']:<12.2f}{int(row['total_sa']):<12}"
        )
    nulls = int(np.count_nonzero(cohort.fu_month_present == 0))
    print(f"  ({nulls} rows have a null fu_month and appear in neither numerator nor denominator)")


def fit_all(
    cohorts: dict[str, Cohort], only_step: str | None
) -> dict[tuple[str, str, str], float]:
    """Fit every model this oracle covers, keyed by (step, population, term)."""
    out: dict[tuple[str, str, str], float] = {}
    for scope, cohort in cohorts.items():
        for spec in all_specs(scope):
            if only_step and spec.step != only_step:
                continue
            matrix, outcome, mask, names = design(cohort, spec)
            if spec.random_intercept:
                beta, sigma2, _ = fit_laplace_glmm(matrix, outcome, mask, cohort.subject_id)
                out[(spec.step, scope, "random_intercept_variance")] = sigma2
            else:
                beta, _ = fit_irls(matrix, outcome, mask)
            for name, estimate in zip(names, beta):
                out[(spec.step, scope, name)] = float(estimate)
    return out


def report_models(cohorts: dict[str, Cohort], only_step: str | None) -> None:
    """Fit and print every model this oracle covers."""
    print(f"\n{'model':<8}{'population':<18}{'term':<32}{'estimate':<14}{'SE':<12}")
    for scope, cohort in cohorts.items():
        for spec in all_specs(scope):
            if only_step and spec.step != only_step:
                continue
            matrix, outcome, mask, names = design(cohort, spec)
            if spec.random_intercept:
                beta, sigma2, _ = fit_laplace_glmm(matrix, outcome, mask, cohort.subject_id)
                errors: Sequence[float] = [float("nan")] * beta.size
            else:
                beta, errors = fit_irls(matrix, outcome, mask)
                sigma2 = float("nan")
            for name, estimate, error in zip(names, beta, errors):
                error_text = "n/a" if np.isnan(error) else f"{error:<12.5f}"
                print(f"{spec.step:<8}{scope:<18}{name:<32}{estimate:<14.5f}{error_text}")
            if not np.isnan(sigma2):
                print(f"{'':<8}{'':<18}{'random intercept variance':<32}{sigma2:<14.5f}")


@dataclass
class MpcFit:
    """One fit as reported by the MPC program's RESULT lines."""

    step: str
    population: str
    terms: list[str] = field(default_factory=list)
    estimates: list[float] = field(default_factory=list)
    standard_errors: list[float] = field(default_factory=list)
    sigma2: float | None = None


def parse_mpc_results(path: Path) -> dict[tuple[str, str], MpcFit]:
    """Read the RESULT lines the MPC program prints at the end of a model run.

    Term order is preserved, because the two implementations name categorical
    dummies differently (label versus level code) and are matched positionally.
    """
    out: dict[tuple[str, str], MpcFit] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("RESULT\t"):
            continue
        _, step, population, term, estimate, error = line.split("\t")
        fit = out.setdefault((step, population), MpcFit(step, population))
        if term == "random_intercept_variance":
            fit.sigma2 = float(estimate)
        else:
            fit.terms.append(term)
            fit.estimates.append(float(estimate))
            fit.standard_errors.append(float("nan") if error == "nan" else float(error))
    return out


def compare(
    mpc: dict[tuple[str, str], MpcFit],
    cohorts: dict[str, Cohort],
    only_step: str | None,
    coefficient_tolerance: float,
    objective_tolerance: float,
    se_tolerance: float,
    nuisance_se_tolerance: float,
) -> bool:
    """Score the secure fits against the plaintext oracle.

    The two model families are judged on different criteria, because only one of
    them has a well-determined answer to compare against:

    * The step 6 models are plain logistic regressions, and the oracle runs the
      same IRLS estimator, so the coefficients themselves must agree to roughly
      the fixed-point floor.

    * The mixed models are optimisation problems whose likelihood has genuinely
      flat directions -- a near-collinear sparse dummy, or a variance sitting on
      the sigma^2 = 0 boundary that the secure code's positivity floor cannot
      reach. Two optimisers can then stop at visibly different coefficients that
      fit equally well, so comparing coefficients would reject a correct
      implementation. What has to hold is that the secure parameters attain the
      same log-likelihood, which is what is checked here.
    """
    print(f"\n{'model':<7}{'population':<12}{'criterion':<26}{'value':<14}{'tolerance':<12}")
    passed = True

    for (step, population), fit in sorted(mpc.items()):
        if only_step and step != only_step:
            continue
        cohort = cohorts[population]
        spec = next(s for s in all_specs(population) if s.step == step)
        matrix, outcome, mask, _names = design(cohort, spec)

        if not spec.random_intercept:
            beta, _errors = fit_irls(matrix, outcome, mask)
            worst = max(
                abs(secure - plain) for secure, plain in zip(fit.estimates, beta)
            )
            ok = worst <= coefficient_tolerance
            passed &= ok
            print(
                f"{step:<7}{population:<12}{'max |coef difference|':<26}"
                f"{worst:<14.3e}{coefficient_tolerance:<12.1e}"
                f"{'' if ok else '  <-- FAIL'}"
            )
            continue

        objective = laplace_objective(matrix, outcome, mask, cohort.subject_id)
        _beta, _sigma2, best = fit_laplace_glmm(matrix, outcome, mask, cohort.subject_id)
        # The secure code holds sigma^2 above a floor for Log's domain; respect it
        # so the comparison is against a point the secure optimiser could reach.
        sigma2 = max(fit.sigma2 if fit.sigma2 is not None else 1.0, SIGMA2_FLOOR)
        point = np.concatenate([np.array(fit.estimates), [0.5 * np.log(sigma2)]])
        gap = objective(point) - best
        ok = gap <= objective_tolerance
        passed &= ok
        print(
            f"{step:<7}{population:<12}{'-logL excess over optimum':<26}"
            f"{gap:<14.3e}{objective_tolerance:<12.1e}"
            f"{'' if ok else '  <-- FAIL'}"
        )

        # Standard errors are scored at the SECURE optimum, so this measures the
        # finite-difference machinery rather than the gap between the two optima.
        #
        # The time coefficient and the nuisance terms are held to different
        # standards, for a reason that shows up clearly in the data rather than
        # being a convenience. The Hessian is near-singular in the collinear
        # directions -- `hispanic` is about 90% ones and so nearly the intercept
        # -- and inverting it there amplifies the objective's fixed-point noise
        # without bound. Both implementations suffer, and they suffer
        # differently. The time term sits in a well-determined direction and
        # must agree closely; the intercept and the sparse dummies get a loose
        # bound, and the secure program prints an ill-conditioning warning
        # alongside them.
        if fit.standard_errors and not all(np.isnan(fit.standard_errors)):
            oracle_se = observed_information_se(objective, point, len(fit.estimates))
            scored = [
                (index, secure, plain)
                for index, (secure, plain) in enumerate(zip(fit.standard_errors, oracle_se))
                if np.isfinite(secure) and np.isfinite(plain)
            ]
            # Index 1 is the time axis; index 0 is the intercept.
            time_errors = [
                abs(secure - plain) / max(plain, 1e-6)
                for index, secure, plain in scored
                if index == 1
            ]
            other_errors = [
                abs(secure - plain) / max(plain, 1e-6)
                for index, secure, plain in scored
                if index != 1
            ]
            if time_errors:
                worst = max(time_errors)
                ok_se = worst <= se_tolerance
                passed &= ok_se
                print(
                    f"{step:<7}{population:<12}{'time-term SE, rel. diff':<26}"
                    f"{worst:<14.3e}{se_tolerance:<12.1e}"
                    f"{'' if ok_se else '  <-- FAIL'}"
                )
            if other_errors:
                worst = max(other_errors)
                ok_se = worst <= nuisance_se_tolerance
                passed &= ok_se
                print(
                    f"{step:<7}{population:<12}{'other SEs, rel. diff':<26}"
                    f"{worst:<14.3e}{nuisance_se_tolerance:<12.1e}"
                    f"{'' if ok_se else '  <-- FAIL'}"
                )

    print("\nVALIDATION: PASS" if passed else "\nVALIDATION: *** FAIL ***")
    return passed


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="directory written by mpc-analysis -O")
    parser.add_argument("--model", default=None, help="only fit this step, e.g. 2a")
    parser.add_argument("--skip-models", action="store_true", help="descriptive nodes only")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="captured stdout of `mpc-analysis -S models`; diff its RESULT lines against the oracle",
    )
    parser.add_argument(
        "--coefficient-tolerance",
        type=float,
        default=1e-3,
        help="max coefficient difference for the fixed-effects models (default 1e-3)",
    )
    parser.add_argument(
        "--objective-tolerance",
        type=float,
        default=0.25,
        help="max -logL excess over the oracle optimum for the mixed models (default 0.25)",
    )
    parser.add_argument(
        "--se-tolerance",
        type=float,
        default=0.10,
        help="max relative SE difference on the time coefficient (default 0.10)",
    )
    parser.add_argument(
        "--nuisance-se-tolerance",
        type=float,
        default=0.40,
        help=(
            "max relative SE difference on the intercept and covariate dummies, whose "
            "Hessian directions are near-singular (default 0.40)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    cohorts = {
        "any": read_cohort(args.dump_dir / "any_system.csv"),
        "umass": read_cohort(args.dump_dir / "umass_system.csv"),
        "nonumass": read_cohort(args.dump_dir / "nonumass_system.csv"),
    }
    for name, cohort in cohorts.items():
        LOGGER.info("%s: %d rows, %d patients", name, cohort.rows, cohort.patients)

    if args.compare is not None:
        mpc_results = parse_mpc_results(args.compare)
        if not mpc_results:
            raise SystemExit(f"no RESULT lines found in {args.compare}")
        ok = compare(
            mpc_results,
            cohorts,
            args.model,
            args.coefficient_tolerance,
            args.objective_tolerance,
            args.se_tolerance,
            args.nuisance_se_tolerance,
        )
        raise SystemExit(0 if ok else 1)

    report_descriptive(cohorts["any"])
    for name, label in (
        ("any", "all systems"),
        ("umass", "UMass only"),
        ("nonumass", "non-UMass only"),
    ):
        report_counts(cohorts[name], label)

    if not args.skip_models:
        report_models(cohorts, args.model)


if __name__ == "__main__":
    main()
