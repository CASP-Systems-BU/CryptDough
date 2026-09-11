#!/usr/bin/env python3
"""Independent plaintext oracle for ``playground/mpc-analysis.cpp``.

The MPC program can dump the two data owners' halves of the base table::

    ./mpc-analysis -S describe -r 200 -O /tmp/dump

This script reads that dump and recomputes every node of the pipeline, so the
secure results can be checked against an implementation that shares no code with
them.

Two stages, two mechanisms:

* The **relational** stage -- lineage node ``flagged_dx`` pass 2 and the per-system
  re-sequencing -- is recomputed by running ``playground/sql/sequencing.sql``
  through the standard-library ``sqlite3`` module. That is the same file
  ``playground/sqlite_oracle.h`` embeds, so the C++ and Python oracles execute
  identical SQL and cannot drift apart.
* The **aggregate and model** nodes are recomputed here in double precision with
  numpy.

The aggregate nodes are exact counting problems and must match to the last digit.
The regression coefficients are compared with a tolerance, because the secure
fits run in 16-bit fixed point.

Note this needs the UNION of both owners' halves, which only exists on the
synthetic path -- ``-O`` is a no-op under ``-D``. In a real deployment no party
holds the union, so neither this oracle nor the in-process C++ one can run.

Usage::

    python3 validate_mpc_analysis.py /tmp/dump
    python3 validate_mpc_analysis.py /tmp/dump --model 2a
    python3 validate_mpc_analysis.py /tmp/dump --compare /tmp/models.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

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
NEWTON_ITERATIONS: int = 3
IRLS_ITERATIONS: int = 8
IRLS_RIDGE: float = 1e-3

# The secure code clamps sigma^2 into a public band so Log keeps a positive
# argument; mirror the lower bound when scoring its parameters.
SIGMA2_FLOOR: float = 1e-3

# Step for the REFERENCE Hessian. Small on purpose: this runs in double
# precision, so truncation error dominates and a smaller step is strictly
# better. It is deliberately NOT the secure code's step, which has to be large
# to keep fixed-point noise from swamping the difference.
REFERENCE_HESSIAN_STEP: float = 1e-3


@dataclass
class Cohort:
    """One input table, as dumped by the MPC program."""

    subject_id: np.ndarray
    # The ordering key the sequencing is defined on. The removed per-cohort CSVs
    # did not carry it; it is needed now that the cohorts are derived here.
    encounter_dt: np.ndarray
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


# Columns of flagged_dx_owner_{a,b}.csv (etl.h WriteFlaggedCsv). The sequencing
# needs a subset; `sisa` and `sa` are renamed to the analysis names here, exactly
# as playground/sqlite_oracle.h does when it fills the same table.
_FLAGGED_COLUMNS: dict[str, str] = {
    "subject_id": "subject_id",
    "encounter_dt": "encounter_dt",
    "newage": "newage",
    "gender": "gender",
    "hispanic": "hispanic",
    "data_source": "data_source",
    "suicide_acutecare_icd_narrow": "sisa",
    "sa_icd_narrow": "sa",
}

# Scope name -> the ?1 the queries bind. 0 selects every system.
_SCOPE_PARAM: dict[str, int] = {"any": 0, "umass": 1, "nonumass": 2}

# Repository-relative location of the shared query. This file is the single
# source of truth: playground/sqlite_oracle.h embeds the same text via cmake.
DEFAULT_SQL_PATH = Path(__file__).resolve().parents[2] / "playground" / "sql" / "sequencing.sql"


def read_flagged(path: Path) -> list[tuple[int, ...]]:
    """Read one owner's post-pass-1 half, as rows ready for the sqlite insert."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [name for name in _FLAGGED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return [tuple(int(row[header]) for header in _FLAGGED_COLUMNS) for row in rows]


def build_cohorts(dump_dir: Path, sql_path: Path) -> dict[str, Cohort]:
    """Derive the three analysis cohorts from the two owners' halves.

    This is the relational stage -- lineage node ``flagged_dx`` pass 2 plus the
    ``umass`` / ``nonumass`` re-sequencing -- and it is deliberately NOT
    reimplemented here. The query in ``sql_path`` is the same text
    ``playground/sqlite_oracle.h`` runs, so this is a genuine second execution of
    one oracle rather than a third hand-transcription that could drift.

    The query is run once per scope. That matters: applying the scope filter
    before the window is what produces the per-system re-sequencing, so
    ``visit_num`` and ``fu_month`` genuinely differ between the three cohorts.
    """
    query = sql_path.read_text()

    connection = sqlite3.connect(":memory:")
    try:
        # No PRIMARY KEY or UNIQUE on (subject_id, encounter_dt): same-day
        # repeat encounters are legitimate and a constraint would drop them.
        connection.execute(
            """
            CREATE TABLE flagged_dx (
                owner        INTEGER NOT NULL,
                rid          INTEGER NOT NULL,
                subject_id   INTEGER NOT NULL,
                encounter_dt INTEGER NOT NULL,
                newage       INTEGER NOT NULL,
                gender       INTEGER NOT NULL,
                hispanic     INTEGER NOT NULL,
                data_source  INTEGER NOT NULL,
                sisa         INTEGER NOT NULL,
                sa           INTEGER NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX ix_flagged ON flagged_dx (subject_id, encounter_dt)")
        for owner, name in enumerate(("flagged_dx_owner_a.csv", "flagged_dx_owner_b.csv")):
            path = dump_dir / name
            if not path.exists():
                raise SystemExit(
                    f"{path} not found. Write it with `mpc-analysis -S describe -O {dump_dir}`; "
                    "note -O only fires on the synthetic path."
                )
            rows = read_flagged(path)
            LOGGER.info("%s: %d rows", name, len(rows))
            connection.executemany(
                "INSERT INTO flagged_dx (owner, rid, subject_id, encounter_dt, newage,"
                " gender, hispanic, data_source, sisa, sa)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(owner, rid, *row) for rid, row in enumerate(rows)],
            )

        cohorts: dict[str, Cohort] = {}
        for scope, param in _SCOPE_PARAM.items():
            # The query uses the numbered placeholder ?1 (it appears twice, and the
            # C++ side binds it by index). Python's sqlite3 treats ?NNN as a NAMED
            # parameter, so it must be bound by mapping: a positional sequence is
            # deprecated since 3.12.
            result = connection.execute(query, {"1": param}).fetchall()
            cohorts[scope] = _cohort_from_rows(result)
        return cohorts
    finally:
        connection.close()


# Column order is fixed by the SELECT list in sequencing.sql, and is the same
# order playground/sqlite_oracle.h reads.
_SQL_COLUMNS: tuple[str, ...] = (
    "subject_id", "encounter_dt", "newage", "gender", "hispanic",
    "index_visit", "visit_num", "final_visit", "fu_month", "data_source",
    "sisa", "sa",
)


def _cohort_from_rows(rows: Sequence[Sequence[int | None]]) -> Cohort:
    """Turn the query result into a Cohort, translating SQL NULL."""
    index = {name: position for position, name in enumerate(_SQL_COLUMNS)}

    def column(name: str) -> np.ndarray:
        return np.array([row[index[name]] for row in rows], dtype=np.int64)

    # fu_month arrives as a genuine SQL NULL. MPC has no NULL, so the pipeline
    # carries presence in its own column; mirror that here.
    raw_fu = [row[index["fu_month"]] for row in rows]
    fu_present = np.array([0 if value is None else 1 for value in raw_fu], dtype=np.int64)
    fu_month = np.array([0 if value is None else value for value in raw_fu], dtype=np.int64)

    return Cohort(
        subject_id=column("subject_id"),
        encounter_dt=column("encounter_dt"),
        newage=column("newage"),
        gender=column("gender"),
        hispanic=column("hispanic"),
        index_visit=column("index_visit"),
        visit_num=column("visit_num"),
        final_visit=column("final_visit"),
        fu_month=fu_month,
        fu_month_present=fu_present,
        data_source=column("data_source"),
        sisa=column("sisa"),
        sa=column("sa"),
    )


# --------------------------------------------------------------------------
# Descriptive and aggregate nodes
# --------------------------------------------------------------------------


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
    """Accurate standard errors from the observed information at ``point``.

    This is a *reference*, not a mirror of the secure construction. It runs in
    double precision, where the objective is good to about 1e-15, so it can
    afford a small step and take second differences of the objective directly.

    The secure code cannot do that: at 16-bit fixed point the objective is good
    to only about 1e-4, and dividing that by ``step**2`` is what made its
    earlier objective-difference Hessian unusable -- 17% to 46% off the truth,
    and indefinite in some directions purely from noise. It now takes first
    differences of its analytic gradient instead, which divides the noise by
    ``step`` only once. It is scored here against the accurate answer rather
    than against a deliberately degraded copy of its own method.
    """
    dim = point.size
    step = REFERENCE_HESSIAN_STEP
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
        # The reference above is accurate, so one tolerance covers every term.
        if fit.standard_errors and not all(np.isnan(fit.standard_errors)):
            oracle_se = observed_information_se(objective, point, len(fit.estimates))
            scored = [
                (index, secure, plain)
                for index, (secure, plain) in enumerate(zip(fit.standard_errors, oracle_se))
                if np.isfinite(secure) and np.isfinite(plain)
            ]
            # A single tolerance now covers every term. The split that used to
            # be here -- tight on the time coefficient, loose on the intercept
            # and the near-collinear dummies -- was an artefact of the old
            # objective-difference Hessian. Differencing the analytic gradient
            # divides the noise by the step once instead of twice, and the
            # weakly determined directions came back inside the same bound as
            # everything else.
            worst = max(
                abs(secure - plain) / max(plain, 1e-6) for _i, secure, plain in scored
            )
            ok_se = worst <= se_tolerance
            passed &= ok_se
            missing = len(fit.estimates) - len(scored)
            label = "max relative SE difference"
            if missing:
                label += f" ({missing} n/a)"
            print(
                f"{step:<7}{population:<12}{label:<26}"
                f"{worst:<14.3e}{se_tolerance:<12.1e}"
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
        help="max relative standard-error difference for the mixed models (default 0.10)",
    )
    parser.add_argument(
        "--sql",
        type=Path,
        default=DEFAULT_SQL_PATH,
        help=f"the sequencing query shared with the C++ oracle (default {DEFAULT_SQL_PATH})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.sql.exists():
        raise SystemExit(f"sequencing query not found: {args.sql}")
    cohorts = build_cohorts(args.dump_dir, args.sql)
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
        )
        raise SystemExit(0 if ok else 1)

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
