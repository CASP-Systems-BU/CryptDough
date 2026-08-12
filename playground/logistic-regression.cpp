// Mixed-effects (random-intercept) logistic regression fitted by the Laplace
// approximation to the marginal likelihood, maximized with a quasi-Newton (BFGS)
// optimizer.
//
// This mirrors the intent of SAS `PROC GLIMMIX` with `METHOD=LAPLACE` and
// `TECHNIQUE=QUANEW`, restricted to a single (scalar) random intercept per group.
// Everything runs in plaintext (no MPC / secret sharing) and depends only on the
// C++ standard library. See tasks/0001_mixed-effects-logistic-regression-laplace.md
// for the design document.
//
// Model (observation j in group i):
//   y_ij ~ Bernoulli(p_ij),   logit(p_ij) = x_ij . beta + u_i,   u_i ~ N(0, sigma^2)
//
// Marginal log-likelihood (random effects integrated out) is approximated per
// group by Laplace's method around the conditional mode u_hat_i:
//   log L_i ~= cll(u_hat_i) - 0.5 * u_hat_i^2 / sigma^2
//             - 0.5 * log(sigma^2) - 0.5 * log(A_i),
// where cll is the conditional Bernoulli log-likelihood and
//   A_i = sum_j p_ij (1 - p_ij) + 1 / sigma^2
// is the curvature (negative second derivative) of the integrand at u_hat_i.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// One cluster/group of observations sharing a common random intercept.
struct Group {
  std::vector<Vector> x;  // Fixed-effect design rows; each row has length p.
  Vector y;               // Binary outcomes (0/1), one per row of x.
};

// A full grouped dataset.
struct Dataset {
  std::vector<Group> groups;
  std::size_t num_fixed = 0;  // Number of fixed-effect coefficients (p).
};

// Result of the outer quasi-Newton optimization.
struct OptResult {
  Vector params;         // [beta_0..beta_{p-1}, s] with sigma = exp(s).
  double value = 0.0;    // Final objective value (negative marginal log-lik).
  int iterations = 0;    // Number of BFGS iterations performed.
  bool converged = false;
};

// ---------------------------------------------------------------------------
// Transcendental functions: exp, log, log1p.
//
// When USE_TAYLOR_MATH is defined to a nonzero value at compile time
// (e.g. `-DUSE_TAYLOR_MATH=1`), these are evaluated with in-file Taylor-series
// approximations whose range reduction is implemented entirely with
// multiply/divide-by-two loops -- the custom path calls no libm functions.
// Otherwise the standard-library versions are used. The standard library is the
// default, so builds without the flag are unchanged. This mirrors an eventual
// secure (MPC) port, where transcendental primitives are unavailable and must be
// approximated by polynomial series.
// ---------------------------------------------------------------------------
#ifndef USE_TAYLOR_MATH
#define USE_TAYLOR_MATH 0
#endif

// Trust-region cap on the per-iteration Newton step. A nearly flat prior (large
// sigma^2) combined with (quasi-)separable binary data pushes the mode far from
// 0; without a cap the undamped Newton step overshoots into overflow and then
// oscillates. Capping keeps every iterate finite while leaving normal steps
// (well under the cap) unaffected.
constexpr double kMaxNewtonStep = 4.0;

#if USE_TAYLOR_MATH

// High-precision constants for the range-reduction steps.
// constexpr double kLn2 = 0.6931471805599453094172321214582;
// constexpr double kSqrt2 = 1.4142135623730950488016887242097;
// constexpr double kSqrt1_2 = 0.7071067811865475244008443621048;

constexpr double kLn2 = 0.693;
constexpr double kSqrt2 = 1.414;
constexpr double kSqrt1_2 = 0.707;

constexpr double SmallEpsilon = 0.0001;

// Number of significant terms is bounded because all series arguments are small
// after range reduction; the loops break early on negligible terms.
constexpr double kSeriesTolerance = 0.0001;   // constexpr double kSeriesTolerance = 1e-18;
constexpr int kMaxSeriesTerms = 3; // Maximum number of terms in the Taylor series (best 60)

// exp(x) via range reduction x = k*ln2 + r with |r| <= ln2/2, then the Maclaurin
// series exp(r) = sum_{n>=0} r^n / n!, scaled by 2^k. The 2^k factor is built by
// repeated doubling/halving (no libm, no std::round).
double Exp(double x) {
  const double quotient = x / kLn2;
  const long k =
      static_cast<long>(quotient >= 0.0 ? quotient + 0.5 : quotient - 0.5);
  const double r = x - static_cast<double>(k) * kLn2;

  double term = 1.0;
  double series = 1.0;
  for (int n = 1; n <= kMaxSeriesTerms; ++n) {
    term *= r / static_cast<double>(n);
    series += term;
    // TODO: measure effect of early termination on accuracy.
    // if (term < kSeriesTolerance && term > -kSeriesTolerance) {
    //   break;
    // }
  }

  double scale = 1.0;
  for (long i = 0; i < k; ++i) {
    scale *= 2.0;
  }
  for (long i = 0; i > k; --i) {
    scale *= 0.5;
  }
  return series * scale;
}

// log(x) for x > 0 via reduction x = m * 2^e with m in [sqrt(1/2), sqrt(2)), then
// log(m) = 2 * sum_{n odd} w^n / n with w = (m - 1) / (m + 1) (the Taylor series
// of log). Centering m keeps |w| <= 0.172, so the series converges quickly.
double Log(double x) {
  double m = x;
  long e = 0;

// TODO: measure effect on accuracy of the range reduction.
  while (m >= kSqrt2) {
    m *= 0.5;
    e += 1;
  }
  while (m < kSqrt1_2) {
    m *= 2.0;
    e -= 1;
  }

  const double w = (m - 1.0) / (m + 1.0);
  const double w_squared = w * w;
  double power = w;  // Holds w^(2i+1) across iterations.
  double series = 0.0;
  for (int i = 0; i < kMaxSeriesTerms; ++i) {
    const double increment = power / static_cast<double>(2 * i + 1);
    series += increment;
    // TODO: measure effect of early termination on accuracy.
    // if (increment < kSeriesTolerance && increment > -kSeriesTolerance) {
    //   break;
    // }
    power *= w_squared;
  }
  return static_cast<double>(e) * kLn2 + 2.0 * series;
}

// log(1 + x) routed through Log for accuracy over the (0, 1] range used here.
double Log1p(double x) { return Log(1.0 + x); }

#else  // USE_TAYLOR_MATH

double Exp(double x) { return std::exp(x); }
double Log(double x) { return std::log(x); }
double Log1p(double x) { return std::log1p(x); }

constexpr double SmallEpsilon = 1e-14;

#endif  // USE_TAYLOR_MATH

// Numerically stable logistic function 1 / (1 + exp(-eta)).
double Sigmoid(double eta) {
  if (eta >= 0.0) {
    const double z = Exp(-eta);
    return 1.0 / (1.0 + z);
  }
  const double z = Exp(eta);
  return z / (1.0 + z);
}

// Numerically stable log(1 + exp(eta)) (the "softplus" function).
double LogOnePlusExp(double eta) {
  if (eta > 0.0) {
    return eta + Log1p(Exp(-eta));
  }
  return Log1p(Exp(eta));
}

// Standard dot product of two equal-length vectors.
double Dot(const Vector& a, const Vector& b) {
  assert(a.size() == b.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// Infinity norm (maximum absolute component) of a vector.
double InfNorm(const Vector& v) {
  double norm = 0.0;
  for (const double value : v) {
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

// Finds the conditional mode u_hat = argmax_u g_i(u) for one group, given the
// fixed effects `beta` and variance `sigma2`. The objective g_i is strictly
// concave in u, so a damped Newton iteration converges reliably.
double ConditionalMode(const Group& group, const Vector& beta, double sigma2) {
  const double inv_sigma2 = 1.0 / sigma2;
  double u = 0.0;
  for (int iteration = 0; iteration < 100; ++iteration) {
    double gradient = -u * inv_sigma2;  // Derivative of the Gaussian prior term.
    double curvature = inv_sigma2;      // A = -g''(u), always positive.
    for (std::size_t j = 0; j < group.y.size(); ++j) {
      const double eta = Dot(group.x[j], beta) + u;
      const double p = Sigmoid(eta);
      gradient += group.y[j] - p;  // Random-intercept design entry z = 1.
      curvature += p * (1.0 - p);
    }
    double step = gradient / curvature;  // Newton step (g'' = -curvature).
    step = std::max(-kMaxNewtonStep, std::min(kMaxNewtonStep, step));
    u += step;
    if (!std::isfinite(u)) {
      break;  // Degenerate trial; caller rejects it via the finiteness guard.
    }
    if (std::abs(step) < SmallEpsilon) {
      break;
    }
  }
  return u;
}

// Laplace-approximated marginal log-likelihood contribution of a single group.
double GroupLaplaceLogLik(const Group& group, const Vector& beta, double sigma2) {
  const double inv_sigma2 = 1.0 / sigma2;
  const double u_hat = ConditionalMode(group, beta, sigma2);

  double conditional_log_lik = 0.0;
  double curvature = inv_sigma2;  // A_i evaluated at the mode.
  for (std::size_t j = 0; j < group.y.size(); ++j) {
    const double eta = Dot(group.x[j], beta) + u_hat;
    const double p = Sigmoid(eta);
    conditional_log_lik += group.y[j] * eta - LogOnePlusExp(eta);
    curvature += p * (1.0 - p);
  }

  return conditional_log_lik - 0.5 * u_hat * u_hat * inv_sigma2 -
         0.5 * Log(sigma2) - 0.5 * Log(curvature);
}

// Splits a packed parameter vector [beta..., s] into `beta` and sigma^2, where
// the variance is parameterized as sigma = exp(s) to keep it strictly positive.
void UnpackParameters(const Vector& params, std::size_t num_fixed, Vector& beta,
                      double& sigma2) {
  beta.assign(params.begin(), params.begin() + num_fixed);
  const double s = params[num_fixed];
  sigma2 = Exp(2.0 * s);
}

// Negative marginal log-likelihood over the whole dataset (the BFGS objective).
double NegMarginalLogLik(const Dataset& data, const Vector& params) {
  Vector beta;
  double sigma2 = 0.0;
  UnpackParameters(params, data.num_fixed, beta, sigma2);

  double total = 0.0;
  for (const Group& group : data.groups) {
    total += GroupLaplaceLogLik(group, beta, sigma2);
  }
  // Reject degenerate trial parameters (overflow / NaN) by reporting a value the
  // minimizer will never accept, so the line search backtracks away from them.
  if (!std::isfinite(total)) {
    return std::numeric_limits<double>::infinity();
  }
  return -total;
}

// Central finite-difference gradient of a scalar objective `f` at `x`.
Vector NumericalGradient(const std::function<double(const Vector&)>& f,
                         const Vector& x) {
  Vector gradient(x.size(), 0.0);
  Vector perturbed = x;
  for (std::size_t k = 0; k < x.size(); ++k) {
    const double h = 1e-5 * (1.0 + std::abs(x[k]));
    perturbed[k] = x[k] + h;
    const double f_plus = f(perturbed);
    perturbed[k] = x[k] - h;
    const double f_minus = f(perturbed);
    perturbed[k] = x[k];
    gradient[k] = (f_plus - f_minus) / (2.0 * h);
  }
  return gradient;
}

// Identity matrix of size n x n.
Matrix Identity(std::size_t n) {
  Matrix m(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    m[i][i] = 1.0;
  }
  return m;
}

// Matrix-vector product m * v.
Vector MatVec(const Matrix& m, const Vector& v) {
  Vector result(m.size(), 0.0);
  for (std::size_t i = 0; i < m.size(); ++i) {
    result[i] = Dot(m[i], v);
  }
  return result;
}

// BFGS update of the inverse-Hessian approximation:
//   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1 / (y^T s).
Matrix BfgsInverseUpdate(const Matrix& h_inv, const Vector& s, const Vector& y,
                         double rho) {
  const std::size_t n = s.size();
  Matrix left = Identity(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      left[i][j] -= rho * s[i] * y[j];
    }
  }

  // temp = left * h_inv
  Matrix temp(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        sum += left[i][k] * h_inv[k][j];
      }
      temp[i][j] = sum;
    }
  }

  // updated = temp * left^T + rho * s s^T
  Matrix updated(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        sum += temp[i][k] * left[j][k];  // left^T[k][j] == left[j][k]
      }
      updated[i][j] = sum + rho * s[i] * s[j];
    }
  }
  return updated;
}

// Minimizes `f` starting from `x0` using BFGS with a backtracking (Armijo) line
// search and numerical gradients.
OptResult MinimizeBFGS(const std::function<double(const Vector&)>& f,
                       const Vector& x0, int max_iterations = 300,
                       double gradient_tol = 1e-6) {
  const std::size_t n = x0.size();
  Vector x = x0;
  double fx = f(x);
  Vector gradient = NumericalGradient(f, x);
  Matrix h_inv = Identity(n);

  OptResult result;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    result.iterations = iteration + 1;
    if (InfNorm(gradient) < gradient_tol) {
      result.converged = true;
      break;
    }

    // Search direction d = -H_inv * gradient; fall back to steepest descent if
    // it is not a descent direction.
    Vector direction = MatVec(h_inv, gradient);
    for (double& component : direction) {
      component = -component;
    }
    double directional_derivative = Dot(gradient, direction);
    if (directional_derivative >= 0.0) {
      h_inv = Identity(n);
      direction = gradient;
      for (double& component : direction) {
        component = -component;
      }
      directional_derivative = Dot(gradient, direction);
    }

    // Backtracking line search satisfying the Armijo sufficient-decrease rule.
    constexpr double c1 = 1e-4;
    double alpha = 1.0;
    Vector x_new(n);
    double fx_new = 0.0;
    while (true) {
      for (std::size_t i = 0; i < n; ++i) {
        x_new[i] = x[i] + alpha * direction[i];
      }
      fx_new = f(x_new);
      // Only accept a finite objective that meets the sufficient-decrease rule.
      if (std::isfinite(fx_new) &&
          fx_new <= fx + c1 * alpha * directional_derivative) {
        break;
      }
      alpha *= 0.5;
      if (alpha < SmallEpsilon) {
        // Line search stalled; reject the step and keep the current point rather
        // than accepting a poisoned (non-finite or non-decreasing) trial.
        x_new = x;
        fx_new = fx;
        break;
      }
    }

    Vector step(n);
    for (std::size_t i = 0; i < n; ++i) {
      step[i] = x_new[i] - x[i];
    }
    const Vector gradient_new = NumericalGradient(f, x_new);
    Vector gradient_delta(n);
    for (std::size_t i = 0; i < n; ++i) {
      gradient_delta[i] = gradient_new[i] - gradient[i];
    }

    const double curvature = Dot(step, gradient_delta);
    if (curvature > SmallEpsilon) {
      h_inv = BfgsInverseUpdate(h_inv, step, gradient_delta, 1.0 / curvature);
    }

    x = x_new;
    gradient = gradient_new;
    const double objective_change = std::abs(fx - fx_new);
    fx = fx_new;

    std::cout << "[BFGS] iter " << std::setw(3) << (iteration + 1)
        << "  neg_log_lik=" << std::fixed << std::setprecision(6) << fx
        << "  |grad|=" << std::scientific << std::setprecision(3)
        << InfNorm(gradient) << "  alpha=" << alpha
        << "  |step|=" << InfNorm(step)
        << "  d_obj=" << objective_change << std::endl;

    if (InfNorm(step) < SmallEpsilon || objective_change < SmallEpsilon) {
      result.converged = true;
      break;
    }
  }

  result.params = x;
  result.value = fx;
  return result;
}

// Generates a synthetic random-intercept logistic dataset with a known ground
// truth so that the recovered estimates can be validated.
Dataset GenerateSyntheticData(std::size_t num_groups, std::size_t obs_per_group,
                              const Vector& true_beta, double true_sigma,
                              unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> covariate_dist(0.0, 1.0);
  std::normal_distribution<double> random_effect_dist(0.0, true_sigma);
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

  const std::size_t num_fixed = true_beta.size();
  Dataset data;
  data.num_fixed = num_fixed;
  data.groups.reserve(num_groups);

  for (std::size_t i = 0; i < num_groups; ++i) {
    Group group;
    group.x.reserve(obs_per_group);
    group.y.reserve(obs_per_group);
    const double u = random_effect_dist(rng);
    for (std::size_t j = 0; j < obs_per_group; ++j) {
      Vector row(num_fixed, 0.0);
      row[0] = 1.0;  // Intercept column.
      for (std::size_t k = 1; k < num_fixed; ++k) {
        row[k] = covariate_dist(rng);
      }
      const double eta = Dot(row, true_beta) + u;
      const double p = Sigmoid(eta);
      group.x.push_back(std::move(row));
      group.y.push_back(uniform_dist(rng) < p ? 1.0 : 0.0);
    }
    data.groups.push_back(std::move(group));
  }
  return data;
}

// Prints a labeled fixed-effect estimate alongside its true value.
void PrintEstimate(const std::string& label, double estimate, double truth) {
  std::cout << "  " << std::left << std::setw(12) << label << std::right
            << std::setw(12) << std::fixed << std::setprecision(4) << estimate
            << std::setw(12) << truth << '\n';
}

}  // namespace

int main() {
  // Ground truth: intercept plus two covariates and a moderate random-intercept
  // standard deviation.
  const Vector true_beta = {-0.5, 1.0, -0.75, 0.3, -0.5, 1.0, -0.75, 0.3, -0.5, 1.0, -0.75, 0.3};
  const double true_sigma = 0.7;
  constexpr std::size_t num_groups = 300;
  constexpr std::size_t obs_per_group = 25;
  constexpr unsigned int seed = 20260730u;

  const Dataset data = GenerateSyntheticData(num_groups, obs_per_group, true_beta,
                                             true_sigma, seed);

  std::cout << "Mixed-effects logistic regression (Laplace + BFGS quasi-Newton)\n";
  std::cout << "Groups: " << num_groups << ", observations/group: "
            << obs_per_group << ", total: " << num_groups * obs_per_group
            << "\n\n";

  // Objective: negative Laplace marginal log-likelihood as a function of the
  // packed parameter vector [beta_0, beta_1, beta_2, s] with sigma = exp(s).
  const std::function<double(const Vector&)> objective =
      [&data](const Vector& params) { return NegMarginalLogLik(data, params); };

  // Start fixed effects at 0 and sigma at 1 (s = 0).
  Vector initial_params(data.num_fixed + 1, 0.0);
  const OptResult fit = MinimizeBFGS(objective, initial_params);

  Vector beta_hat;
  double sigma2_hat = 0.0;
  UnpackParameters(fit.params, data.num_fixed, beta_hat, sigma2_hat);
  const double sigma_hat = std::sqrt(sigma2_hat);
  const double final_log_lik = -fit.value;

  std::cout << "Converged: " << (fit.converged ? "yes" : "no")
            << " in " << fit.iterations << " iterations\n";
  std::cout << "Final marginal log-likelihood: " << std::fixed
            << std::setprecision(4) << final_log_lik << "\n\n";

  std::cout << "  " << std::left << std::setw(12) << "Parameter" << std::right
            << std::setw(12) << "Estimate" << std::setw(12) << "Truth" << '\n';
  PrintEstimate("Intercept", beta_hat[0], true_beta[0]);
  for (std::size_t k = 1; k < data.num_fixed; ++k) {
    PrintEstimate("beta_" + std::to_string(k), beta_hat[k], true_beta[k]);
  }
  PrintEstimate("sigma", sigma_hat, true_sigma);

  // Lightweight sanity checks: with this much data the Laplace estimates should
  // land close to the ground truth.
  assert(fit.converged && "BFGS optimization failed to converge");
  for (std::size_t k = 0; k < data.num_fixed; ++k) {
    assert(std::abs(beta_hat[k] - true_beta[k]) < 0.25);
  }
  assert(std::abs(sigma_hat - true_sigma) < 0.25);
  assert(std::abs(sigma_hat - true_sigma) < 0.25);

  return 0;
}
