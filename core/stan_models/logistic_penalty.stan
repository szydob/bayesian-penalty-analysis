// ---------------------------------------------------------------
// Bayesian Logistic Regression for Penalty Shootouts (Baskets with Intercept)
// ---------------------------------------------------------------
// Goal_i ~ Bernoulli(logit^{-1}(alpha + beta_basket[Penalty_Basket_i]
//                                     + beta_elim * Elimination_i
//                                     + beta_left * is_left_i))
//
// Priors (on logit scale):
//   beta_basket[1] ~ Normal(0.0, 1.0)  — Round 1 deviation
//   beta_basket[2] ~ Normal(0.0, 1.0)  — Round 2 deviation
//   beta_basket[3] ~ Normal(0.0, 1.0)  — Round 3 deviation
//   beta_basket[4] ~ Normal(0.0, 1.0)  — Round 4 & 5 deviation
//   beta_basket[5] ~ Normal(0.0, 1.0)  — Round 6-10 deviation
//   beta_basket[6] ~ Normal(0.0, 1.0)  — Round 11+ deviation
//   beta_elim      ~ Normal(0,   1.0)  — effect of elimination round
//   beta_left      ~ Normal(0,   0.5)  — effect of left foot
// ---------------------------------------------------------------

data {
  int<lower=0> N;                       // number of observations
  array[N] int<lower=0, upper=1> y;     // Goal (1 = scored, 0 = missed)
  vector[N] elimination;                // Elimination round indicator
  array[N] int<lower=1, upper=6> penalty_basket; // Penalty basket indicator (1 to 6)
  vector[N] is_left;      // Indicator for left foot (1 = left, 0 = right)
}

parameters {
  vector[6] beta_basket; // basket-specific deviations
  real beta_elim;       // coefficient for Elimination
  real beta_left;       // coefficient for left foot
}

model {
  // ----- Priors -----
  beta_basket[1] ~ normal(0.0, 1.0);
  beta_basket[2] ~ normal(0.0, 1.0);
  beta_basket[3] ~ normal(0.0, 1.0);
  beta_basket[4] ~ normal(0.0, 1.0);
  beta_basket[5] ~ normal(0.0, 1.0);
  beta_basket[6] ~ normal(0.0, 1.0);

  beta_elim ~ normal(0.0, 1.0);
  beta_left ~ normal(0.0, 0.5);  // Left foot effect

  // ----- Likelihood -----
  y ~ bernoulli_logit(beta_basket[penalty_basket]
                      + beta_elim * elimination
                      + beta_left * is_left);
}

generated quantities {
  // Posterior linear predictor
  vector[N] eta = beta_basket[penalty_basket]
                  + beta_elim * elimination
                  + beta_left * is_left;

  // Pointwise log-likelihood (required for WAIC / LOO-CV)
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta[n]);
  }

  // 1. POSTERIOR Predictive Check
  array[N] int y_rep;
  for (n in 1:N) {
    y_rep[n] = bernoulli_logit_rng(eta[n]);
  }

  // 2. PRIOR Predictive Check
  array[N] int y_prior;
  vector[6] beta_basket_prior;
  beta_basket_prior[1] = normal_rng(0, 1.0);
  beta_basket_prior[2] = normal_rng(0, 1.0);
  beta_basket_prior[3] = normal_rng(0, 1.0);
  beta_basket_prior[4] = normal_rng(0, 1.0);
  beta_basket_prior[5] = normal_rng(0, 1.0);
  beta_basket_prior[6] = normal_rng(0, 1.0);

  real beta_elim_prior = normal_rng(0, 1.0);
  real beta_left_prior = normal_rng(0, 0.5);  // Left foot prior

  for (n in 1:N) {
    real eta_prior = beta_basket_prior[penalty_basket[n]]
                     + beta_elim_prior * elimination[n]
                     + beta_left_prior * is_left[n];

    y_prior[n] = bernoulli_logit_rng(eta_prior);
  }
}