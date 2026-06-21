// ---------------------------------------------------------------
// Bayesian Logistic Regression — UCL Penalty Shootouts (Position & Basket)
// ---------------------------------------------------------------
// Goal_i ~ Bernoulli(logit^{-1}(alpha_pos[Position_i]
//                                     + beta_basket[Penalty_Basket_i]
//                                     + beta_elim * Elimination_i
//                                     + beta_left * is_left_i))
//
// Position-specific intercept priors (logit scale):
//   alpha_pos[1] GK  ~ Normal(0.62, 0.5) — ~65% base chance
//   alpha_pos[2] DEF ~ Normal(1.10, 0.5) — ~75% base chance
//   alpha_pos[3] MID ~ Normal(1.39, 0.5) — ~80% base chance
//   alpha_pos[4] FWD ~ Normal(1.73, 0.5) — ~85% base chance
//
// Basket relative deviations (logit scale):
//   beta_basket[1] ~ Normal(-0.25, 0.3) — Basket 1 (Round 1, lower chance)
//   beta_basket[2] ~ Normal(0.0,   0.3) — Basket 2 (Round 2, baseline/typical)
//   beta_basket[3] ~ Normal(0.25,  0.3) — Basket 3 (Round 3, higher chance)
//   beta_basket[4] ~ Normal(-0.15, 0.3) — Basket 4 (Round 4 & 5, high-stress)
//   beta_basket[5] ~ Normal(0.0,   0.3) — Basket 5 (Round 6-10, sudden death)
//   beta_basket[6] ~ Normal(-0.50, 0.5) — Basket 6 (Round 11+, goalies shoot)
//
// Left-foot effect (logit scale):
//   beta_left    ~ Normal(0, 0.5) — effect of kicking with left foot
// ---------------------------------------------------------------

data {
  int<lower=0> N;                              // number of observations
  array[N] int<lower=0, upper=1> y;            // Goal (1 = scored, 0 = missed)
  vector[N] elimination;                       // Elimination round indicator
  array[N] int<lower=1, upper=6> penalty_basket; // Penalty basket indicator (1 to 6)
  vector[N] is_left;                           // Indicator for left foot (1 = left, 0 = right)
  array[N] int<lower=1, upper=4> position_id;  // 1=GK, 2=DEF, 3=MID, 4=FWD
}

parameters {
  vector[4] alpha_pos;    // position-specific intercepts (logit scale)
  vector[6] beta_basket;  // basket-specific effects (logit-scale deviations)
  real beta_elim;         // coefficient for Elimination
  real beta_left;         // coefficient for left foot
}

model {
  // ----- Position-specific intercept priors -----
  alpha_pos[1] ~ normal(0.62, 0.5);
  alpha_pos[2] ~ normal(1.10, 0.5);
  alpha_pos[3] ~ normal(1.39, 0.5);
  alpha_pos[4] ~ normal(1.73, 0.5);

  // ----- Basket-specific relative deviation priors -----
  beta_basket[1] ~ normal(0.0, 1.0);
  beta_basket[2] ~ normal(0.0, 1.0);
  beta_basket[3] ~ normal(0.0, 1.0);
  beta_basket[4] ~ normal(0.0, 1.0);
  beta_basket[5] ~ normal(0.0, 1.0);
  beta_basket[6] ~ normal(0.0, 1.0);

  // ----- Weakly informative coefficient priors -----
  beta_elim  ~ normal(0, 1.0);
  beta_left  ~ normal(0, 0.5);

  // ----- Likelihood -----
  y ~ bernoulli_logit(alpha_pos[position_id]
                      + beta_basket[penalty_basket]
                      + beta_elim * elimination
                      + beta_left * is_left);
}

generated quantities {
  vector[N] eta;
  vector[N] log_lik;
  array[N] int y_rep;
  array[N] int y_prior;

  vector[4] alpha_pos_prior;
  vector[6] beta_basket_prior;
  real beta_elim_prior;
  real beta_left_prior;
  vector[N] eta_prior;

  alpha_pos_prior[1] = normal_rng(0.62, 0.5);
  alpha_pos_prior[2] = normal_rng(1.10, 0.5);
  alpha_pos_prior[3] = normal_rng(1.39, 0.5);
  alpha_pos_prior[4] = normal_rng(1.73, 0.5);

  beta_basket_prior[1] = normal_rng(0.0, 1.0);
  beta_basket_prior[2] = normal_rng(0.0, 1.0);
  beta_basket_prior[3] = normal_rng(0.0, 1.0);
  beta_basket_prior[4] = normal_rng(0.0, 1.0);
  beta_basket_prior[5] = normal_rng(0.0, 1.0);
  beta_basket_prior[6] = normal_rng(0.0, 1.0);

  beta_elim_prior = normal_rng(0, 1.0);
  beta_left_prior = normal_rng(0, 0.5);

  for (n in 1:N) {
    eta[n] = alpha_pos[position_id[n]]
             + beta_basket[penalty_basket[n]]
             + beta_elim * elimination[n]
             + beta_left * is_left[n];

    // Pointwise log-likelihood (required for WAIC / LOO-CV)
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta[n]);

    // Posterior predictive replicates (PPC)
    y_rep[n] = bernoulli_logit_rng(eta[n]);

    eta_prior[n] = alpha_pos_prior[position_id[n]]
                   + beta_basket_prior[penalty_basket[n]]
                   + beta_elim_prior * elimination[n]
                   + beta_left_prior * is_left[n];
    y_prior[n] = bernoulli_logit_rng(eta_prior[n]);
  }
}
