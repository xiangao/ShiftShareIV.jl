# BHJ Shock-Level Inference

```@meta
CurrentModule = ShiftShareIV
```

Borusyak, Hull, and Jaravel (2022) provided an alternative view of the
shift-share instrument: instead of treating shares as the source of
identifying variation (GPSS), treat the **shocks** as as-if-randomly
assigned across industries. Under this view, the shares are just exposure
weights that aggregate the shocks to the location level.

This shock view has a key practical consequence for **inference**. Standard
errors clustered at the location level underestimate uncertainty because the
same industry shock $g_k$ enters thousands of locations, inducing cross-location
correlation that location-level clustering does not capture. The fix: run the
regression at the shock (industry) level rather than the location level.

## Setup

```@example bhj
using ShiftShareIV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using Random
using Printf
using GLM
using StatsModels

Random.seed!(2026)

n_loc, n_ind = 500, 20
beta_true    = 0.5

shares_raw = rand(Gamma(0.3), n_loc, n_ind) .+ 1e-4
shares     = shares_raw ./ sum(shares_raw, dims=2)
shocks     = randn(n_ind)

B = bartik_iv(shares, shocks)

u = randn(n_loc)
X = B .+ 0.3u .+ 0.1randn(n_loc)
Y = beta_true .* X .+ u .+ 0.3randn(n_loc)
nothing
```

## The BHJ collapse

`bhj_collapse` aggregates the location-level outcome `Y` and endogenous
regressor `X` to the shock (industry) level:

$$\bar Y_k = \frac{\sum_\ell s_{\ell k}\, w_\ell\, Y_\ell}{\sum_\ell s_{\ell k}\, w_\ell},
\qquad
\bar X_k = \frac{\sum_\ell s_{\ell k}\, w_\ell\, X_\ell}{\sum_\ell s_{\ell k}\, w_\ell}$$

The denominator $W_k = \sum_\ell s_{\ell k}\, w_\ell$ is the industry's
total exposure, which serves as the observation weight in the shock-level
regression.

```@example bhj
collapsed = bhj_collapse(shares, shocks, Y, X)
println("Collapsed to $(nrow(collapsed)) shock-level observations")
first(collapsed, 6)
```

## Numerical equivalence

The BHJ theorem says the weighted 2SLS at the shock level is numerically
equivalent to the 2SLS at the location level (under no controls). Verify:

```@example bhj
# Location-level 2SLS (Wald estimator)
Bc = B .- mean(B)
Xc = X .- mean(X)
Yc = Y .- mean(Y)
iv_loc = dot(Bc, Yc) / dot(Bc, Xc)

# Shock-level WLS 2SLS using bhj_collapse output
# First-stage: regress X_agg ~ shock, weighted by weight
w     = collapsed.weight
sc    = collapsed.shock
Xagg  = collapsed.X_agg
Yagg  = collapsed.Y_agg
sc_c  = sc  .- sum(w .* sc) / sum(w)
Xa_c  = Xagg .- sum(w .* Xagg) / sum(w)
Ya_c  = Yagg .- sum(w .* Yagg) / sum(w)

iv_bhj = dot(w .* sc_c, Ya_c) / dot(w .* sc_c, Xa_c)

@printf("Location-level 2SLS:  %.6f\n", iv_loc)
@printf("BHJ shock-level 2SLS: %.6f\n", iv_bhj)
@printf("Difference:           %.2e\n",  abs(iv_loc - iv_bhj))
@printf("True β:               %.3f\n",  beta_true)
```

The two estimates agree to machine precision. This is the key BHJ result:
the two formulations are algebraically identical, but the shock-level
representation makes the identifying assumption more transparent — you have
$K$ shock-level observations, not $L$ location-level observations.

## Why shock-level inference matters

With $K = 20$ effective observations at the shock level, the inference
picture is very different from $L = 500$ at the location level.

```@example bhj
# Heteroskedasticity-robust SE at the location level (naive)
b_fs_loc = dot(Bc, Xc) / dot(Bc, Bc)
resid_loc = Yc .- iv_loc .* Xc
sigma2_loc = dot(resid_loc, resid_loc) / (n_loc - 2)
se_loc_naive = sqrt(sigma2_loc / dot(Bc, Xc)^2 * dot(Bc, Bc))

# HC0 SE at the shock level
b_fs_bhj = dot(w .* sc_c, Xa_c) / dot(w .* sc_c, sc_c)
resid_bhj = Ya_c .- iv_bhj .* Xa_c
meat_bhj  = sum(w .* sc_c .* resid_bhj .* w .* sc_c .* resid_bhj)  # HC0
bread_bhj = dot(w .* sc_c, Xa_c)^2
se_bhj    = sqrt(meat_bhj) / abs(bread_bhj)

@printf("%-30s %.4f\n", "Naive location-level SE:", se_loc_naive)
@printf("%-30s %.4f\n", "BHJ shock-level HC SE:", se_bhj)
@printf("%-30s %.2f (95%% CI)\n", "Location-level t-CI half-width:",
        1.96 * se_loc_naive)
@printf("%-30s %.2f (95%% CI)\n", "BHJ t-CI half-width (K-2 df):",
        quantile(TDist(n_ind - 2), 0.975) * se_bhj)
```

The shock-level SE is typically larger because (a) there are only $K$
effective observations and (b) the $t$-critical value uses $K - 2$ degrees
of freedom, not $L - 2$.

## Using external regression packages

In practice, pass the collapsed DataFrame to any regression package that
supports weights. Using `GLM.jl`:

```@example bhj
# First stage (for diagnostics)
fs_bhj = lm(@formula(X_agg ~ shock), collapsed,
             wts=collapsed.weight)

# IV via 2SLS manually (or use a dedicated IV package)
# Add fitted values as a column before the second-stage formula
collapsed2 = copy(collapsed)
collapsed2.Xhat = predict(fs_bhj)
ss_bhj = lm(@formula(Y_agg ~ Xhat), collapsed2,
             wts=collapsed2.weight)

@printf("BHJ 2SLS (via GLM two-stage): %.4f\n", coef(ss_bhj)[2])
@printf("True β:                       %.3f\n", beta_true)
```

!!! note
    The two-stage GLM approach gives the same point estimate but
    **incorrect standard errors** (they treat the first-stage fitted values
    as known). For valid inference, use the closed-form HC/cluster SEs on
    the shock-level data directly, or a dedicated IV package that propagates
    first-stage uncertainty.

## Optional location weights

When location sizes differ (population, employment), pass a `weights` vector
to `bhj_collapse`. The shock-level aggregation uses these as $w_\ell$:

```@example bhj
employment = rand(Gamma(2, 1), n_loc)   # simulate location sizes

collapsed_wtd = bhj_collapse(shares, shocks, Y, X; weights=employment)

# Numerically equivalent to employment-weighted location-level 2SLS
Bwc  = B .- sum(employment .* B) / sum(employment)
Xwc  = X .- sum(employment .* X) / sum(employment)
Ywc  = Y .- sum(employment .* Y) / sum(employment)
iv_wtd = dot(employment .* Bwc, Ywc) / dot(employment .* Bwc, Xwc)

w2   = collapsed_wtd.weight
sc2  = collapsed_wtd.shock .- sum(w2 .* collapsed_wtd.shock) / sum(w2)
Xa2  = collapsed_wtd.X_agg .- sum(w2 .* collapsed_wtd.X_agg) / sum(w2)
Ya2  = collapsed_wtd.Y_agg .- sum(w2 .* collapsed_wtd.Y_agg) / sum(w2)
iv_bhj_wtd = dot(w2 .* sc2, Ya2) / dot(w2 .* sc2, Xa2)

@printf("Employment-weighted 2SLS (location level):  %.6f\n", iv_wtd)
@printf("Employment-weighted 2SLS (shock level):     %.6f\n", iv_bhj_wtd)
@printf("Difference:                                 %.2e\n",
        abs(iv_wtd - iv_bhj_wtd))
```

## Summary

- **BHJ collapse** transforms a location-level dataset into a $K$-row
  shock-level dataset. The mapping is `bhj_collapse(shares, shocks, Y, X)`.
- **Numerical equivalence**: the shock-level weighted 2SLS equals the
  location-level 2SLS exactly. The two formulations are the same estimator.
- **Why bother?**: shock-level inference uses $K$ degrees of freedom, not
  $L$, and makes the identifying assumption transparent — shocks must be
  uncorrelated with the shock-level aggregated error.
- **Location weights**: pass `weights=w` to weight locations by size (e.g.
  employment) before aggregating to the shock level.
