# Introduction to ShiftShareIV.jl

```@meta
CurrentModule = ShiftShareIV
```

Shift-share (or "Bartik") instruments combine local industry shares with
national industry shocks to construct an instrument for local economic
outcomes. They are the workhorse identification strategy in labour economics,
trade, and urban economics.

The core idea: locations that happened to specialise in fast-growing
industries at baseline will, through no fault of their own, receive larger
labour-demand shocks later. If those baseline shares are exogenous (GPSS
view) or the national shocks are as-if-random (BHJ view), the instrument is
valid.

## Setup

```@example bartik
using ShiftShareIV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using Random
using Printf

Random.seed!(2026)
nothing
```

## Constructing the Bartik instrument

For each location $\ell$ and industry $k$, let $s_{\ell k}$ be the share of
local employment in industry $k$ at baseline (rows of the shares matrix sum
to 1). Let $g_k$ be a national industry growth rate. The instrument is

$$B_\ell = \sum_{k=1}^{K} s_{\ell k}\, g_k = (S\, g)_\ell$$

```@example bartik
n_loc, n_ind = 500, 20
beta_true    = 0.5

# Shares: Dirichlet(0.3) draws, so locations concentrate in a few industries
shares_raw = rand(Gamma(0.3), n_loc, n_ind) .+ 1e-4
shares     = shares_raw ./ sum(shares_raw, dims=2)

@assert all(isapprox.(sum(shares, dims=2), 1.0, atol=1e-10))  # rows sum to 1

# National industry shocks
shocks = randn(n_ind)

# Bartik instrument
B = bartik_iv(shares, shocks)

@printf("B: mean=%.3f  std=%.3f  min=%.3f  max=%.3f\n",
        mean(B), std(B), minimum(B), maximum(B))
```

## The identification problem

The shift-share IV is useful because local outcomes are often determined by a
local shock that is itself correlated with unobserved confounders. Simulate a
DGP where `u` is an unobserved location-level confounder:

```@example bartik
u = randn(n_loc)                                    # unobserved confounder
X = B .+ 0.3u .+ 0.1randn(n_loc)                   # endogenous local shock
Y = beta_true .* X .+ u .+ 0.3randn(n_loc)          # outcome

df = DataFrame(X=X, Y=Y, B=B)

nothing  # suppress DataFrame display
```

The confounder `u` enters both `X` and `Y`, so OLS is biased.

## OLS vs Shift-Share IV

A manual 2SLS via the Frisch-Waugh theorem illustrates why `B` recovers
the true effect while OLS does not:

```@example bartik
# OLS: biased because Cov(X, u) ≠ 0
Xc = X .- mean(X); Yc = Y .- mean(Y)
ols_est = dot(Xc, Yc) / dot(Xc, Xc)

# 2SLS (Wald-style with one instrument):
# β̂_IV = Cov(B, Y) / Cov(B, X)
Bc = B .- mean(B)
iv_est = dot(Bc, Yc) / dot(Bc, Xc)

@printf("%-20s %.3f\n", "True β:", beta_true)
@printf("%-20s %.3f  (biased — Cov(X,u) > 0)\n", "OLS:", ols_est)
@printf("%-20s %.3f  (recovers true β)\n", "Shift-share IV:", iv_est)
```

OLS is biased upward because `u` creates a positive correlation between `X`
and the error. The shift-share IV — which varies only through industry shocks
and baseline shares, not through `u` — removes that bias.

## Checking the first stage

A valid instrument must be relevant (correlated with `X`). The first-stage
$F$-statistic should far exceed the weak-instrument threshold of 10 (or 104.7
for the Montiel-Pflueger test):

```@example bartik
# First stage: regress X on B
b_fs  = dot(Bc, Xc) / dot(Bc, Bc)     # slope
res   = Xc .- b_fs .* Bc              # first-stage residuals
s2    = dot(res, res) / (n_loc - 2)
Var_b = s2 / dot(Bc, Bc)
F_stat = b_fs^2 / Var_b

@printf("First-stage slope: %.3f\n", b_fs)
@printf("First-stage F:     %.1f\n", F_stat)
```

A large $F$ here reflects the strong mechanical relationship between the
Bartik instrument and the local shock: locations more exposed to positive-shock
industries receive larger local shocks.

## What's next

- **[Rotemberg Decomposition](02_rotemberg.md)**: Which industries are driving
  the 2SLS estimate? Are those industries' shares credibly exogenous?
- **[BHJ Shock-Level Inference](03_bhj.md)**: Collapse to the shock level to
  obtain inference that is robust to cross-location correlation from shared
  industry shocks.
