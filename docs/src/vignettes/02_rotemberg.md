# Rotemberg Decomposition

```@meta
CurrentModule = ShiftShareIV
```

Goldsmith-Pinkham, Sorkin, and Swift (2020) showed that the shift-share IV
estimate is numerically equivalent to a weighted average of $K$
just-identified IV estimates — one per industry share — where the weights are
the **Rotemberg weights** $\alpha_k$.

$$\hat\beta^{SS} = \sum_{k=1}^{K} \alpha_k \hat\beta_k$$

This decomposition is not just a mathematical curiosity. It tells you
*which industry shares are doing the identifying work* and *what the
industry-specific IV estimates say*. Both pieces of information are needed
to evaluate the plausibility of the identifying assumption.

## Setup

Recreate the DGP from the Introduction vignette:

```@example rw
using ShiftShareIV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using Random
using Printf

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

## Computing Rotemberg weights

```@example rw
rw = rotemberg_weights(shares, shocks, X, Y)
first(rw, 6)
```

The columns are:

- `alpha`: the Rotemberg weight for industry $k$
- `beta_k`: the just-identified IV estimate using $s_{\cdot k}$ alone as the instrument
- `alpha_beta`: the $\alpha_k \hat\beta_k$ contribution to the overall 2SLS estimate

## The GPSS decomposition identity

The sum of the `alpha_beta` column must exactly equal the 2SLS estimate.
Verify:

```@example rw
Bc = B .- mean(B)
Xc = X .- mean(X)
Yc = Y .- mean(Y)
iv_est = dot(Bc, Yc) / dot(Bc, Xc)

decomp = sum(rw.alpha_beta)

@printf("2SLS estimate:         %.6f\n", iv_est)
@printf("Σ αₖ βₖ (GPSS):       %.6f\n", decomp)
@printf("Difference:            %.2e\n",  abs(iv_est - decomp))
@printf("Sum of weights (= 1):  %.6f\n", sum(rw.alpha))
```

The identity holds to machine precision. This is not a coincidence: the
GPSS theorem proves it analytically for any shift-share IV.

## Diagnosing weight concentration

If one or two industries carry most of the weight, the identifying assumption
is really a statement about those specific industries' shares — the
"average" argument that all $K$ shares are exogenous is doing less work than
it appears.

```@example rw
sort!(rw, :alpha, rev=true)

@printf("%-10s %8s %8s %8s\n", "Industry", "α_k", "β_k", "Σ (cum)")
let cumulative = 0.0
    for row in eachrow(rw)
        cumulative += row.alpha
        @printf("%-10d %8.4f %8.4f %8.4f\n",
                row.industry, row.alpha, row.beta_k, cumulative)
        cumulative > 0.8 && break
    end
end
println("(Industries above explain >80% of Σ αₖ = 1)")
```

In this simulation, multiple industries share the weight (the Dirichlet
shares give realistic industry concentration). In real data, a single
industry can carry 50% or more of the weight — those industries warrant
individual scrutiny.

## Diagnosing β_k heterogeneity

Even if weight concentration is acceptable, heterogeneous $\hat\beta_k$
values across industries suggest effect heterogeneity (or, more worryingly,
violation of the exclusion restriction for some shares):

```@example rw
# Summary of beta_k across industries
q10, q50, q90 = quantile(rw.beta_k, [0.1, 0.5, 0.9])
@printf("β_k quantiles:  10th=%.3f  50th=%.3f  90th=%.3f\n", q10, q50, q90)
@printf("True β:         %.3f\n", beta_true)
@printf("Std(β_k):       %.3f\n", std(rw.beta_k))
```

In a well-behaved simulation, all $\hat\beta_k$ are close to the truth and
the spread is sampling noise. In real data, some industries give wildly
different estimates, which could reflect:

1. Heterogeneous treatment effects by industry
2. Endogeneity of specific industry shares

The `alpha`-weighted plot makes the diagnostic visual:

```@example rw
# Print an ASCII weight chart
sort!(rw, :beta_k)
@printf("\n%-10s %6s   %s\n", "Industry", "α_k", "β_k (bar chart, true β=0.50)")
for row in eachrow(rw)
    bar_len = round(Int, max(0, row.beta_k) * 20)
    bar = "█" ^ bar_len
    mark = abs(row.alpha) > 0.08 ? " ← high weight" : ""
    @printf("%-10d %6.3f   %s%.3f%s\n",
            row.industry, row.alpha, bar, row.beta_k, mark)
end
```

## What concentration looks like in a bad case

Artificially give industry 1 a large, negative share that covaries with
unobservables. Watch what happens to the Rotemberg weights:

```@example rw
Random.seed!(11)
v = randn(n_loc)                          # a new confounder
shares_bad = copy(shares)
shares_bad[:, 1] .= max.(0.01, shares[:, 1] .+ 0.4v)   # industry 1 endogenous
shares_bad = shares_bad ./ sum(shares_bad, dims=2)

B_bad = bartik_iv(shares_bad, shocks)
X_bad = B_bad .+ 0.3u .+ 0.1randn(n_loc)
Y_bad = beta_true .* X_bad .+ u .+ 0.6v .+ 0.3randn(n_loc)

Bc_bad = B_bad .- mean(B_bad)
Xc_bad = X_bad .- mean(X_bad)
Yc_bad = Y_bad .- mean(Y_bad)
iv_bad  = dot(Bc_bad, Yc_bad) / dot(Bc_bad, Xc_bad)

rw_bad  = rotemberg_weights(shares_bad, shocks, X_bad, Y_bad)
rw_bad_sorted = sort(rw_bad, :alpha, rev=true)

@printf("IV estimate with endogenous share 1: %.3f  (true β = %.3f)\n",
        iv_bad, beta_true)
@printf("\nTop-3 industries by |α_k|:\n")
@printf("%-10s %8s %8s\n", "Industry", "α_k", "β_k")
for row in eachrow(first(rw_bad_sorted, 3))
    @printf("%-10d %8.4f %8.4f\n", row.industry, row.alpha, row.beta_k)
end
```

Industry 1 carries a large Rotemberg weight and its $\hat\beta_k$ is far from
the truth — a clear red flag in the diagnostic output. In a real application,
this would prompt either dropping industry 1 from the share construction or
switching to the BHJ shock-level strategy.

## Summary

- **Rotemberg weights** ($\alpha_k$) reveal which industry shares drive the
  2SLS estimate.
- **Weight concentration** is a risk: the exclusion restriction is really
  only for the high-weight industries.
- **Heterogeneous $\hat\beta_k$** warns of effect heterogeneity or share
  endogeneity.
- The GPSS identity $\hat\beta^{SS} = \sum_k \alpha_k \hat\beta_k$ holds
  exactly — it is a useful numerical check and the foundation of the diagnostic.
