# ShiftShareIV.jl

Julia toolkit for **shift-share (Bartik) instrumental variables**. Implements the Bartik instrument, the GPSS (2020) Rotemberg weight decomposition, and the BHJ (2022) shock-level collapse for inference.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/ShiftShareIV.jl")
```

## Quick Start

```julia
using ShiftShareIV, DataFrames, Statistics, Random

Random.seed!(2026)
n_loc, n_ind = 500, 20

shares_raw = rand(n_loc, n_ind) .+ 0.1
shares     = shares_raw ./ sum(shares_raw, dims=2)
shocks     = randn(n_ind)

# Bartik instrument
B = bartik_iv(shares, shocks)

# Simulate outcome and endogenous regressor
u = randn(n_loc)
X = B .+ 0.3u .+ 0.1randn(n_loc)
Y = 0.5X .+ u  .+ 0.3randn(n_loc)

# Rotemberg weights (GPSS 2020)
rw = rotemberg_weights(shares, shocks, X, Y)
sum(rw.alpha)       # ≈ 1.0
sum(rw.alpha_beta)  # ≈ the 2SLS estimate

# BHJ shock-level collapse (BHJ 2022)
collapsed = bhj_collapse(shares, shocks, Y, X)
# Regress collapsed.Y_agg ~ collapsed.X_agg | collapsed.shock,
# weighted by collapsed.weight
```

## API

### `bartik_iv(shares, shocks) → Vector`

Construct the Bartik instrument $B_\ell = \sum_k s_{\ell k}\, g_k$ for each location.

- `shares`: L × K matrix (rows = locations, columns = industries; rows should sum to 1)
- `shocks`: K-vector of national industry-level growth rates

### `rotemberg_weights(shares, shocks, X[, Y]) → DataFrame`

GPSS (2020) Rotemberg decomposition. Returns a DataFrame with one row per industry:

| Column | Description |
|--------|-------------|
| `industry` | industry index |
| `shock` | $g_k$ |
| `cov_sk_X` | $\text{Cov}(s_{\cdot k},\, X)$ |
| `alpha` | Rotemberg weight $\alpha_k$ (sums to 1) |
| `beta_k` | just-identified IV estimate *(only with Y)* |
| `alpha_beta` | $\alpha_k \times \hat\beta_k$ contribution *(only with Y)* |

The GPSS decomposition identity: $\hat\beta^{2SLS} = \sum_k \alpha_k \hat\beta_k$.

### `bhj_collapse(shares, shocks, Y, X; weights) → DataFrame`

Borusyak, Hull, Jaravel (2022) shock-level collapse. Returns a DataFrame with one row per industry:

| Column | Description |
|--------|-------------|
| `industry` | industry index |
| `shock` | $g_k$ |
| `Y_agg` | shock-level aggregated outcome |
| `X_agg` | shock-level aggregated endogenous regressor |
| `weight` | $\sum_\ell s_{\ell k}\, w_\ell$ exposure weight |

Run weighted 2SLS on the collapsed data to get the BHJ shock-level estimate.

## Vignettes

- [Introduction](https://xiangao.github.io/ShiftShareIV.jl/dev/vignettes/01_introduction/): Bartik instrument basics and OLS vs IV comparison
- [Rotemberg Decomposition](https://xiangao.github.io/ShiftShareIV.jl/dev/vignettes/02_rotemberg/): GPSS (2020) weight diagnostics and the decomposition identity
- [BHJ Shock-Level Inference](https://xiangao.github.io/ShiftShareIV.jl/dev/vignettes/03_bhj/): BHJ (2022) shock-level collapse and inference

## References

- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development Policies?* Upjohn Institute.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: what, when, why, and how. *American Economic Review*, 110(8), 2586–2624.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share research designs. *Review of Economic Studies*, 89(1), 181–213.
- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: theory and inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
