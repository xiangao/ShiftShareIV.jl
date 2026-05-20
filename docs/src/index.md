# ShiftShareIV.jl

`ShiftShareIV.jl` contains the three pieces I usually want for shift-share
instruments: build the Bartik instrument, inspect the GPSS Rotemberg weights,
and collapse the data to the shock level following BHJ.

## Main functions

- **`bartik_iv`**: Construct the Bartik instrument $B_\ell = \sum_k s_{\ell k}\, g_k$ from a shares matrix and a shocks vector.
- **`rotemberg_weights`**: Decompose the 2SLS estimate into industry-specific just-identified IV estimates and their Rotemberg weights. Verifies the GPSS decomposition identity $\hat\beta^{SS} = \sum_k \alpha_k \hat\beta_k$.
- **`bhj_collapse`**: Collapse a location-level dataset to the shock (industry) level for BHJ-style inference.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/ShiftShareIV.jl")
```

## Quick Start

```julia
using ShiftShareIV, DataFrames, Random

Random.seed!(42)
n_loc, n_ind = 500, 20

shares_raw = rand(n_loc, n_ind) .+ 0.1
shares     = shares_raw ./ sum(shares_raw, dims=2)
shocks     = randn(n_ind)

B  = bartik_iv(shares, shocks)
rw = rotemberg_weights(shares, shocks, X, Y)
collapsed = bhj_collapse(shares, shocks, Y, X)
```

## Vignettes

| Vignette | Description |
|----------|-------------|
| [Introduction](vignettes/01_introduction.md) | Bartik instrument construction and OLS vs IV comparison |
| [Rotemberg Decomposition](vignettes/02_rotemberg.md) | GPSS (2020) diagnostics: weight concentration and β_k heterogeneity |
| [BHJ Shock-Level Inference](vignettes/03_bhj.md) | BHJ (2022) shock-level collapse, numerical equivalence, and inference |

## References

- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development Policies?*
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: what, when, why, and how. *AER*, 110(8).
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share research designs. *ReStud*, 89(1).
- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: theory and inference. *QJE*, 134(4).
