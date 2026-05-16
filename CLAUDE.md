# ShiftShareIV.jl

## Overview

Julia toolkit for shift-share (Bartik) instrumental variables.

- **GitHub**: https://github.com/xiangao/ShiftShareIV.jl
- **Docs**: https://xiangao.github.io/ShiftShareIV.jl/
- **Version**: 0.1.0 (not yet registered in Julia General registry)

## Exported functions

| Function | Purpose |
|---|---|
| `bartik_iv(shares, shocks)` | Construct Bartik instrument B_ℓ = S * g |
| `rotemberg_weights(shares, shocks, X[, Y])` | GPSS (2020) Rotemberg decomposition; returns DataFrame |
| `bhj_collapse(shares, shocks, Y, X; weights)` | BHJ (2022) shock-level collapse; returns DataFrame |

## Structure

- `src/ShiftShareIV.jl` — all code in one file (~130 lines)
- `test/runtests.jl` — 14 tests (4 test sets)
- `docs/` — Documenter.jl site with 3 vignettes
- `.github/workflows/docs.yml` — auto-deploys docs to gh-pages on push to main

## Build/test

```bash
cd ~/projects/software/ShiftShareIV.jl
julia --project -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl   # local docs build
```

## Notes

- Used in the Julia book (`causal_econometrics_julia`) via `include()` rather than Project.toml
  because the book environment has a pre-existing MLJ version conflict (Crumble 0.20 vs
  CausalEstimate 0.23) that blocks `Pkg.develop`.
- The `rotemberg_weights` function has two overloads: without `Y` (returns alpha only) and
  with `Y` (also returns beta_k and alpha_beta for the full GPSS decomposition identity).
- `bhj_collapse` accepts an optional `weights` keyword for location-level size weights.
