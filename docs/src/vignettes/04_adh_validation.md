# Validation: the ADH China Shock (GPSS benchmark)

```@meta
CurrentModule = ShiftShareIV
```

The other vignettes use simulated data. This one runs `ShiftShareIV.jl` on the
**real, canonical shift-share dataset** — the Autor–Dorn–Hanson (2013) "China
shock" — and reproduces, to three decimals, the Rotemberg-weight decomposition
published by Goldsmith-Pinkham, Sorkin & Swift (2020, *AER*). The data here are
GPSS's own replication files (the [`bartik-weight`](https://github.com/paulgp/bartik-weight)
repository), shipped with the docs in compact form under `docs/adh_data/`.

The design stacks two periods (1990, 2000):

- **L = 1,444** commuting-zone × period observations
- **K = 794** industry × period instruments
- population weights `timepwt48`; controls = ADH demographics + census-division
  fixed effects + period.

## Loading the data

The shares matrix is 88% zeros, so it is stored as sparse `(row, col, value)`
triplets and reconstructed.

```@example adh
using ShiftShareIV, DataFrames, DelimitedFiles, LinearAlgebra, Statistics, Printf

dir    = joinpath(pkgdir(ShiftShareIV), "docs", "adh_data")
trip   = readdlm(joinpath(dir, "shares_triplets.csv"), ',')
shocks = vec(readdlm(joinpath(dir, "shocks.csv")))
sic    = vec(Int.(readdlm(joinpath(dir, "industry_sic.csv"))))
M      = readdlm(joinpath(dir, "master.csv"), ',')   # cols: Y X IV w  6 controls  division period

L, K = size(M, 1), length(shocks)
shares = zeros(L, K)
for r in axes(trip, 1)
    shares[Int(trip[r, 1]), Int(trip[r, 2])] = trip[r, 3]
end

Y, X, IVref, w = M[:, 1], M[:, 2], M[:, 3], M[:, 4]
controls = M[:, 5:10]
division = Int.(M[:, 11])
period   = M[:, 12]
@printf("L = %d locations × periods,  K = %d industries × periods\n", L, K)
```

## Step 1 — the Bartik instrument

`bartik_iv` forms $B_\ell = \sum_k s_{\ell k}\, g_k$. It should reproduce the
instrument GPSS constructed (their `IV` column) exactly:

```@example adh
B = bartik_iv(shares, shocks)
@printf("cor(bartik_iv, GPSS IV) = %.4f\n", cor(B, IVref))
```

## Step 2 — residualize on the controls (the step that matters)

`rotemberg_weights` decomposes whatever you give it. To match GPSS you must first
partial the ADH controls $W$ (demographics + division FE + period) out of $X$, $Y$,
**and every share column**, weighted by `timepwt48` — i.e. weighted Frisch–Waugh–Lovell.
Skipping this gives the right *ranking* but the top weight comes out ≈ 0.142
instead of 0.183.

```@example adh
divs = sort(unique(division))[2:end]                         # division FE (drop one)
W = hcat(ones(L), controls,
         reduce(hcat, [Float64.(division .== d) for d in divs]),
         period)

# weighted residual: v - W (W'ΩW)^{-1} W'Ω v, with Ω = diag(w). Works for vec or matrix.
wresid(V) = V .- W * ((W' * (w .* W)) \ (W' * (w .* V)))

# fold population weights in by scaling residuals by √w, then center=false
sw = sqrt.(w)
rw = rotemberg_weights(sw .* wresid(shares), shocks,
                       sw .* wresid(X), sw .* wresid(Y); center = false)
@printf("Σ Rotemberg weights = %.4f\n", sum(rw.alpha))
```

## Step 3 — compare to the published GPSS table

GPSS report the positive/negative split and the top industries at the **industry**
level (the two periods collapsed), so we aggregate `alpha` over periods first.

```@example adh
rw.sic = sic
agg = sort!(combine(groupby(rw, :sic), :alpha => sum => :alpha), :alpha, rev = true)

an = -sum(agg.alpha[agg.alpha .< 0]); ap = sum(agg.alpha[agg.alpha .> 0])
@printf("negative-weight mass share = %.3f   (GPSS 0.059)\n", an / (an + ap))
@printf("positive-weight mass share = %.3f   (GPSS 0.941)\n", ap / (an + ap))
@printf("Σ negative = %+.3f  Σ positive = %+.3f   (GPSS -0.067 / +1.067)\n\n", -an, ap)

names = Dict(3571 => "Electronic Computers", 3944 => "Games, Toys & Children's Veh.",
             3651 => "Household Audio & Video", 3661 => "Telephone & Telegraph App.",
             3577 => "Computer Peripheral Eq.")
gpss  = Dict(3571 => 0.183, 3944 => 0.138, 3651 => 0.085, 3661 => 0.066, 3577 => 0.060)
@printf("%-30s %8s %8s\n", "industry (SIC)", "alpha", "GPSS")
for row in eachrow(first(agg, 5))
    @printf("%-30s %8.3f %8.3f\n", get(names, row.sic, "SIC $(row.sic)"),
            row.alpha, get(gpss, row.sic, NaN))
end
```

Every reported quantity matches GPSS to three decimals: the instrument
(correlation 1.0), the positive/negative weight split (0.059 / 0.941), and the
five most influential industries — Electronic Computers, Games & Toys, Household
Audio/Video, Telephone/Telegraph, Computer Peripherals.

## Takeaways

1. `bartik_iv` and `rotemberg_weights` are numerically correct against the
   field-standard published benchmark on real data — not just internal unit tests.
2. Two reproduction details, both standard: **residualize on the included controls
   before decomposing**, and remember **GPSS's weight summary is at the industry
   level** (periods collapsed).
```
