module ShiftShareIV

using DataFrames
using LinearAlgebra
using Statistics

export bartik_iv, rotemberg_weights, bhj_collapse

"""
    bartik_iv(shares, shocks)

Compute the Bartik (shift-share) instrument for each location.

# Arguments
- `shares`: L × K matrix of industry shares (rows = locations, columns = industries).
  Each row should sum to 1 (or close to 1).
- `shocks`: K-vector of national industry-level growth rates or shocks.

# Returns
L-vector of instrument values B_ℓ = Σ_k s_{ℓk} g_k.

# Reference
Bartik (1991); Goldsmith-Pinkham, Sorkin, Swift (2020).
"""
function bartik_iv(shares::Matrix{<:Real}, shocks::Vector{<:Real})
    size(shares, 2) == length(shocks) ||
        throw(DimensionMismatch("shares has $(size(shares,2)) columns but shocks has $(length(shocks)) elements"))
    return shares * shocks
end

"""
    rotemberg_weights(shares, shocks, X; center=true)

Compute GPSS (2020) Rotemberg weights: the decomposition of the shift-share IV
estimate into a weighted average of K just-identified IV estimates, one per industry.

# Arguments
- `shares`: L × K matrix of industry shares.
- `shocks`: K-vector of shocks.
- `X`: L-vector of the endogenous regressor (the local shock measure).
- `center`: if true (default), demean shares and X before computing covariances,
  matching the within-variation used by OLS/2SLS.

# Returns
DataFrame with columns:
- `industry`: industry index 1…K
- `shock`: g_k
- `alpha`: Rotemberg weight α_k (sums to 1 over all industries)
- `beta_k`: just-identified IV estimate using share k as the sole instrument
- `alpha_beta`: α_k × β_k contribution to the overall IV estimate

# Reference
Goldsmith-Pinkham, Sorkin, Swift (2020), AER.
"""
function rotemberg_weights(shares::Matrix{<:Real}, shocks::Vector{<:Real},
                           X::Vector{<:Real}; center::Bool=true)
    L, K = size(shares)
    length(shocks) == K || throw(DimensionMismatch("shocks length mismatch"))
    length(X) == L     || throw(DimensionMismatch("X length mismatch"))

    S = center ? shares .- mean(shares, dims=1) : shares
    Xc = center ? X .- mean(X) : X

    cov_sk_X = [dot(S[:, k], Xc) / (L - 1) for k in 1:K]
    cov_sk_Y_num = [dot(S[:, k], Xc) for k in 1:K]  # proportional to Cov(s_k, X)

    denom = dot(shocks, cov_sk_X)
    abs(denom) < 1e-12 && error("Denominator near zero: shocks and shares may be collinear with X")

    alpha  = shocks .* cov_sk_X ./ denom

    # β_k = Cov(s_k, Y) / Cov(s_k, X) — but Y is not passed here.
    # Return Cov(s_k, X) for the caller to compute β_k = Cov(s_k, Y)/Cov(s_k, X).
    # We expose cov_sk_X so callers can complete the decomposition.
    return DataFrame(
        industry  = 1:K,
        shock     = shocks,
        cov_sk_X  = cov_sk_X,
        alpha     = alpha,
    )
end

"""
    rotemberg_weights(shares, shocks, X, Y; center=true)

Extended version that also returns β_k (just-identified IV per industry) and
the α_k × β_k contribution. Requires both X (endogenous regressor) and Y (outcome).
"""
function rotemberg_weights(shares::Matrix{<:Real}, shocks::Vector{<:Real},
                           X::Vector{<:Real}, Y::Vector{<:Real}; center::Bool=true)
    L, K = size(shares)
    length(Y) == L || throw(DimensionMismatch("Y length mismatch"))

    S  = center ? shares .- mean(shares, dims=1) : shares
    Xc = center ? X .- mean(X) : X
    Yc = center ? Y .- mean(Y) : Y

    cov_sk_X = [dot(S[:, k], Xc) / (L - 1) for k in 1:K]
    cov_sk_Y = [dot(S[:, k], Yc) / (L - 1) for k in 1:K]

    denom = dot(shocks, cov_sk_X)
    abs(denom) < 1e-12 && error("Denominator near zero")

    alpha  = shocks .* cov_sk_X ./ denom
    beta_k = cov_sk_Y ./ cov_sk_X

    return DataFrame(
        industry   = 1:K,
        shock      = shocks,
        cov_sk_X   = cov_sk_X,
        alpha      = alpha,
        beta_k     = beta_k,
        alpha_beta = alpha .* beta_k,
    )
end

"""
    bhj_collapse(shares, shocks, Y, X; weights=nothing)

Collapse a location-level dataset to a shock (industry) level dataset, following
Borusyak, Hull, Jaravel (2022). The resulting shock-level regression is equivalent
to the location-level shift-share IV regression but makes the identifying assumption
(shocks are as-if-random) more transparent.

# Arguments
- `shares`: L × K matrix of shares (rows = locations, cols = industries).
- `shocks`: K-vector of industry-level shocks.
- `Y`: L-vector of outcome at location level.
- `X`: L-vector of endogenous regressor at location level.
- `weights`: optional L-vector of location weights (e.g. employment); default = uniform.

# Returns
DataFrame with one row per industry (K rows):
- `industry`: industry index
- `shock`: g_k
- `Y_agg`: aggregated outcome at shock level (Σ_ℓ s_{ℓk} w_ℓ Y_ℓ / Σ_ℓ s_{ℓk} w_ℓ)
- `X_agg`: aggregated endogenous variable at shock level
- `weight`: Σ_ℓ s_{ℓk} w_ℓ  (the "exposure" weight for industry k)

To run the BHJ shock-level IV: regress Y_agg ~ X_agg with instrument shock,
weighting by weight (use WLS/weighted 2SLS).

# Reference
Borusyak, Hull, Jaravel (2022), Review of Economic Studies.
"""
function bhj_collapse(shares::Matrix{<:Real}, shocks::Vector{<:Real},
                      Y::Vector{<:Real}, X::Vector{<:Real};
                      weights::Union{Vector{<:Real}, Nothing}=nothing)
    L, K = size(shares)
    length(shocks) == K || throw(DimensionMismatch("shocks length mismatch"))
    length(Y) == L      || throw(DimensionMismatch("Y length mismatch"))
    length(X) == L      || throw(DimensionMismatch("X length mismatch"))

    w = weights === nothing ? ones(L) : weights
    length(w) == L || throw(DimensionMismatch("weights length mismatch"))

    # For each industry k: w_k = Σ_ℓ s_{ℓk} w_ℓ
    industry_weight = [dot(shares[:, k], w) for k in 1:K]

    # Weighted average of Y and X at the shock level
    Y_agg = [dot(shares[:, k] .* w, Y) / max(industry_weight[k], 1e-12) for k in 1:K]
    X_agg = [dot(shares[:, k] .* w, X) / max(industry_weight[k], 1e-12) for k in 1:K]

    return DataFrame(
        industry = 1:K,
        shock    = shocks,
        Y_agg    = Y_agg,
        X_agg    = X_agg,
        weight   = industry_weight,
    )
end

end # module
