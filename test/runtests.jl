using Test
using Random
using LinearAlgebra
using Statistics
using DataFrames
using ShiftShareIV

@testset "bartik_iv" begin
    shares = [0.3 0.7; 0.6 0.4; 0.5 0.5]   # 3 regions × 2 industries
    shocks = [1.0, -1.0]
    B = bartik_iv(shares, shocks)
    @test length(B) == 3
    @test B ≈ [-0.4, 0.2, 0.0]  # row × column dot products

    # Dimension mismatch
    @test_throws DimensionMismatch bartik_iv(shares, [1.0, 2.0, 3.0])
end

@testset "rotemberg_weights — no Y" begin
    Random.seed!(1)
    L, K = 200, 5
    shares_raw = rand(L, K) .+ 0.1
    shares = shares_raw ./ sum(shares_raw, dims=2)
    shocks = randn(K)
    X = shares * shocks + 0.2 * randn(L)

    rw = rotemberg_weights(shares, shocks, X)
    @test nrow(rw) == K
    @test isapprox(sum(rw.alpha), 1.0, atol=1e-10)
    @test all(rw.industry .== 1:K)
end

@testset "rotemberg_weights — with Y, decomposition identity" begin
    Random.seed!(42)
    L, K = 500, 8
    beta_true = 0.7
    shares_raw = rand(L, K) .+ 0.1
    shares = shares_raw ./ sum(shares_raw, dims=2)
    shocks = randn(K)
    B = bartik_iv(shares, shocks)
    u = randn(L)
    X = B + 0.3 * u + 0.1 * randn(L)
    Y = beta_true * X + u + 0.3 * randn(L)

    rw = rotemberg_weights(shares, shocks, X, Y)

    # GPSS decomposition: Σ_k alpha_k * beta_k == 2SLS estimate
    # 2SLS estimate: Cov(B, Y) / Cov(B, X)
    Bc = B .- mean(B); Xc = X .- mean(X); Yc = Y .- mean(Y)
    iv_est = dot(Bc, Yc) / dot(Bc, Xc)
    decomp = sum(rw.alpha_beta)
    @test isapprox(decomp, iv_est, atol=1e-10)
    @test isapprox(sum(rw.alpha), 1.0, atol=1e-10)
end

@testset "bhj_collapse" begin
    Random.seed!(7)
    L, K = 100, 6
    shares_raw = rand(L, K) .+ 0.1
    shares = shares_raw ./ sum(shares_raw, dims=2)
    shocks = randn(K)
    B = bartik_iv(shares, shocks)
    u = randn(L)
    X = B + 0.3 * u + 0.1 * randn(L)
    Y = 0.5 * X + u + 0.3 * randn(L)

    collapsed = bhj_collapse(shares, shocks, Y, X)
    @test nrow(collapsed) == K
    @test all(collapsed.weight .> 0)
    @test !any(isnan.(collapsed.Y_agg))
    @test !any(isnan.(collapsed.X_agg))

    # Weighted version
    w = rand(L) .+ 0.5
    collapsed_w = bhj_collapse(shares, shocks, Y, X; weights=w)
    @test nrow(collapsed_w) == K
    # Weights should differ from uniform
    @test collapsed_w.weight != collapsed.weight
end
