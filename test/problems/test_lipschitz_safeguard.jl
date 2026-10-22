using LinearAlgebra
using Random
using Test

using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: NormL1
using ProximalAlgorithms
using JLArrays: JLArrays, JLArray
using RecursiveArrayTools: ArrayPartition

@testset "Lipschitz safeguard ($T)" for T in [Float32, Float64, ComplexF64]
    R = real(T)
    rng = Xoshiro(7)
    A = randn(rng, T, 60, 40)
    b = randn(rng, T, 60)
    f = ProximalAlgorithms.AutoDifferentiable(x -> (norm(A * x - b)^2) / 2, AutoZygote())
    g = NormL1(R(0.1) * norm(A' * b, Inf))
    Lf = opnorm(A)^2

    # The reference solution, from a valid stepsize run long.
    x_star, _ = ProximalAlgorithms.FastForwardBackward(maxit = 20_000, tol = 1.0e-10)(
        x0 = zeros(T, 40), f = f, g = g, Lf = Lf,
    )
    tol = T <: Float32 ? 1.0e-3 : 1.0e-6

    # A third of the true constant: unguarded, FISTA never settles. (POGM's restart backoff rescues
    # it on its own, at a stepsize halved from the wrong one.)
    x, _ = ProximalAlgorithms.FastForwardBackward(maxit = 2000, tol = 1.0e-10)(
        x0 = zeros(T, 40), f = f, g = g, Lf = Lf / 3, lipschitz_safeguard = false,
    )
    @test !(norm(x - x_star) <= tol * norm(x_star))

    for (alg, Iteration) in (
            (ProximalAlgorithms.FastForwardBackward, ProximalAlgorithms.FastForwardBackwardIteration),
            (ProximalAlgorithms.POGM, ProximalAlgorithms.POGMIteration),
        )
        # Guarded, the secant quotients expose the low constant and the run converges.
        x, _ = alg(maxit = 2000, tol = 1.0e-10)(x0 = zeros(T, 40), f = f, g = g, Lf = Lf / 3)
        @test norm(x - x_star) <= tol * norm(x_star)
        # The stepsize it settles on lies between the safe 1/(1.01 Lf) and the wrong 3/Lf.
        gamma = foldl((_, s) -> s.gamma, Iterators.take(Iteration(x0 = zeros(T, 40), f = f, g = g, Lf = Lf / 3), 200); init = nothing)
        @test 1 / (1.01 * Lf) * (1 - 1.0e-4) <= gamma < 3 / Lf

        # A valid stepsize is never touched: the iterates are those of the unguarded run.
        iter_on = Iteration(x0 = zeros(T, 40), f = f, g = g, Lf = Lf)
        iter_off = Iteration(x0 = zeros(T, 40), f = f, g = g, Lf = Lf, lipschitz_safeguard = false)
        for (s_on, s_off) in Iterators.take(zip(iter_on, iter_off), 300)
            @test s_on.gamma == s_off.gamma
            @test s_on.z == s_off.z
        end
    end
end

# Partitioned iterates of device arrays (JLArrays refuse scalar indexing), as the components of a
# decomposition solved on a GPU: the sums go part by part and agree with those of the whole arrays.
@testset "Lipschitz safeguard on an ArrayPartition of device arrays ($T)" for T in [Float32, ComplexF64]
    R = real(T)
    rng = Xoshiro(11)
    JLArrays.allowscalar(false)
    whole = [randn(rng, T, 30) for _ in 1:4]
    split2(v) = ArrayPartition(JLArray(v[1:12]), JLArray(v[13:end]))
    S = promote_type(R, Float64)
    @test collect(ProximalAlgorithms._safeguard_sums_in(S, split2.(whole)...)) ≈
        collect(ProximalAlgorithms._safeguard_sums_in(S, whole...))
    # A stepsize the quotient exceeds is shortened alike; one it does not is returned as is.
    for gamma in (R(10), R(1.0e-3))
        @test ProximalAlgorithms.lipschitz_safeguard(gamma, split2.(whole)...) ≈
            ProximalAlgorithms.lipschitz_safeguard(gamma, whole...)
    end
end
