using LinearAlgebra
using Random
using Test

using GPUEnv

GPUEnv.activate(; include_jlarrays = true, persist = true)

using AbstractOperators: MatrixOp
using ProximalOperators: NormL1
using ProximalAlgorithms: PANOC, PANOCplus, ZeroFPR, FastForwardBackward, ADMM
import ProximalAlgorithms: value_and_gradient, value_and_gradient!

# Generic-array coverage for the solver iterate loops themselves: this package's own test
# suite exercises every solver only with CPU arrays (`f` built from `AutoDifferentiable` or
# `ProximalOperators` functions bridged in by StructuredOptimization.jl), so this file is the
# only place a device array ever reaches PANOC/PANOCplus/ZeroFPR/FastForwardBackward/ADMM's
# internal `similar`-based scratch buffers. `QuadTest` is a minimal hand-written smooth
# quadratic (no autodiff, no StructuredOptimization dependency) so a failure here always
# points at this package's own iterate/state code, never at a bridge or backend library.
struct QuadTest{TA, Tb}
    A::TA
    b::Tb
end
(q::QuadTest)(x) = 0.5 * real(dot(q.A * x - q.b, q.A * x - q.b))
function value_and_gradient(q::QuadTest, x)
    r = q.A * x - q.b
    return 0.5 * real(dot(r, r)), q.A' * r
end
function value_and_gradient!(grad, q::QuadTest, x)
    r = q.A * x - q.b
    mul!(grad, q.A', r)
    return 0.5 * real(dot(r, r))
end

for backend in gpu_backends(; include_jlarrays = true)
    @testset "GPU backend: $(backend.name)" begin
        Random.seed!(0)
        A, b = randn(6, 5), randn(6)
        Ag, bg = to_gpu(backend, A), to_gpu(backend, b)
        f = QuadTest(Ag, bg)
        g = NormL1(0.05)

        x0 = gpu_zeros(backend, Float64, 5)
        sol, _ = PANOCplus(tol = 1.0e-8)(x0 = x0, f = f, g = g)
        @test typeof(sol) == typeof(x0)

        @testset "$name matches PANOCplus" for (name, run) in (
                ("FastForwardBackward", () -> FastForwardBackward(tol = 1.0e-8)(x0 = gpu_zeros(backend, Float64, 5), f = f, g = g)),
                ("ZeroFPR", () -> ZeroFPR(tol = 1.0e-8)(x0 = gpu_zeros(backend, Float64, 5), f = f, g = g)),
                ("PANOC", () -> PANOC(tol = 1.0e-8)(x0 = gpu_zeros(backend, Float64, 5), f = f, g = g)),
                ("ADMM", () -> ADMM(maxit = 2000, rho = 1.0)(x0 = gpu_zeros(backend, Float64, 5), A = Ag, b = bg, g = g)),
            )
            other, _ = run()
            @test typeof(other) == typeof(x0)
            @test Array(other) ≈ Array(sol) rtol = 1.0e-3
        end

        @testset "MatrixOp as the linear operator" begin
            Aop = MatrixOp(Ag)
            x0m = gpu_zeros(backend, Float64, 5)
            solm, _ = PANOCplus(tol = 1.0e-8)(x0 = x0m, f = QuadTest(Aop, bg), g = g)
            @test Array(solm) ≈ Array(sol) rtol = 1.0e-3
        end
    end
end
