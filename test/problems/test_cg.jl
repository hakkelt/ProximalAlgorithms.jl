using Test
using LinearAlgebra
using ProximalAlgorithms
using Random

@testset "CG" begin
    @testset "Real inputs" begin
        n = 100
        A = rand(n,n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        
        # Test basic CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
        
        # Test with preconditioner
        P = Diagonal(diag(A))  # Jacobi preconditioner
        pcg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, P=P)
        x, it = pcg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Complex inputs" begin
        n = 100
        A = rand(ComplexF64, n,n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Custom operator" begin
        # Define simple operator that implements mul!
        struct DiagonalOperator{T}
            diag::Vector{T}
        end
        
        function LinearAlgebra.mul!(y, A::DiagonalOperator, x)
            y .= A.diag .* x
            return y
        end

        function Base.:*(A::DiagonalOperator, x)
            return A.diag .* x
        end
        
        n = 100
        d = rand(n) .+ 1  # Ensure positive diagonal
        A = DiagonalOperator(d)
        b = rand(n)
        x0 = zeros(n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(d .* x - b) < 1e-6
    end
    
    @testset "Ridge regression" begin
        n = 100
        A = rand(n, n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        λ = 0.1  # Regularization parameter

        # Test ridge regression with CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2

        # Test ridge regression with complex inputs
        A = rand(ComplexF64, n, n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)

        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2
    end

    @testset "CGNR with a caller-supplied AᴴA" begin
        Random.seed!(0)
        A = randn(ComplexF64, 80, 50)
        b = rand(ComplexF64, 80)
        x0 = zeros(ComplexF64, 50)
        AHA = A' * A
        @test ProximalAlgorithms.CGNRIteration(; x0, A, b, AHA).A === AHA

        x, it = ProximalAlgorithms.CGNR(; x0, A, b, maxit = 100, tol = 0.0)()
        x_aha, it_aha = ProximalAlgorithms.CGNR(; x0, A, b, AHA, maxit = 100, tol = 0.0)()
        @test x_aha == x
        @test it_aha == it
        @test norm(A' * (A * x - b)) < 1e-6 * norm(A' * b)

        P = Diagonal(diag(AHA))
        xp, _ = ProximalAlgorithms.CGNR(; x0, A, b, P, maxit = 100, tol = 0.0)()
        xp_aha, _ = ProximalAlgorithms.CGNR(; x0, A, b, AHA, P, maxit = 100, tol = 0.0)()
        @test ProximalAlgorithms.PCGNRIteration(; x0, A, b, AHA, P).A === AHA
        @test xp_aha == xp
        @test xp ≈ x
    end

    @testset "CGNR with a caller-supplied Aᴴb" begin
        Random.seed!(0)
        A = randn(ComplexF64, 80, 50)
        b = rand(ComplexF64, 80)
        x0 = zeros(ComplexF64, 50)
        AHb = A' * b
        @test ProximalAlgorithms.CGNRIteration(; x0, A, b, AHb).b === AHb

        x, it = ProximalAlgorithms.CGNR(; x0, A, b, maxit = 100, tol = 0.0)()
        x_ahb, it_ahb = ProximalAlgorithms.CGNR(; x0, A, b, AHb, maxit = 100, tol = 0.0)()
        @test x_ahb == x
        @test it_ahb == it
        # The supplied vector is the right-hand side actually solved for, not `A' * b`.
        x_zero, _ = ProximalAlgorithms.CGNR(; x0, A, b, AHb = zero(AHb), maxit = 100, tol = 0.0)()
        @test iszero(x_zero)

        P = Diagonal(diag(A' * A))
        xp, _ = ProximalAlgorithms.CGNR(; x0, A, b, P, maxit = 100, tol = 0.0)()
        xp_ahb, _ = ProximalAlgorithms.CGNR(; x0, A, b, AHb, P, maxit = 100, tol = 0.0)()
        @test ProximalAlgorithms.PCGNRIteration(; x0, A, b, AHb, P).b === AHb
        @test xp_ahb == xp
    end
end
