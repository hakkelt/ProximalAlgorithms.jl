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

    @testset "Level-1 BLAS grant" begin
        NT = ProximalAlgorithms.NestedThreading
        PA = ProximalAlgorithms
        n = 100
        A = rand(n, n)
        A = A'A + I
        b = rand(n)
        x0 = zeros(n)
        reference, _ = PA.CG(x0 = x0, A = A, b = b)()

        blas = BLAS.get_num_threads()
        old = PA.CG_BLAS_THREAD_BYTES[]
        try
            # Defaults: nothing this small is granted, and non-BLAS storage never is.
            @test !PA._grants_level1(x0)
            @test !PA._grants_level1(zeros(BigFloat, 2^21))
            PA.CG_BLAS_THREAD_BYTES[] = sizeof(x0)
            @test PA._grants_level1(x0)
            @test !PA._grants_level1(zeros(n - 1))

            # The grant restores BLAS's own count inside a serial default, and the solve,
            # with and without a preconditioner, computes the same thing through it.
            NT.with_thread_default(1; only = (:blas,)) do
                observed = PA._with_level1_threads(BLAS.get_num_threads, x0)
                @test observed == blas
                x, _ = PA.CG(x0 = x0, A = A, b = b)()
                @test x ≈ reference
                x, _ = PA.CG(x0 = x0, A = A, b = b, P = Diagonal(diag(A)))()
                @test norm(A * x - b) < 1e-6
                @test BLAS.get_num_threads() == 1
            end
            @test BLAS.get_num_threads() == blas
        finally
            PA.CG_BLAS_THREAD_BYTES[] = old
        end
    end
end
