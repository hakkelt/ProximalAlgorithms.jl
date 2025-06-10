using Test
using LinearAlgebra
using ProximalAlgorithms
using ProximalOperators: NormL1, NormL2
using Random

@testset "ADMM" begin
    @testset "Least squares with L1 regularization" begin
        n = 100
        m = 80
        A = randn(m,n)
        b = randn(m)
        x0 = zeros(n)
        λ = 0.1
        
        admm = ProximalAlgorithms.ADMM(;
            x0 = x0,
            A = A,
            b = b,
            g = (NormL1(λ),),
            B = (I,),
            ρ = [1.0],
        )
        
        x, it = admm()
        @test norm(A*x - b) < 1e-3
    end
    
    @testset "Multiple regularizers" begin
        n = 100
        m = 80
        A = randn(m,n)
        b = randn(m)
        x0 = zeros(n)
        
        admm = ProximalAlgorithms.ADMM(;
            x0 = x0,
            A = A,
            b = b,
            g = (NormL1(0.1), NormL2(0.05)),
            B = (I, I),
            ρ = [1.0, 1.0],
        )
        
        x, it = admm()
        @test norm(A*x - b) < 1e-3
    end
end
