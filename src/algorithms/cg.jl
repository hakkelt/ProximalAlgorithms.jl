# Linear conjugate gradient method for solving Ax = b
# Method of Hestenes and Stiefel, "Methods of conjugate gradients for solving linear systems."
# Journal of Research of the National Bureau of Standards 49.6 (1952).

abstract type AbstractCGIteration end
abstract type AbstractCGState end

mutable struct CGState{Tx,R<:Real} <: AbstractCGState
    x::Tx
    r::Tx
    p::Tx
    Ap::Tx
    α::R
    β::R
    rr::R
    res_norm::R
end

CGState(x0) = CGState{typeof(x0), real(eltype(x0))}(
    copy(x0), # x
    similar(x0), # r
    similar(x0), # p
    similar(x0), # Ap
    zero(real(eltype(x0))), # α
    zero(real(eltype(x0))), # β
    zero(real(eltype(x0))), # rr
    zero(real(eltype(x0))), # res_norm
)

mutable struct PCGState{Tx,R<:Real} <: AbstractCGState
    x::Tx
    r::Tx
    p::Tx
    Ap::Tx
    z::Tx
    α::R
    β::R
    rz::R
    res_norm::R
end

PCGState(x0) = PCGState{typeof(x0), real(eltype(x0))}(
    copy(x0), # x
    similar(x0), # r
    similar(x0), # p
    similar(x0), # Ap
    similar(x0), # z
    zero(real(eltype(x0))), # α
    zero(real(eltype(x0))), # β
    zero(real(eltype(x0))), # rz
    zero(real(eltype(x0))), # res_norm
)

struct CGIteration{Tx,TA,Tb,R} <: AbstractCGIteration
    x0::Tx
    A::TA
    b::Tb
    state::CGState{Tx,R}
end

function CGIteration(; x0::Tx, A::TA, b::Tb, state::CGState{Tx,R} = CGState(x0)) where {Tx,TA,Tb,R}
    return CGIteration{Tx,TA,Tb,R}(x0, A, b, state)
end

struct PCGIteration{Tx,TA,Tb,TP,R} <: AbstractCGIteration
    x0::Tx
    A::TA
    b::Tb
    P::TP
    P_is_inverse::Bool
    state::PCGState{Tx,R}
end

function PCGIteration(; x0::Tx, A::TA, b::Tb, P::TP, P_is_inverse = false, state::PCGState{Tx,R} = PCGState(x0)) where {Tx,TA,Tb,TP,R}
    return PCGIteration{Tx,TA,Tb,TP,R}(x0, A, b, P, P_is_inverse, state)
end

function Base.iterate(iter::CGIteration)
    state = iter.state

    # Reset state
    copyto!(state.x, iter.x0)
    
    # r = b - Ax
    mul!(state.r, iter.A, state.x)
    state.r .= iter.b .- state.r

    # p = r
    copyto!(state.p, state.r)

    # Initialize parameters
    state.rr = real(dot(vec(state.r), vec(state.r)))
    state.res_norm = sqrt(real(state.rr))
    
    return state, state
end

function Base.iterate(iter::PCGIteration)
    state = iter.state
    # Reset state
    copyto!(state.x, iter.x0)
    
    # r = b - Ax
    mul!(state.r, iter.A, state.x)
    state.r .= iter.b .- state.r

    # z = P\r or z = P*r
    if iter.P_is_inverse
        mul!(state.z, iter.P, state.r)
    else
        ldiv!(state.z, iter.P, state.r)
    end
    
    # p = z
    copyto!(state.p, state.z)

    # Initialize parameters
    state.rz = real(dot(vec(state.r), vec(state.z)))
    state.res_norm = norm(vec(state.r))

    return state, state
end

function Base.iterate(iter::CGIteration, state::CGState)
    # Ap = A*p
    mul!(state.Ap, iter.A, state.p)
    
    # α = (r'r)/(p'Ap)
    pAp = real(dot(vec(state.p), vec(state.Ap)))
    state.α = state.rr / pAp
    
    # x = x + αp
    axpy!(state.α, state.p, state.x)
    
    # r = r - αAp
    axpy!(-state.α, state.Ap, state.r)
    
    # β = (r'r)/(r_old'r_old) with proper conjugation
    rr_new = real(dot(vec(state.r), vec(state.r)))
    state.β = rr_new / state.rr
    state.rr = rr_new
    
    # p = r + βp
    state.p .= state.r .+ state.β .* state.p
    
    # Update residual norm
    state.res_norm = sqrt(real(rr_new))
    
    return state, state
end

function Base.iterate(iter::PCGIteration, state::PCGState)
    # Ap = A*p
    mul!(state.Ap, iter.A, state.p)
    
    # α = (r'z)/(p'Ap)
    pAp = real(dot(vec(state.p), vec(state.Ap)))
    state.α = state.rz / pAp
    
    # x = x + αp
    axpy!(state.α, state.p, state.x)
    
    # r = r - αAp
    axpy!(-state.α, state.Ap, state.r)
    
    # z = P\r or z = P*r depending on P_is_inverse
    if iter.P_is_inverse
        mul!(state.z, iter.P, state.r)
    else
        ldiv!(state.z, iter.P, state.r)
    end
    
    # β = (r'z)/(r_old'z_old)
    rz_new = real(dot(vec(state.r), vec(state.z)))
    state.β = rz_new / state.rz
    state.rz = rz_new
    
    # p = z + βp
    state.p .= state.z .+ state.β .* state.p
    
    # Update residual norm
    state.res_norm = norm(vec(state.r))
    
    return state, state
end

default_stopping_criterion(tol, ::AbstractCGIteration, state::AbstractCGState) =
    state.res_norm <= tol

default_solution(::AbstractCGIteration, state::AbstractCGState) = state.x

default_display(it, ::AbstractCGIteration, state::AbstractCGState) =
    @printf("%5d | %.3e\n", it, state.res_norm)

"""
    CG(; <keyword-arguments>)

Constructs the Conjugate Gradient algorithm.

This algorithm solves linear systems of the form

    Ax = b

where `A` is a symmetric positive definite linear operator.

The returned object has type `IterativeAlgorithm{CGIteration}`.

# Arguments
- `maxit::Int=1000`: maximum number of iterations
- `tol::Float64=1e-8`: tolerance for the stopping criterion
- `stop::Function`: custom stopping criterion
- `solution::Function`: solution mapping
- `verbose::Bool=false`: whether to display iteration information
- `freq::Int=100`: frequency of iteration display
- `display::Function`: custom display function
- `kwargs...`: additional keyword arguments for CGIteration

# References
1. Hestenes, M.R. and Stiefel, E., "Methods of conjugate gradients for solving linear systems."
   Journal of Research of the National Bureau of Standards 49.6 (1952): 409-436.
"""
function CG(;
    maxit = 1000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    display = default_display,
    kwargs...,
)
    is_preconditioned = (:P in keys(kwargs) && kwargs[:P] !== nothing)
    if !is_preconditioned
        iterType = CGIteration
        kwargs = filter(kv -> kv[1] !== :P && kv[1] !== :P_is_inverse, kwargs)
    else
        iterType = PCGIteration
    end
    IterativeAlgorithm(
        iterType;
        maxit,
        stop,
        solution,
        verbose,
        freq,
        display,
        kwargs...,
    )
end

get_assumptions(::Type{<:AbstractCGIteration}) = (
    LeastSquaresTerm(:A => (is_linear, is_symmetric, is_positive_definite), :b),
)

# Aliases
const ConjugateGradientIteration = CGIteration
const ConjugateGradient = CG

"""
Solve CG system using existing state
"""
function solve!(iter::AbstractCGIteration, alg::IterativeAlgorithm)
    state = iterate(iter)[1]
    alg.verbose && alg.display(0, iter, state)

    it = 1
    for (st, _) in Iterators.drop(iter, 1)
        alg.verbose && it % alg.freq == 0 && alg.display(it, alg, st)
        alg.stop(iter, st) && break
        it += 1
    end

    return iter.x
end
