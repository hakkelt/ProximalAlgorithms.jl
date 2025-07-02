# Linear conjugate gradient method for solving Ax = b
# Method of Hestenes and Stiefel, "Methods of conjugate gradients for solving linear systems."
# Journal of Research of the National Bureau of Standards 49.6 (1952).

abstract type AbstractCGIteration end
abstract type AbstractCGState end

mutable struct CGState{Tx,R<:Real} <: AbstractCGState
	x::Tx       # current iterate
	r::Tx       # residual (b - Ax)
	p::Tx       # search direction
	Ap::Tx      # A*p
	α::R        # step size
	β::R        # conjugate direction parameter
	r²::R       # squared norm of residual
end

function CGState(x0)
	CGState{typeof(x0),real(eltype(x0))}(
		copy(x0), # x
		similar(x0), # r
		similar(x0), # p
		similar(x0), # Ap
		0, # α
		0, # β
		0, # r²
	)
end

mutable struct PCGState{Tx,R<:Real} <: AbstractCGState
	x::Tx
	r::Tx
	p::Tx
	Ap::Tx
	z::Tx
	α::R
	β::R
	rz::R
	r²::R
end

function PCGState(x0)
	PCGState{typeof(x0),real(eltype(x0))}(
		copy(x0), # x
		similar(x0), # r
		similar(x0), # p
		similar(x0), # Ap
		similar(x0), # z
		0, # α
		0, # β
		0, # rz
		0, # r²
	)
end

"""
	CGIteration(; <keyword-arguments>)

Iterator implementing the Conjugate Gradient (CG) algorithm.

This iterator solves linear systems of the form

	argminₓ ||Ax - b||₂² + ||λx||₂² 

where `A` is a symmetric positive definite linear operator, and `b` is the measurement vector,
and `λ` is the L2 regularization parameter. `λ` might be scalar or an array of the same size
as `x`. If `λ` is zero, the problem reduces to a least-squares problem:

    argminₓ ||Ax - b||₂²

# Arguments
- `x0`: initial point
- `A`: symmetric positive definite linear operator
- `b`: measurement vector
- `λ=0`: L2 regularization parameter (default: 0)

# References
1. Hestenes, M.R. and Stiefel, E., "Methods of conjugate gradients for solving linear systems."
   Journal of Research of the National Bureau of Standards 49.6 (1952): 409-436.
"""
struct CGIteration{Tx,TA,Tb,R} <: AbstractCGIteration
	x0::Tx
	A::TA
	b::Tb
    λ::R
	state::CGState{Tx,R}
end

function CGIteration(;
	x0::Tx, A::TA, b::Tb, λ::R=0, state::CGState=CGState(x0)
) where {Tx,TA,Tb,R}
	return CGIteration{Tx,TA,Tb,real(eltype(x0))}(x0, A, b, λ, state)
end

"""
	PCGIteration(; <keyword-arguments>)

Iterator implementing the Preconditioned Conjugate Gradient (PCG) algorithm.

This iterator solves linear systems of the form

	argminₓ ||Ax - b||₂² + ||λx||₂² 

where `A` is a symmetric positive definite linear operator, and `b` is the measurement vector,
and `λ` is the L2 regularization parameter. `λ` might be scalar or an array of the same size
as `x`. If `λ` is zero, the problem reduces to a least-squares problem:

    argminₓ ||Ax - b||₂²

A preconditioner `P` is used to accelerate convergence.

# Arguments
- `x0`: initial point
- `A`: symmetric positive definite linear operator
- `b`: measurement vector
- `λ=0`: L2 regularization parameter (default: 0)
- `P`: preconditioner (optional)
- `P_is_inverse`: whether `P` is the inverse of the preconditioner (default: `false`)

# References
1. Hestenes, M.R. and Stiefel, E., "Methods of conjugate gradients for solving linear systems."
   Journal of Research of the National Bureau of Standards 49.6 (1952): 409-436.
"""
struct PCGIteration{Tx,TA,Tb,TP,R} <: AbstractCGIteration
	x0::Tx
	A::TA
	b::Tb
	P::TP
	P_is_inverse::Bool
    λ::R
	state::PCGState{Tx,R}
end

function PCGIteration(;
	x0::Tx, A::TA, b::Tb, P::TP, P_is_inverse=false, λ::R=0, state::PCGState=PCGState(x0)
) where {Tx,TA,Tb,TP,R}
	return PCGIteration{Tx,TA,Tb,TP,real(eltype(x0))}(x0, A, b, P, P_is_inverse, λ, state)
end

function Base.iterate(iter::CGIteration)
	state = iter.state

	# Reset state
	copyto!(state.x, iter.x0)

	# Compute residual r = b - Ax - λx
	mul!(state.r, iter.A, state.x)
	@. state.r = iter.b - state.r
	if iter.λ > 0
		@. state.r -= iter.λ * state.x
	end

	copyto!(state.p, state.r)

	state.r² = real(dot(vec(state.r), vec(state.r)))

	return state, state
end

function Base.iterate(iter::PCGIteration)
	state = iter.state
	# Reset state
	copyto!(state.x, iter.x0)

	# r = b - Ax
	mul!(state.r, iter.A, state.x)
	@. state.r = iter.b - state.r

	# z = P\r or z = P*r
	if iter.P_is_inverse
		mul!(state.z, iter.P, state.r)
	else
		ldiv!(state.z, iter.P, state.r)
	end

	copyto!(state.p, state.z)

	state.rz = real(dot(vec(state.r), vec(state.z)))
	state.r² = real(dot(vec(state.r), vec(state.r)))

	return state, state
end

function Base.iterate(iter::CGIteration, state::CGState)
	# Ap = A*p
	mul!(state.Ap, iter.A, state.p) # compute A*p

	# Add regularization term if λ > 0
	if iter.λ > 0
		@. state.Ap += iter.λ * state.p # add regularization term λp
	end

	# α = (r'r)/(p'Ap)
	pAp = real(dot(vec(state.p), vec(state.Ap))) # compute p'Ap
	state.α = state.r² / pAp # compute step size α

	# x = x + αp
	axpy!(state.α, state.p, state.x) # update solution x

	# r = r - αAp
	axpy!(-state.α, state.Ap, state.r) # update residual r

	# β = (r'r)/(r_old'r_old)
	r²_new = real(dot(vec(state.r), vec(state.r))) # compute new squared norm of residual
	state.β = r²_new / state.r² # compute conjugate direction parameter β
	state.r² = r²_new # update squared norm of residual

	# p = r + βp
	@. state.p = state.r + state.β * state.p # update search direction p

	return state, state
end

function Base.iterate(iter::PCGIteration, state::PCGState)
	mul!(state.Ap, iter.A, state.p) # Ap = A*p

	pAp = real(dot(vec(state.p), vec(state.Ap)))
	state.α = state.rz / pAp # α = (r'z)/(p'Ap)

	axpy!(state.α, state.p, state.x) # x = x + αp
	axpy!(-state.α, state.Ap, state.r) # r = r - αAp

	# z = P\r or z = P*r depending on P_is_inverse
	if iter.P_is_inverse
		mul!(state.z, iter.P, state.r)
	else
		ldiv!(state.z, iter.P, state.r)
	end

	rz_new = real(dot(vec(state.r), vec(state.z)))
	state.β = rz_new / state.rz # β = (r'z)/(r_old'z_old)
	state.rz = rz_new
	state.r² = real(dot(vec(state.r), vec(state.r))) # r² = r'r

	@. state.p = state.z + state.β * state.p # p = z + βp

	return state, state
end

function default_stopping_criterion(tol, ::AbstractCGIteration, state::AbstractCGState)
	sqrt(state.r²) <= tol
end

default_solution(::AbstractCGIteration, state::AbstractCGState) = state.x

function default_display(it, ::AbstractCGIteration, state::AbstractCGState)
	@printf("%5d | %.3e\n", it, sqrt(state.r²))
end

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
	maxit=1000,
	tol=1e-8,
	stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
	solution=default_solution,
	verbose=false,
	freq=100,
	display=default_display,
	kwargs...,
)
	is_preconditioned = (:P in keys(kwargs) && kwargs[:P] !== nothing)
	if !is_preconditioned
		iterType = CGIteration
		kwargs = filter(kv -> kv[1] !== :P && kv[1] !== :P_is_inverse, kwargs)
	else
		iterType = PCGIteration
	end
	IterativeAlgorithm(iterType; maxit, stop, solution, verbose, freq, display, kwargs...)
end

function get_assumptions(::Type{<:AbstractCGIteration})
	(
		LeastSquaresTerm(:A => (is_linear, is_symmetric, is_positive_definite), :b),
		SquaredL2Term(:λ),
	)
end

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
