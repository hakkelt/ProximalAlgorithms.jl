#
# This file contains code that is derived from RegularizedLeastSquares.jl.
# Original source: https://github.com/JuliaImageRecon/RegularizedLeastSquares.jl
#
# RegularizedLeastSquares.jl is licensed under the MIT License:
#
# Copyright (c) 2018: Tobias Knopp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

struct ADMMIteration{R,Tx,TAHb,Tg<:Tuple,TB<:Tuple,TP,TC,TCGS}
    x0::Tx
    AHb::TAHb
    g::Tg
    B::TB
    rho::Vector{R}
    P::TP
    P_is_inverse::Bool
    cg_operator::TC
    cg_tol::R
    cg_maxiter::Int
    y0::Vector{Tx}
    z0::Vector{Tx}
    cg_state::TCGS
end

"""
    ADMMIteration(; <keyword-arguments>)

Iterator implementing the Alternating Direction Method of Multipliers (ADMM) algorithm.

This iterator solves optimization problems of the form

    minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(Bᵢx)

where:
- `A` is a linear operator
- `b` is the measurement vector
- `gᵢ` are proximable functions with associated linear operators `Bᵢ`

See also: [`ADMM`](@ref).
    
# Arguments
- `x0`: initial point
- `A=nothing`: forward operator. If `A` is not provided, ½‖Ax - b‖²₂ is not computed, and the algorithm will only minimize the regularization terms.
- `b=nothing`: measurement vector. If `A` is provided, `b` must also be provided.
- `g=()`: tuple of proximable regularization functions
- `B=()`: tuple of regularization operators
- `rho=ones(length(g))`: vector of augmented Lagrangian parameters (one per regularizer)
- `P=nothing`: preconditioner for CG (optional)
- `cg_tol=1e-6`: CG tolerance
- `cg_maxiter=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
"""
function ADMMIteration(;
        x0,
        A = nothing,
        b = nothing,
        g = (),
        B = nothing,
        rho = nothing,
        P = nothing,
        P_is_inverse = false,
        cg_tol = 1e-6,
        cg_maxiter = 100,
        y0 = nothing,
        z0 = nothing,
    )
    if isnothing(A) && !isnothing(b)
        throw(ArgumentError("A must be provided if b is given"))
    end
    if !isnothing(A) && isnothing(b)
        throw(ArgumentError("b must be provided if A is given"))
    end
    if !(g isa Tuple)
        g = (g,)
    end
    if isnothing(B)
        B = tuple(fill(LinearAlgebra.I, length(g))...)  # Default to identity operators
    elseif !(B isa Tuple)
        B = (B,)
    end
    if length(B) != length(g)
        throw(ArgumentError("B and g must have the same length"))
    end
    if isnothing(rho)
        rho = ones(real(eltype(x0)), length(g))
    elseif rho isa Number
        rho = fill(rho, length(g))
    elseif !(rho isa Vector) || !all(isreal, rho)
        throw(ArgumentError("rho must be a vector of real numbers"))
    end
    if length(rho) != length(g)
        throw(ArgumentError("rho must have the same length as g"))
    end
    # Build the CG operator for the x update
    # If A is not provided, we assume a simple identity operator
    # cg_operator = A'*A + sum(rho[i] * (B[i]' * B[i]) for i in eachindex(g))
    cg_operator = isnothing(A) ? nothing : A' * A
    for i in eachindex(g)
        new_op = rho[i] * (B[i]' * B[i])
        if isnothing(cg_operator)
            cg_operator = new_op
        else
            cg_operator += new_op
        end
    end
    if isnothing(y0)
        y0 = [zero(x0) for _ in 1:length(g)]
    elseif length(y0) != length(g)
        throw(ArgumentError("y0 must have the same length as g"))
    end
    if isnothing(z0)
        z0 = [zero(x0) for _ in 1:length(g)]
    elseif length(z0) != length(g)
        throw(ArgumentError("z0 must have the same length as g"))
    end
    AHb = isnothing(A) ? nothing : (A' * b)
    if size(AHb) != size(x0)
        throw(ArgumentError("A'b must have the same size as x0"))
    end

    # Create initial CGState
    cg_state = isnothing(P) ? CGState(x0) : PCGState(x0)

    return ADMMIteration{eltype(rho),typeof(x0),typeof(AHb),typeof(g),typeof(B),
                        typeof(P),typeof(cg_operator),typeof(cg_state)}(
        x0, AHb, g, B, rho, P, P_is_inverse, cg_operator, cg_tol, cg_maxiter, y0, z0, cg_state
    )
end

Base.@kwdef mutable struct ADMMState{R,Tx}
    x::Tx                # primal variable
    y::Vector{Tx}       # scaled dual variables
    z::Vector{Tx}       # auxiliary variables
    u::Tx               # temporary variable for x update
    v::Tx               # temporary variable for normal equations
    w::Vector{Tx}       # temporary variables for residuals
    res_primal::Vector{R} # primal residual norms
    res_dual::Vector{R}   # dual residual norms
end

function ADMMState(iter::ADMMIteration)
    n_reg = length(iter.g)
    
    # Initialize variables and CG state
    x = iter.cg_state.x  # Start with initial guess
    y = isnothing(iter.y0) ? [zero(x) for _ in 1:n_reg] : copy.(iter.y0)
    z = isnothing(iter.z0) ? [zero(x) for _ in 1:n_reg] : copy.(iter.z0)
    
    # Allocate temporary variables
    u = similar(x)
    v = similar(x)
    w = [similar(x) for _ in 1:n_reg]
    
    # Initialize residuals
    res_primal = zeros(real(eltype(x)), n_reg)
    res_dual = zeros(real(eltype(x)), n_reg)
    
    return ADMMState(;x, y, z, u, v, w, res_primal, res_dual)
end

function Base.iterate(iter::ADMMIteration, state::ADMMState = ADMMState(iter))
    # Store old z for computing dual residuals
    z_old = copy.(state.z)
    
    # Update x using CG
    if !isnothing(iter.AHb)
        copyto!(state.v, iter.AHb)  # v = A'b
    else
        fill!(state.v, 0)  # no least squares term
    end

    # Add contributions from regularizers
    fill!(state.u, 0)
    for i in eachindex(iter.g)
        mul!(state.w[i], adjoint(iter.B[i]), state.z[i] .- state.y[i])
        state.u .+= iter.rho[i] .* state.w[i]
    end
    state.v .+= state.u
    
    # Create new CGIteration but reuse state
    cg = CG(
        x0 = state.x,
        A = iter.cg_operator,
        b = state.v,
        P = iter.P,
        P_is_inverse = iter.P_is_inverse,
        state = iter.cg_state,
        tol = iter.cg_tol,
        maxit = iter.cg_maxiter,
    )
    cg() # this works in-place, updating state.x == iter.cg_state.x
    
    # z-updates
    for i in eachindex(iter.g)
        mul!(state.w[i], iter.B[i], state.x)
        state.w[i] .+= state.y[i]
        prox!(state.z[i], iter.g[i], state.w[i], 1/iter.rho[i])
    end
    
    # Update dual variables and compute residuals
    for i in eachindex(iter.g)
        mul!(state.w[i], iter.B[i], state.x)
        state.w[i] .-= state.z[i]
        state.y[i] .+= state.w[i]
        
        state.res_primal[i] = norm(state.w[i])
        state.res_dual[i] = iter.rho[i] * norm(state.z[i] - z_old[i])
    end
    
    return state, state
end

default_stopping_criterion(tol, ::ADMMIteration, state::ADMMState) =
    all(r -> r <= tol, state.res_primal) && all(r -> r <= tol, state.res_dual)
default_solution(::ADMMIteration, state::ADMMState) = state.x
default_display(it, ::ADMMIteration, state::ADMMState) =
    @printf("%5d | Primal: %.3e, Dual: %.3e\n", it,
            maximum(state.res_primal), maximum(state.res_dual))

"""
    ADMM(; <keyword-arguments>)

Create an instance of the ADMM algorithm.

This algorithm solves optimization problems of the form

    minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(Bᵢx)

where `A` is a linear operator, `b` is the measurement vector, and `gᵢ` are proximable functions with associated linear operators `Bᵢ`.

The returned object has type `IterativeAlgorithm{ADMMIteration}`,
and can called be with the problem's arguments to trigger its solution.

# Arguments
- `x0`: initial point
- `A=nothing`: forward operator. If `A` is not provided, ½‖Ax - b‖²₂ is not computed, and the algorithm will only minimize the regularization terms. 
- `b=nothing`: measurement vector. If `A` is provided, `b` must also be provided.
- `g=()`: tuple of proximable regularization functions
- `B=()`: tuple of regularization operators
- `rho=ones(length(g))`: vector of augmented Lagrangian parameters (one per regularizer)
- `P=nothing`: preconditioner for CG (optional)
- `cg_tol=1e-6`: CG tolerance
- `cg_maxiter=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
"""
ADMM(;
    maxit = 10_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    ADMMIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    display,
    kwargs...,
)

get_assumptions(::Type{<:ADMMIteration}) = (
    LeastSquaresTerm(:A => (is_linear,), :b),
    RepeatedOperatorTerm(:g => (is_proximable,), :B => (is_linear,)),
)