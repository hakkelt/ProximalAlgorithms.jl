# Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
# Optimization" (2008).
#
# Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
# for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2,
# no. 1, pp. 183-202 (2009).
#
# Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for
# Convex Optimization", J. Optim. Theory Appl. (2018) (POGM, without restart).

"""
    FastForwardBackwardIteration(; <keyword-arguments>)

Iterator implementing the accelerated forward-backward splitting algorithm [1, 2].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`FastForwardBackward`](@ref), [`POGM`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `mf=0`: convexity modulus of `f`.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize, defaults to `1/Lf` if `Lf` is set, and `nothing` otherwise.
- `adaptive=true`: makes `gamma` adaptively adjust during the iterations; this is by default `gamma === nothing`. Not supported when `pogm=true`.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `reduce_gamma=0.5`: factor by which to reduce `gamma` in case `adaptive == true`, during backtracking.
- `increase_gamma=1.0`: factor by which to increase `gamma` in case `adaptive == true`, before backtracking.
- `extrapolation_sequence=nothing`: sequence (iterator) of extrapolation coefficients to use for acceleration. Must be left as `nothing` when `pogm=true`.
- `pogm=false`: use the momentum update of the proximal optimized gradient method (POGM) [3] instead of the standard (FISTA-type) one; see [`POGM`](@ref). Requires `mf == 0` and `extrapolation_sequence === nothing`; the `adaptive` stepsize backtracking is ignored in this case.

# References
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).
3. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
Base.@kwdef struct FastForwardBackwardIteration{R,Tx,Tf,Tg,TLf,Tgamma,Textr}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    mf::R = real(eltype(x0))(0)
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
    reduce_gamma::R = real(eltype(x0))(0.5)
    increase_gamma::R = real(eltype(x0))(1.0)
    extrapolation_sequence::Textr = nothing
    pogm::Bool = false
end

Base.IteratorSize(::Type{<:FastForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct FastForwardBackwardState{R,Tx,Textr}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    z_prev::Tx = copy(x)
    extrapolation_sequence::Textr
    theta::R = one(gamma)   # POGM only: extrapolation "theta" parameter
    y_prev::Tx = copy(y)   # POGM only: forward point at the previous iteration
    w_prev::Tx = copy(x)   # POGM only: pre-prox composite point at the previous iteration
    zeta_prev::R = gamma   # POGM only: prox stepsize at the previous iteration
end

function Base.iterate(iter::FastForwardBackwardIteration)
    if iter.pogm
        iter.extrapolation_sequence === nothing ||
            error("pogm=true does not support a custom extrapolation_sequence")
        iszero(iter.mf) || error("pogm=true is currently only implemented for mf == 0")
    end
    x = copy(iter.x0)
    y = similar(x)
    f_x, grad_f_x = value_and_gradient(iter.f, x)
    R = real(eltype(x))
    gamma = R(iter.gamma === nothing ? 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma)
    @. y = x - gamma .* grad_f_x
    extrapolation_sequence = if iter.extrapolation_sequence !== nothing
        Iterators.Stateful(iter.extrapolation_sequence)
    else
        AdaptiveNesterovSequence(iter.mf)
    end
    if iter.pogm
        # First POGM update: theta_0 = 1, so beta = 0 and only the "eta" (OGM)
        # momentum term is active; see [3] in the docstring above.
        theta = R(1)
        theta_new = (1 + sqrt(1 + 4 * theta^2)) / 2
        eta = theta / theta_new
        w = y .+ eta .* (y .- x)
        zeta = gamma * (1 + eta)
        z, g_z = prox(iter.g, w, zeta)
        state = FastForwardBackwardState(
            x = x,
            f_x = f_x,
            grad_f_x = grad_f_x,
            gamma = gamma,
            y = y,
            z = z,
            g_z = g_z,
            res = x - z,
            extrapolation_sequence = extrapolation_sequence,
            theta = theta_new,
            y_prev = copy(y),
            w_prev = w,
            zeta_prev = zeta,
        )
    else
        z, g_z = prox(iter.g, y, gamma)
        state = FastForwardBackwardState(
            x = x,
            f_x = f_x,
            grad_f_x = grad_f_x,
            gamma = gamma,
            y = y,
            z = z,
            g_z = g_z,
            res = x - z,
            extrapolation_sequence = extrapolation_sequence,
        )
    end
    return state, state
end

get_next_extrapolation_coefficient!(
    state::FastForwardBackwardState{R,Tx,<:Iterators.Stateful},
) where {R,Tx} = first(state.extrapolation_sequence)
get_next_extrapolation_coefficient!(
    state::FastForwardBackwardState{R,Tx,<:AdaptiveNesterovSequence},
) where {R,Tx} = next!(state.extrapolation_sequence, state.gamma)

function Base.iterate(
    iter::FastForwardBackwardIteration{R},
    state::FastForwardBackwardState{R,Tx},
) where {R,Tx}
    if iter.pogm
        # Carry the previous prox output forward as the point where the
        # gradient is evaluated: POGM has no separate momentum-on-x step,
        # the momentum is folded directly into the pre-prox point below.
        state.x .= state.z
        state.f_x = value_and_gradient!(state.grad_f_x, iter.f, state.x)
        state.y .= state.x .- state.gamma .* state.grad_f_x

        theta_new = (1 + sqrt(1 + 4 * state.theta^2)) / 2
        beta = (state.theta - 1) / theta_new
        eta = state.theta / theta_new

        # Pre-prox composite point; safe to update w_prev in place since the
        # update is elementwise (see [3] in the docstring above).
        coef = beta * state.gamma / state.zeta_prev
        state.w_prev .=
            state.y .+ beta .* (state.y .- state.y_prev) .+ eta .* (state.y .- state.x) .-
            coef .* (state.x .- state.w_prev)
        zeta = state.gamma * (1 + beta + eta)

        state.y_prev .= state.y
        state.g_z = prox!(state.z, iter.g, state.w_prev, zeta)
        state.res .= state.x .- state.z
        state.zeta_prev = zeta
        state.theta = theta_new

        return state, state
    end

    state.gamma = if iter.adaptive == true
        state.gamma *= iter.increase_gamma
        gamma, state.g_z = backtrack_stepsize!(
            state.gamma,
            iter.f,
            nothing,
            iter.g,
            state.x,
            state.f_x,
            state.grad_f_x,
            state.y,
            state.z,
            state.g_z,
            state.res,
            state.z,
            nothing,
            minimum_gamma = iter.minimum_gamma,
            reduce_gamma = iter.reduce_gamma,
        )
        gamma
    else
        iter.gamma
    end

    beta = get_next_extrapolation_coefficient!(state)
    state.x .= state.z .+ beta .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev

    state.f_x = value_and_gradient!(state.grad_f_x, iter.f, state.x)
    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.res .= state.x .- state.z

    return state, state
end

default_stopping_criterion(
    tol,
    ::FastForwardBackwardIteration,
    state::FastForwardBackwardState,
) = norm(state.res, Inf) / state.gamma <= tol
default_solution(::FastForwardBackwardIteration, state::FastForwardBackwardState) = state.z
default_iteration_summary(it, iter::FastForwardBackwardIteration, state::FastForwardBackwardState) = begin
    if iter.adaptive && !iter.pogm
        ("" => it, "f(x)" => state.f_x, "g(z)" => state.g_z, "γ" => state.gamma, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)
    else
        ("" => it, "f(x)" => state.f_x, "g(z)" => state.g_z, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)
    end
end

"""
    FastForwardBackward(; <keyword-arguments>)

Constructs the accelerated forward-backward splitting algorithm [1, 2].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{FastForwardBackwardIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`FastForwardBackwardIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `FastForwardBackwardIteration` constructor upon call

# References
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).
"""
FastForwardBackward(;
    maxit = 10_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    FastForwardBackwardIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:FastForwardBackwardIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex,))
)

# Aliases

const FastProximalGradientIteration = FastForwardBackwardIteration
const FastProximalGradient = FastForwardBackward

"""
    POGM(; <keyword-arguments>)

Constructs the proximal optimized gradient method (POGM) [3].

This is a shortcut for [`FastForwardBackward`](@ref) with `pogm=true`: it solves the
same class of problems, using the same keyword arguments (except `pogm`, which should
not be passed), but replaces the FISTA-type momentum update with the (asymptotically
faster, by a factor 2) POGM one.

See also: [`FastForwardBackward`](@ref), [`FastForwardBackwardIteration`](@ref).

# References
3. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
POGM(; kwargs...) = FastForwardBackward(; kwargs..., pogm = true)
