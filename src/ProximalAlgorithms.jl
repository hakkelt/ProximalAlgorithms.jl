module ProximalAlgorithms

using ADTypes: ADTypes
using DifferentiationInterface: DifferentiationInterface
using ProximalCore
using ProximalCore: Zero, IndZero, convex_conjugate, prox, prox!, is_smooth, is_locally_smooth, is_convex, is_strongly_convex, is_proximable
using OperatorCore: is_linear, is_symmetric, is_positive_definite
using LinearAlgebra
using Base.Iterators
using Printf

import Base: show
import Base: *
import LinearAlgebra: mul!

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

"""
    AutoDifferentiable(f, backend)

Callable struct wrapping function `f` to be auto-differentiated using `backend`.

When called, it evaluates the same as `f`, while its gradient
is implemented using `backend` for automatic differentiation.
The backend can be any of those supported by [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl).
"""
struct AutoDifferentiable{F,B<:ADTypes.AbstractADType}
    f::F
    backend::B
end

(f::AutoDifferentiable)(x) = f.f(x)

"""
    value_and_gradient(f, x)

Return a tuple containing the value of `f` at `x` and the gradient of `f` at `x`.
"""
value_and_gradient

function value_and_gradient(f::AutoDifferentiable, x)
    return DifferentiationInterface.value_and_gradient(f.f, f.backend, x)
end

function value_and_gradient(f::ProximalCore.Zero, x)
    return f(x), zero(x)
end

# various utilities

include("utilities/fb_tools.jl")
include("utilities/iteration_tools.jl")

# acceleration utilities

include("accel/traits.jl")
include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")
include("accel/broyden.jl")
include("accel/noaccel.jl")

# algorithm interface

struct IterativeAlgorithm{IteratorType,H,S,D,K}
    maxit::Int
    stop::H
    solution::S
    verbose::Bool
    freq::Int
    display::D
    kwargs::K
end

"""
    IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...)

Wrapper for an iterator type `T`, adding termination and verbosity options on top of it.

This is a conveniency constructor to allow for "partial" instantiation of an iterator of type `T`.
The resulting "algorithm" object `alg` can be called on a set of keyword arguments, which will be merged
to `kwargs` and passed on to `T` to construct an iterator which will be looped over.
Specifically, if an algorithm is constructed as

    alg = IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...)

then calling it with

    alg(; more_kwargs...)

will internally loop over an iterator constructed as

    T(; alg.kwargs..., more_kwargs...)

# Note
This constructor is not meant to be used directly: instead, algorithm-specific constructors
should be defined on top of it and exposed to the user, that set appropriate default functions
for `stop`, `solution`, `display`.

# Arguments
* `T::Type`: iterator type to use
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
* `kwargs...`: keyword arguments to pass on to `T` when constructing the iterator
"""
IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...) =
    IterativeAlgorithm{T,typeof(stop),typeof(solution),typeof(display),typeof(kwargs)}(
        maxit,
        stop,
        solution,
        verbose,
        freq,
        display,
        kwargs,
    )

"""
    get_iterator(alg::IterativeAlgorithm{IteratorType}) where {IteratorType}

Return an iterator of type `IteratorType` constructed from the algorithm `alg`.
This is a convenience function to allow for easy access to the iterator type
associated with an `IterativeAlgorithm`.

# Example
```julia
julia> using ProximalAlgorithms: CG, get_iterator

julia> alg = CG(maxit=3, tol=1e-8);

julia> iter = get_iterator(alg, A=reshape(collect(1:25)), b=collect(1:5));

julia>  for (k, state) in enumerate(iter)
            if k >= alg.maxit || alg.stop(iter, state)
                alg.verbose && alg.display(k, iter, state)
                return (alg.solution(iter, state), k)
            end
            alg.verbose && mod(k, alg.freq) == 0 && alg.display(k, iter, state)
        end
    1 | 7.416e+00
    2 | 2.742e+00
    3 | 2.300e+01
([0.5581699346405239, 0.31633986928104635, 0.07450980392156867, -0.16732026143790907, -0.4091503267973867], 3)
```
"""
get_iterator(alg::IterativeAlgorithm{IteratorType}; kwargs...) where {IteratorType} =
    IteratorType(; alg.kwargs..., kwargs...)

function (alg::IterativeAlgorithm{IteratorType})(; kwargs...) where {IteratorType}
    iter = get_iterator(alg; kwargs...)
    for (k, state) in enumerate(iter)
        if k >= alg.maxit || alg.stop(iter, state)
            alg.verbose && alg.display(k, iter, state)
            return (alg.solution(iter, state), k)
        end
        alg.verbose && mod(k, alg.freq) == 0 && alg.display(k, iter, state)
    end
end

include("utilities/get_assumptions.jl")

# algorithm implementations

include("algorithms/cg.jl")
include("algorithms/admm.jl")
include("algorithms/forward_backward.jl")
include("algorithms/fast_forward_backward.jl")
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/douglas_rachford.jl")
include("algorithms/drls.jl")
include("algorithms/primal_dual.jl")
include("algorithms/davis_yin.jl")
include("algorithms/li_lin.jl")
include("algorithms/sfista.jl")
include("algorithms/panocplus.jl")

get_algorithms() = [
    SFISTA(),
    FastForwardBackward(),
    ZeroFPR(),
    PANOCplus(),
    DavisYin(),
    VuCondat(),
    DouglasRachford(),
    DRLS(),
    ChambollePock(),
    LiLin(),
    PANOC(),
    ForwardBackward(),
]

end # module
