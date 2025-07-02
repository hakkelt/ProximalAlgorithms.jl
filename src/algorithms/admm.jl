# M. V. Afonso, J. M. Bioucas-Dias and M. A. T. Figueiredo, "An Augmented
# Lagrangian Approach to the Constrained Optimization Formulation of Imaging
# Inverse Problems," in IEEE Transactions on Image Processing, vol. 20, no. 3,
# pp. 681-695, March 2011, doi: 10.1109/TIP.2010.2076294.

struct ADMMIteration{R,Tx,TA,Tb,TAHb,Tg,TB,TP,Tyz,TCGS,Tps}
	x0::Tx
	A::TA
	b::Tb
	AHb::TAHb
	g::Tg
	B::TB
	P::TP
	P_is_inverse::Bool
	cg_tol::R
	cg_maxiter::Int
	y0::Tyz
	z0::Tyz
	cg_state::TCGS
	penalty_sequence::Tps
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
- `P=nothing`: preconditioner for CG (optional)
- `P_is_inverse=false`: whether `P` is the inverse of the preconditioner
- `eps_abs=0`: absolute tolerance for convergence
- `eps_rel=1`: relative tolerance for convergence
- `cg_tol=1e-6`: CG tolerance
- `cg_maxiter=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
- `penalty_sequence=nothing`: penalty sequence for adaptive rho updating. The following options are available:
  - `FixedPenalty(rho)`: fixed penalty sequence with specified rho values
  - `ResidualBalancingPenalty(rho; mu=10.0, tau=2.0)`: adaptive penalty sequence based on residual balancing [2]
  - `SpectralRadiusBoundPenalty(rho; tau=10.0, eta=100.0)`: adaptive penalty sequence based on spectral radius bounds [3]
  - `SpectralRadiusApproximationPenalty(rho; tau=10.0)`: adaptive penalty sequence based on spectral radius approximation [4]
  Note: rho can be specified either as the `rho` parameter or within the penalty sequence constructor, but not both.

The adaptive penalty parameter schemes are implemented through the penalty sequence types, 
following various strategies from the literature. See the individual penalty sequence types 
for their specific update rules and references.

# References
1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1), 1-122.
2. He, B. S., Yang, H., & Wang, S. L. (2000). Alternating direction method with self-adaptive penalty parameters for monotone variational inequalities. Journal of Optimization Theory and applications, 106(2), 337-356.
3. Lorenz, D. A., & Tran-Dinh, Q. (2019). Non-stationary Douglas–Rachford and alternating direction method of multipliers: Adaptive step-sizes and convergence. Computational Optimization and Applications, 74(1), 67–92. https://doi.org/10.1007/s10589-019-00106-9
4. Mccann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection. IEEE Open Journal of Signal Processing, 5, 402–420. https://doi.org/10.1109/OJSP.2023.3349115
"""
function ADMMIteration(;
	x0,
	A=nothing,
	b=nothing,
	g=(),
	B=nothing,
	rho=nothing,
	P=nothing,
	P_is_inverse=false,
	cg_tol=1e-6,
	cg_maxiter=100,
	y0=nothing,
	z0=nothing,
	penalty_sequence=nothing,
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
	if length(g) == 0
		throw(ArgumentError("g must be a non-empty tuple of proximable functions"))
	end
	if isnothing(B)
		B = ntuple(_ -> LinearAlgebra.I, length(g))  # Default to identity operators
	elseif !(B isa Tuple)
		B = (B,)
	end
	if length(B) != length(g)
		throw(ArgumentError("B and g must have the same length"))
	end
	if isnothing(rho)
		# Only set default rho if penalty_sequence doesn't already have it
		if isnothing(penalty_sequence) || isnothing(penalty_sequence.rho)
			rho = ones(length(g))
		end
	elseif rho isa Number
		rho = fill(rho, length(g))
	elseif !all(isreal, rho)
		throw(ArgumentError("rho must be a tuple of real numbers"))
	end

	# Only process rho if it's not nothing
	if !isnothing(rho)
		R = real(eltype(x0))  # Ensure rho is of the same type as x0
		rho = Tuple(R.(rho)) # Ensure rho is of the same type as x0 and is a tuple
		if length(rho) != length(g)
			throw(ArgumentError("rho must have the same length as g"))
		end
	end
	if !isnothing(y0) && length(y0) != length(g)
		throw(ArgumentError("y0 must have the same length as g"))
	end
	if !isnothing(z0) && length(z0) != length(g)
		throw(ArgumentError("z0 must have the same length as g"))
	end

	AHb = isnothing(A) ? nothing : A' * b
	if !isnothing(AHb) && size(AHb) != size(x0)
		throw(ArgumentError("A'b must have the same size as x0"))
	end

	# Create initial CGState
	cg_state = isnothing(P) ? CGState(x0) : PCGState(x0)

	# Initialize penalty sequence
	R = real(eltype(x0))
	ps = if isnothing(penalty_sequence)
		# No penalty sequence provided, create default ResidualBalancingPenalty
		# Use default rho if none provided
		default_rho = isnothing(rho) ? ones(R, length(g)) : collect(R.(rho))
		ResidualBalancingPenalty(; rho=default_rho)
	else
		# Check for ambiguous rho specification
		if !isnothing(rho) && !isnothing(penalty_sequence.rho)
			throw(
				ArgumentError(
					"Ambiguous rho specification: rho is provided both as a parameter ($rho) and in the penalty sequence ($(penalty_sequence.rho)). Please specify rho in only one location.",
				),
			)
		end

		# Determine final rho: use penalty_sequence.rho if non-empty, otherwise use constructor rho
		final_rho = if isnothing(penalty_sequence.rho)
			isnothing(rho) ? ones(R, length(g)) : collect(R.(rho))
		elseif penalty_sequence.rho isa Number
			fill(R(penalty_sequence.rho), length(g))  # Convert single value to tuple
		else
			collect(R.(penalty_sequence.rho))  # Ensure it's a tuple of the right type
		end

		# Convert all non-integer fields to match the precision of x0 and set rho if needed
		reinstantiate_penalty_sequence(penalty_sequence, R, final_rho)
	end

	return ADMMIteration(
		x0, A, b, AHb, g, B, P, P_is_inverse, R(cg_tol), cg_maxiter, y0, z0, cg_state, ps
	)
end

Base.@kwdef mutable struct ADMMState{R,Tx,NTx,NTBHx,TCGS}
	x::Tx                  # primal variable
	u::NTBHx               # scaled dual variables
	z::NTBHx               # auxiliary variables
	z_old::NTBHx           # previous auxiliary variables
	rᵏ::NTBHx              # temporary variables
	sᵏ::NTx                # temporary variables
	tempˣ::NTx             # temporary variables
	Bx::NTBHx              # temporary variables
	Δx_norm::R             # change in primal variable (for convergence checks)
	rᵏ_norm::Vector{R}     # primal residual norms
	sᵏ_norm::Vector{R}     # dual residual norms
	ϵᵖʳⁱ::Vector{R}        # primal residual thresholds
	ϵᵈᵘᵃ::Vector{R}        # dual residual thresholds
	cg_operator::TCGS      # CG operator for x update
end

function ADMMState(iter::ADMMIteration)
	n_reg = length(iter.g)

	# Initialize variables and CG state
	x = iter.cg_state.x # CGState's x field can be shared with the ADMMState
	if isnothing(iter.y0)
		u = Tuple(
			similar(x, B_ isa UniformScaling ? size(x) : size(B_, 1)) for B_ in iter.B
		)
		for y_ in u
			fill!(y_, 0)
		end
	else
		u = copy.(iter.y0)
	end
	if isnothing(iter.z0)
		z = Tuple(
			similar(x, B_ isa UniformScaling ? size(x) : size(B_, 1)) for B_ in iter.B
		)
		for z_ in z
			fill!(z_, 0)
		end
	else
		z = copy.(iter.z0)
	end
	z_old = similar.(z)

	# Allocate temporary variables
	sᵏ = ntuple(_ -> similar(x), n_reg)
	tempˣ = ntuple(_ -> similar(x), n_reg)
	rᵏ = similar.(u)
	Bx = similar.(u)

	# Initialize residuals
	R = real(eltype(x)) # Ensure residuals are of the same type as x
	Δx_norm = zero(R)
	rᵏ_norm = Vector{R}(undef, n_reg)
	sᵏ_norm = Vector{R}(undef, n_reg)
	ϵᵖʳⁱ = Vector{R}(undef, n_reg)
	ϵᵈᵘᵃ = Vector{R}(undef, n_reg)

	# Build the CG operator for the x update
	# If A is not provided, we assume a simple identity operator
	# cg_operator = A'*A + sum(rho[i] * (B[i]' * B[i]) for i in eachindex(g))
	rho = iter.penalty_sequence.rho
	cg_operator = isnothing(iter.A) ? nothing : iter.A' * iter.A
	for i in eachindex(iter.g)
		new_op = rho[i] * (iter.B[i]' * iter.B[i])
		if isnothing(cg_operator)
			cg_operator = new_op
		else
			cg_operator += new_op
		end
	end

	return ADMMState(; x, u, z, z_old, rᵏ, sᵏ, tempˣ, Bx, Δx_norm, rᵏ_norm, sᵏ_norm, ϵᵖʳⁱ, ϵᵈᵘᵃ, cg_operator)
end

"""
	Base.iterate(iter::ADMMIteration, state::ADMMState=ADMMState(iter))

Performs a single iteration of the Alternating Direction Method of Multipliers (ADMM) algorithm,
for problems of the form

	minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(Bᵢx)

where `A` is a linear operator, `b` is the measurement vector, and `gᵢ` are proximable functions with associated linear operators `Bᵢ`.

ADMM formulation of this problem:

	minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(zᵢ) s.t. Bᵢx = zᵢ

This function advances the ADMM optimization process by sequentially executing four main stages:

- **1. CG-step (x-update):**  
	Updates the main variable `x` by (approximately) solving the linear system  
	```
	xᵢ ← argminₓ (AᴴA + ∑ᵢ ρᵢ BᵢᴴBᵢ) x = Aᴴb + ∑ᵢ ρᵢ Bᵢᴴ(zᵢ - yᵢ)
	```
	typically using a conjugate gradient (CG) method for efficiency.

- **2. Prox-step (z-update):**  
	Updates each auxiliary variable `zᵢ` by applying the proximal operator of the regularizer `gᵢ`:  
	```
	zᵢ ← prox_{gᵢ, 1/ρᵢ}(Bᵢ⋅x + 1/ρᵢ⋅yᵢ)
	```

- **3. Dual-step (y-update):**  
	Updates each dual variable `yᵢ` to enforce consistency:  
	```
	yᵢ ← yᵢ + ρᵢ⋅(Bᵢ⋅xᵢ - zᵢ)
	```

- **4. Residuals computation:**  
	Computes the primal and dual residuals for convergence checks:  
	```
	rᵢ ← Bᵢ⋅x - zᵢ
	ϵᵖʳⁱ ← √p ϵᵃᵇˢ + ϵʳᵉˡ max{norm(Bᵢ⋅x), norm(zᵢ)}
	sᵢ ← ρᵢ⋅Bᵢᴴ⋅(zᵢ - zᵢ₋₁)
	ϵᵈᵘᵃ ← √n ϵᵃᵇˢ + ϵʳᵉˡ norm(yᵢ₊₁)
	```
	where n and p are the length of the primal and dual variables, respectively,
	and `ϵᵃᵇˢ` and `ϵʳᵉˡ` are the absolute and relative tolerances specified in the ADMM iteration.
	In this implementation `ϵᵃᵇˢ` is set to 0, and the tolerance passed to the algorithm is used as `ϵʳᵉˡ`.
	The iterations continue until the stopping criterion is met, which, by default, is:
	```
	norm(rᵢ) ≤ ϵᵖʳⁱ
	norm(sᵢ) ≤ ϵᵈᵘᵃ
	```
Note: This function implements the scaled ADMM algorithm, where the dual variables `yᵢ`
are scaled by the penalty parameter `ρᵢ`: `uᵢ = 1/ρ ⋅ yᵢ`. This simplifies some of the
formulas.

The function returns the updated state, allowing the ADMM algorithm to proceed iteratively until convergence.
"""
function Base.iterate(iter::ADMMIteration, state::ADMMState=ADMMState(iter))
	# Get current rho values
	rho, rho_changed = get_next_rho!(iter.penalty_sequence, iter, state)

	# Swap z and z_old at start of iteration
	state.z, state.z_old = state.z_old, state.z

	# 1. GC-step (x-update): xᵢ ← argminₓ (AᴴA + ∑ᵢ ρᵢ BᵢᴴBᵢ) x = Aᴴb + ∑ᵢ ρᵢ Bᵢᴴ(zᵢ - yᵢ)
	# Compute the right-hand side: b = Aᴴb + ∑ᵢ ρᵢ Bᵢᴴ(zᵢ - yᵢ)
	rhs = state.sᵏ[1] # reusing the first element of sᵏ for the right-hand side
	if !isnothing(iter.AHb)
		copyto!(rhs, iter.AHb)
	else
		fill!(rhs, 0)
	end
	Threads.@threads for i in eachindex(iter.g)
		temp = state.rᵏ[i] # reusing array of previous iteration's rᵏ as a temporary variable
		temp .= state.z_old[i] .- state.u[i]
		mul!(state.tempˣ[i], adjoint(iter.B[i]), temp)
	end
	for i in eachindex(iter.g)
		rhs .+= rho[i] .* state.tempˣ[i]
	end

	# The CG operator is defined as:
	# AᴴA + ∑ᵢ ρᵢ BᵢᴴBᵢ
	# For adaptive penalty sequences, we need to reconstruct the operator with new rho values
	if rho_changed
		new_terms = sum(rho[i] * (iter.B[i]' * iter.B[i]) for i in eachindex(iter.g))
		cg_operator = isnothing(iter.A) ? new_terms : (iter.A' * iter.A) + new_terms
	else
		cg_operator = state.cg_operator
	end
	cg_solver = CG(;
		x0=state.x,
		A=cg_operator,
		b=rhs,
		P=iter.P,
		P_is_inverse=iter.P_is_inverse,
		state=iter.cg_state,
		tol=iter.cg_tol,
		maxit=iter.cg_maxiter,
	)
	x_old = state.tempˣ[1] # reusing the first element of tempˣ for the change in x
	x_old .= state.x # Initialize Δx with the current x value
	state.x, _ = cg_solver() # this actually works in-place, but we set state.x for readability
	state.tempˣ[1] .= state.x .- x_old # Compute the change in x
	state.Δx_norm = norm(state.tempˣ[1]) # Store the norm of the change in x

	Threads.@threads for i in eachindex(iter.g)
		# 2. Prox-step (z-update): zᵢ ← prox_{gᵢ, 1/ρᵢ}(Bᵢ⋅x + 1/ρᵢ⋅yᵢ)
		mul!(state.Bx[i], iter.B[i], state.x)
		temp = state.rᵏ[i] # reusing array of previous iteration's rᵏ as a temporary variable
		temp .= state.Bx[i] .+ state.u[i] # remember that u[i] = 1/ρᵢ * yᵢ, so we can skip the division
		prox!(state.z[i], iter.g[i], temp, 1/rho[i])

		# 3. Dual-step (y-update): yᵢ ← yᵢ + ρᵢ⋅(Bᵢ⋅xᵢ - zᵢ)
		state.rᵏ[i] .= state.Bx[i] .- state.z[i] # Bᵢ * x - zᵢ -> this is the primal residual
		state.u[i] .+= state.rᵏ[i] # again, we can skip the multiplication by ρᵢ

		# compute normalized residuals
		# Raw primal residual: rᵏ = Bᵢ * x - zᵢ₊₁
		# Normalization factor: ϵᵖʳⁱ = max{norm(Bᵢ * x), norm(zᵢ₊₁))
		# Normalized primal residual: rᵏ_norm[i] = norm(rᵏ) / ϵᵖʳⁱ
		state.rᵏ_norm[i] = norm(state.rᵏ[i]) # We already computed the primal residual in the previous step
		state.ϵᵖʳⁱ[i] = max(norm(state.Bx[i]), norm(state.z[i]))

		# Raw dual residual: sᵏ = ρ * Bᵢᴴ * (zᵢ₊₁ - zᵢ)
		# Normalization factor: ϵᵈᵘᵃˡ = ρ * norm(yᵢ₊₁)
		# Normalized dual residual: sᵏ_norm[i] = norm(sᵏ) / ϵᵈᵘᵃ
		Δz = state.Bx[i]  # we don't need Bx anymore, so we can reuse it to store Δz
		Δz .= state.z[i] .- state.z_old[i]
		mul!(state.sᵏ[i], iter.B[i]', Δz) # by definition, we should multiply by ρᵢ, but it is cheaper to multiple the norms later
		state.sᵏ_norm[i] = rho[i] * norm(state.sᵏ[i])
		state.ϵᵈᵘᵃ[i] = rho[i] * norm(state.u[i])
	end

	return state, state
end

function default_stopping_criterion(tol, ::ADMMIteration, state::ADMMState)
	return !any(isnan.(state.x)) && state.Δx_norm < tol && all(state.rᵏ_norm .< tol * state.ϵᵖʳⁱ) && all(state.sᵏ_norm .< tol * state.ϵᵈᵘᵃ)
end
default_solution(::ADMMIteration, state::ADMMState) = state.x
function default_display(it, iteration::ADMMIteration, state::ADMMState)
	if !(iteration.penalty_sequence isa FixedPenalty)
		rho_values = iteration.penalty_sequence.rho
		@printf(
			"%5d | %.3e, %.3e, %.3e\n",
			it,
			maximum(state.rᵏ_norm),
			maximum(state.sᵏ_norm),
			maximum(rho_values)
		)
	else
		@printf("%5d | %.3e, %.3e\n", it, maximum(state.rᵏ_norm), maximum(state.sᵏ_norm))
	end
end

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
- `P=nothing`: preconditioner for CG (optional)
- `P_is_inverse=false`: whether `P` is the inverse of the preconditioner
- `cg_tol=1e-6`: CG tolerance
- `cg_maxiter=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
- `penalty_sequence=nothing`: penalty sequence for adaptive rho updating. Options include:
  - `FixedPenalty(rho)`: fixed penalty sequence with specified rho values
  - `ResidualBalancingPenalty(rho; mu=10.0, tau=2.0)`: adaptive penalty sequence based on residual balancing [2]
  - `SpectralRadiusBoundPenalty(rho; tau=10.0, eta=100.0)`: adaptive penalty sequence based on spectral radius bounds [3]
  - `SpectralRadiusApproximationPenalty(rho; tau=10.0)`: adaptive penalty sequence based on spectral radius approximation [4]
- `maxit=10_000`: maximum number of iterations
- `tol=1e-8`: tolerance for stopping criterion
- `stop=...`: stopping criterion function. Use `normalized_stopping_criterion` for normalized residuals.

The adaptive penalty parameter schemes are implemented through the penalty sequence types, 
following various strategies from the literature. See the individual penalty sequence types 
for their specific update rules and references.

# References
1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1), 1-122.
2. He, B. S., Yang, H., & Wang, S. L. (2000). Alternating direction method with self-adaptive penalty parameters for monotone variational inequalities. Journal of Optimization Theory and applications, 106(2), 337-356.
3. Lorenz, D. A., & Tran-Dinh, Q. (2019). Non-stationary Douglas–Rachford and alternating direction method of multipliers: Adaptive step-sizes and convergence. Computational Optimization and Applications, 74(1), 67–92. https://doi.org/10.1007/s10589-019-00106-9
4. Mccann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection. IEEE Open Journal of Signal Processing, 5, 402–420. https://doi.org/10.1109/OJSP.2023.3349115
"""
function ADMM(;
	maxit=10_000,
	tol=1e-8,
	stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
	solution=default_solution,
	verbose=false,
	freq=100,
	display=default_display,
	kwargs...,
)
	IterativeAlgorithm(
		ADMMIteration; maxit, stop, solution, verbose, freq, display, kwargs...
	)
end

function get_assumptions(::Type{<:ADMMIteration})
	(
		LeastSquaresTerm(:A => (is_linear,), :b),
		RepeatedOperatorTerm(:g => (is_proximable,), :B => (is_linear,)),
	)
end
