"""
    ICLLeastSquaresUpdateLaw <: IdentificationUpdateLaw

Perform least-squares on prediction error using integral concurrent learning.

# Fields
- `β::Float64` : forgetting factor
- `Γ̄::Float64` : bound on covariance matrix
- `update_law::Function` : function defining the parameter update law
- `gamma_update::Function` : function defining the covariance matrix update law
- `stack::ICLHistoryStack` : history stack of input-output data
- `dt::Float64` : sampling period for adding data to history stack
- `Δt::Float64` : length of integration window
"""
mutable struct ICLLeastSquaresUpdateLaw <: IdentificationUpdateLaw
    β::Float64
    Γ̄::Float64
    update_law::Function
    gamma_update::Function
    stack::ICLHistoryStack
    dt::Float64
    Δt::Float64
end

"""
    (τ::ICLLeastSquaresUpdateLaw)(θ̂, Γ)

Compute parameter update law.
"""
(τ::ICLLeastSquaresUpdateLaw)(θ̂, Γ) = τ.update_law(θ̂, Γ, τ.stack)

"""
    (τ::ICLLeastSquaresUpdateLaw)(Γ)

Update covariance/least-squares matrix.
"""
(τ::ICLLeastSquaresUpdateLaw)(Γ) = τ.gamma_update(Γ, τ.stack)

"""
    ICLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::ICLHistoryStack, 
        dt::Float64, 
        β::Float64, 
        Γ̄::Float64
        )
    ICLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::DCLHistoryStack, 
        dt::Float64 
        )
    ICLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::DCLHistoryStack, 
        dt::Float64,
        β::Float64
        )

Construct a recursive least squares update law using integral concurrent learning.
"""
function ICLLeastSquaresUpdateLaw(
    stack::ICLHistoryStack, 
    dt::Float64,
    Δt::Float64,
    β::Float64, 
    Γ̄::Float64
    )

    # Construct parameter update law
    function update_law(θ̂, Γ, stack::ICLHistoryStack)
        θ̂̇ = length(θ̂) == 1 ? zeros(stack.M) : [zeros(length(θ̂)) for i in 1:stack.M]
        for i in 1:stack.M
            Δx, f, F, g = stack.Δx[i], stack.f[i], stack.F[i], stack.g[i]
            θ̂̇[i] = F'*(Δx - f - F*θ̂ - g)
        end

        return Γ * sum(θ̂̇)
    end

    # Construct covariance update law
    function gamma_update(Γ, stack::ICLHistoryStack)
        Λ = stack_sum(stack)
        return β*(1 - norm(Γ)/Γ̄)*Γ - Γ*Λ*Γ
    end

    return ICLLeastSquaresUpdateLaw(β, Γ̄, update_law, gamma_update, stack, dt, Δt)
end

function ICLLeastSquaresUpdateLaw(
    stack::ICLHistoryStack, 
    dt::Float64,
    Δt::Float64
    )
    β = 0.0
    Γ̄ = Inf

    return ICLLeastSquaresUpdateLaw(stack, dt, Δt, β, Γ̄)
end

function ICLLeastSquaresUpdateLaw(
    stack::ICLHistoryStack, 
    dt::Float64,
    Δt::Float64,
    β::Float64
    )
    Γ̄ = Inf

    return ICLLeastSquaresUpdateLaw(stack, dt, Δt, β, Γ̄)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::UncertainParameters,
        k::AdaptiveController,
        τ::ICLLeastSquaresUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}},
        Γ::Union{Float64, Matrix{Float64}}
        )

Simulate a system using a least-squares based ICL update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::UncertainParameters,
    k::AdaptiveController,
    τ::ICLLeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}},
    Γ::Union{Float64, Matrix{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side!(dX, X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
        Γ = P.p == 1 ? X[end] : X[Σ.n + P.p + 1 : end]
        Γ = P.p == 1 ? Γ : reshape(Γ, P.p, P.p)

        # Dynamics
        if Σ.n == 1
            dX[1] = closed_loop_dynamics(x, θ̂, Σ, P, k)
        else
            dX[1:Σ.n] = closed_loop_dynamics(x, θ̂, Σ, P, k)
        end

        # Update law
        if P.p == 1
            dX[Σ.n + 1] = τ(θ̂, Γ)
            dX[end] = τ(Γ)
        else
            dX[Σ.n + 1 : Σ.n + P.p] = τ(θ̂, Γ)
            dX[Σ.n + P.p + 1 : end] = vec(τ(Γ))
        end

        return nothing
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    Γ = P.p == 1 ? Γ : vec(Γ)
    problem = ODEProblem(right_hand_side!, vcat(x, θ̂, Γ), [S.t0, S.tf], p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end