"""
    DCLGradientUpdateLaw <: IdentificationUpdateLaw

Perform gradient descent on estimation error using derivative concurrent learning.

# Fields
- `β::Float64` : forgetting factor
- `Γ̄::Float64` : bound on covariance matrix
- `update_law::Function` : function defining the parameter update law
- `gamma_update::Function` : function defining the covariance matrix update law
- `stack::DCLHistoryStack` : history stack of input-output data
- `dt::Float64` : sampling period for adding data to history stack
"""
mutable struct DCLLeastSquaresUpdateLaw <: IdentificationUpdateLaw
    β::Float64
    Γ̄::Float64
    update_law::Function
    gamma_update::Function
    stack::DCLHistoryStack
    dt::Float64
end

"""
    (τ::DCLLeastSquaresUpdateLaw)(θ̂, Γ)

Compute parameter update law.
"""
(τ::DCLLeastSquaresUpdateLaw)(θ̂, Γ) = τ.update_law(θ̂, Γ, τ.stack)

"""
    (τ::DCLLeastSquaresUpdateLaw)(Γ)

Update covariance/least-squares matrix.
"""
(τ::DCLLeastSquaresUpdateLaw)(Γ) = τ.gamma_update(Γ, τ.stack)

"""
    DCLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::DCLHistoryStack, 
        dt::Float64, 
        β::Float64, 
        Γ̄::Float64
        )
    DCLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::DCLHistoryStack, 
        dt::Float64 
        )
    DCLLeastSquaresUpdateLaw(
        Σ::ControlAffineSystem,
        P::MatchedParameters, 
        stack::DCLHistoryStack, 
        dt::Float64,
        β::Float64
    )

Construct a recursive least squares update law using derivative concurrent learning.
"""
function DCLLeastSquaresUpdateLaw(
    Σ::ControlAffineSystem,
    P::MatchedParameters, 
    stack::DCLHistoryStack, 
    dt::Float64, 
    β::Float64, 
    Γ̄::Float64
    )

    # Construct parameter update law
    function update_law(θ̂, Γ, stack::DCLHistoryStack)
        θ̂̇ = length(θ̂) == 1 ? zeros(stack.M) : [zeros(length(θ̂)) for i in 1:stack.M]
        for i in 1:stack.M
            x, u, ẋ = stack.x[i], stack.u[i], stack.ẋ[i]
            f, g, φ = Σ.f(x), Σ.g(x), P.φ(x)
            θ̂̇[i] = (g*φ)'*(ẋ - f - g*(u + φ*θ̂))
        end

        return Γ * sum(θ̂̇)
    end

    # Construct covariance update law
    function gamma_update(Γ, stack::DCLHistoryStack)
        Λ = stack_sum(stack, Σ, P)
        return β*(1 - norm(Γ)/Γ̄)*Γ - Γ*Λ*Γ
    end

    return DCLLeastSquaresUpdateLaw(β, Γ̄, update_law, gamma_update, stack, dt)
end

function DCLLeastSquaresUpdateLaw(
    Σ::ControlAffineSystem,
    P::MatchedParameters, 
    stack::DCLHistoryStack, 
    dt::Float64 
    )
    β = 0.0
    Γ̄ = Inf

    return DCLLeastSquaresUpdateLaw(Σ, P, stack, dt, β, Γ̄)
end

function DCLLeastSquaresUpdateLaw(
    Σ::ControlAffineSystem,
    P::MatchedParameters, 
    stack::DCLHistoryStack, 
    dt::Float64,
    β::Float64
    )
    Γ̄ = Inf

    return DCLLeastSquaresUpdateLaw(Σ, P, stack, dt, β, Γ̄)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::DCLLeastSquaresUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}},
        Γ::Union{Float64, Matrix{Float64}}
        )

Simulate a system using a least-squares based DCL update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::DCLLeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}},
    Γ::Union{Float64, Matrix{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
        Γ = P.p == 1 ? X[end] : X[Σ.n + P.p + 1 : end]
        Γ = P.p == 1 ? Γ : reshape(Γ, P.p, P.p)

        # Dynamics
        u = k(x, θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ(θ̂, Γ)
        Γ̇ = τ(Γ)
        Γ̇ = P.p == 1 ? Γ̇ : vec(Γ̇)

        return vcat(ẋ, θ̂̇, Γ̇)
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = S.t0:τ.dt:S.tf
    affect! = dcl_least_squares_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    Γ = P.p == 1 ? Γ : vec(Γ)
    problem = ODEProblem(right_hand_side, vcat(x, θ̂, Γ), [S.t0, S.tf], p)
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5(), callback=cb, tstops=ts)

    return trajectory
end


"""
    dcl_least_squares_affect!(integrator)

Stop integration and update history stack with current data.
"""
function dcl_least_squares_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current state and parameter estimates
    x = Σ.n == 1 ? integrator.u[1] : integrator.u[1:Σ.n]
    θ̂ = P.p == 1 ? integrator.u[Σ.n + 1] : integrator.u[Σ.n + 1 : Σ.n + P.p]

    # Compute control input
    u = k(x, θ̂)

    # Update stack
    update_stack!(τ.stack, Σ, P, x, u)

    return integrator
end