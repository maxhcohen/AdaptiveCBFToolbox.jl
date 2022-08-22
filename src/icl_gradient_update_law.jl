"""
    ICLGradientUpdateLaw <: IdentificationUpdateLaw

Perform gradient descent on prediction error using integral concurrent learning.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}`: adaptation gain
- `update_law::Function`: function representing the update law
- `stack::DCLHistoryStack` : history stack of input-output data
- `dt::Float64` : sampling period for adding data to history stack
- `Δt::Float64` : length of integration window
"""
mutable struct ICLGradientUpdateLaw <: IdentificationUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
    stack::ICLHistoryStack
    dt::Float64
    Δt::Float64
end

"""
    (τ::ICLGradientUpdateLaw)(θ̂)

Compute update law with current stack and parameter estimates.
"""
(τ::ICLGradientUpdateLaw)(θ̂) = τ.update_law(θ̂, τ.stack)

"""
    ICLGradientUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        dt::Float64,
        Δt::Float64,
        stack::ICLHistoryStack,
        )

Construct a gradient-based update law using integral concurrent learning.
"""
function ICLGradientUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    dt::Float64,
    Δt::Float64,
    stack::ICLHistoryStack,
    )

    function update_law(θ̂, stack::ICLHistoryStack)
        θ̂̇ = length(θ̂) == 1 ? zeros(stack.M) : [zeros(length(θ̂)) for i in 1:stack.M]
        for i in 1:stack.M
            Δx, f, F, g = stack.Δx[i], stack.f[i], stack.F[i], stack.g[i]
            θ̂̇[i] = F'*(Δx - f - F*θ̂ - g)
        end

        return Γ * sum(θ̂̇)
    end

    return ICLGradientUpdateLaw(Γ, update_law, stack, dt, Δt)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::ICLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τCLF::CLFUpdateLaw,
        τ::ICLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::RACBFQuadProg,
        τ::ICLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::RACBFQuadProg,
        τCLF::CLFUpdateLaw,
        τ::ICLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate a system under an ICL-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        u = k(x, θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ(θ̂)

        return vcat(ẋ, θ̂̇)
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf], p)
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5(), callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τCLF::CLFUpdateLaw,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        u = k(x, θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τCLF(x) + τCLF.Γ * τ(θ̂)

        return vcat(ẋ, θ̂̇)
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf], p)
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5(), callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::RACBFQuadProg,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[Σ.n+1] : X[Σ.n+1:Σ.n+P.p]
        ϑ = X[end]

        # Dynamics
        u = k(x, θ̂, ϑ)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ(θ̂)

        # Update estimation error
        ϑ̇ = estimation_error_dyn(ϑ, τ)

        return vcat(ẋ, θ̂̇, ϑ̇)
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_cbf_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(right_hand_side, vcat(x, θ̂, P.ϑ), [S.t0, S.tf], p)
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5(), callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::RACBFQuadProg,
    τCLF::CLFUpdateLaw,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct right-hand-side function
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1 : Σ.n]
        θ̂cbf = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
        θ̂clf = P.p == 1 ? X[Σ.n + P.p + 1] : X[Σ.n + P.p + 1 : Σ.n + 2*P.p]
        ϑ = X[end]

        # Dynamics
        u = k(x, θ̂cbf, θ̂clf, ϑ)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇cbf = τ(θ̂cbf)
        θ̂̇clf = τCLF(x) + τCLF.Γ * τ(θ̂clf)

        # Update estimation error
        ϑ̇ = estimation_error_dyn(ϑ, τ)

        return vcat(ẋ, θ̂̇cbf, θ̂̇clf, ϑ̇)
    end

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_cbf_clf_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(right_hand_side, vcat(x, θ̂, θ̂, P.ϑ), [S.t0, S.tf], p)
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5(), callback=cb, tstops=ts)

    return trajectory
end

"""
    icl_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function icl_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current time
    t = integrator.t
    tp = t - τ.Δt

    # Construct trajectory function
    x(t) = Σ.n == 1 ? integrator.sol(t, idxs=1) : integrator.sol(t, idxs=1:Σ.n)

    # Integrate regressor
    F = integrate_regressor(x, t, tp, Σ, P)

    # Check if regressor is different enough to add data
    diff = stack_difference(τ.stack, F)
    if diff >= τ.stack.ε
        # If so integrate other objects of interest and update stack
        θ̂(t) = P.p == 1 ? integrator.sol(t, idxs=Σ.n+1) : integrator.sol(t, idxs=(Σ.n+1):(Σ.n+P.p))
        Δx = integrate_state_derivative(x, t, tp)
        f = integrate_drift(x, t, tp, Σ)
        g = integrate_control(x, θ̂, t, tp, Σ, k)
        update_stack!(τ.stack, Δx, f, F, g)
    end

    return integrator
end

"""
    icl_cbf_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function icl_cbf_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current time
    t = integrator.t
    tp = t - τ.Δt

    # Construct trajectory function
    x(t) = Σ.n == 1 ? integrator.sol(t, idxs=1) : integrator.sol(t, idxs=1:Σ.n)

    # Integrate regressor
    F = integrate_regressor(x, t, tp, Σ, P)

    # Check if regressor is different enough to add data
    diff = stack_difference(τ.stack, F)
    if diff >= τ.stack.ε
        # If so integrate other objects of interest and update stack
        θ̂(t) = P.p == 1 ? integrator.sol(t, idxs=Σ.n+1) : integrator.sol(t, idxs=(Σ.n+1):(Σ.n+P.p))
        ϑ(t) = integrator.sol(t, idxs = Σ.n + P.p + 1)
        Δx = integrate_state_derivative(x, t, tp)
        f = integrate_drift(x, t, tp, Σ)
        g = integrate_control(x, θ̂, ϑ, t, tp, Σ, k)
        update_stack!(τ.stack, Δx, f, F, g)
    end

    return integrator
end

"""
    icl_cbf_clf_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function icl_cbf_clf_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current time
    t = integrator.t
    tp = t - τ.Δt

    # Construct trajectory function
    x(t) = Σ.n == 1 ? integrator.sol(t, idxs=1) : integrator.sol(t, idxs=1:Σ.n)

    # Integrate regressor
    F = integrate_regressor(x, t, tp, Σ, P)

    # Check if regressor is different enough to add data
    diff = stack_difference(τ.stack, F)
    if diff >= τ.stack.ε
        # If so integrate other objects of interest and update stack
        θ̂cbf(t) = P.p == 1 ? integrator.sol(t, idxs=Σ.n+1) : integrator.sol(t, idxs=(Σ.n+1):(Σ.n+P.p))
        θ̂clf(t) = P.p == 1 ? integrator.sol(t, idxs=Σ.n+P.p+1) : integrator.sol(t, idxs=(Σ.n+P.p+1):(Σ.n+2*P.p))
        ϑ(t) = integrator.sol(t, idxs = Σ.n + 2*P.p + 1)
        Δx = integrate_state_derivative(x, t, tp)
        f = integrate_drift(x, t, tp, Σ)
        g = integrate_control(x, θ̂cbf, θ̂clf, ϑ, t, tp, Σ, k)
        update_stack!(τ.stack, Δx, f, F, g)
    end

    return integrator
end

"""
    estimation_error_dyn(ϑ::Float64, τ::ICLGradientUpdateLaw)

Compute dynamics of worst-case parameter estimation error.
"""
function estimation_error_dyn(ϑ::Float64, τ::ICLGradientUpdateLaw)
    Λ = stack_sum(τ.stack)

    return -eigmin(τ.Γ * Λ)*ϑ
end