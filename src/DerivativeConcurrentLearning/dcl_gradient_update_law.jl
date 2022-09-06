"""
    DCLGradientUpdateLaw <: IdentificationUpdateLaw

Perform gradient descent on estimation error using derivative concurrent learning.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}`: adaptation gain
- `update_law::Function`: function representing the update law
- `stack::DCLHistoryStack` : history stack of input-output data
- `dt::Float64` : sampling period for adding data to history stack
"""
mutable struct DCLGradientUpdateLaw <: IdentificationUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
    stack::DCLHistoryStack
    dt::Float64
end

"""
    (τ::DCLGradientUpdateLaw)(θ̂)

Evaluate update law with current stack and parameter estimates.
"""
(τ::DCLGradientUpdateLaw)(θ̂) = τ.update_law(θ̂, τ.stack)

"""
    DCLGradientUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        dt::Float64,
        P::MatchedParameters,
        Σ::ControlAffineSystem,
        stack::DCLHistoryStack
        )

    DCLGradientUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        dt::Float64,
        P::MatchedParameters,
        Σ::ControlAffineSystem,
        M::Int,
        )

    DCLGradientUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        dt::Float64,
        P::MatchedParameters,
        Σ::ControlAffineSystem,
        M::Int,
        ε::Float64,
        )

Construct a gradient-based update law using derivative concurrent learning.
"""
function DCLGradientUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    dt::Float64,
    P::MatchedParameters,
    Σ::ControlAffineSystem,
    stack::DCLHistoryStack,
    )

    function update_law(θ̂, stack::DCLHistoryStack)
        θ̂̇ = length(θ̂) == 1 ? zeros(stack.M) : [zeros(length(θ̂)) for i in 1:stack.M]
        for i in 1:stack.M
            x, u, ẋ = stack.x[i], stack.u[i], stack.ẋ[i]
            f, g, φ = Σ.f(x), Σ.g(x), P.φ(x)
            θ̂̇[i] = (g*φ)'*(ẋ - f - g*(u + φ*θ̂))
        end

        return Γ * sum(θ̂̇)
    end

    return DCLGradientUpdateLaw(Γ, update_law, stack, dt)
end

function DCLGradientUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    dt::Float64,
    P::MatchedParameters,
    Σ::ControlAffineSystem,
    M::Int,
    )
    stack = DCLHistoryStack(M, Σ)

    return DCLGradientUpdateLaw(Γ, dt, P, Σ, stack)
end

function DCLGradientUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    dt::Float64,
    P::MatchedParameters,
    Σ::ControlAffineSystem,
    M::Int,
    ε::Float64,
    )
    stack = DCLHistoryStack(M, Σ, ε)

    return DCLGradientUpdateLaw(Γ, dt, P, Σ, stack)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::DCLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τCLF::CLFUpdateLaw,
        τ::DCLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::RACBFQuadProg,
        τ::DCLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::RACBFQuadProg,
        τCLF::CLFUpdateLaw,
        τ::DCLGradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate system with matched uncertainties under a DCL-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::DCLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct RHS function
    rhs!(dX, X, p, t) = rhs_cl_gradient!(dX, X, p, t, Σ, P, k, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = S.t0:τ.dt:S.tf
    affect! = dcl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(rhs!, vcat(x, θ̂), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τCLF::CLFUpdateLaw,
    τ::DCLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # RHS function
    rhs!(dX, X, p, t) = rhs_cl_clf_gradient!(dX, X, p, t, Σ, P, k, τ, τCLF)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = S.t0:τ.dt:S.tf
    affect! = dcl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(rhs!, vcat(x, θ̂), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::RACBFQuadProg,
    τ::DCLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct RHS function
    rhs!(dX, X, p, t) = rhs_cl_cbf_gradient!(dX, X, p, t, Σ, P, k, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = S.t0:τ.dt:S.tf
    affect! = dcl_cbf_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(rhs!, vcat(x, θ̂, P.ϑ), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::RACBFQuadProg,
    τCLF::CLFUpdateLaw,
    τ::DCLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )

    # Construct right-hand-side function
    rhs!(dX, X, p, t) = rhs_cl_cbf_clf_gradient!(dX, X, p, t, Σ, P, k, τ, τCLF)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k)

    # Set up callback for updating history stack
    ts = S.t0:τ.dt:S.tf
    affect! = dcl_cbf_clf_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Set up ODEProblem and solve
    problem = ODEProblem(rhs!, vcat(x, θ̂, θ̂, P.ϑ), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end


"RHS function to pass to ODE solver when using CL with gradient update."
function rhs_cl_gradient!(dX, X, p, t, Σ::ControlAffineSystem, P::MatchedParameters, k::AdaptiveController, τ::DCLGradientUpdateLaw)
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x,θ̂) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x,θ̂) + P.φ(x)*P.θ)
    end

    # Update law
    if P.p == 1
        dX[end] = τ(θ̂)
    else
        dX[Σ.n+1 : end] = τ(θ̂)
    end

    return nothing
end

"RHS function to pass to ODE solver when using CL with gradient update and CLF."
function rhs_cl_clf_gradient!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::AdaptiveController, 
    τ::DCLGradientUpdateLaw, 
    τCLF::CLFUpdateLaw
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x,θ̂) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x,θ̂) + P.φ(x)*P.θ)
    end

    # Update law
    if P.p == 1
        dX[end] = τCLF(x) + τCLF.Γ * τ(θ̂)
    else
        dX[Σ.n+1 : end] = τCLF(x) + τCLF.Γ * τ(θ̂)
    end

    return nothing
end

"RHS function to pass to ODE solver when using CL with CBF."
function rhs_cl_cbf_gradient!(dX, X, p, t, Σ::ControlAffineSystem, P::MatchedParameters, k::RACBFQuadProg, τ::DCLGradientUpdateLaw)
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    θ̂ = P.p == 1 ? X[Σ.n+1] : X[Σ.n+1:Σ.n+P.p]
    ϑ = X[end]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, θ̂, ϑ) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, θ̂, ϑ) + P.φ(x)*P.θ)
    end

    # Update law
    if P.p == 1
        dX[Σ.n+1] = τ(θ̂)
    else
        dX[Σ.n+1:Σ.n+P.p] = τ(θ̂)
    end

    # Estimation error
    dX[end] = estimation_error_dyn(ϑ, τ, Σ, P)

    return nothing
end

"RHS function to pass to ODE solver when using CL with CBF and CLF."
function rhs_cl_cbf_clf_gradient!(dX, X, p, t, Σ::ControlAffineSystem, P::MatchedParameters, k::RACBFQuadProg, τ::DCLGradientUpdateLaw, τCLF::CLFUpdateLaw)
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1 : Σ.n]
    θ̂cbf = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
    θ̂clf = P.p == 1 ? X[Σ.n + P.p + 1] : X[Σ.n + P.p + 1 : Σ.n + 2*P.p]
    ϑ = X[end]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, θ̂cbf, θ̂clf, ϑ) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, θ̂cbf, θ̂clf, ϑ) + P.φ(x)*P.θ)
    end

    # Update law
    if P.p == 1
        dX[Σ.n + 1] = τ(θ̂cbf)
        dX[Σ.n + P.p + 1] = τCLF(x) + τCLF.Γ * τ(θ̂clf)
    else
        dX[Σ.n + 1 : Σ.n + P.p] = τ(θ̂cbf)
        dX[Σ.n + P.p + 1 : Σ.n + 2*P.p] = τCLF(x) + τCLF.Γ * τ(θ̂clf)
    end

    # Estimation error
    dX[end] = estimation_error_dyn(ϑ, τ, Σ, P)

    return nothing
end

"""
    dcl_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function dcl_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current state and parameter estimates
    x = Σ.n == 1 ? integrator.u[1] : integrator.u[1:Σ.n]
    θ̂ = P.p == 1 ? integrator.u[end] : integrator.u[Σ.n+1:end]

    # Compute control input
    u = k(x, θ̂)

    # Update stack
    update_stack!(τ.stack, Σ, P, x, u)

    return integrator
end

"""
    dcl_cbf_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function dcl_cbf_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current state and parameter estimates
    x = Σ.n == 1 ? integrator.u[1] : integrator.u[1:Σ.n]
    θ̂ = P.p == 1 ? integrator.u[Σ.n+1] : integrator.u[Σ.n+1:Σ.n+P.p]
    ϑ = integrator.u[end]

    # Compute control input
    u = k(x, θ̂, ϑ)

    # Update stack
    update_stack!(τ.stack, Σ, P, x, u)

    return integrator
end

"""
    dcl_cbf_clf_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function dcl_cbf_clf_gradient_affect!(integrator)
    # Pull out structs from parameter dictionary
    Σ = integrator.p["Σ"]
    P = integrator.p["P"]
    τ = integrator.p["τ"]
    k = integrator.p["k"]

    # Get current state and parameter estimates
    x = Σ.n == 1 ? integrator.u[1] : integrator.u[1 : Σ.n]
    θ̂cbf = P.p == 1 ? integrator.u[Σ.n + 1] : integrator.u[Σ.n + 1 : Σ.n + P.p]
    θ̂clf = P.p == 1 ? integrator.u[Σ.n + P.p + 1] : integrator.u[Σ.n + P.p + 1 : Σ.n + 2*P.p]
    ϑ = integrator.u[end]

    # Compute control input
    u = k(x, θ̂cbf, θ̂clf, ϑ)

    # Update stack
    update_stack!(τ.stack, Σ, P, x, u)

    return integrator
end

"""
    estimation_error_dyn(ϑ::Float64, τ::DCLGradientUpdateLaw, Σ::ControlAffineSystem, P::MatchedParameters)

Compute dynamics of worst-case parameter estimation error.
"""
function estimation_error_dyn(ϑ::Float64, τ::DCLGradientUpdateLaw, Σ::ControlAffineSystem, P::MatchedParameters)
    Λ = stack_sum(τ.stack, Σ, P)

    return -eigmin(τ.Γ * Λ)*ϑ
end