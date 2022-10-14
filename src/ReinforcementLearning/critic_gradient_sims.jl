function (S::Simulation)(
    Σ::ControlAffineSystem,
    k::Union{MBRLController, CBFToolbox.CBFQuadProg, SafeGuardingController},
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_gradient!(dX, X, p, t, Σ, k, ϕ, τC, τA)
    problem = ODEProblem(rhs!, vcat(x, Wc, Wa), S.tf)
    trajectory = solve(problem)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_gradient!(dX, X, p, t, Σ, P, k, ϕ, τC, τA, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k, "ϕ" => ϕ)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_mbrl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    problem = ODEProblem(rhs!, vcat(x, Wc, Wa, θ̂), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    τ::ICLLeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    θ̂::Union{Float64, Vector{Float64}},
    Γ::Union{Float64, Matrix{Float64}}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_gradient!(dX, X, p, t, Σ, P, k, ϕ, τC, τA, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k, "ϕ" => ϕ)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_mbrl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    # Setup ODE problem
    Γ = P.p == 1 ? Γ : vec(Γ)
    problem = ODEProblem(rhs!, vcat(x, Wc, Wa, θ̂, Γ), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

# function (S::Simulation)(
#     Σ::ControlAffineSystem,
#     k::CBFToolbox.CBFQuadProg, 
#     ϕ::BasisFunctions,
#     τC::CriticGradient,
#     τA::ActorGradient,
#     x::Union{Float64, Vector{Float64}},
#     Wc::Vector{Float64},
#     Wa::Vector{Float64},
#     )
#     # Construct right-hand-side
#     rhs!(dX, X, p, t) = rhs_mbrl_gradient!(dX, X, p, t, Σ, k, ϕ, τC, τA)
#     problem = ODEProblem(rhs!, vcat(x, Wc, Wa), S.tf)
#     trajectory = solve(problem)

#     return trajectory
# end

function rhs_mbrl_gradient!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem, 
    k::Union{MBRLController, CBFToolbox.CBFQuadProg, SafeGuardingController},
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*k(x,Wa)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*k(x,Wa)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC(x, Wc, Wa)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)

    return nothing
end

function rhs_mbrl_gradient!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem,
    P::MatchedParameters, 
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    τ::ICLGradientUpdateLaw,
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]
    θ̂ = P.p == 1 ? X[Σ.n + 2*ϕ.L + 1] : X[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + P.p]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC(x, θ̂, Wc, Wa)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)

    # Update law
    if P.p == 1
        dX[Σ.n + 2*ϕ.L + 1] = τ(θ̂)
    else
        dX[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + P.p] = τ(θ̂)
    end

    return nothing
end

function rhs_mbrl_gradient!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem,
    P::MatchedParameters, 
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticGradient,
    τA::ActorGradient,
    τ::ICLLeastSquaresUpdateLaw,
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]
    θ̂ = P.p == 1 ? X[Σ.n + 2*ϕ.L + 1] : X[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + P.p]
    Γ = P.p == 1 ? X[Σ.n + 2*ϕ.L + P.p + 1] : X[Σ.n + 2*ϕ.L + P.p + 1 : Σ.n + 2*ϕ.L + P.p + P.p^2]
    Γ = P.p == 1 ? Γ : reshape(Γ, P.p, P.p)

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC(x, θ̂, Wc, Wa)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)

    # Update law
    if P.p == 1
        dX[Σ.n + 2*ϕ.L + 1] = τ(θ̂, Γ)
        dX[Σ.n + 2*ϕ.L + P.p + 1] = τ(Γ)
    else
        dX[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + P.p] = τ(θ̂, Γ)
        dX[Σ.n + 2*ϕ.L + P.p + 1 : Σ.n + 2*ϕ.L + P.p + P.p^2] = vec(τ(Γ))
    end

    return nothing
end

# function rhs_mbrl_gradient!(
#     dX, 
#     X, 
#     p, 
#     t, 
#     Σ::ControlAffineSystem, 
#     k::CBFToolbox.CBFQuadProg, 
#     ϕ::BasisFunctions,
#     τC::CriticGradient,
#     τA::ActorGradient,
#     )
#     # Pull out states
#     x = Σ.n == 1 ? X[1] : X[1:Σ.n]
#     Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
#     Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]

#     # Dynamics
#     if Σ.n == 1
#         dX[1] = Σ.f(x) + Σ.g(x)*k(x, Wa)
#     else
#         dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*k(x, Wa)
#     end

#     # Update laws
#     dX[Σ.n + 1 : Σ.n + ϕ.L] = τC(x, Wc, Wa)
#     dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)

#     return nothing
# end


# Integrator callbacks
"""
    icl_mbrl_gradient_affect!(integrator)

Stop integration and update history stack with current data.
"""
function icl_mbrl_gradient_affect!(int)
    # Construct trajectory function
    x(t) = int.p["Σ"].n == 1 ? int.sol(t, idxs=1) : int.sol(t, idxs=1:int.p["Σ"].n)

    # Integrate regressor
    F = integrate_regressor(x, int.t, int.t - int.p["τ"].Δt, int.p["Σ"], int.p["P"])

    # Check if regressor is different enough to add data
    if stack_difference(int.p["τ"].stack, F) >= int.p["τ"].stack.ε
        # If so integrate other objects of interest and update stack
        Wa(t) = int.sol(t, idxs=(int.p["Σ"].n + int.p["ϕ"].L + 1 : int.p["Σ"].n + 2*int.p["ϕ"].L))
        update_stack!(
            int.p["τ"].stack, 
            integrate_state_derivative(x,  int.t, int.t - int.p["τ"].Δt), 
            integrate_drift(x, int.t, int.t - int.p["τ"].Δt, int.p["Σ"]),
            F, 
            integrate_control(x, Wa, int.t, int.t - int.p["τ"].Δt, int.p["Σ"], int.p["k"])
            )
    end

    return int
end