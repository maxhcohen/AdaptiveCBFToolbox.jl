function (S::Simulation)(
    Σ::ControlAffineSystem,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    Γ::Matrix{Float64}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_rls!(dX, X, p, t, Σ, k, ϕ, τC, τA)
    problem = ODEProblem(rhs!, vcat(x, Wc, Wa, vec(Γ)), S.tf)
    trajectory = solve(problem)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    τ::ICLGradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    Γ::Matrix{Float64},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_rls!(dX, X, p, t, Σ, P, k, ϕ, τC, τA, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k, "ϕ" => ϕ)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_mbrl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    problem = ODEProblem(rhs!, vcat(x, Wc, Wa, vec(Γ), θ̂), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    τ::ICLLeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    Wc::Vector{Float64},
    Wa::Vector{Float64},
    Γ::Matrix{Float64},
    θ̂::Union{Float64, Vector{Float64}},
    Γc::Union{Float64, Matrix{Float64}}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_mbrl_rls!(dX, X, p, t, Σ, P, k, ϕ, τC, τA, τ)

    # Set up parameter dictionary
    p = Dict("Σ" => Σ, "P" => P, "τ" => τ, "k" => k, "ϕ" => ϕ)

    # Set up callback for updating history stack
    ts = (S.t0 + τ.Δt) : τ.dt : S.tf
    affect! = icl_mbrl_gradient_affect!
    cb = PresetTimeCallback(ts, affect!)

    Γc = P.p == 1 ? Γc : vec(Γc)
    problem = ODEProblem(rhs!, vcat(x, Wc, Wa, vec(Γ), θ̂, Γc), S.tf, p)
    trajectory = solve(problem, callback=cb, tstops=ts)

    return trajectory
end

function rhs_mbrl_rls!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem, 
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]
    Γ = X[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2]
    Γ = reshape(Γ, ϕ.L, ϕ.L)

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*k(x,Wa)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*k(x,Wa)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC.update_law(x, Wc, Wa, Γ)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)
    dX[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2] = vec(τC.gamma_update(x, Wa, Γ))

    return nothing
end

function rhs_mbrl_rls!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    τ::ICLGradientUpdateLaw
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]
    Γ = X[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2]
    Γ = reshape(Γ, ϕ.L, ϕ.L)
    θ̂ = P.p == 1 ? X[Σ.n + 2*ϕ.L + ϕ.L^2 + 1] : X[Σ.n + 2*ϕ.L + ϕ.L^2 + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p]

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC.update_law(x, θ̂, Wc, Wa, Γ)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)
    dX[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2] = vec(τC.gamma_update(x, θ̂, Wa, Γ))

    # Update law
    if P.p == 1
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + 1] = τ(θ̂)
    else
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p] = τ(θ̂)
    end

    return nothing
end

function rhs_mbrl_rls!(
    dX, 
    X, 
    p, 
    t, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    k::MBRLController, 
    ϕ::BasisFunctions,
    τC::CriticLeastSquares,
    τA::ActorGradient,
    τ::ICLLeastSquaresUpdateLaw
    )
    # Pull out states
    x = Σ.n == 1 ? X[1] : X[1:Σ.n]
    Wc = X[Σ.n + 1 : Σ.n + ϕ.L]
    Wa = X[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L]
    Γ = X[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2]
    Γ = reshape(Γ, ϕ.L, ϕ.L)
    θ̂ = P.p == 1 ? X[Σ.n + 2*ϕ.L + ϕ.L^2 + 1] : X[Σ.n + 2*ϕ.L + ϕ.L^2 + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p]
    Γc = P.p == 1 ? X[Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + 1] : X[Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + P.p^2]
    Γc = P.p == 1 ? Γc : reshape(Γc, P.p, P.p)

    # Dynamics
    if Σ.n == 1
        dX[1] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    else
        dX[1:Σ.n] = Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*P.θ)
    end

    # Update laws
    dX[Σ.n + 1 : Σ.n + ϕ.L] = τC.update_law(x, θ̂, Wc, Wa, Γ)
    dX[Σ.n + ϕ.L + 1 : Σ.n + 2*ϕ.L] = τA(Wa, Wc)
    dX[Σ.n + 2*ϕ.L + 1 : Σ.n + 2*ϕ.L + ϕ.L^2] = vec(τC.gamma_update(x, θ̂, Wa, Γ))

    # Update law
    if P.p == 1
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + 1] = τ(θ̂, Γc)
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + 1] = τ(Γc)
    else
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p] = τ(θ̂, Γc)
        dX[Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + 1 : Σ.n + 2*ϕ.L + ϕ.L^2 + P.p + P.p^2] = vec(τ(Γc))
    end

    return nothing
end