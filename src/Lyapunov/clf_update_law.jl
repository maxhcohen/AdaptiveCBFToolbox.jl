"""
    CLFUpdateLaw <: LyapunovUpdateLaw

Update law associated with adaptive CLF controllers.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}`: adaptation gain
- `update_law::Function`: function representing the update law
"""
struct CLFUpdateLaw <: LyapunovUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
end

"""
    (τ::CLFUpdateLaw)(x)

Evaluate update law at state `x`.
"""
(τ::CLFUpdateLaw)(x) = τ.update_law(x)

"""
    CLFUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        P::MatchedParameters,
        Σ::ControlAffineSystem,
        CLF::ControlLyapunovFunction,
        )

Construct a `CLFUpdateLaw` for a system with matched parameters.
"""
function CLFUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    P::MatchedParameters,
    Σ::ControlAffineSystem,
    CLF::ControlLyapunovFunction,
    )

    update_law(x) = Γ * regressor_lie_derivative(CLF, Σ, P, x)'

    return CLFUpdateLaw(Γ, update_law)
end 

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::CLFUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate a system with matched parameters under a CLF-based controller.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::CLFUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    rhs!(dX, X, p, t) = rhs_aclf!(dX, X, p, t, Σ, P, k, τ)
    problem = ODEProblem(rhs!, vcat(x, θ̂), S.tf)
    trajectory = solve(problem)

    return trajectory
end

"RHS function to pass to ODE solver when using adaptive CLF."
function rhs_aclf!(dX, X, p, t, Σ::ControlAffineSystem, P::MatchedParameters, k::AdaptiveController, τ::CLFUpdateLaw)
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
        dX[end] = τ(x)
    else
        dX[Σ.n+1 : end] = τ(x)
    end

    return nothing
end