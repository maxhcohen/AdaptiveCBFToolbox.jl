"""
    GradientUpdateLaw <: IdentificationUpdateLaw

Update law associated with performing gradient descent on the estimation error.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}`: adaptation gain
- `update_law::Function`: function representing the update law
"""
struct GradientUpdateLaw <: IdentificationUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
end

"""
    (τ::GradientUpdateLaw)(x, ẋ, θ̂)

Compute gradient-based update law at state `x`, measurement `ẋ`, and estimate `θ̂`.
"""
(τ::GradientUpdateLaw)(x, u, θ̂) = τ.update_law(x, u, θ̂)

"""
    GradientUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}}, 
        P::MatchedParameters, 
        Σ::ControlAffineSystem
        )

Construct a gradient-based update law for a control affine system with matched uncertainty.
"""
function GradientUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}}, 
    P::MatchedParameters, 
    Σ::ControlAffineSystem
    )
    function update_law(x, u, θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)
        x̂̇ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*θ̂)
        F = Σ.g(x) * P.φ(x)

        return Γ * F' * (ẋ - x̂̇)
    end

    return GradientUpdateLaw(Γ, update_law)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::GradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate a control affine system with matched parameters using a gradient-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::GradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        u = k(x,θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ(x, u, θ̂)

        return vcat(ẋ, θ̂̇)
    end
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τCLF::CLFUpdateLaw,
        τ::GradientUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate a system with matched parameters using a composite gradient-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τCLF::CLFUpdateLaw,
    τ::GradientUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        u = k(x,θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τCLF(x) + τCLF.Γ * τ(x, u, θ̂)

        return vcat(ẋ, θ̂̇)
    end
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end

