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
    function update_law(x)
        LgV = CBFToolbox.control_lie_derivative(CLF, Σ, x)
        LFV = LgV*P.φ(x)

        return Γ * LFV'
    end

    return CLFUpdateLaw(Γ, update_law)
end 

"""
    CLFUpdateLaw(
        Γ::Union{Float64, Matrix{Float64}},
        P::UnmatchedParameters,
        Σ::ControlAffineSystem,
        CLF::ControlLyapunovFunction,
        )

Construct a `CLFUpdateLaw` for a system with matched parameters.
"""
function CLFUpdateLaw(
    Γ::Union{Float64, Matrix{Float64}},
    P::UnmatchedParameters,
    CLF::ControlLyapunovFunction,
    )
    function update_law(x)
        LFV = regressor_lie_derivative(CLF, P, x)

        return Γ * LFV'
    end

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
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        ẋ = Σ.f(x) + Σ.g(x)*(k(x,θ̂) + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ(x)

        return vcat(ẋ, θ̂̇)
    end
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::UnmatchedParameters,
        k::AdaptiveController,
        τ::CLFUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}}
        )

Simulate a system with unmatched parameters under a CLF-based controller.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::UnmatchedParameters,
    k::AdaptiveController,
    τ::CLFUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[end] : X[Σ.n+1:end]

        # Dynamics
        ẋ = Σ.f(x) + P.F(x)*P.θ + Σ.g(x)*k(x,θ̂)

        # Update law
        θ̂̇ = τ(x)

        return vcat(ẋ, θ̂̇)
    end
    problem = ODEProblem(right_hand_side, vcat(x, θ̂), [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end