"""
    LeastSquaresUpdateLaw <: IdentificationUpdateLaw

Recursive least squares update law for the prediction error.

# Fields
- `β::Float64` : forgetting factor
- `Γ̄::Float64` : bound on covariance matrix
- `update_law::Function` : function defining the parameter update law
- `gamma_update::Function` : function defining the covariance matrix update law
"""
struct LeastSquaresUpdateLaw <: IdentificationUpdateLaw
    β::Float64
    Γ̄::Float64
    update_law::Function
    gamma_update::Function
end

"""
    LeastSquaresUpdateLaw(β::Float64, Γ̄::Float64, P::MatchedParameters, Σ::ControlAffineSystem)
    LeastSquaresUpdateLaw(β::Float64, P::MatchedParameters, Σ::ControlAffineSystem)
    LeastSquaresUpdateLaw(P::MatchedParameters, Σ::ControlAffineSystem)

Construct a recursive least-squares (RLS) update law.

If `β` not provided, then pure RLS. If `β` is provided, then RLS with forgetting factor. If
both `β` and `Γ̄` provided, then RLS with bounded gain forgetting factor.
"""
function LeastSquaresUpdateLaw(β::Float64, Γ̄::Float64, P::MatchedParameters, Σ::ControlAffineSystem)
    function update_law(x, u, θ̂, Γ)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)
        x̂̇ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*θ̂)
        F = Σ.g(x) * P.φ(x)

        return Γ * F' * (ẋ - x̂̇)
    end

    function gamma_update(x, Γ)
        F = Σ.g(x) * P.φ(x)

        return β*(1 - norm(Γ)/Γ̄)*Γ - Γ*F'*F*Γ
    end

    return LeastSquaresUpdateLaw(β, Γ̄, update_law, gamma_update)
end

function LeastSquaresUpdateLaw(β::Float64, P::MatchedParameters, Σ::ControlAffineSystem)
    Γ̄ = Inf

    return LeastSquaresUpdateLaw(β, Γ̄, P, Σ)
end

function LeastSquaresUpdateLaw(P::MatchedParameters, Σ::ControlAffineSystem)
    β = 0.0
    Γ̄ = Inf

    return LeastSquaresUpdateLaw(β, Γ̄, P, Σ)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τ::LeastSquaresUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}},
        Γ::Union{Float64, Matrix{Float64}}
        )

Simulate a control affine system with matched parameters using a gradient-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τ::LeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}},
    Γ::Union{Float64, Matrix{Float64}}
    )
    # Construct right-hand-side
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
        Γ = P.p == 1 ? X[end] : X[Σ.n + P.p + 1 : end]
        Γ = P.p == 1 ? Γ : reshape(Γ, P.p, P.p)

        # Dynamics
        u = k(x,θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τ.update_law(x, u, θ̂, Γ)
        Γ̇ = τ.gamma_update(x, Γ)
        Γ̇ = P.p == 1 ? Γ̇ : vec(Γ̇)

        return vcat(ẋ, θ̂̇, Γ̇)
    end
    Γ = P.p == 1 ? Γ : vec(Γ)
    problem = ODEProblem(right_hand_side, vcat(x, θ̂, Γ), [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem,
        P::MatchedParameters,
        k::AdaptiveController,
        τCLF::CLFUpdateLaw,
        τ::LeastSquaresUpdateLaw,
        x::Union{Float64, Vector{Float64}},
        θ̂::Union{Float64, Vector{Float64}},
        Γ::Union{Float64, Matrix{Float64}}
        )

Simulate a control affine system with matched parameters using a gradient-based update law.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::MatchedParameters,
    k::AdaptiveController,
    τCLF::CLFUpdateLaw,
    τ::LeastSquaresUpdateLaw,
    x::Union{Float64, Vector{Float64}},
    θ̂::Union{Float64, Vector{Float64}},
    Γ::Union{Float64, Matrix{Float64}}
    )
    # Construct right-hand-side
    function right_hand_side(X, p, t)
        # Pull out states
        x = Σ.n == 1 ? X[1] : X[1:Σ.n]
        θ̂ = P.p == 1 ? X[Σ.n + 1] : X[Σ.n + 1 : Σ.n + P.p]
        Γ = P.p == 1 ? X[end] : X[Σ.n + P.p : end]
        Γ = P.p == 1 ? Γ : reshape(Γ, P.p, P.p)

        # Dynamics
        u = k(x,θ̂)
        ẋ = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

        # Update law
        θ̂̇ = τCLF(x) + τCLF.Γ * τ.update_law(x, u, θ̂, Γ)
        Γ̇ = τ.gamma_update(x, Γ)
        Γ̇ = P.p == 1 ? Γ̇ : vec(Γ̇)

        return vcat(ẋ, θ̂̇, Γ̇)
    end
    Γ = P.p == 1 ? Γ : vec(Γ)
    problem = ODEProblem(right_hand_side, vcat(x, θ̂, Γ), [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end