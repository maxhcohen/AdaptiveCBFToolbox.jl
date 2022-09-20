"""
    CriticGradient <: CriticUpdateLaw

Perform gradient descent on the squared Bellman error.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}` : learning rate
- `update_law::Function` : update law
"""
struct CriticGradient <: CriticUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
end

"""
    (τ::CriticGradient)(x, Wc, Wa)
    (τ::CriticGradient)(x, θ̂, Wc, Wa)

Evaluate update law with current weight estimates
"""
(τ::CriticGradient)(x, Wc, Wa) = τ.update_law(x, Wc, Wa)
(τ::CriticGradient)(x, θ̂, Wc, Wa) = τ.update_law(x, θ̂, Wc, Wa)

"""
    CriticGradient(
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanGrid
        )

    CriticGradient(
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanGrid
        )

    CriticGradient(
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanSampling
        )

    CriticGradient(
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanSampling
        )

Construct a `CriticGradient` object.
"""
function CriticGradient(
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanExtrapolation
    )
    update_law(x, Wc, Wa) = critic_gradient_update(x, Wc, Wa, Γ, Σ, ϕ, k, cost, B)

    return CriticGradient(Γ, update_law)
end

function CriticGradient(
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanExtrapolation
    )
    update_law(x, θ̂, Wc, Wa) = critic_gradient_update(x, θ̂, Wc, Wa, Γ, Σ, P, ϕ, k, cost, B)

    return CriticGradient(Γ, update_law)
end

"""
    critic_gradient_update(
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanGrid
        )

    critic_gradient_update(
        θ̂::Union{Vector{Float64}, Float64},
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanGrid
        )

    critic_gradient_update(
        x::Union{Vector{Float64}, Float64},
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanSampling
        )

    critic_gradient_update(
        x::Union{Vector{Float64}, Float64},
        θ̂::Union{Vector{Float64}, Float64},
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Γ::Union{Float64, Matrix{Float64}}, 
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanSampling
        )

Update law for critic when using gradient descent.
"""
function critic_gradient_update(
    x::Union{Vector{Float64}, Float64},
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanGrid
    )
    δ = mean_bellman_error(B.X, Wc, Wa, Σ, ϕ, k, cost)

    return -Γ * δ
end

function critic_gradient_update(
    x::Union{Vector{Float64}, Float64},
    θ̂::Union{Vector{Float64}, Float64},
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanGrid
    )
    δ = mean_bellman_error(B.X, θ̂, Wc, Wa, Σ, P, ϕ, k, cost)

    return -Γ * δ
end

function critic_gradient_update(
    x::Union{Vector{Float64}, Float64},
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanSampling
    )
    δ = mean_bellman_error(B(x), Wc, Wa, Σ, ϕ, k, cost)

    return -Γ * δ
end

function critic_gradient_update(
    x::Union{Vector{Float64}, Float64},
    θ̂::Union{Vector{Float64}, Float64},
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanSampling
    )
    δ = mean_bellman_error(B(x), θ̂, Wc, Wa, Σ, P, ϕ, k, cost)

    return -Γ * δ
end
