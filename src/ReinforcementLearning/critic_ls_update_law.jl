"""
    CriticLeastSquares <: CriticUpdateLaw

Minimize Bellman error using recursive least-squares

# Fields
- `β::Float64` : forgetting factor
- `update_law::Function` : function defining the parameter update law
- `gamma_update::Function` : function defining the covariance matrix update law
"""
struct CriticLeastSquares <: CriticUpdateLaw
    β::Float64
    update_law::Function
    gamma_update::Function
end

"""
    CriticLeastSquares(
        β::Float64,
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanExtrapolation
        )

    CriticLeastSquares(
        β::Float64,
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction,
        B::BellmanExtrapolation
        )
"""
function CriticLeastSquares(
    β::Float64,
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanExtrapolation
    )
    update_law(x, Wc, Wa, Γ) = critic_gradient_update(x, Wc, Wa, Γ, Σ, ϕ, k, cost, B)
    gamma_update(x, Wa, Γ) = critic_gamma_update(β, x, Wa, Γ, Σ, ϕ, k, B)

    return CriticLeastSquares(β, update_law, gamma_update)
end

function CriticLeastSquares(
    β::Float64,
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction,
    B::BellmanExtrapolation
    )
    update_law(x, θ̂, Wc, Wa, Γ) = critic_gradient_update(x, θ̂, Wc, Wa, Γ, Σ, P, ϕ, k, cost, B)
    gamma_update(x, θ̂, Wa, Γ) = critic_gamma_update(β, x, Wa, Γ, θ̂, Σ, P, ϕ, k, B)

    return CriticLeastSquares(β, update_law, gamma_update)
end

function critic_gamma_update(
    β::Float64,
    x::Union{Vector{Float64}, Float64},
    Wa::Union{Vector{Float64}, Float64},
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    B::BellmanGrid
    )
    Λ = normalized_bellman_matrix(B.X, Wa, Σ, ϕ, k)

    return β*Γ - Γ*Λ*Γ
end

function critic_gamma_update(
    β::Float64,
    x::Union{Vector{Float64}, Float64},
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    θ̂::Union{Float64, Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    B::BellmanGrid
    )
    Λ = normalized_bellman_matrix(B.X, θ̂, Wa, Σ, P, ϕ, k)

    return β*Γ - Γ*Λ*Γ
end

function critic_gamma_update(
    β::Float64,
    x::Union{Vector{Float64}, Float64},
    Wa::Union{Vector{Float64}, Float64},
    Γ::Union{Float64, Matrix{Float64}}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    B::BellmanSampling
    )
    Λ = normalized_bellman_matrix(B(x), Wa, Σ, ϕ, k)

    return β*Γ - Γ*Λ*Γ
end

function critic_gamma_update(
    β::Float64,
    x::Union{Vector{Float64}, Float64},
    Wa::Union{Vector{Float64}, Float64}, 
    Γ::Union{Float64, Matrix{Float64}}, 
    θ̂::Union{Float64, Vector{Float64}}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    B::BellmanSampling
    )
    Λ = normalized_bellman_matrix(B(x), θ̂, Wa, Σ, P, ϕ, k)

    return β*Γ - Γ*Λ*Γ
end