"""
    bellman_error(x, Wc, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController, cost::CostFunction)
    bellman_error(x, θ̂, Wc, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController, cost::CostFunction)

Compute the Bellman error
"""
function bellman_error(
    x::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    ω = bellman_regressor(x, Wa, Σ, ϕ, k)
    δ = Wc'*ω + cost(x, k(x,Wa))

    return δ
end

function bellman_error(
    x::Union{Vector{Float64}, Float64}, 
    θ̂::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    ω = bellman_regressor(x, θ̂, Wa, Σ, P, ϕ, k)
    δ = Wc'*ω + cost(x, k(x,Wa))

    return δ
end

function bellman_error(
    X::Vector{Vector{Float64}}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    return [bellman_error(x, Wc, Wa, Σ, ϕ, k, cost) for x in X]
end

function bellman_error(
    X::Vector{Vector{Float64}}, 
    θ̂::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    return [bellman_error(x, θ̂, Wc, Wa, Σ, P, ϕ, k, cost) for x in X]
end

"""
    normalized_bellman_error(x, Wc, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController, cost::CostFunction)
    normalized_bellman_error(x, θ̂, Wc, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController, cost::CostFunction)

Compute normalized Bellman error.
"""
function normalized_bellman_error(
    x::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    ω, ρ = bellman_normalizer(x, Wa, Σ, ϕ, k)
    δ = Wc'*ω + cost(x, k(x,Wa))

    return (ω/ρ^2) * δ
end

function normalized_bellman_error(
    x::Union{Vector{Float64}, Float64}, 
    θ̂::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    ω, ρ = bellman_normalizer(x, θ̂, Wa, Σ, P, ϕ, k)
    δ = Wc'*ω + cost(x, k(x,Wa))

    return (ω/ρ^2) * δ
end

function normalized_bellman_error(
    X::Vector{Vector{Float64}}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    return [normalized_bellman_error(x, Wc, Wa, Σ, ϕ, k, cost) for x in X]
end

function normalized_bellman_error(
    X::Vector{Vector{Float64}}, 
    θ̂::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    return [normalized_bellman_error(x, θ̂, Wc, Wa, Σ, P, ϕ, k, cost) for x in X]
end

"""
    mean_bellman_error(
        X::Vector{Vector{Float64}}, 
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Σ::ControlAffineSystem, 
        ϕ::BasisFunctions, 
        k::MBRLController, 
        cost::CostFunction
    )

    mean_bellman_error(
        X::Vector{Vector{Float64}}, 
        θ̂::Union{Vector{Float64}, Float64}, 
        Wc::Union{Vector{Float64}, Float64}, 
        Wa::Union{Vector{Float64}, Float64}, 
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        ϕ::BasisFunctions, 
        k::MBRLController,
        cost::CostFunction
    )

Compute the average Bellman error across all points in `X`.
"""
function mean_bellman_error(
    X::Vector{Vector{Float64}}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController, 
    cost::CostFunction
    )
    return mean(normalized_bellman_error(X, Wc, Wa, Σ, ϕ, k, cost))
end

function mean_bellman_error(
    X::Vector{Vector{Float64}}, 
    θ̂::Union{Vector{Float64}, Float64}, 
    Wc::Union{Vector{Float64}, Float64}, 
    Wa::Union{Vector{Float64}, Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    ϕ::BasisFunctions, 
    k::MBRLController,
    cost::CostFunction
    )
    return mean(normalized_bellman_error(X, θ̂, Wc, Wa, Σ, P, ϕ, k, cost))
end

"""
    bellman_regressor(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    bellman_regressor(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)

Compute the Bellman error regressor.
"""
function bellman_regressor(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    return jacobian(ϕ, x) * (Σ.f(x) + Σ.g(x)*k(x, Wa))
end

function bellman_regressor(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)
    return jacobian(ϕ, x) * (Σ.f(x) + Σ.g(x)*(k(x, Wa) + P.φ(x)*θ̂))
end

"""
    bellman_normalizer(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    bellman_normalizer(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)

Compute the Bellman error regressor and the regressor's normalizer.
"""
function bellman_normalizer(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    ω = bellman_regressor(x, Wa, Σ, ϕ, k)
    ρ = 1 + ω' * ω

    return ω, ρ
end

function bellman_normalizer(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)
    ω = bellman_regressor(x, θ̂, Wa, Σ, P, ϕ, k)
    ρ = 1 + ω' * ω

    return ω, ρ
end

"""
    bellman_matrix(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    bellman_matrix(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)
"""
function bellman_matrix(x, Wa, Σ::ControlAffineSystem, ϕ::BasisFunctions, k::MBRLController)
    ω = bellman_regressor(x, Wa, Σ, ϕ, k)
    return ω*ω'
end

function bellman_matrix(x, θ̂, Wa, Σ::ControlAffineSystem, P::MatchedParameters, ϕ::BasisFunctions, k::MBRLController)
    ω = bellman_regressor(x, θ̂, Wa, Σ, P, ϕ, k)
    return ω*ω'
end

function normalized_bellman_matrix(
    x::Union{Float64, Vector{Float64}}, 
    Wa::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController
    )
    ω = bellman_regressor(x, Wa, Σ, ϕ, k)
    ρ = 1 + ω' * ω

    return (ω*ω')/ρ^2
end

function normalized_bellman_matrix(
    x::Union{Float64, Vector{Float64}},  
    θ̂::Union{Float64, Vector{Float64}},  
    Wa::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController
    )
    ω = bellman_regressor(x, θ̂, Wa, Σ, P, ϕ, k)
    ρ = 1 + ω' * ω

    return (ω*ω')/ρ^2
end

function normalized_bellman_matrix(
    X::Vector{Vector{Float64}}, 
    Wa::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    ϕ::BasisFunctions, 
    k::MBRLController
    )
    return mean([normalized_bellman_matrix(x, Wa, Σ, ϕ, k) for x in X])
end

function normalized_bellman_matrix(
    X::Vector{Vector{Float64}}, 
    θ̂::Union{Float64, Vector{Float64}},  
    Wa::Vector{Float64}, 
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    ϕ::BasisFunctions, 
    k::MBRLController
    )
    return mean([normalized_bellman_matrix(x, θ̂, Wa, Σ, P, ϕ, k) for x in X])
end