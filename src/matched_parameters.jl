"""
    MatchedParameters <: UncertainParameters

Represents parametrric uncertainty that is matched by the control input.

An uncertain system with matched uncertain parameters is governed by the dynamics

    ẋ = f(x) + g(x)(u + φ(x)θ)

# Fields
- `p::Int`: number of uncertain parameters
- `θ::Union{Float64, Vector{Float64}}`: uncertain parameters
- `φ::Function`: function representing the basis or regressor of the parameters
- `𝚯::Vector{Vector{Float64}}` : list of bounds on each parameter
- `θ̃::Union{Float64, Vector{Float64}}` : absolute value of max estimation error for each param
- `ϑ::Float64` : norm of θ̃
"""
struct MatchedParameters
    p::Int
    θ::Union{Float64, Vector{Float64}}
    φ::Function
    𝚯::Vector{Vector{Float64}}
    θ̃::Union{Float64, Vector{Float64}}
    ϑ::Float64
end

"""
    MatchedParameters(θ::Union{Float64, Vector{Float64}}, φ::Function)
    MatchedParameters(θ::Union{Float64, Vector{Float64}}, φ::Function, 𝚯::Vector{Vector{Float64}})

Construct a MatchedParameters object from a vector of parameters and regresssor function.
"""
function MatchedParameters(θ::Union{Float64, Vector{Float64}}, φ::Function)
    𝚯 = [zeros(2) for i in 1:length(θ)]

    return MatchedParameters(θ, φ, 𝚯)
end

function MatchedParameters(θ::Union{Float64, Vector{Float64}}, φ::Function, 𝚯::Vector{Vector{Float64}})
    # Compute worst-case estimation error for each parameter
    θ̃ = [abs(θ[2] - θ[1]) for θ in 𝚯]
    ϑ = norm(θ̃)

    return MatchedParameters(length(θ), θ, φ, 𝚯, θ̃, ϑ)
end 

"""
    regressor_lie_derivative(CLF::ControlLyapunovFunction, P::MatchedParameters, x)

Compute Lie derivative along regressor.
"""
function regressor_lie_derivative(CLF::ControlLyapunovFunction, Σ::ControlAffineSystem, P::MatchedParameters, x)
    LgV = CBFToolbox.control_lie_derivative(CLF, Σ, x)
    return LgV * P.φ(x)
end

############################################################################################
#                                      Simulations                                         #
############################################################################################

"""
    (S::Simulation)(Σ::ControlAffineSystem, P::MatchedParameters, x::Union{Float64, Vector{Float64}})

Run open-loop simulation of control affine system from initial state x.
"""
function (S::Simulation)(Σ::ControlAffineSystem, P::MatchedParameters, x::Union{Float64, Vector{Float64}})
    right_hand_side(x, p, t) = Σ.f(x) + Σ.g(x)*P.φ(x)*P.θ
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::CBFToolbox.FeedbackController, 
    x::Union{Float64, Vector{Float64}}
    )
    right_hand_side(x, p, t) = Σ.f(x) + Σ.g(x)*(k(x) + P.φ(x)*P.θ)
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end