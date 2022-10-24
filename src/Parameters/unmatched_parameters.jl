"""
    UnmatchedParameters <: UncertainParameters

Represents parametric uncertainty that is not matched by the control input.

An uncertain system with unmatched uncertain parameters is governed by the dynamics

    ẋ = f(x) + F(x)θ + g(x)u

Note that a system with matched uncertainty can always be represented as above by defining
`F(x)=g(x)φ(x)`

# Fields
- `p::Int`: number of uncertain parameters
- `θ::Union{Float64, Vector{Float64}}`: uncertain parameters
- `F::Function`: function representing the basis or regressor of the parameters
- `𝚯::Vector{Vector{Float64}}` : list of bounds on each parameter
- `θ̃::Union{Float64, Vector{Float64}}` : absolute value of max estimation error for each param
- `ϑ::Float64` : norm of θ̃
"""
struct UnmatchedParameters <: UncertainParameters
    p::Int
    θ::Union{Float64, Vector{Float64}}
    F::Function
    𝚯::Vector{Vector{Float64}}
    θ̃::Union{Float64, Vector{Float64}}
    ϑ::Float64
end

"""
    UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function)
    UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function, 𝚯::Vector{Vector{Float64}})
    UnmatchedParameters(P::MatchedParameters, Σ::ControlAffineSystem)

Construct a UnmatchedParameters object from a vector of parameters and regresssor function.
"""
function UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function)
    𝚯 = [zeros(2) for i in 1:length(θ)]

    return UnmatchedParameters(θ, F, 𝚯)
end

function UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function, 𝚯::Vector{Vector{Float64}})
    # Compute worst-case estimation error for each parameter
    θ̃ = [abs(θ[2] - θ[1]) for θ in 𝚯]
    ϑ = norm(θ̃)

    return UnmatchedParameters(length(θ), θ, F, 𝚯, θ̃, ϑ)
end 

function UnmatchedParameters(P::MatchedParameters, Σ::ControlAffineSystem)
    F(x) = Σ.g(x)*P.φ(x)

    return UnmatchedParameters(P.θ, F, P.𝚯)
end

"""
    regressor_lie_derivative(CLF::ControlLyapunovFunction, P::UnmatchedParameters, x)
    regressor_lie_derivative(CBF::ControlBarrierFunction, P::UnmatchedParameters, x)
    regressor_lie_derivative(HOCBF::SecondOrderCBF, P::UnmatchedParameters, x)

Compute Lie derivative along regressor.
"""
function regressor_lie_derivative(CLF::ControlLyapunovFunction, P::UnmatchedParameters, x)
    return CBFToolbox.gradient(CLF, x) * P.F(x)
end

function regressor_lie_derivative(CBF::ControlBarrierFunction, P::UnmatchedParameters, x)
    return CBFToolbox.gradient(CBF, x) * P.F(x)
end

function regressor_lie_derivative(HOCBF::SecondOrderCBF, P::UnmatchedParameters, x)
    return CBFToolbox.gradient(HOCBF, x) * P.F(x)
end

"Compute closed-loop dynamics under adaptive controller."
function closed_loop_dynamics(x, θ̂, Σ::ControlAffineSystem, P::UnmatchedParameters, k::AdaptiveController)
    return Σ.f(x) + P.F(x)*P.θ + Σ.g(x)*k(x,θ̂)
end

############################################################################################
#                                      Simulations                                         #
############################################################################################

"""
    (S::Simulation)(Σ::ControlAffineSystem, P::UnmatchedParameters, x::Union{Float64, Vector{Float64}})
Run open-loop simulation of control affine system from initial state x.
"""
function (S::Simulation)(Σ::ControlAffineSystem, P::UnmatchedParameters, x::Union{Float64, Vector{Float64}})
    function rhs!(dx, x, θ, t)
        dx .= Σ.f(x) + P.F(x)*θ
        nothing
    end
    problem = ODEProblem(rhs!, x, [S.t0, S.tf], P.θ)
    trajectory = solve(problem)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem, 
    P::UnmatchedParameters, 
    k::CBFToolbox.FeedbackController, 
    x::Union{Float64, Vector{Float64}}
    )
    function rhs!(dx, x, θ, t)
        dx .= Σ.f(x) + P.F(x)*θ + Σ.g(x)*k(x)
        nothing
    end
    problem = ODEProblem(rhs!, x, [S.t0, S.tf], P.θ)
    trajectory = solve(problem)

    return trajectory
end