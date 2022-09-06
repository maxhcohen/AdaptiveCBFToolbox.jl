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
"""
struct UnmatchedParameters
    p::Int
    θ::Union{Float64, Vector{Float64}}
    F::Function
end

"""
    UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function)
    UnmatchedParameters(P::MatchedParameters, Σ::ControlAffineSystem)

Construct a UnmatchedParameters object from a vector of parameters and regresssor function.
"""
function UnmatchedParameters(θ::Union{Float64, Vector{Float64}}, F::Function)
    return UnmatchedParameters(length(θ), θ, F)
end

function UnmatchedParameters(P::MatchedParameters, Σ::ControlAffineSystem)
    F(x) = Σ.g(x)*P.φ(x)

    return UnmatchedParameters(P.p, P.θ, F)
end

"""
    regressor_lie_derivative(CLF::ControlLyapunovFunction, P::UnmatchedParameters, x)

Compute Lie derivative along regressor.
"""
function regressor_lie_derivative(CLF::ControlLyapunovFunction, P::UnmatchedParameters, x)
    return CBFToolbox.gradient(CLF, x) * P.F(x)
end

############################################################################################
#                                      Simulations                                         #
############################################################################################

"""
    (S::Simulation)(Σ::ControlAffineSystem, P::UnmatchedParameters, x::Union{Float64, Vector{Float64}})
Run open-loop simulation of control affine system from initial state x.
"""
function (S::Simulation)(Σ::ControlAffineSystem, P::UnmatchedParameters, x::Union{Float64, Vector{Float64}})
    right_hand_side(x, p, t) = Σ.f(x) + P.F(x)*P.θ
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem, 
    P::UnmatchedParameters, 
    k::CBFToolbox.FeedbackController, 
    x::Union{Float64, Vector{Float64}}
    )
    right_hand_side(x, p, t) = Σ.f(x) + P.F(x)*P.θ + Σ.g(x)*k(x)
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem, OrdinaryDiffEq.Tsit5())

    return trajectory
end