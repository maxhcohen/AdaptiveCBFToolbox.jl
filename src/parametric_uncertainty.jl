"""
    ParametricUncertainty <: Uncertainty

Represents linearly parameterized uncertainty of the form `F(x)θ`m=, 
where `F(x)` is a function representing the basis for the uncertainty,
and `θ` represents the uncertain parameters.

# Fields
- `p::Int`: number of uncertain parameters
- `θ::Union{Float64, Vector{Float64}}`: uncertain parameters
- `F::Function`: function representing the basis or regressor of the parameters
"""
struct ParametricUncertainty
    p::Int
    θ::Union{Float64, Vector{Float64}}
    F::Function
end

function ParametricUncertainty(θ::Union{Float64, Vector{Float64}}, F::Function)
    return ParametricUncertainty(length(θ), θ, F)
end

"Compute Lie derivative of CLF along regressor dynamics ``LFV(x) = ∇V(x) * F(x)``."
function regressor_lie_derivative(CLF::ControlLyapunovFunction, P::ParametricUncertainty, x)
    return CBFToolbox.gradient(CLF, x) * P.F(x)
end

"""
    (S::Simulation)(
        Σ::ControlAffineSystem, 
        P::ParametricUncertainty, 
        x::Union{Float64, Vector{Float64}}
        )

Run open-loop simulation of uncertain control affine system from initial state x.
"""
function (S::Simulation)(
    Σ::ControlAffineSystem, 
    P::ParametricUncertainty, 
    x::Union{Float64, Vector{Float64}}
    )
    # Construct right-hand-side function
    right_hand_side(x, p, t) = Σ.f(x) + P.F(x)*P.θ
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end

function (S::Simulation)(
    Σ::ControlAffineSystem,
    P::ParametricUncertainty,
    k::FeedbackController,
    x::Union{Float64, Vector{Float64}}
    )
    right_hand_side(x, p, t) = Σ.f(x) + P.F(x)*P.θ +  Σ.g(x)*k(x)
    problem = ODEProblem(right_hand_side, x, [S.t0, S.tf])
    trajectory = solve(problem)

    return trajectory
end
