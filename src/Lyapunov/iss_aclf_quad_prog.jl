"""
    ISSaCLFQuadProg <: AdaptiveController

Input-to-state adaptive control Lyapunov function (ISS-aCLF)-based quadratic program (QP).

# Fields
- `solve` : function that solves the QP for the control input
- `H` : quadratic weight in QP objective
- `F` : linear weight in QP objective
"""
struct ISSaCLFQuadProg <: AdaptiveController
    solve::Function
    H::Union{Float64, Matrix{Float64}, Function}
    F::Union{Float64, Vector{Float64}, Function}
end

"Solve `ISSaCLFQuadProg` at state `x` with parameter estimates `θ̂`."
(k::ISSaCLFQuadProg)(x, θ̂) = k.solve(x, θ̂)

"""
    ISSaCLFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction, ε::Float64)

Construct an `ISSaCLFQuadProg`.
"""
function ISSaCLFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction, ε::Float64)
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F = Σ.m == 1 ? 0.0 : zeros(Σ.m)

    # Construct quadratic program
    function solve(x, θ̂)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Compute Lie derivatives
        LfV = CBFToolbox.drift_lie_derivative(CLF, Σ, x)
        LgV = CBFToolbox.control_lie_derivative(CLF, Σ, x)
        γ = CLF.α(x) + (1/ε)*(norm(P.φ(x))^2)*LgV*LgV'

        # Check if we're relaxing the CLF constraint
        if CLF.relax
            @variable(model, δ)
            @constraint(model, LfV + LgV*(u + P.φ(x)*θ̂) <= -γ + δ)
            @objective(model, Min, 0.5*u'*H*u + F'*u + CLF.p*δ^2)
        else
            @constraint(model, LfV + LgV*(u + P.φ(x)*θ̂) <= -γ)
            @objective(model, Min, 0.5*u'*H*u + F'*u)
        end

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return ISSaCLFQuadProg(solve, H, F)
end