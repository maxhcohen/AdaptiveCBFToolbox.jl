"""
    ACLFQuadProg <: AdaptiveController

Adaptive Control Lyapunov Function (aCLF)-based quadratic program (QP) to compute control inputs
for a control affine system.

# Fields
- `solve` : function that solves the QP for the control input
- `H` : quadratic weight in QP objective
- `F` : linear weight in QP objective
"""
struct ACLFQuadProg <: AdaptiveController
    solve::Function
    H::Union{Float64, Matrix{Float64}, Function}
    F::Union{Float64, Vector{Float64}, Function}
end

"Solve `ACLFQuadProg` at state `x` with parameter estimates `θ̂`."
(k::ACLFQuadProg)(x, θ̂) = k.solve(x, θ̂)

"""
    ACLFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction)

Construct an adaptive CLF-based QP for a system with matched parameters.
"""
function ACLFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction)
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
        γ = CLF.α(x)

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

    return ACLFQuadProg(solve, H, F)
end

"""
    ACLFQuadProg(Σ::ControlAffineSystem, P::UnmatchedParameters, CLF::ControlLyapunovFunction)

Construct an adaptive CLF-based QP for a system with unmatched parameters.
"""
function ACLFQuadProg(Σ::ControlAffineSystem, P::UnmatchedParameters, CLF::ControlLyapunovFunction)
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
        γ = CLF.α(x)
        LFV = regressor_lie_derivative(CLF, P, x)

        # Check if we're relaxing the CLF constraint
        if CLF.relax
            @variable(model, δ)
            @constraint(model, LfV + LFV*θ̂ + LgV*u <= -γ + δ)
            @objective(model, Min, 0.5*u'*H*u + F'*u + CLF.p*δ^2)
        else
            @constraint(model, LfV + LFV*θ̂ + LgV*u <= -γ)
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

    return ACLFQuadProg(solve, H, F)
end