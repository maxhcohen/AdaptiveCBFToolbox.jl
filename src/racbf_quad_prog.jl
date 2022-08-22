"""
    RACBFQuadProg <: AdaptiveController

Robust Adaptive Control Barrier Function (RaCBF)-based quadratic program (QP).

# Fields
- `solve` : function that solves the QP for the control input
- `H` : quadratic weight in QP objective
- `F` : linear weight in QP objective
"""
struct RACBFQuadProg <: AdaptiveController
    solve::Function
    H::Union{Float64, Matrix{Float64}, Function}
    F::Union{Float64, Vector{Float64}, Function}
end

"""
    (k::RACBFQuadProg)(x, θ̂, ϑ)
    (k::RACBFQuadProg)(x, θ̂cbf, θ̂clf, ϑ)

Solve robust adaptive control barrier function (RaCBF) quadratic program.
"""
(k::RACBFQuadProg)(x, θ̂, ϑ) = k.solve(x, θ̂, ϑ)
(k::RACBFQuadProg)(x, θ̂cbf, θ̂clf, ϑ) = k.solve(x, θ̂cbf, θ̂clf, ϑ)

"""
    RACBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        CBFs::Vector{ControlBarrierFunction}
        )

    RACBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        k::AdaptiveController, 
        CBFs::Vector{ControlBarrierFunction}
        )

    RACBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        k::ACLFQuadProg, 
        CBFs::Vector{ControlBarrierFunction}
        )

Construct a robust adaptive control barrier function (RaCBF) quadratic program.
"""
function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBFs::Vector{ControlBarrierFunction}
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F = Σ.m == 1 ? 0.0 : zeros(Σ.m)

    # Construct quadratic program
    function solve(x, θ̂, ϑ)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x))
            @constraint(model, Lfh + Lgh*(u + P.φ(x)*θ̂) >= -α + norm(Lgh*P.φ(x))*ϑ)
        end
        @objective(model, Min, 0.5*u'*H*u + F'*u)

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return RACBFQuadProg(solve, H, F)
end

function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::AdaptiveController, 
    CBFs::Vector{ControlBarrierFunction}
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, θ̂) = -H*k(x, θ̂)

    # Construct quadratic program
    function solve(x, θ̂, ϑ)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x))
            @constraint(model, Lfh + Lgh*(u + P.φ(x)*θ̂) >= -α + norm(Lgh*P.φ(x))*ϑ)
        end
        @objective(model, Min, 0.5*u'*H*u + F(x, θ̂)'*u)

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return RACBFQuadProg(solve, H, F)
end

function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::ACLFQuadProg, 
    CBFs::Vector{ControlBarrierFunction}
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, θ̂) = -H*k(x, θ̂)

    # Construct quadratic program
    function solve(x, θ̂cbf, θ̂clf, ϑ)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x))
            @constraint(model, Lfh + Lgh*(u + P.φ(x)*θ̂cbf) >= -α + norm(Lgh*P.φ(x))*ϑ)
        end
        @objective(model, Min, 0.5*u'*H*u + F(x, θ̂clf)'*u)

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return RACBFQuadProg(solve, H, F)
end

function RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, CBF::ControlBarrierFunction)
    return RACBFQuadProg(Σ, P, [CBF])
end

function RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::AdaptiveController, CBF::ControlBarrierFunction)
    return RACBFQuadProg(Σ, P, k, [CBF])
end