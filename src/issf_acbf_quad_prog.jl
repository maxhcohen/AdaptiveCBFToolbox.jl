"""
    ISSfaCBFQuadProg <: AdaptiveController

Input-to-state-safe adaptive control barrier function quadratic program.

# Fields
- `solve` : function that solves the QP for the control input
- `H` : quadratic weight in QP objective
- `F` : linear weight in QP objective
"""
struct ISSfaCBFQuadProg <: AdaptiveController
    solve::Function
    H::Union{Float64, Matrix{Float64}, Function}
    F::Union{Float64, Vector{Float64}, Function}
end

"Solve `ISSfaCBFQuadProg` at state `x` with parameter estimates `θ̂`."
(k::ISSfaCBFQuadProg)(x, θ̂) = k.solve(x, θ̂)

"""
    ISSfaCBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        CBFs::Vector{ControlBarrierFunction},
        ε::Function
        )
    ISSfaCBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        CBFs::Vector{ControlBarrierFunction}, 
        ε0::Float64
        )
    ISSfaCBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters,
        CBFs::Vector{ControlBarrierFunction}, 
        ε0::Float64, 
        λ::Float64
        )

Construct an `ISSfaCBFQuadProg`.
"""
function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBFs::Vector{ControlBarrierFunction},
    ε::Function
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F = Σ.m == 1 ? 0.0 : zeros(Σ.m)

    # Construct quadratic program
    function solve(x, θ̂)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x)) - (1/ε(CBF.h(x)))*(norm(P.φ(x))^2)*Lgh*Lgh'
            @constraint(model, Lfh + Lgh*(u + P.φ(x)*θ̂) >= -α)
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

    return ISSfaCBFQuadProg(solve, H, F)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBFs::Vector{ControlBarrierFunction}, 
    ε0::Float64
    )
    # Construct damping function
    ε(s) = ε0

    return ISSfaCBFQuadProg(Σ, P, CBFs, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    CBFs::Vector{ControlBarrierFunction}, 
    ε0::Float64, 
    λ::Float64
    )
    # Construct damping function
    ε(s) = ε0*exp(λ*s)

    return ISSfaCBFQuadProg(Σ, P, CBFs, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    CBFs::Vector{ControlBarrierFunction},
    k::AdaptiveController,
    ε::Function
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, θ̂) = -H*k(x, θ̂)

    # Construct quadratic program
    function solve(x, θ̂)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x)) - (1/ε(CBF.h(x)))*(norm(P.φ(x))^2)*Lgh*Lgh'
            @constraint(model, Lfh + Lgh*(u + P.φ(x)*θ̂) >= -α)
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

    return ISSfaCBFQuadProg(solve, H, F)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBFs::Vector{ControlBarrierFunction}, 
    k::AdaptiveController,
    ε0::Float64
    )
    # Construct damping function
    ε(s) = ε0

    return ISSfaCBFQuadProg(Σ, P, CBFs, k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    CBFs::Vector{ControlBarrierFunction}, 
    k::AdaptiveController,
    ε0::Float64, 
    λ::Float64
    )
    # Construct damping function
    ε(s) = ε0*exp(λ*s)

    return ISSfaCBFQuadProg(Σ, P, CBFs, k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    ε::Function
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    ε::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    ε::Float64,
    λ::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε, λ)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    k::AdaptiveController,
    ε::Function
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    k::AdaptiveController,
    ε::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::ControlBarrierFunction,
    k::AdaptiveController,
    ε::Float64,
    λ::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε, λ)
end