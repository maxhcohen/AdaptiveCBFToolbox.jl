"Solve CBF-QP with MBRL policy"
(k::CBFToolbox.CBFQuadProg)(x, Wa) = k.solve(x, Wa)

"""
    CBFQuadProg(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction}, k::MBRLController)
    CBFQuadProg(Σ::ControlAffineSystem, CBF::ControlBarrierFunction, k::MBRLController)

Filter MBRL controller through CBF-QP.
"""
function CBFToolbox.CBFQuadProg(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction}, k::MBRLController)
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, Wa) = -H*k(x, Wa)

    # Construct quadratic program
    function solve(x, Wa)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for CBF in CBFs
            Lfh = CBFToolbox.drift_lie_derivative(CBF, Σ, x)
            Lgh = CBFToolbox.control_lie_derivative(CBF, Σ, x)
            α = CBF.α(CBF.h(x))
            @constraint(model, Lfh + Lgh*u >= -α)
        end
        @objective(model, Min, 0.5*u'*H*u + F(x, Wa)'*u)

        # Add control bounds on system - recall these default to unbounded controls
        if ~(Inf in Σ.b)
            @constraint(model, Σ.A * u .<= Σ.b)
        end

        # Solve QP
        optimize!(model)

        return Σ.m == 1 ? value(u) : value.(u)
    end

    return CBFToolbox.CBFQuadProg(solve, H, F)
end

function CBFToolbox.CBFQuadProg(Σ::ControlAffineSystem, CBF::ControlBarrierFunction, k::MBRLController)
    return CBFQuadProg(Σ, [CBF], k)
end

"""
    SafeGuardingController <: CBFToolbox.FeedbackController

Safeguarding controller used to shield an RL policy using a LyapunovLikeCBF.

# Fields
- `control::Function` : function representing the safeguarding controller
- `α::Float64` : gain on the safeguarding component of the controller
"""
struct SafeGuardingController <: CBFToolbox.FeedbackController
    control::Function
    α::Float64
end

"""
    (k::SafeGuardingController)(x)
    (k::SafeGuardingController)(x, Wa)

Evaluate safeguarding controller.
"""
(k::SafeGuardingController)(x) = k.control(x)
(k::SafeGuardingController)(x, Wa) = k.control(x, Wa)

"""
    SafeGuardingController(Σ::ControlAffineSystem, LCBF::LyapunovLikeCBF, α::Float64)
    SafeGuardingController(Σ::ControlAffineSystem, LCBF::LyapunovLikeCBF, k0::FeedbackController, α::Float64)
    SafeGuardingController(Σ::ControlAffineSystem, LCBF::LyapunovLikeCBF, k0::MBRLController, α::Float64)

Constructors for SafeGuardingController.
"""
function SafeGuardingController(Σ::ControlAffineSystem, LCBF::LyapunovLikeCBF, α::Float64)
    control(x) = -α*control_lie_derivative(LCBF, Σ, x)'

    return SafeGuardingController(control, α)
end

function SafeGuardingController(
    Σ::ControlAffineSystem, 
    LCBF::LyapunovLikeCBF, 
    k0::CBFToolbox.FeedbackController, 
    α::Float64
    )
    control(x) = k0(x) - α*control_lie_derivative(LCBF, Σ, x)'

    return SafeGuardingController(control, α)
end

function SafeGuardingController(Σ::ControlAffineSystem, LCBF::LyapunovLikeCBF, k0::MBRLController, α::Float64)
    control(x, Wa) = k0(x, Wa) - α*control_lie_derivative(LCBF, Σ, x)'

    return SafeGuardingController(control, α)
end

"""
    RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, CBFs::Vector{ControlBarrierFunction})
    RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, CBF::ControlBarrierFunction)

Filter an MBRL controller through a RaCBF-QP.
"""
function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::MBRLController, 
    CBFs::Vector{ControlBarrierFunction}
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, Wa) = -H*k(x, Wa)

    # Construct quadratic program
    function solve(x, Wa, θ̂, ϑ)
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
        @objective(model, Min, 0.5*u'*H*u + F(x, Wa)'*u)

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

function RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, CBF::ControlBarrierFunction)
    return RACBFQuadProg(Σ, P, k, [CBF])
end

"""
    RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, HOCBFs::Vector{SecondOrderCBF})
    RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, HOCBF::SecondOrderCBF)

Filter an MBRL controller through a high order RaCBF-QP.
"""
function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    k::MBRLController, 
    HOCBFs::Vector{SecondOrderCBF}
    )
    # Set parameters for objective function
    H = Σ.m == 1 ? 1.0 : Matrix(1.0I, Σ.m, Σ.m)
    F(x, Wa) = -H*k(x, Wa)

    # Construct quadratic program
    function solve(x, Wa, θ̂, ϑ)
        # Build QP and instantiate control decision variable
        model = Model(OSQP.Optimizer)
        set_silent(model)
        Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:Σ.m])

        # Set CBF constraint and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            α = HOCBF.α2(HOCBF.ψ1(x))
            @constraint(model, Lfψ + Lgψ*(u + P.φ(x)*θ̂) >= -α + norm(Lgψ*P.φ(x))*ϑ)
        end
        @objective(model, Min, 0.5*u'*H*u + F(x, Wa)'*u)

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

function RACBFQuadProg(Σ::ControlAffineSystem, P::MatchedParameters, k::MBRLController, HOCBF::SecondOrderCBF)
    return RACBFQuadProg(Σ, P, k, [HOCBF])
end