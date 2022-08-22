function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    HOCBFs::Vector{SecondOrderCBF},
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

        # Set HOCBF constraints and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            α = HOCBF.α2(HOCBF.ψ1(x)) - (1/ε(HOCBF.ψ1(x)))*(norm(P.φ(x))^2)*Lgψ*Lgψ'
            @constraint(model, Lfψ + Lgψ*(u + P.φ(x)*θ̂) >= -α)
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
    HOCBFs::Vector{SecondOrderCBF},
    ε0::Float64
    )
    # Construct damping function
    ε(s) = ε0

    return ISSfaCBFQuadProg(Σ, P, HOCBFs, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    HOCBFs::Vector{SecondOrderCBF},
    ε0::Float64, 
    λ::Float64
    )
    # Construct damping function
    ε(s) = ε0*exp(λ*s)

    return ISSfaCBFQuadProg(Σ, P, HOCBFs, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    HOCBFs::Vector{SecondOrderCBF},
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

        # Set HOCBF constraints and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            α = HOCBF.α2(HOCBF.ψ1(x)) - (1/ε(HOCBF.ψ1(x)))*(norm(P.φ(x))^2)*Lgψ*Lgψ'
            @constraint(model, Lfψ + Lgψ*(u + P.φ(x)*θ̂) >= -α)
        end
        @objective(model, Min, 0.5*u'*H*u + F(x,θ̂)'*u)

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
    HOCBFs::Vector{SecondOrderCBF},
    k::AdaptiveController,
    ε0::Float64
    )
    # Construct damping function
    ε(s) = ε0

    return ISSfaCBFQuadProg(Σ, P, HOCBFs, k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters,
    HOCBFs::Vector{SecondOrderCBF},
    k::AdaptiveController,
    ε0::Float64, 
    λ::Float64
    )
    # Construct damping function
    ε(s) = ε0*exp(λ*s)

    return ISSfaCBFQuadProg(Σ, P, HOCBFs, k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    ε::Function
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    ε::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    ε::Float64,
    λ::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], ε, λ)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    k::AdaptiveController,
    ε::Function
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    k::AdaptiveController,
    ε::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε)
end

function ISSfaCBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    CBF::SecondOrderCBF,
    k::AdaptiveController,
    ε::Float64,
    λ::Float64
    )

    return ISSfaCBFQuadProg(Σ, P, [CBF], k, ε, λ)
end