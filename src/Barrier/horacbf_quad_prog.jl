"""
    RACBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        HOCBFs::Vector{SecondOrderCBF}
        )

    RACBFQuadProg(
        Σ::ControlAffineSystem, 
        P::MatchedParameters, 
        k::ACLFQuadProg, 
        HOCBFs::Vector{SecondOrderCBF}
        )

Construct a high order robust adaptive control barrier function (HO-RaCBF) quadratic program.
"""
function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::MatchedParameters, 
    HOCBFs::Vector{SecondOrderCBF}
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

        # Set HOCBF constraint and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            α = HOCBF.α2(HOCBF.ψ1(x))
            @constraint(model, Lfψ + Lgψ*(u + P.φ(x)*θ̂) >= -α + norm(Lgψ*P.φ(x))*ϑ)
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
    k::ACLFQuadProg, 
    HOCBFs::Vector{SecondOrderCBF}
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

        # Set HOCBF constraint and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            α = HOCBF.α2(HOCBF.ψ1(x))
            @constraint(model, Lfψ + Lgψ*(u + P.φ(x)*θ̂cbf) >= -α + norm(Lgψ*P.φ(x))*ϑ)
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

function RACBFQuadProg(
    Σ::ControlAffineSystem, 
    P::UnmatchedParameters, 
    HOCBFs::Vector{SecondOrderCBF}
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

        # Set HOCBF constraint and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            LFψ = regressor_lie_derivative(HOCBF, P, x)
            α = HOCBF.α2(HOCBF.ψ1(x))
            @constraint(model, Lfψ + LFψ*θ̂ + Lgψ*u >= -α + norm(LFψ)*ϑ)
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
    P::UnmatchedParameters, 
    k::ACLFQuadProg, 
    HOCBFs::Vector{SecondOrderCBF}
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

        # Set HOCBF constraint and objective
        for HOCBF in HOCBFs
            Lfψ = CBFToolbox.drift_lie_derivative(HOCBF, Σ, x)
            Lgψ = CBFToolbox.control_lie_derivative(HOCBF, Σ, x)
            LFψ = regressor_lie_derivative(HOCBF, P, x)
            α = HOCBF.α2(HOCBF.ψ1(x))
            @constraint(model, Lfψ + LFψ*θ̂cbf + Lgψ*u >= -α + norm(LFψ)*ϑ)
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

function RACBFQuadProg(Σ::ControlAffineSystem, P::UncertainParameters, HOCBF::SecondOrderCBF)
    return RACBFQuadProg(Σ, P, [HOCBF])
end

function RACBFQuadProg(Σ::ControlAffineSystem, P::UncertainParameters, k::ACLFQuadProg, HOCBF::SecondOrderCBF)
    return RACBFQuadProg(Σ, P, k, [HOCBF])
end