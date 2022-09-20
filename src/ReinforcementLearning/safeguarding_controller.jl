(k::CBFToolbox.CBFQuadProg)(x, Wa) = k.solve(x, Wa)

"""
    CBFQuadProg(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction}, k::MBRLController)
    CBFQuadProg(Σ::ControlAffineSystem, CBF::ControlBarrierFunction, k::MBRLController)

Filter MBRL controller through CBF-QP.
"""
function CBFQuadProg(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction}, k::MBRLController)
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

function CBFQuadProg(Σ::ControlAffineSystem, CBF::ControlBarrierFunction, k::MBRLController)
    return CBFQuadProg(Σ, [CBF], k)
end