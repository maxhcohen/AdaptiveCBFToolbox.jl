"""
    ACLFSontag <: AdaptiveController

Adaptive control Lyapunov function controller based on Sontag's universal formula.
"""
struct ACLFSontag <: AdaptiveController
    control_law::Function
end

"""
    (k::ACLFSontag)(x, θ̂)

Evaluate adaptive controller with universal formula at state `x` with estimate `θ̂`.
"""
(k::ACLFSontag)(x, θ̂) = k.control_law(x, θ̂)

"""
    ACLFSontag(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction)

Construct an `ACLFSontag` controller.
"""
function ACLFSontag(Σ::ControlAffineSystem, P::MatchedParameters, CLF::ControlLyapunovFunction)
    function control_law(x, θ̂)
        LfV = CBFToolbox.drift_lie_derivative(CLF, Σ, x)
        LgV = CBFToolbox.control_lie_derivative(CLF, Σ, x)
        a = LfV + LgV*P.φ(x)*θ̂
        b = LgV'
        u = CBFToolbox.clf_universal_formula(a, b)

        return u
    end

    return ACLFSontag(control_law)
end