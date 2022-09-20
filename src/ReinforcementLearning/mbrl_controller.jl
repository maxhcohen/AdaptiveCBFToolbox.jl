"""
    MBRLController <: AdaptiveController

Model-based reinforcement learning controller.
"""
struct MBRLController <: AdaptiveController
    control::Function
end

"Compute control at state `x` with estimated weight `W`."
(k::MBRLController)(x, W) = k.control(x, W)

"""
    MBRLController(Σ::ControlAffineSystem, cost::CostFunction, ϕ::BasisFunctions)

Construct an MBRLController based on a system, cost function, and basis.
"""
function MBRLController(Σ::ControlAffineSystem, cost::CostFunction, ϕ::BasisFunctions)
    control(x, W) = -(1/2) * cost.R⁻¹ * Σ.g(x)' * jacobian(ϕ, x)' * W  

    return MBRLController(control)
end