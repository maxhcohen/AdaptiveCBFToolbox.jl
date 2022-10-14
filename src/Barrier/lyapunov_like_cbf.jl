"""
    LyapunovLikeCBF <: CBFToolbox.CertificateFunction

Lyapunov-like CBF - essentially a reciprocal CBF that is also positive semi-definite

# Fields
- `B::Function` : function `B(x)` that represents the Lyapunov-like CBF
"""
struct LyapunovLikeCBF <: CBFToolbox.CertificateFunction
    B::Function
end

"Evaluate B at state x"
(B::LyapunovLikeCBF)(x) = B.B(x)

"""
    LyapunovLikeCBF(Σ::ControlAffineSystem, CBF::ControlBarrierFunction)
    LyapunovLikeCBF(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction})

Construct a LyapunovLikeCBF from a CBF.
"""
function LyapunovLikeCBF(Σ::ControlAffineSystem, CBF::ControlBarrierFunction)
    h = CBF.h
    b(x) = 1/h(x)
    B(x) = (b(x) - b(zeros(Σ.n)))^2

    return LyapunovLikeCBF(B)
end

function LyapunovLikeCBF(Σ::ControlAffineSystem, CBFs::Vector{ControlBarrierFunction})
    B(x) = sum([((1/CBF.h(x)) - (1/CBF.h(zeros(Σ.n))))^2 for CBF in CBFs ])

    return LyapunovLikeCBF(B)
end

"Compute the gradient of the LyapunovLikeCBF at x"
function gradient(LCBF::LyapunovLikeCBF, x)
    return length(x) == 1 ? ForwardDiff.derivative(LCBF.B, x) : ForwardDiff.gradient(LCBF.B, x)'
end

"Compute the drift Lie derivative of the LyapunovLikeCBF at x"
function drift_lie_derivative(LCBF::LyapunovLikeCBF, Σ::ControlAffineSystem, x)
    return gradient(LCBF, x) * Σ.f(x)
end

"Compute the control directions Lie derivative of the LyapunovLikeCBF at x"
function control_lie_derivative(LCBF::LyapunovLikeCBF, Σ::ControlAffineSystem, x)
    return gradient(LCBF, x) * Σ.g(x)
end