"""
    integrate_regressor(x, t, tp, Σ::ControlAffineSystem, P::MatchedParameters)

Integrate regressor along trajectory `x(t)` from time `tp` to time `t`.
"""
function integrate_regressor(x, t, tp, Σ::ControlAffineSystem, P::MatchedParameters)
    prob = IntegralProblem((t,p) -> Σ.g(x(t))*P.φ(x(t)), tp, t)
    F = solve(prob, HCubatureJL(), reltol=1e-3,abstol=1e-3)

    return F 
end

"""
    integrate_state_derivative(x, t, tp)

Integrate state derivative `ẋ` along trajectory `x(t)` from time `tp` to time `t`.

We don't perform any numerical integration here, just compute `Δx(t) = x(t) - x(t-Δt)`.
"""
function integrate_state_derivative(x, t, tp)
    Δx = x(t) - x(tp)

    return Δx
end

"""
    integrate_drift(x, t, tp, Σ::ControlAffineSystem)

Integrate drift dynamics along trajectory `x(t)` from time `tp` to time `t`.
"""
function integrate_drift(x, t, tp, Σ::ControlAffineSystem)
    prob = IntegralProblem((t,p) -> Σ.f(x(t)), tp, t)
    f = solve(prob, HCubatureJL(), reltol=1e-3,abstol=1e-3)

    return f
end

"""
    integrate_control(x, θ̂, t, tp, Σ::ControlAffineSystem, k::AdaptiveController)
    integrate_control(x, θ̂, ϑ, t, tp, Σ::ControlAffineSystem, k::RACBFQuadProg)
    integrate_control(x, θ̂cbf, θ̂clf, ϑ, t, tp, Σ::ControlAffineSystem, k::RACBFQuadProg)

Integrate control directions multiplied by control input along trajectory `x(t)`.
"""
function integrate_control(x, θ̂, t, tp, Σ::ControlAffineSystem, k::AdaptiveController)
    prob = IntegralProblem((t,p) -> Σ.g(x(t))*k(x(t),θ̂(t)), tp, t)
    g = solve(prob, HCubatureJL(), reltol=1e-3,abstol=1e-3)

    return g
end

function integrate_control(x, θ̂, ϑ, t, tp, Σ::ControlAffineSystem, k::RACBFQuadProg)
    prob = IntegralProblem((t,p) -> Σ.g(x(t))*k(x(t),θ̂(t),ϑ(t)), tp, t)
    g = solve(prob, HCubatureJL(), reltol=1e-3,abstol=1e-3)

    return g
end

function integrate_control(x, θ̂cbf, θ̂clf, ϑ, t, tp, Σ::ControlAffineSystem, k::RACBFQuadProg)
    prob = IntegralProblem((t,p) -> Σ.g(x(t))*k(x(t),θ̂cbf(t),θ̂clf(t),ϑ(t)), tp, t)
    g = solve(prob, HCubatureJL(), reltol=1e-3,abstol=1e-3)

    return g
end