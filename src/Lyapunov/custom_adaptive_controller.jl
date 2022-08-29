"""
    CustomAdaptiveController <: AdaptiveController

User-defined adaptive controller.

The only field is a `control_law`, which is a function `u = k(x, θ̂)` that takes as input a
state `x` and parameter estimate `θ̂` and returns a control input.
"""
struct CustomAdaptiveController <: AdaptiveController
    control_law::Function
end

"Evaluate a `CustomAdaptiveController` at state `x` with estimate `θ̂`."
(k::CustomAdaptiveController)(x, θ̂) = k.control_law(x, θ̂)

