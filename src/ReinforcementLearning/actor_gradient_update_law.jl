"""
    ActorGradient <: CriticUpdateLaw

Perform gradient descent on the squared difference between actor and critic weights.

# Fields
- `Γ::Union{Float64, Matrix{Float64}}` : learning rate
- `update_law::Function` : update law
"""
struct ActorGradient <: ActorUpdateLaw
    Γ::Union{Float64, Matrix{Float64}}
    update_law::Function
end

(τ::ActorGradient)(Wa, Wc) = τ.update_law(Wa, Wc)

"""
    ActorGradient(Γ::Union{Float64, Matrix{Float64}})

Construct an `ActorGradient` from a learning rate `Γ`.
"""
function ActorGradient(Γ::Union{Float64, Matrix{Float64}})
    update_law(Wa, Wc) = actor_gradient_update(Wa, Wc, Γ)

    return ActorGradient(Γ, update_law)
end

function actor_gradient_update(Wa, Wc, Γ)
    return -Γ * (Wa - Wc)
end