"""
    CostFunction

Struct representing a cost function of the form `Q(x) + u'*R*u`

# Fields:
- `Q::Function` : positive definite function representing the state cost.
- `R::Union{Matrix{Float64}, Float64}` : positive definite matrix representing control cost.
- `R⁻¹::::Union{Matrix{Float64}, Float64}` : inverse of `R`.
"""
struct CostFunction
    Q::Function
    R::Union{Matrix{Float64}, Float64}
    R⁻¹::Union{Matrix{Float64}, Float64}
end

"""
    (cost::CostFunction)(x,u)

Evaluate cost function with at state `x` with control `u`.
"""
(cost::CostFunction)(x,u) = cost.Q(x) + u'*cost.R*u

"""
    CostFunction(Q::Function, R::Matrix{Float64})
    CostFunction(Q::Function, R::Float64)
    CostFunction(Q::Union{Float64, Matrix{Float64}}, R::Union{Float64, Matrix{Float64}})

Cost constructors.
"""
CostFunction(Q::Function, R::Matrix{Float64}) = CostFunction(Q, R, inv(R))
CostFunction(Q::Function, R::Float64) = CostFunction(Q, R, 1/R)
function CostFunction(Q::Union{Float64, Matrix{Float64}}, R::Union{Float64, Matrix{Float64}})
    return CostFunction(x -> x'*Q*x, R)
end