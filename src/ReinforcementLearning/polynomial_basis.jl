"""
    PolynomialBasis <: BasisFunctions

Multi-variate polynomial basis functions.

# Fields
- `n::Int` : number of state variables
- `degrees::Union{Int, Vector{Int}}` : degrees of the polynomial basis
- `L::Int` : number of basis functions
- `ϕ::Function` : function `ϕ(x)` that computes the basis at state `x`.
"""
struct PolynomialBasis <: BasisFunctions
    n::Int
    degrees::Union{Int, Vector{Int}}
    L::Int
    ϕ::Function
end

"""
    (ϕ::PolynomialBasis)(x)

Evaluate basis at state `x`.
"""
(ϕ::PolynomialBasis)(x) = ϕ.ϕ(x)

"""
    PolynomialBasis(n::Int, d::Union{Int, Vector{Int}})

Construct a multi-variate polynomial basis of degree d.
"""
function PolynomialBasis(n::Int, d::Union{Int, Vector{Int}})
    ϕ = generate_polynomial_basis(n, d)
    L = length(ϕ(zeros(n)))

    return PolynomialBasis(n, d, L, ϕ)
end

"Automatically generate polynomial basis functions using `DynamicPolynomials.jl`"
function generate_polynomial_basis(n::Int, d::Union{Int, Vector{Int}})
    @polyvar xₚ[1:n]
    m = monomials(xₚ, d)
    ϕ(x) = evaluate_polynomial_basis(m, xₚ, x)

    return ϕ
end

"Helper function to evaluate a symbolic monomial vector at state `x`."
function evaluate_polynomial_basis(m::MonomialVector, xₚ, x)
    S = subs(m, xₚ => x)
    return [s.α for s in S]
end

"Compute jacobian of basis at state `x`."
function jacobian(ϕ::BasisFunctions, x)
    return ForwardDiff.jacobian(ϕ.ϕ, x)
end

"Get number of basis functions in basis."
Base.length(ϕ::BasisFunctions) = ϕ.L 