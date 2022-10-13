"""
    StateFollowing <: BasisFunctions

State following basis of kernel functions compatabile with any of those defined in KernelFunctions.jl.

For more details on state following basis functions see
R. Kamalapurkar, J. A. Rosenfeld, and W. E. Dixon, "Efficient model–based reinforcement 
learning for approximate online optimal control," Automatica, 74:247–258, 2016.

# Fields
- `L::Int` : number of basis functions
- `ϕ::Function` : vector of basis functions
"""
struct StateFollowingBasis <: BasisFunctions
    L::Int
    ϕ::Function
end

"""
    StateFollowingBasis(n::Int, k::Kernel, d::Float64)
    StateFollowingBasis(n::Int, k::Kernel, d::Function)
    StateFollowingBasis(n::Int, k::Kernel, centers::Matrix{Float64})
    StateFollowingBasis(n::Int, k::Kernel, centers::Vector{Vector{Float64}})

Construct a state following basis for an `n`-dimensional system with kernel function `k`,
where the center offsets from the current state are scaled by `d`.
"""
function StateFollowingBasis(n::Int, k::Kernel, d::Float64)
    centers = d*simplex_coordinates(n)
    ϕ(x) = [k(x, x + vec(c)) for c in eachcol(centers)]
    L = length(ϕ(zeros(n)))

    return StateFollowingBasis(L, ϕ)
end

function StateFollowingBasis(n::Int, k::Kernel, d::Function)
    centers(x) = d(x)*simplex_coordinates(n)
    ϕ(x) = [k(x, x + vec(c)) for c in eachcol(centers(x))]
    L = length(ϕ(zeros(n)))

    return StateFollowingBasis(L, ϕ)
end

function StateFollowingBasis(n::Int, k::Kernel, centers::Matrix{Float64})
    ϕ(x) = [k(x, x + vec(c)) for c in eachcol(centers)]
    L = length(ϕ(zeros(n)))

    return StateFollowingBasis(L, ϕ)
end

function StateFollowingBasis(n::Int, k::Kernel, centers::Vector{Vector{Float64}})
    ϕ(x) = [k(x, x + c) for c in centers]
    L = length(ϕ(zeros(n)))

    return StateFollowingBasis(L, ϕ)
end

StateFollowingBasis(n::Int, k::Kernel) = StateFollowingBasis(n, k, 1.0)

"""
    simplex_coordinates(n::Int)

Compute the vertices of an n+1 dimensional simplex centered at the origin, based upon the
code at https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html
"""
function simplex_coordinates(n::Int)
    c = zeros(n, n + 1)
    for i in 1:n
        c[i, i] = sqrt(1 - sum(c[1:(i - 1), i] .^ 2))
        for j in (i + 1):(n + 1)
            c[i, j] = (-1 / n - (c[1:(i - 1), i]'c[1:(i - 1), j])) / c[i, i]
        end
    end

    return reverse(c; dims=1)
end