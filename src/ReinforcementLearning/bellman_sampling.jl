"""
    BellmanSampling <: BellmanExtrapolation

Bellman error extrapolation by sampling `N` points from a distribution `D` centered at `x`.'

Note that we shift the distribution to the current state whenever we sample so it suffices
to specify the distribution with zero mean.

# Fields
- `N::Int` : number of samples per timestep
- `D::Distribution` : distribution to sample from
"""
struct BellmanSampling <: BellmanExtrapolation
    N::Int
    D::Distribution
end

"""
    (B::BellmanSampling)(x)

Sample from the distribution using current state `x` as the mean.
"""
(B::BellmanSampling)(x) = mat_to_vecvec(rand(B.D, B.N) .+ x)

"Convert a matrix into a vector of vectors."
function mat_to_vecvec(X::Matrix{Float64})
    return [X[:,i] for i in 1:size(X, 2)]
end