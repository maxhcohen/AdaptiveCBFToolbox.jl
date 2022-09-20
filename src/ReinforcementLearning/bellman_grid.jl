"""
    BellmanGrid <: BellmanExtrapolation

Bellman error extrapolation on a grid.

# Fields
- `N::Int` : number of extrapolation points
- `X::Vector{Vector{Float64}}` : vector of extrapolation points
"""
struct BellmanGrid <: BellmanExtrapolation
    N::Int
    X::Vector{Vector{Float64}}
end

"""
    BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen)
    BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen)
    BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen, x4s::StepRangeLen)

Construct a Bellman grid from points specified along each axis.
"""
function BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen)
    X = collect_meshgrid(x1s, x2s)
    N = length(X)

    return BellmanGrid(N, X)
end

function BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen)
    X = collect_meshgrid(x1s, x2s, x3s)
    N = length(X)

    return BellmanGrid(X, N)
end

function BellmanGrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen, x4s::StepRangeLen)
    X = collect_meshgrid(x1s, x2s, x3s, x4s)
    N = length(X)

    return BellmanGrid(X, N)
end

"""
    meshgrid(x1s::StepRangeLen, x2s::StepRangeLen)
    meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen)
    meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen, x4s::StepRangeLen)

Create a meshgrid along each axis.
"""
function meshgrid(x1s::StepRangeLen, x2s::StepRangeLen)
    X1s = [x1 for x1 in x1s for x2 in x2s]
    X2s = [x2 for x1 in x1s for x2 in x2s]

    return X1s, X2s
end

function meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen)
    X1s = [x1 for x1 in x1s for x2 in x2s for x3 in x3s]
    X2s = [x2 for x1 in x1s for x2 in x2s for x3 in x3s]
    X3s = [x3 for x1 in x1s for x2 in x2s for x3 in x3s]

    return X1s, X2s, X3s
end

function meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen, x4s::StepRangeLen)
    X1s = [x1 for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s]
    X2s = [x2 for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s]
    X3s = [x3 for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s]
    X4s = [x4 for x1 in x1s for x2 in x2s for x3 in x3s for x4 in x4s]

    return X1s, X2s, X3s, X4s
end

"""
    collect_meshgrid(x1s, x2s)
    collect_meshgrid(x1s, x2s, x3s)
    collect_meshgrid(x1s, x2s, x3s, x4s)

Convert an n-dimensional meshgrid into a vector containing the points of the grid.
"""
function collect_meshgrid(x1s::StepRangeLen, x2s::StepRangeLen)
    X1s, X2s = meshgrid(x1s::StepRangeLen, x2s::StepRangeLen)
    return [[x1, x2] for (x1, x2) in zip(X1s, X2s)]
end

function collect_meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen)
    X1s, X2s, X3s = meshgrid(x1s, x2s, x3s)
    return [[x1, x2, x3] for (x1, x2, x3) in zip(X1s, X2s, X3s)]
end

function collect_meshgrid(x1s::StepRangeLen, x2s::StepRangeLen, x3s::StepRangeLen, x4s::StepRangeLen)
    X1s, X2s, X3s, X4s = meshgrid(x1s, x2s, x3s, x4s)
    return [[x1, x2, x3, x4] for (x1, x2, x3, x4) in zip(X1s, X2s, X3s, X4s)]
end