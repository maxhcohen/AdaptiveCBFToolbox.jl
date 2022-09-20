"""
    KernelBasis <: BasisFunctions

Basis of kernel functions compatabile with any of those defined in KernelFunctions.jl.

# Fields
- `L::Int` : number of basis functions
- `ϕ::Function` : vector of basis functions
"""
struct KernelBasis <: BasisFunctions
    L::Int
    ϕ::Function
end

"""
    KernelBasis(n::Int, k::Kernel, centers::Matrix{Float64})
    KernelBasis(n::Int, k::Kernel, centers::Vector{Vector{Float64}})
    KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen)
    KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen, x3::StepRangeLen)
    KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen, x3::StepRangeLen, x4::StepRangeLen)

Construct a `KernelBasis` for an `n` dimensional system from kernels at specified locations.
"""
function KernelBasis(n::Int, k::Kernel, centers::Matrix{Float64})
    ϕ(x) = [k(x, vec(c)) for c in eachcol(centers)]
    L = length(ϕ(zeros(n)))

    return KernelBasis(L, ϕ)
end

function KernelBasis(n::Int, k::Kernel, centers::Vector{Vector{Float64}})
    ϕ(x) = [k(x, c) for c in centers]
    L = length(ϕ(zeros(n)))

    return KernelBasis(L, ϕ)
end

function KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen)
    n = 2
    c = collect_meshgrid(x1, x2)

    return KernelBasis(n, k, c)
end

function KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen, x3::StepRangeLen)
    n = 3
    c = collect_meshgrid(x1, x2, x3)

    return KernelBasis(n, k, c)
end

function KernelBasis(k::Kernel, x1::StepRangeLen, x2::StepRangeLen, x3::StepRangeLen, x4::StepRangeLen)
    n = 4
    c = collect_meshgrid(x1, x2, x3, x4)

    return KernelBasis(n, k, c)
end