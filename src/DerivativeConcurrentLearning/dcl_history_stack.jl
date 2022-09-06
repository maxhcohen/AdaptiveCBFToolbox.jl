"""
    DCLHistoryStack <: HistoryStack

History stack of input-output data for use in concurrent learning adaptive control.
"""
mutable struct DCLHistoryStack <: HistoryStack
    idx::Int
    M::Int
    ẋ::Union{Vector{Float64}, Vector{Vector{Float64}}}
    x::Union{Vector{Float64}, Vector{Vector{Float64}}}
    u::Union{Vector{Float64}, Vector{Vector{Float64}}}
    ε::Float64
end

"""
    DCLHistoryStack(M::Int, Σ::ControlAffineSystem, ε::Float64)
    DCLHistoryStack(M::Int, Σ::ControlAffineSystem)

Construct a derivative concurrent leaarning history stack with `M` entries. 

If no tolerance `ε` is provided it defaults to `0.01`.
"""
function DCLHistoryStack(M::Int, Σ::ControlAffineSystem, ε::Float64)
    # Set initial index to 1
    idx = 1

    # Allocate arrays
    ẋ = Σ.n == 1 ? zeros(M) : [zeros(Σ.n) for i in 1:M]
    x = Σ.n == 1 ? zeros(M) : [zeros(Σ.n) for i in 1:M]
    u = Σ.m == 1 ? zeros(M) : [zeros(Σ.m) for i in 1:M]

    return DCLHistoryStack(idx, M, ẋ, x, u, ε)
end

function DCLHistoryStack(M::Int, Σ::ControlAffineSystem)
    ε = 0.01

    return DCLHistoryStack(M, Σ, ε)
end

"""
    update_stack!(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x, u)

Update stack with current data using singular value maximizing algorithm.
"""
function update_stack!(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x, u)
    # Check if current data is different enough from older data
    diff = stack_difference(stack, Σ, P, x)
    if diff >= stack.ε
        # Check if stack is full yet
        if stack.idx < stack.M 
            # If not full bump stack index by 1 and save current data to new index
            stack.idx += 1
            save_data!(stack.idx, stack, Σ, P, x, u)
        else
            # If stack is full perform singular value decomposition on current stack
            λs = zeros(stack.M)
            temp_stack = deepcopy(stack) # Copy current state of stack to temp stack.
            λold = stack_eig(temp_stack, Σ, P)
            for i in 1:stack.M
                save_data!(i, stack, Σ, P, x, u)
                λs[i] = stack_eig(stack, Σ, P)
                stack = deepcopy(temp_stack)
            end
            λmax = maximum(λs)
            imax = argmax(λs)
            if λmax > λold
                save_data!(imax, stack, Σ, P, x, u)
            else
                stack = deepcopy(temp_stack)
            end
        end
    end

    return stack
end

"""
    stack_difference(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x)

Check if new data is different enough from old data.
"""
function stack_difference(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x)
    xold = stack.x[stack.idx]
    Fnew = Σ.g(x)*P.φ(x)
    Fold = Σ.g(xold)*P.φ(xold)
    diff = norm(Fnew - Fold)^2 / norm(Fnew)^2

    return diff
end

"""
    save_data!(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x, u)

Save current input-output data to stack.
"""
function save_data!(idx::Int, stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x, u)
    stack.x[idx] = x
    stack.u[idx] = u
    stack.ẋ[idx] = Σ.f(x) + Σ.g(x)*(u + P.φ(x)*P.θ)

    return stack
end

"""
    stack_sum(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x)

Sum all the regressors in the current history stack.
"""
function stack_sum(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters)
    F = [Σ.g(stack.x[i])*P.φ(stack.x[i]) for i in 1:stack.M]

    return sum([F[i]' * F[i] for i in 1:stack.M])
end

"""
    stack_eig(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters, x)

Compute minimum eigenvalue of the matrix composed of the sum of regressors in current stack.
"""
function stack_eig(stack::DCLHistoryStack, Σ::ControlAffineSystem, P::MatchedParameters)
    return eigmin(stack_sum(stack, Σ, P))
end