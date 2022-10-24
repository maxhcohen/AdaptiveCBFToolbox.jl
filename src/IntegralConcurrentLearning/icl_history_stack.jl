"""
    ICLHistoryStack <: HistoryStack

History stack of input-output data for use in integral concurrent learning adaptive control.
"""
mutable struct ICLHistoryStack <: HistoryStack
    idx::Int
    M::Int
    Δx::Union{Vector{Float64}, Vector{Vector{Float64}}}
    f::Union{Vector{Float64}, Vector{Vector{Float64}}}
    F
    g::Union{Vector{Float64}, Vector{Vector{Float64}}}
    ε::Float64
end

"""
    ICLHistoryStack(M::Int, Σ::ControlAffineSystem, P::UncertainParameters, ε::Float64)
    ICLHistoryStack(M::Int, Σ::ControlAffineSystem, P::UncertainParameters)

Construct an integral concurrent learning history stack with `M` entries.
"""
function ICLHistoryStack(M::Int, Σ::ControlAffineSystem, P::UncertainParameters, ε::Float64)
    # Set initial index to 1
    idx = 1

    # Allocate arrays
    Δx = Σ.n == 1 ? zeros(M) : [zeros(Σ.n) for i in 1:M]
    f = Σ.n == 1 ? zeros(M) : [zeros(Σ.n) for i in 1:M]
    g = Σ.n == 1 ? zeros(M) : [zeros(Σ.n) for i in 1:M]
    if Σ.n == 1 && P.p == 1
        F = zeros(M)
    elseif Σ.n == 1 && P.p > 1
        F = [zeros(P.p)' for i in 1:M]
    elseif Σ.n > 1 && P.p == 1
        F = [zeros(Σ.n) for i in 1:M]
    elseif Σ.n > 1 && P.p > 1
        F = [zeros(Σ.n, P.p) for i in 1:M]
    end

    return ICLHistoryStack(idx, M, Δx, f, F, g, ε)
end

function ICLHistoryStack(M::Int, Σ::ControlAffineSystem, P::UncertainParameters)
    return ICLHistoryStack(M, Σ, P, 0.01)
end

"""
    update_stack!(stack::ICLHistoryStack, Δx, f, F, g)

Update stack with current data using singular value maximizing algorithm.
"""
function update_stack!(stack::ICLHistoryStack, Δx, f, F, g)
    # Check if current data is different enough from older data
    if stack_difference(stack, F) >= stack.ε
        # Check if stack is full yet
        if stack.idx < stack.M 
            # If not full bump stack index by 1 and save current data to new index
            stack.idx += 1
            save_data!(stack.idx, stack, Δx, f, F, g)
        else
            # If stack is full perform singular value decomposition on current stack
            λs = zeros(stack.M)
            temp_stack = deepcopy(stack) # Copy current state of stack to temp stack.
            λold = stack_eig(temp_stack)
            for i in 1:stack.M
                save_data!(i, stack, Δx, f, F, g)
                λs[i] = stack_eig(stack)
                stack = deepcopy(temp_stack)
            end
            λmax = maximum(λs)
            imax = argmax(λs)
            if λmax > λold
                save_data!(imax, stack, Δx, f, F, g)
            else
                stack = deepcopy(temp_stack)
            end
        end
    end

    return stack
end

"""
    stack_difference(stack::ICLHistoryStack, Fnew)

Check if new data is different enough from old data.
"""
stack_difference(stack::ICLHistoryStack, Fnew) = norm(Fnew - stack.F[stack.idx])^2 / norm(Fnew)^2

"""
    save_data!(idx::Int, stack::ICLHistoryStack, Δx, f, F, g)

Save current input-output data to stack.
"""
function save_data!(idx::Int, stack::ICLHistoryStack, Δx, f, F, g)
    stack.Δx[idx] = Δx
    stack.f[idx] = f
    stack.F[idx] = F
    stack.g[idx] = g

    return stack
end

"""
    stack_sum(stack::ICLHistoryStack)

Sum all the regressors in the current history stack.
"""
function stack_sum(stack::ICLHistoryStack)
    return sum([stack.F[i]' * stack.F[i] for i in 1:stack.M])
end

"""
    stack_eig(stack::ICLHistoryStack)

Compute minimum eigenvalue of the matrix composed of the sum of regressors in current stack.
"""
function stack_eig(stack::ICLHistoryStack)
    return eigmin(stack_sum(stack))
end