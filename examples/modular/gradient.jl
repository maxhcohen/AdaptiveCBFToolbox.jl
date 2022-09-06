# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
gr()
default(fontfamily="Computer Modern", grid=false, framestyle=:box, lw=2, label="")

# How many simulations we want to run
N = 25
sols = []

# Define system: planar double integrator
n = 4
m = 2
f(x) = vcat(x[3:4], zeros(2))
g(x) = vcat(zeros(2,2), diagm(ones(2)))
Σ = ControlAffineSystem(n, m, f, g)

# Define uncertain parameters: friction coefficients
μ1 = 0.8
μ2 = 1.4
θ = [μ1, μ2]

# Define the regressor function and construct a MatchedParameters object
φ(x) = -diagm(x[3:4]*norm(x[3:4]))
P = MatchedParameters(θ, φ)

# Define a CLF and an adaptive CLF-QP controller for reaching the origin
Q = [2.0 0.0 1.0 0.0; 0.0 2.0 0.0 1.0; 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
V(x) = x'Q*x
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)

# Define ISS controller
ε = 10.0
kISS = ISSaCLFQuadProg(Σ, P, CLF, ε)

# Define safety constraints for two obstacles
xo = [-1.0, 1.0]
ro = 0.5
h(x) = norm(x[1:2] - xo)^2 - ro^2

# Construct HOCBFs for each obstacle and HO-RaCBF QP
α1(s) = s
α2(s) = 0.5s
HOCBF = SecondOrderCBF(Σ, h, α1, α2)

# Define an ISSf controller
ε0 = 0.5
λ = 10.0
kISSf = ISSfaCBFQuadProg(Σ, P, HOCBF, kISS, ε0, λ)

for i in 1:N
    # Define an ICL history stack and update law
    Γ = 100.0*diagm(ones(length(θ)))
    M = 20
    dt = 0.5
    Δt = 0.5
    H = ICLHistoryStack(M, Σ, P)
    τ = ICLGradientUpdateLaw(Γ, dt, Δt, H)

    # Initial conditions
    x10 = rand(-2.2:0.01:-1.8)
    x20 = rand(1.8:0.01:2.2)
    x0 = [x10, x20, 0.0, 0.0]
    θ̂10 = rand(0.0:0.1:2)
    θ̂20 = rand(0.0:0.1:2)
    θ̂0 = [θ̂10, θ̂20]

    # Run simulation
    T = 15.0
    S = Simulation(T)
    sol = S(Σ, P, kISSf, τ, x0, θ̂0)

    # Save data
    push!(sols, sol)
end

# Set up some plotting stuff
ts = 0.0:0.01:15

begin
    fig = plot()
    for sol in sols
        plot!(sol, idxs=(1,2), label="")
    end
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
    display(fig)
end

begin
    fig = plot()
    for sol in sols
        θ̂(t) = sol(t, idxs = Σ.n + 1 : Σ.n + P.p)
        θ̃(t) = norm(θ - θ̂(t))
        plot!(ts, θ̃.(ts), label="")
    end
    xlabel!(L"t")
    ylabel!(L"||\tilde{\theta}(t)||")
    display(fig)
end

# Compute average estimation error
θ̃avg(t) = mean([norm(θ - sol(t, idxs = Σ.n + 1 : Σ.n + P.p)) for sol in sols])
θ̃std(t) = std([norm(θ - sol(t, idxs = Σ.n + 1 : Σ.n + P.p)) for sol in sols])
begin
    fig = plot(ts, θ̃avg.(ts), ribbon=θ̃std.(ts), fillalpha=0.2, c=1)
    xlabel!(L"t")
    ylabel!(L"||\tilde{\theta}_{\text{avg}}(t)||")
    title!("Gradient estimator")
end 
