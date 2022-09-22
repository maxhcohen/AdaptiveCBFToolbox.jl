# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Statistics

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
V(x) = 0.5*norm(x[1:2])^2 + 0.5*norm(x[3:4] + x[1:2])^2
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)
ε = 10.0
kISS = ISSaCLFQuadProg(Σ, P, CLF, ε)

# CBF
xo = [-1.0, 1.0]
ro = 0.5
h(x) = norm(x[1:2] - xo)^2 - ro^2
α1(s) = s
α2(s) = 0.5s
HOCBF = SecondOrderCBF(Σ, h, α1, α2)
ε0 = 1.0
λ = 0.0
kISSf = ISSfaCBFQuadProg(Σ, P, HOCBF, kISS, ε0, λ)

# Parameters associated with ICL
β = 1.0
Γ̄ = 1000.0
M = 20
dt = 0.5
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLLeastSquaresUpdateLaw(H, dt, Δt, β)

# Initial conditions
x0 = [-2.1, 2.0, 0.0, 0.0]
θ̂0 = 3*ones(2)
Γ0 = 100.0*diagm(ones(2))

# Run sim
T = 15.0
S = Simulation(T)
sol = S(Σ, P, kISSf, τ, x0, θ̂0, Γ0)

using Plots
using LaTeXStrings
gr()
default(fontfamily="Computer Modern", grid=false, framestyle=:box, lw=2, label="", palette=:julia)
# Set up some plotting stuff
ts = 0.0:0.01:15

begin
    fig = plot()
    plot!(sol, idxs=(1,2), label="")
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(-2.2, 0.2)
    ylims!(-0.7, 2.2)
    display(fig)
end

begin
    fig = plot()
    θ̂(t) = sol(t, idxs = Σ.n + 1 : Σ.n + P.p)
    θ̃(t) = norm(θ - θ̂(t))
    plot!(ts, θ̃.(ts), label="")
    xlabel!(L"t")
    ylabel!(L"||\tilde{\theta}(t)||")
    display(fig)
end

# Plot control effort
begin
    fig = plot()
    x(t) = sol(t, idxs = 1 : Σ.n)
    θ̂(t) = sol(t, idxs = Σ.n + 1 : Σ.n + P.p)
    u(t) = kISSf(x(t),  θ̂(t))
    unorm(t) = norm(u(t))
    plot!(ts, unorm.(ts), label="")
    xlabel!(L"t")
    ylabel!(L"\|u(t)\|")
    display(fig)
end

# Plot state convergence?
begin
    fig = plot()
    x(t) = sol(t, idxs = 1 : Σ.n)
    xnorm(t) = norm(x(t))
    plot!(ts, xnorm.(ts), label="")
    xlabel!(L"t")
    ylabel!(L"\|x(t)\|")
end

