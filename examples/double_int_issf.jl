# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Plots
using LaTeXStrings
default(fontfamily="Computer Modern", grid=false, framestyle=:box, lw=2, label="")

# Define system: planar double integrator
n = 4
m = 2
f(x) = vcat(x[3:4], zeros(2))
g(x) = vcat(zeros(2,2), diagm(ones(2)))
Σ = ControlAffineSystem(n, m, f, g)

# Define uncertain parameters: friction coefficients
μ1 = 1.0
μ2 = 1.0
θ = [μ1, μ2]

# Define the regressor function and construct a MatchedParameters object
φ(x) = -diagm(x[3:4]*norm(x[3:4]))
P = MatchedParameters(θ, φ, 𝚯)

# Define a CLF and an adaptive CLF-QP controller for reaching the origin
Q = [2.0 0.0 1.0 0.0; 0.0 2.0 0.0 1.0; 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
V(x) = x'Q*x
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)

# Define ISS controller
ε = 1.0
kISS = ISSaCLFQuadProg(Σ, P, CLF, ε)

# Define safety constraints for two obstacles
xo1 = [-1.75, 2.0]
xo2 = [-1.0, 0.5]
ro1 = 0.5
ro2 = 0.5
h1(x) = norm(x[1:2] - xo1)^2 - ro1^2
h2(x) = norm(x[1:2] - xo2)^2 - ro2^2

# Construct HOCBFs for each obstacle and HO-RaCBF QP
α1(s) = s
α2(s) = s
HOCBF1 = SecondOrderCBF(Σ, h1, α1, α2)
HOCBF2 = SecondOrderCBF(Σ, h2, α1, α2)
HOCBFs = [HOCBF1, HOCBF2]

# Define an ISSf controller
ε0 = 1.0
λ = 1.0
kISSf = ISSfaCBFQuadProg(Σ, P, HOCBFs, kISS, ε0, λ)

# Define an ICL history stack and update law
Γ = 10.0
M = 20
dt = 0.1
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLGradientUpdateLaw(Γ, dt, Δt, H)

# Initial conditions
x0 = [-2.5, 2.5, 0.0, 0.0]
θ̂0 = zeros(length(θ))

# Run simulation
T = 15.0
S = Simulation(T)
sol = S(Σ, P, kISSf, τ, x0, θ̂0)

# Plot system states
begin
    fig = plot(sol, idxs=1:Σ.n, label="")
    xlabel!(L"t")
    ylabel!(L"x(t)")
    display(fig)
end

# Plot evolution of system's configuration
begin
    fig = plot(sol, idxs=(1,2), label="")
    plot_circle!(xo1[1], xo1[2], ro1)
    plot_circle!(xo2[1], xo2[2], ro2)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(fig)
end

# Plot trajectory of parameter estimates
begin
    fig = plot(sol, idxs = Σ.n + 1 : Σ.n + P.p, label="")
    hline!(θ, c=:black, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"\hat{\theta}(t)")
    display(fig)
end