# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Plots
using LaTeXStrings
default(fontfamily="Computer Modern", grid=false, framestyle=:box, lw=2, label="")

# Define system: inverted pendulum
mass = 0.7
l = 0.7
grav = 9.8
c = 0.2
f(x) = [x[2], 0.0]
g(x) = [0.0, 1/(mass*l^2)]
Σ = ControlAffineSystem(2, 1, f, g)

# Define uncertain parameters
θ = [grav, c]
𝚯1 = [7.0, 12.0]
𝚯2 = [0.0, 2.0]
𝚯 = [𝚯1, 𝚯2]
φ(x) = [sin(x[1])/l, -(l^2)*x[2]]'
P = MatchedParameters(θ, φ, 𝚯)

# Define CLF
V(x) = x[1]^2 + x[1]*x[2] + 0.5*x[2]^2
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)
k = ACLFQuadProg(Σ, P, CLF)

# Define update law associated with CLF
Γ = 1.0
τCLF = CLFUpdateLaw(Γ, P, Σ, CLF)

# Define an ICL history stack and update law
Γc = 10.0
M = 20
dt = 0.1
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLGradientUpdateLaw(Γc, dt, Δt, H)

# Initial conditions
x0 = [π/6, 0.0]
θ̂0 = [7.5, 1.0]

# Run simulation
T = 15.0
S = Simulation(T)
sol = S(Σ, P, k, τCLF, τ, x0, θ̂0)

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
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(fig)
end

# Plot trajectory of parameter estimates
begin
    fig = plot(sol, idxs = Σ.n + 1 : Σ.n + P.p, label=[L"\hat{\theta}_1" L"\hat{\theta}_2"])
    hline!(θ, c=:black, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"\hat{\theta}(t)")
    display(fig)
end