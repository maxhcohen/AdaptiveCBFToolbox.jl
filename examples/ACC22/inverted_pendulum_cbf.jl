# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Plots
using LaTeXStrings
julia_palette = deleteat!(distinguishable_colors(10, [c for c in palette(:julia)]), 5:6)
default(fontfamily="Computer Modern", grid=false, framestyle=:box, lw=2, label="", palette=julia_palette)

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
𝚯1 = [7.0, 13.0]
𝚯2 = [0.0, 3.0]
𝚯 = [𝚯1, 𝚯2]
φ(x) = [sin(x[1])/l, -(l^2)*x[2]]'
P = MatchedParameters(θ, φ, 𝚯)

# Define CLF
V(x) = x[1]^2 + x[1]*x[2] + 0.5*x[2]^2
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)
k0 = ACLFQuadProg(Σ, P, CLF)

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

# Define safety constraints and HOCBFs
h1(x) = x[1] + π/4
h2(x) = -x[1] + π/4
α1(s) = 5s
α2(s) = 5s^3
HOCBF1 = SecondOrderCBF(Σ, h1, α1, α2)
HOCBF2 = SecondOrderCBF(Σ, h2, α1, α2)
HOCBFs = [HOCBF1, HOCBF2]
k = RACBFQuadProg(Σ, P, k0, HOCBFs)

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
    hline!([-π/4, π/4], c=:black, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"x(t)")
    display(fig)
end

# Plot evolution of system's configuration
begin
    fig = plot(sol, idxs=(1,2), label="")
    vline!([-π/4, π/4], c=:black, ls=:dot)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(-π/3, π/3)
    display(fig)
end

# Plot trajectory of parameter estimates
begin
    fig = plot(sol, idxs = Σ.n + 1 : Σ.n + P.p, label=[L"\hat{\theta}_1^h" L"\hat{\theta}_2^h"])
    plot!(sol, idxs = Σ.n + P.p + 1 : Σ.n + 2*P.p, label=[L"\hat{\theta}_1^V" L"\hat{\theta}_2^V"])
    hline!(θ, c=:black, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"\hat{\theta}(t)")
    display(fig)
end

# Plot trajectory of estimation error
begin
    θ̂(t) = sol(t, idxs = Σ.n + 1 : Σ.n + P.p)
    θ̃(t) = norm(θ - θ̂(t))
    tspan = 0.0:dt:T
    fig = plot(sol, idxs = Σ.n + 2*P.p + 1, label=L"\tilde{\vartheta}(t)")
    plot!(tspan, θ̃.(tspan), label=L"||\tilde{\theta}(t)||")
    xlabel!(L"t")
    ylabel!("Estimation error")
    display(fig)
end