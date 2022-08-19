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

# Bounds on each parameter
μ1bounds = [0.0, 3.0]
μ2bounds = [0.0, 3.0]
𝚯 = [μ1bounds, μ2bounds]

# Define the regressor function and construct a MatchedParameters object
φ(x) = -diagm(x[3:4])
P = MatchedParameters(θ, φ, 𝚯)

# Define a CLF and an adaptive CLF-QP controller for reaching the origin
Q = [2.0 0.0 1.0 0.0; 0.0 2.0 0.0 1.0; 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
V(x) = x'Q*x
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)
k0 = ACLFQuadProg(Σ, P, CLF)

# Define update law associated with CLF
Γ = 1.0
τCLF = CLFUpdateLaw(Γ, P, Σ, CLF)

# Define an ICL history stack and update law
Γc = 10.0
M = 20
dt = 0.01
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLGradientUpdateLaw(Γc, dt, Δt, H)

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
k = RACBFQuadProg(Σ, P, k0, HOCBFs)

# Initial conditions
x0 = [-2.5, 2.5, 0.0, 0.0]
θ̂0 = zeros(length(θ))

# Simulate
T = 20.0
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
    fig = plot(sol, idxs=(1,2), label="", dpi=200)
    plot_circle!(xo1[1], xo1[2], ro1)
    plot_circle!(xo2[1], xo2[2], ro2)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(fig)
    # savefig(fig, "dbl_int_traj.png")
end

# Plot trajectory of parameter estimates
begin
    fig = plot(sol, idxs = Σ.n + 1 : Σ.n + P.p, label=[L"\hat{\theta}_1^h" L"\hat{\theta}_2^h"], dpi=200)
    plot!(sol, idxs = Σ.n + P.p + 1 : Σ.n + 2*P.p, label=[L"\hat{\theta}_1^V" L"\hat{\theta}_2^V"])
    hline!(θ, c=:black, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"\hat{\theta}(t)")
    display(fig)
    # savefig(fig, "dbl_int_params.png")
end

# Plot trajectory of estimation error
θ̂(t) = sol(t, idxs = Σ.n + 1 : Σ.n + P.p)
θ̃(t) = norm(θ - θ̂(t))
tspan = 0.0:dt:T
begin
    fig = plot(sol, idxs = Σ.n + 2*P.p + 1, label=L"\tilde{\vartheta}(t)")
    plot!(tspan, θ̃.(tspan), label=L"||\tilde{\theta}(t)||")
    xlabel!(L"t")
    ylabel!("Estimation error")
    display(fig)
end