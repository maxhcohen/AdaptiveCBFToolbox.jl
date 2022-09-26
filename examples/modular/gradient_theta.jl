# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra

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
ε = 20.0
kISS = ISSaCLFQuadProg(Σ, P, CLF, ε)

# Define a HOCBF to avoid an obstacle
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
Γ = 100.0*diagm(ones(length(θ)))
M = 20
dt = 0.5
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLGradientUpdateLaw(Γ, dt, Δt, H)

# Initial conditions
x0 = [-2.1, 2.0, 0.0, 0.0]
θ̂0s = [i*ones(2) for i in 2:4]
Γ0 = 100.0*diagm(ones(2))

# Run sim for each initial set of parameter estimates
T = 15.0
S = Simulation(T)
sols = []
for θ̂0 in θ̂0s
    local H = ICLHistoryStack(M, Σ, P)
    local τ = ICLGradientUpdateLaw(Γ, dt, Δt, H)
    push!(sols, S(Σ, P, kISSf, τ, x0, θ̂0))
end

using Plots
using LaTeXStrings
gr() # Used to quickly vizualize results
# pgfplotsx() # Used to save figures as .tex files and directly embed in LaTeX file

# Define colors and plot default settings to make things look nice
begin
    myblue = RGB(7/255, 114/255, 179/255)
    myred = RGB(240/255, 97/255, 92/255)
    mygreen = RGB(0/255, 159/255, 115/255)
    mypurple = RGB(120/255, 110/255, 179/255)
    myyellow = RGB(231/255, 161/255, 34/255)
    mycyan = RGB(93/255, 180/255, 229/255)
    mypink = RGB(217/255, 91/255, 161/255)
    mypalette = [myblue, myred, mygreen, mypurple, myyellow, mycyan, mypink]
end
default(grid=false, framestyle=:box, lw=2, label="", palette=mypalette, fontfamily="Computer Modern", legend=:topright)

# Discretization (is that spelled right?) of simulation timespan
ts = 0.0:0.01:15

# Plot system trajectory
begin
    fig = plot()
    plot!(sols[1], idxs=(1,2), label=L"\hat{\theta}(0)=(2,2)", ls=:solid, c=1)
    plot!(sols[2], idxs=(1,2), label=L"\hat{\theta}(0)=(3,3)", ls=:dash, c=1)
    plot!(sols[3], idxs=(1,2), label=L"\hat{\theta}(0)=(4,4)", ls=:dot, c=1)
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(-2.2, 0.2)
    ylims!(-0.7, 2.2)
    # savefig("gradient_theta_traj.tex")
    display(fig)
end

# Plot evolution of parameter estimation error
begin
    fig = plot()
    θ̂(t, i) = sols[i](t, idxs = Σ.n + 1 : Σ.n + P.p)
    θ̃(t, i) = norm(θ - θ̂(t, i))
    plot!(ts, θ̃.(ts, 1), label="", c=1, ls=:solid)
    plot!(ts, θ̃.(ts, 2), label="", c=1, ls=:dash)
    plot!(ts, θ̃.(ts, 3), label="", c=1, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"||\tilde{\theta}(t)||")
    # savefig("gradient_theta_theta.tex")
    display(fig)
end

# Plot control effort
begin
    fig = plot()
    x(t, i) = sols[i](t, idxs = 1 : Σ.n)
    θ̂(t, i) = sols[i](t, idxs = Σ.n + 1 : Σ.n + P.p)
    u(t, i) = kISSf(x(t, i),  θ̂(t, i))
    unorm(t, i) = norm(u(t, i))
    plot!(ts, unorm.(ts, 1), label="", c=1, ls=:solid)
    plot!(ts, unorm.(ts, 2), label="", c=1, ls=:dash)
    plot!(ts, unorm.(ts, 3), label="", c=1, ls=:dot)
    xlabel!(L"t")
    ylabel!(L"\|u(t)\|")
    # savefig("gradient_theta_control.tex")
    display(fig)
end

