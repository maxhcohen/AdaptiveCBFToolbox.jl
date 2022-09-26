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
Q = [2.0 0.0 1.0 0.0; 0.0 2.0 0.0 1.0; 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0]
V(x) = x'Q*x
γ(x) = V(x)
CLF = ControlLyapunovFunction(V, γ)

# Define ISS controller
ε = 1.0
kISS = ISSaCLFQuadProg(Σ, P, CLF, ε)

# Define safety constraints for two obstacles
xo = [-1.0, 1.0]
ro = 0.5
h(x) = norm(x[1:2] - xo)^2 - ro^2

# Construct HOCBFs for each obstacle and HO-RaCBF QP
α1(s) = s
α2(s) = s
HOCBF = SecondOrderCBF(Σ, h, α1, α2)

# Define an ISSf controller
ε0 = 0.5
λ = 1.0
kISSf = ISSfaCBFQuadProg(Σ, P, HOCBF, kISS, ε0, λ)

# Define an ICL history stack and update law
M = 20
dt = 0.1
Δt = 0.5
H = ICLHistoryStack(M, Σ, P)
τ = ICLLeastSquaresUpdateLaw(H, dt, Δt)

# Initial conditions
x10 = rand(-2.2:0.01:-1.8)
x20 = rand(1.8:0.01:2.2)
x0 = [x10, x20, 0.0, 0.0]
θ̂10 = rand(0.0:0.1:2)
θ̂20 = rand(0.0:0.1:2)
θ̂0 = [θ̂10, θ̂20]
Γ0 = 10.0*diagm(ones(length(θ)))

# Run simulation
T = 15.0
S = Simulation(T)
sol = S(Σ, P, kISSf, τ, x0, θ̂0, Γ0)

using Plots
using LaTeXStrings
gr()
# Define colors and plot default settings
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

# Set up some plotting stuff
ts = 0.0:0.01:15

begin
    fig = plot()
    plot!(sol, idxs=(1,2), label="")
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
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

