# Import packages
using Revise
using AdaptiveCBFToolbox
using LinearAlgebra
using Statistics

# Number of sims to run
N = 25

# Control-affine system under consideration
n = 4
m = 2
f(x) = vcat(x[3:4], zeros(2))
g(x) = vcat(zeros(2,2), diagm(ones(2)))
Σ = ControlAffineSystem(n, m, f, g)

# Uncertain parameters for system
μ1 = 0.8
μ2 = 1.4
θ = [μ1, μ2]
φ(x) = -diagm(x[3:4]*norm(x[3:4]))
P = MatchedParameters(θ, φ)

# CLF
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
ε0 = 0.5
λ = 10.0
kISSf = ISSfaCBFQuadProg(Σ, P, HOCBF, kISS, ε0, λ)

# Parameters associated with ICL
Γ0 = 100.0
β = 1.0
Γ̄ = 1000.0
M = 20
dt = 0.5
Δt = 0.5

# Generate initial conditions
x10 = rand(-2.2:0.01:-1.8, N)
x20 = rand(1.8:0.01:2.2, N)
θ̂10 = rand(0.0:0.1:2, N)
θ̂20 = rand(0.0:0.1:2, N)
x0 = [vcat([x10[i], x20[i]], zeros(2)) for i in 1:N]
θ̂0 = [[θ̂10[i], θ̂20[i]] for i in 1:N]

# Construct simulator
T = 15.0
sim = Simulation(T)

# Put dictionary together
D = Dict(
    "x0" => x0,
    "θ̂0" => θ̂0,
    "Σ" => Σ,
    "P" => P,
    "k" => kISSf,
    "M" => M,
    "β" => β,
    "Γ̄" => Γ̄,
    "dt" => dt,
    "Δt" => Δt,
    "Γ0" => Γ0,
    "sim" => sim
)

function modular_sims(D::Dict, estimator::String)
    # Pull out variables
    x0s = D["x0"]
    θ̂0s = D["θ̂0"]
    Σ = D["Σ"]
    P = D["P"]
    k = D["k"]
    M = D["M"]
    β = D["β"]
    Γ̄ = D["Γ̄"]
    dt = D["dt"]
    Δt = D["Δt"]
    Γ0 = D["Γ0"]
    sim = D["sim"]

    # Start looping through sims
    sols = []
    for i in 1:N
        # Construct history stack
        H = ICLHistoryStack(M, Σ, P)

        # Construct parameter estimator
        if estimator == "gradient"
            Γ = Γ0*diagm(ones(length(θ)))
            τ = ICLGradientUpdateLaw(Γ, dt, Δt, H)
            sol = sim(Σ, P, k, τ, x0s[i], θ̂0s[i])
        elseif estimator == "rls"
            Γ = Γ0*diagm(ones(length(θ)))
            τ = ICLLeastSquaresUpdateLaw(H, dt, Δt)
            sol = sim(Σ, P, k, τ, x0s[i], θ̂0s[i], Γ)
        elseif estimator == "rls_forgetting"
            Γ = Γ0*diagm(ones(length(θ)))
            τ = ICLLeastSquaresUpdateLaw(H, dt, Δt, β)
            sol = sim(Σ, P, k, τ, x0s[i], θ̂0s[i], Γ)
        elseif estimator == "rls_bounded_forgetting"
            Γ = Γ0*diagm(ones(length(θ)))
            τ = ICLLeastSquaresUpdateLaw(H, dt, Δt, β, Γ̄)
            sol = sim(Σ, P, k, τ, x0s[i], θ̂0s[i], Γ)
        else
            return error("Please select a valid estimator.")
        end

        # Save data
        push!(sols, sol)
    end

    return sols
end

# Run sims
sol_gradient = modular_sims(D, "gradient");
sol_rls = modular_sims(D, "rls");
sol_rlsf = modular_sims(D, "rls_forgetting");
sol_rlsbf = modular_sims(D, "rls_bounded_forgetting");

# Do some plots
using Plots
using LaTeXStrings
gr()
default(grid=false, framestyle=:box, lw=2, label="", palette=:julia, legend = :topright)
# Construct trajectory functions
ts = 0.0:0.01:T
Σidxs = 1 : Σ.n
θ̂idxs = Σ.n + 1 : Σ.n + P.p
x(t, sol) = sol(t, idxs = Σidxs)
q(t, sol) = x(t, sol)[1:2]
θ̂(t, sol) = sol(t, idxs = θ̂idxs)
u(t, sol) = kISSf(x(t,sol), θ̂(t, sol))

# Compute average estimation error
θ̃avg_gradient(t) = mean([norm(θ - θ̂(t, sol)) for sol in sol_gradient])
θ̃std_gradient(t) = std([norm(θ - θ̂(t, sol)) for sol in sol_gradient])
θ̃avg_rls(t) = mean([norm(θ - θ̂(t, sol)) for sol in sol_rls])
θ̃std_rls(t) = std([norm(θ - θ̂(t, sol)) for sol in sol_rls])
θ̃avg_rlsf(t) = mean([norm(θ - θ̂(t, sol)) for sol in sol_rlsf])
θ̃std_rlsf(t) = std([norm(θ - θ̂(t, sol)) for sol in sol_rlsf])
θ̃avg_rlsbf(t) = mean([norm(θ - θ̂(t, sol)) for sol in sol_rlsbf])
θ̃std_rlsbf(t) = std([norm(θ - θ̂(t, sol)) for sol in sol_rlsbf])
begin
    fig = plot(legend = :topright)
    plot!(ts, θ̃avg_gradient.(ts), ribbon=θ̃std_gradient.(ts), fillalpha=0.2, c=1, label="Gradient")
    plot!(ts, θ̃avg_rls.(ts), ribbon=θ̃std_rls.(ts), fillalpha=0.2, c=2, label="RLS")
    plot!(ts, θ̃avg_rlsf.(ts), ribbon=θ̃std_rlsf.(ts), fillalpha=0.2, c=3, label="RLS w/ forgetting")
    plot!(ts, θ̃avg_rlsbf.(ts), ribbon=θ̃std_rlsbf.(ts), fillalpha=0.2, c=4, label="RLS w/ bounded forgetting")
    xlabel!(L"t")
    ylabel!(L"||\tilde{\theta}_{\mathrm{avg}}(t)||")
    display(fig)
end

# Plot trajectories
begin
    fig = plot()
    for sol in sol_gradient
        plot!(sol, idxs=(1,2), label="", c=1, lw=0.1)
    end
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    title!("Gradient descent")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
    display(fig)
end

begin
    fig = plot()
    for sol in sol_rls
        plot!(sol, idxs=(1,2), label="", c=2, lw=0.1)
    end
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    title!("Recursive least squares")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
    display(fig)
end

begin
    fig = plot()
    for sol in sol_rlsf
        plot!(sol, idxs=(1,2), label="", c=3, lw=0.1)
    end
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    title!("RLS w/ forgetting factor")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
    display(fig)
end

begin
    fig = plot()
    for sol in sol_rlsbf
        plot!(sol, idxs=(1,2), label="", c=4, lw=0.1)
    end
    plot_circle!(xo[1], xo[2], ro)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    title!("RLS w/ bounded-gain forgetting factor")
    xlims!(-2.5, 0.5)
    ylims!(-0.5, 2.5)
    display(fig)
end

# Compute average safety constraint
havg_gradient(t) = mean([h(x(t, sol)) for sol in sol_gradient])
hstd_gradient(t) = std([h(x(t, sol)) for sol in sol_gradient])
hmin_gradient(t) = minimum([h(x(t, sol)) for sol in sol_gradient])
havg_rls(t) = mean([h(x(t, sol)) for sol in sol_rls])
hstd_rls(t) = std([h(x(t, sol)) for sol in sol_rls])
hmin_rls(t) = minimum([h(x(t, sol)) for sol in sol_rls])
havg_rlsf(t) = mean([h(x(t, sol)) for sol in sol_rlsf])
hstd_rlsf(t) = std([h(x(t, sol)) for sol in sol_rlsf])
havg_rlsbf(t) = mean([h(x(t, sol)) for sol in sol_rlsbf])
hstd_rlsbf(t) = std([h(x(t, sol)) for sol in sol_rlsbf])
begin
    fig = plot()
    plot!(ts, havg_gradient.(ts), ribbon=hstd_gradient.(ts), fillalpha=0.2, c=1, label="Gradient")
    plot!(ts, havg_rls.(ts), ribbon=hstd_rls.(ts), fillalpha=0.2, c=2, label="RLS")
    plot!(ts, havg_rlsf.(ts), ribbon=hstd_rlsf.(ts), fillalpha=0.2, c=3, label="RLS w/ forgetting")
    plot!(ts, havg_rlsbf.(ts), ribbon=hstd_rlsbf.(ts), fillalpha=0.2, c=4, label="RLS w/ bounded forgetting")
    xlabel!(L"t")
    ylabel!(L"h(x(t))")
    display(fig)
end

# Plot minimum value of h
begin
    fig = plot()
    plot!(ts, hmin_gradient.(ts), c=1, label="Gradient Descent")
    plot!(ts, hmin_rls.(ts), c=2, label="RLS")
    hline!([0.0], c=:black)
    ylims!(-0.1, 1.0)
    display(fig)
end