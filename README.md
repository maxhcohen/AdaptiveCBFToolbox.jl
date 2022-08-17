# AdaptiveCBFToolbox.jl
A package for implementing adaptive control algorithms in Julia, with an emphasis on methods that leverage control Lyapunov functions (CLFs) and control barrier functions (CBFs).

## Overview
This toolbox is intended to serve as an extension of [CBFToolbox.jl](https://github.com/maxhcohen/CBFToolbox.jl/tree/dev2) to adaptive control systems and implements the methods described in our ACC 2022 paper

>M. H. Cohen and C. Belta, "[High order robust adaptive control barrier functions and exponentially stabilizing adaptive control Lyapunov functions](https://arxiv.org/abs/2203.01999)," in Proceedings of the American Control Conference, pp. 2233-2238, 2022.

The scripts in the examples folder allow one to reproduce the results from the above paper.


The toolbox essentially adds to [CBFToolbox.jl](https://github.com/maxhcohen/CBFToolbox.jl/tree/dev2) a suite of parameter estimation routines that can be used to define adaptive controllers that stabilize an uncertain dynamical system while satisfying safety constraints expressed as [CBFs](https://arxiv.org/abs/1903.11199) or [high order CBFs](https://ieeexplore.ieee.org/abstract/document/9516971). The parameter estimation algorithms used in this toolbox are based on those developed in

>G. Chowdhary and E. Johnson, "[Concurrent learning for convergence in adaptive control without persistency of excitation](https://ieeexplore.ieee.org/abstract/document/5717148)," in Proceedings of the IEEE Conference on Decision and Control, pp. 3674-3679, 2010;

>G. Chowdhary and E. Johnson, "[A singular value maximizing data recording algorithm for concurrent learning](https://ieeexplore.ieee.org/abstract/document/5991481)," in Proceedings of the American Control Conference, pp. 3547 - 3552, 2011;

>A. Parikh, R. Kamalapurkar, and W. E. Dixon, "[Integral concurrent learning: Adaptive control with parameter convergence using finite excitation](https://onlinelibrary.wiley.com/doi/full/10.1002/acs.2945)," International Journal of Adaptive Control and Signal Processing, vol. 33, no. 12, pp. 1775-1787, 2019.

## Tutorial
The usage of this package is similar to that of [CBFToolbox.jl](https://github.com/maxhcohen/CBFToolbox.jl/tree/dev2): we define a `ControlAffineSystem` and a `Controller` based on a `ControlLyapunovFunction`, `ControlBarrierFunction`, or both, and then run a simulation using a `Simulation` object. This package integrates the new abstract types `UncertainParameters` and `UpdateLaw` into the previous workflow that allows for specifying unknown parameters of the underlying system and a parameter estimator to learn such parameters, respectively. The following code shows a simple example of the typical workflow.
```julia
# Import packages
using AdaptiveCBFToolbox
using LinearAlgebra

# Define system: planar double integrator
n = 4 # State dimension
m = 2 # Constrol dimension
f(x) = vcat(x[3:4], zeros(2)) # Nominal drift dynamics
g(x) = vcat(zeros(2,2), diagm(ones(2))) # Control directions
Σ = ControlAffineSystem(n, m, f, g) # Construct a control affine system 

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
V(x) = x'Q*x # Quadratic CLF candidate
γ(x) = V(x) # Function defining the rate of CLF decay V̇(x) ≤ -γ(x)
CLF = ControlLyapunovFunction(V, γ)
k0 = ACLFQuadProg(Σ, P, CLF) # adaptive CLF-QP controller

# Define update law associated with CLF
Γ = 1.0 # Learning rate
τCLF = CLFUpdateLaw(Γ, P, Σ, CLF) # Parameter estimation object

# Define history stack to store input-output data
Γc = 10.0
M = 20 # Number of entries in history stack
H = ICLHistoryStack(M, Σ, P)

# Define update law associated with data in history stack
dt = 0.01 # Sampling time used for data collection
Δt = 0.5 # Length of integration window
τ = ICLGradientUpdateLaw(Γc, dt, Δt, H) # Parameter estimation object

# Define safety constraints for two circular obstacles
xo1 = [-1.75, 2.0] # Center of obstacle
xo2 = [-1.0, 0.5]
ro1 = 0.5 # Radius of obstacle
ro2 = 0.5
h1(x) = norm(x[1:2] - xo1)^2 - ro1^2 # Distance to obstacle
h2(x) = norm(x[1:2] - xo2)^2 - ro2^2

# Construct HOCBFs for each obstacle and HO-RaCBF QP
α1(s) = s # Extended class K function
α2(s) = s
HOCBF1 = SecondOrderCBF(Σ, h1, α1, α2) # Relative degree 2 HOCBF
HOCBF2 = SecondOrderCBF(Σ, h2, α1, α2)
HOCBFs = [HOCBF1, HOCBF2]
k = RACBFQuadProg(Σ, P, k0, HOCBFs) # Filter solution to CLF-QP through CBF-QP

# Initial conditions
x0 = [-2.5, 2.5, 0.0, 0.0]
θ̂0 = zeros(length(θ))

# Run simulation
T = 20.0
S = Simulation(T)
sol = S(Σ, P, k, τCLF, τ, x0, θ̂0)
```
The solution output from a simulation is simply a solution of an ODE generated by [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). We can thus access the different indices of the solution object to plot results like the evolution of the system's position

![](https://github.com/maxhcohen/AdaptiveCBFToolbox.jl/blob/main/figures/dbl_int_traj.png)

or the estimates of the uncertain parameters

![](https://github.com/maxhcohen/AdaptiveCBFToolbox.jl/blob/main/figures/dbl_int_params.png)

## Questions and Contributions
If you have any questions about the toolbox, have suggestions for improvements, or would like to make your own contribution to the toolbox feel free to reach out to maxcohen@bu.edu.
