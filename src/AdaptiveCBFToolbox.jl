module AdaptiveCBFToolbox

# Required modules
using LinearAlgebra
using DifferentialEquations
using Integrals
using ForwardDiff
using JuMP
using OSQP
using CBFToolbox

# Abstract types used throughout module
abstract type UncertainParameters end
abstract type UpdateLaw end
abstract type LyapunovUpdateLaw <: UpdateLaw end
abstract type IdentificationUpdateLaw <: UpdateLaw end
abstract type AdaptiveController <: CBFToolbox.Controller end
abstract type HistoryStack end

# Export CBFToolbox types
export ControlAffineSystem
export Simulation
export ControlLyapunovFunction
export ControlBarrierFunction
export SecondOrderCBF

# Export functions from CBFToolbox
export plot_circle
export plot_circle!

# Export AdaptiveCBFToolbox types
export MatchedParameters
export UnmatchedParameters
export ACLFQuadProg
export ISSaCLFQuadProg
export RACBFQuadProg
export CLFUpdateLaw
export GradientUpdateLaw
export LeastSquaresUpdateLaw
export DCLHistoryStack
export DCLGradientUpdateLaw
export ICLHistoryStack
export ICLGradientUpdateLaw

# Source code
include("matched_parameters.jl")
include("unmatched_parameters.jl")
include("aclf_quad_prog.jl")
include("iss_aclf_quad_prog.jl")
include("racbf_quad_prog.jl")
include("horacbf_quad_prog.jl")
include("clf_update_law.jl")
include("gradient_update_law.jl")
include("least_squares_update_law.jl")
include("dcl_history_stack.jl")
include("dcl_gradient_update_law.jl")
include("icl_history_stack.jl")
include("icl_gradient_update_law.jl")

end # module
