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
export CustomAdaptiveController
export ACLFSontag
export ACLFQuadProg
export ISSaCLFQuadProg
export RACBFQuadProg
export ISSfaCBFQuadProg
export CLFUpdateLaw
export GradientUpdateLaw
export LeastSquaresUpdateLaw
export DCLHistoryStack
export DCLGradientUpdateLaw
export DCLLeastSquaresUpdateLaw
export ICLHistoryStack
export ICLGradientUpdateLaw
export ICLLeastSquaresUpdateLaw

# Source code
include("Parameters/matched_parameters.jl")
include("Parameters/unmatched_parameters.jl")
include("Lyapunov/custom_adaptive_controller.jl")
include("Lyapunov/aclf_sontag.jl")
include("Lyapunov/aclf_quad_prog.jl")
include("Lyapunov/iss_aclf_quad_prog.jl")
include("Barrier/racbf_quad_prog.jl")
include("Barrier/horacbf_quad_prog.jl")
include("Barrier/issf_acbf_quad_prog.jl")
include("Barrier/high_order_issf_acbf_qp.jl")
include("Lyapunov/clf_update_law.jl")
include("IdentificationUpdateLaws/gradient_update_law.jl")
include("IdentificationUpdateLaws/least_squares_update_law.jl")
include("DerivativeConcurrentLearning/dcl_history_stack.jl")
include("DerivativeConcurrentLearning/dcl_gradient_update_law.jl")
include("DerivativeConcurrentLearning/dcl_least_squares_update.jl")
include("IntegralConcurrentLearning/icl_history_stack.jl")
include("IntegralConcurrentLearning/integration_utils.jl")
include("IntegralConcurrentLearning/icl_gradient_update_law.jl")
include("IntegralConcurrentLearning/icl_least_squares_update.jl")

end # module
