# ----- Gibbs / Glauber dynamics -----
# Delegates the actual conditional sampling to the model via gibbs_sample!()

"""
    Gibbs <: AbstractDynamics

Gibbs (Glauber) sampling dynamics for discrete models.
Samples a component from its exact conditional distribution.

# Fields
- `beta::Float64` — inverse temperature
- `local_energy_only::Bool` — if true, use local energy for conditional; if false, use global energy
"""
struct Gibbs <: AbstractDynamics
    beta::Float64
    local_energy_only::Bool
end

"""Constructor: Gibbs(beta) defaults to local energy only."""
Gibbs(beta::Float64) = Gibbs(beta, true)

"""
    step!(dyn::Gibbs, model::AbstractDiscreteModel, g::AbstractDynGraph; rng) -> Bool

Perform one Gibbs sampling step: pick a random node, sample one component
from its conditional distribution. Returns `true` if the state changed.
"""
function step!(dyn::Gibbs, model::AbstractDiscreteModel, g::AbstractDynGraph;
               rng::AbstractRNG=Random.default_rng())
    N = nv(g)
    i = rand(rng, 1:N)
    return gibbs_sample!(
        model, g, i;
        local_only=dyn.local_energy_only,
        beta=dyn.beta,
        rng=rng
        )
end
