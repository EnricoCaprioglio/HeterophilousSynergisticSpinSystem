# ----- Metropolis dynamics -----
# Generic over any AbstractDiscreteModel that implements:
#   local_energy(model, g, i), propose!(model, g, i; rng), revert!(model, g, i, proposal)

"""
    Metropolis <: AbstractDynamics

Metropolis-Hastings dynamics for discrete models.
Proposes a local move, accepts with probability min(1, exp(-β ΔE)).

- `beta::Float64` — inverse temperature
"""
struct Metropolis <: AbstractDynamics
    beta::Float64
end

"""
    step!(dyn::Metropolis, model::AbstractDiscreteModel, g::AbstractDynGraph; rng) -> Bool

Perform one Metropolis step: pick a random node, propose a move, accept/reject.
Returns `true` if the move was accepted.
"""
function step!(dyn::Metropolis, model::AbstractDiscreteModel, g::AbstractDynGraph;
               rng::AbstractRNG=Random.default_rng())
    N = nv(g)
    i = rand(rng, 1:N)

    H_old = local_energy(model, g, i)
    proposal = propose!(model, g, i; rng=rng)
    H_new = local_energy(model, g, i)

    ΔH = H_new - H_old
    accept = rand(rng) < min(1.0, exp(-dyn.beta * ΔH))

    if !accept
        revert!(model, g, i, proposal)
    end
    return accept
end
