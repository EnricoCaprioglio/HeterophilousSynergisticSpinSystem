# ---- Abstract type hierarchy ----
# Basically here we define all the abstract types to keep things in order

"""
    AbstractDynGraph

Abstract type for all graph structures.
"""
abstract type AbstractDynGraph end

"""
    AbstractModel

Abstract type for all models. 
A model defines the energy function or update rule for a given system.
"""
abstract type AbstractModel end

"""
    AbstractDiscreteModel <: AbstractModel

Models with discrete state spaces.
These typically evolve via Glauber or Metropolis dynamics.

# Required methods

- `init_state!(model, g::AbstractDynGraph; rng)` — initialize random state on the graph
- `get_state(model, g::AbstractDynGraph)` — return a copy of the current state
- `set_state!(model, g::AbstractDynGraph, state)` — restore a previously saved state
- `local_energy(model, g::AbstractDynGraph, i::Int)` — energy contribution of node i (temperature-independent)
- `global_energy(model, g::AbstractDynGraph)` — total energy (sum of local energies)
- `propose!(model, g::AbstractDynGraph, i::Int; rng)` — propose and apply a local move at node i, return an undo token
- `revert!(model, g::AbstractDynGraph, i::Int, proposal)` — undo a proposed move using the undo token

# Optional methods

- `gibbs_sample!(model, g::AbstractDynGraph, i::Int; local_only::Bool, beta::Float64, rng)` — sample one component from its conditional distribution
"""
abstract type AbstractDiscreteModel <: AbstractModel end

"""
    AbstractContinuousModel <: AbstractModel

Models with continuous state spaces.
These typically evolve via numerical integration.

# Required methods

- `init_state!(model, g::AbstractDynGraph; rng)` — initialize random state
- `get_state(model, g::AbstractDynGraph)` — return a copy of the current state
- `set_state!(model, g::AbstractDynGraph, state)` — restore a previously saved state
- `dstate_dt(model, g::AbstractDynGraph, i::Int)` — time derivative of node i's state
"""
abstract type AbstractContinuousModel <: AbstractModel end

"""
    AbstractDynamics

Abstract type for dynamics / integration schemes.
Dynamics must carry the inverse temperature `beta` (for discrete models) or the time step `dt` (for continuous models).

# Required methods

- `step!(dynamics, model::AbstractModel, g::AbstractDynGraph; rng) -> Bool` — perform one update step, return whether the state changed
"""
abstract type AbstractDynamics end