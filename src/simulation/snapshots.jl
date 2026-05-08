# ----- Snapshot and history system

"""
    Snapshot{T}

A recorded state at a given time step.

# Fields
- `t::Int` — the time step at which this snapshot was taken
- `state::T` — a copy of the model state
"""
struct Snapshot{T}
    t::Int
    state::T
end

"""
    SimHistory

Collects state snapshots and named observables during a simulation run.

# Fields
- `snapshots::Vector{Snapshot}` — recorded states at specific time steps
- `observables::Dict{Symbol, Vector}` — named time series of observables
"""
mutable struct SimHistory
    snapshots::Vector{Snapshot}
    observables::Dict{Symbol, Vector}
end

SimHistory() = SimHistory(Snapshot[], Dict{Symbol, Vector}())

"""Record a snapshot of the current model state."""
function record!(hist::SimHistory, t::Int, model::AbstractModel, g::AbstractDynGraph)
    state = get_state(model, g)
    push!(hist.snapshots, Snapshot(t, state))
    return hist
end

"""Record a named observable value."""
function record_observable!(hist::SimHistory, name::Symbol, value)
    if !haskey(hist.observables, name)
        hist.observables[name] = Any[]
    end
    push!(hist.observables[name], value)
    return hist
end

"""Extract all states from history as a vector."""
get_states(hist::SimHistory) = [s.state for s in hist.snapshots]

"""Extract all time steps from history."""
get_times(hist::SimHistory) = [s.t for s in hist.snapshots]
