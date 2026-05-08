# ----- Simulation runner

"""
    SimConfig

Configuration for a simulation run.

# Fields
- `steps::Int` — total number of MC steps
- `seed::Int` — random seed for reproducibility
- `save_dt::Int` — save a snapshot every `save_dt` steps
- `save_state::Bool` — whether to record state snapshots (default true)
"""
@kwdef struct SimConfig
    steps::Int
    seed::Int = 123
    save_dt::Int
    save_state::Bool = true
end

"""
    SimResult

Output of a simulation run.

# Fields
- `model` — the model after simulation (final state)
- `graph` — the graph used
- `history` — recorded snapshots and observables
- `accept_rate` — fraction of accepted moves
- `config` — the simulation configuration used
"""
struct SimResult
    model::AbstractModel
    graph::AbstractDynGraph
    history::SimHistory
    accept_rate::Float64
    config::SimConfig
end

"""
    simulate(model, dynamics, g, config) -> SimResult

Run a simulation. Initializes the model state, then loops for `config.steps` steps,
recording snapshots at intervals of `config.save_dt`.

The model state is initialized via `init_state!` using the seed from `config`.
"""
function simulate(
    model::AbstractModel,
    dynamics::AbstractDynamics,
    g::AbstractDynGraph,
    config::SimConfig
)
    rng = MersenneTwister(config.seed)

    # initialize state
    init_state!(model, g; rng=rng)

    history = SimHistory()
    accepts = 0

    for t in 1:config.steps
        accepted = step!(dynamics, model, g; rng=rng)
        accepts += accepted ? 1 : 0

        if t % config.save_dt == 0 && config.save_state
            record!(history, t, model, g)
        end
    end

    return SimResult(
        model,
        g,
        history,
        accepts / config.steps,
        config
    )
end
