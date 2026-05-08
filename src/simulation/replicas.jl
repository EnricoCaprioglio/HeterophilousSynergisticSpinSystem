# ----- Replica collection

"""
    ReplicaConfig

Configuration for collecting replicas (multiple independent runs of the same system).

# Fields
- `n_replicas::Int` — number of independent replicas to run
- `starting_seed::Int` — seed for the first replica (increments by 1 per replica)
- `t_record::Int` — which snapshot index to extract from each replica's history
- `sim_config::SimConfig` — base simulation config (seed will be overridden per replica)
"""
@kwdef struct ReplicaConfig
    n_replicas::Int
    starting_seed::Int = 1
    t_record::Int
    sim_config::SimConfig
end

"""
    collect_replicas(model_fn, dynamics, g, rconfig) -> Vector

Run `n_replicas` independent simulations, each with a different seed but the same
graph and parameters. Extract the snapshot at index `t_record` from each.

`model_fn` is a zero-argument function that creates a fresh model instance,
so each replica starts with independent random initialization.

# Example
```julia
model_fn = () -> AttractionRepulsionSpinGlass(G=3, alpha=0.5, lambda=fill(Int8(-1), N), N=N)
rconfig = ReplicaConfig(n_replicas=100, starting_seed=1, t_record=50, sim_config=sim_config)
replicas = collect_replicas(model_fn, Metropolis(3.0), g, rconfig)
```
"""
function collect_replicas(
    model_fn::Function,
    dynamics::AbstractDynamics,
    g::AbstractDynGraph,
    rconfig::ReplicaConfig
)
    # run first replica to determine state type
    first_config = SimConfig(
        steps = rconfig.sim_config.steps,
        seed = rconfig.starting_seed,
        save_dt = rconfig.sim_config.save_dt,
        save_state = true
    )
    first_model = model_fn()
    first_result = simulate(first_model, dynamics, g, first_config)
    @assert rconfig.t_record <= length(first_result.history.snapshots) "t_record=$(rconfig.t_record) but only $(length(first_result.history.snapshots)) snapshots were recorded"
    first_state = first_result.history.snapshots[rconfig.t_record].state

    # allocate typed vector
    T = typeof(first_state)
    replicas = Vector{T}(undef, rconfig.n_replicas)
    replicas[1] = first_state

    # run remaining replicas
    for r in 2:rconfig.n_replicas
        seed = rconfig.starting_seed + r - 1
        model = model_fn()
        config = SimConfig(
            steps = rconfig.sim_config.steps,
            seed = seed,
            save_dt = rconfig.sim_config.save_dt,
            save_state = true
        )
        result = simulate(model, dynamics, g, config)
        replicas[r] = result.history.snapshots[rconfig.t_record].state
    end

    return replicas
end
