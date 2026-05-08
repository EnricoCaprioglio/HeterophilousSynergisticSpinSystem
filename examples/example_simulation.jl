# AttractionRepulsionSpinGlass — Basic Example
#
# This tutorial shows how to:
#   1. Create a graph and model
#   2. Run a simulation with Metropolis dynamics
#   3. Run with Gibbs dynamics
#   4. Collect replicas
#   5. Convert states for info-theory analysis

include(joinpath(@__DIR__, "..", "src", "HeterophilySynergy.jl"))

# ---- 1. Create a triangle graph and model ----

N = 3; G = 3
A = falses(N, N)
A[1,2] = A[2,1] = true
A[1,3] = A[3,1] = true
A[2,3] = A[3,2] = true

g = DynGraph(A)
println("Graph: ", nv(g), " nodes, ", ne(g), " edges")

# all homophilous (λ = -1 for all nodes), α = 0.5
lambda = fill(Int8(-1), N)
model = AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda, N=N)

# initialize state manually to inspect it
rng = MersenneTwister(42)
init_state!(model, g; rng=rng)
println("Initial state:\n", model.S)
println("local_energy(1) = ", local_energy(model, g, 1))
println("global_energy   = ", global_energy(model, g))

# ---- 2. Run simulation with Metropolis ----

model_metro = AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda, N=N)
config = SimConfig(steps=10_000, seed=42, save_dt=1000)
result = simulate(model_metro, Metropolis(3.0), g, config)

println("\n--- Metropolis ---")
println("Snapshots: ", length(result.history.snapshots))
println("Accept rate: ", round(result.accept_rate, digits=3))
println("Final state:\n", get_state(result.model, g))

# ---- 3. Run simulation with Gibbs ----

model_gibbs = AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda, N=N)
result_gibbs = simulate(model_gibbs, Gibbs(3.0), g, config)

println("\n--- Gibbs (local energy) ---")
println("Snapshots: ", length(result_gibbs.history.snapshots))
println("Accept rate: ", round(result_gibbs.accept_rate, digits=3))

# Gibbs with global energy
result_gibbs_global = simulate(
    AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda, N=N),
    Gibbs(3.0, false),  # false = use global energy
    g,
    SimConfig(steps=10_000, seed=42, save_dt=1000)
)
println("\n--- Gibbs (global energy) ---")
println("Accept rate: ", round(result_gibbs_global.accept_rate, digits=3))

# ---- 4. Collect replicas ----

model_fn = () -> AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda, N=N)

sim_config = SimConfig(steps=5000, save_dt=500)
rconfig = ReplicaConfig(
    n_replicas=100,
    starting_seed=1,
    t_record=10,          # last snapshot (5000 / 500 = 10 snapshots)
    sim_config=sim_config
)

replicas = collect_replicas(model_fn, Metropolis(3.0), g, rconfig)
println("\n--- Replicas ---")
println("Collected: ", length(replicas))
println("Each replica shape: ", size(replicas[1]))

# ---- 5. Convert states for analysis ----

# convert replicas to integer-encoded matrix (n_replicas × N)
int_matrix = states_to_int_matrix(replicas)
println("\nInteger-encoded replica matrix shape: ", size(int_matrix))
println("First 5 replicas:\n", int_matrix[1:5, :])

# single state conversion round-trip
state = replicas[1]
int_state = state_convert(state, Vector{Int})
back = state_convert(int_state, Matrix{Int8}; G=G)
println("\nRound-trip conversion OK: ", back == state)

# ---- 6. Heterogeneous model ----

# mix of homophilous and heterophilous nodes
lambda_mixed = random_lambda(N, 1; rng=MersenneTwister(99))  # 1 heterophilous node
println("\n--- Heterogeneous λ ---")
println("lambda = ", lambda_mixed)

model_het = AttractionRepulsionSpinGlass(G=G, alpha=0.5, lambda=lambda_mixed, N=N)
result_het = simulate(model_het, Metropolis(3.0), g, config)
println("Accept rate: ", round(result_het.accept_rate, digits=3))

# per-node alpha
alpha_vec = [0.3, 0.5, 0.8]
model_alpha = AttractionRepulsionSpinGlass(G=G, alpha=alpha_vec, lambda=lambda, N=N)
result_alpha = simulate(model_alpha, Metropolis(3.0), g, config)
println("\n--- Heterogeneous α ---")
println("alpha = ", alpha_vec)
println("Accept rate: ", round(result_alpha.accept_rate, digits=3))

# stress test replicas
N = 3; G = 3;
model_stress = () -> AttractionRepulsionSpinGlass(G=G, alpha=0.4, lambda=fill(Int8(1), N), N=N)

sim_config = SimConfig(steps=1000, save_dt=N)
rconfig = ReplicaConfig(
    n_replicas=10_000,
    starting_seed=1,
    t_record=10,
    sim_config=sim_config
)

replicas = collect_replicas(model_stress, Metropolis(10.0), g, rconfig)
println("\n--- Replicas ---")
println("Collected: ", length(replicas))
println("Each replica shape: ", size(replicas[1]))

data = states_to_int_matrix(replicas)
ho_summary(data)