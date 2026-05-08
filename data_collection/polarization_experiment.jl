### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 91d3521e-e9a1-46ef-b121-31023097b286
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()

    include(joinpath(@__DIR__, "..", "src", "HeterophilySynergy.jl"))
	println("HeterophilySynergy\nPS: do not run this cell twice!")
end

# ╔═╡ 7943a7ef-e474-412b-b7d7-1a759b5d4416
begin
	# essentials
	using PlutoUI
	using Distributions
	using Printf
	using Graphs
	using Random
	using LinearAlgebra
	using Combinatorics
	using StatsBase

	# plotting
	using LaTeXStrings
	using CairoMakie
	using GraphMakie
	# using NetworkLayout

	# extra
	using BenchmarkTools
	using ProgressLogging
	using DelimitedFiles
	using TerminalLoggers

	# saving files
	using FileIO
	using JLD2
end

# ╔═╡ 68650560-587f-4f86-9d40-b966b60e2e71
using Logging

# ╔═╡ 0fb40e5c-4ad5-11f1-905e-8fedd15a49b4
md"""
---

## Contact Details

**Author:** Enrico Caprioglio

**Email:** ec627@sussex.ac.uk

---
"""

# ╔═╡ 3331b66c-f4ff-467e-a2fd-54a3ecc54a75
PlutoUI.TableOfContents()

# ╔═╡ 2de4ce6c-03ad-45b2-b911-3e009607e500
const mk = CairoMakie

# ╔═╡ 114fab7e-a45d-47a7-9f86-cfbdd199ceba
md"""
# Disrupting polarization experiment

## Experiment details

We start from a homophilous system (each individual is homophilous) using $\mu_\alpha  = 0.4$.

After $10000$ sweeps (at this point the system will have self-organized into a polarized state) we convert $p\;\%$ individuals from homophilous to heterophilous.

See the paper for more details!
"""

# ╔═╡ 3bb4ea65-79ac-4c6f-991d-9f16d7634af0
function get_alpha(rng, μ_α, σ_α, N)::Vector{Float64}
	α = rand(rng, Normal(μ_α, σ_α), N)
	if maximum(α) > 1
		α = [i > 1 ? 1 : i for i in α]
	end
	if minimum(α) < 0
		α = [i < 0 ? 0 : i for i in α]
	end
	return α
end

# ╔═╡ 9fa3dac8-dee8-4eea-aa1c-749c8a60fe8f
md"""
Example polarization after $10000$ sweeps.

[below we plot the last $1000$ steps (x-axis) and each single opinion (y-axis)]
"""

# ╔═╡ 6b2bf870-1954-4092-9e28-3947f1a7b5d4
let
	seed = 1
	N = 30
	L = 0
	
	rng = MersenneTwister(seed)
		
	# underlying network set-up
	k = 4; ϵ = 0.175;
	if N == 3
		A = BitMatrix(adjacency_matrix(complete_graph(N)))
	else
		A = BitMatrix(adjacency_matrix(watts_strogatz(N, k, ϵ; rng=rng)))
	end
	g = DynGraph(A)
	
	# model
	G = 3
	λ = random_lambda(N, L; rng)
	μ_α = 0.4; σ_α = 0.05;
	α = get_alpha(rng, μ_α, σ_α, N)
	model = AttractionRepulsionSpinGlass(;G, alpha=α, lambda=λ, N)
	
	# dynamics
	β = 5 # 3.4
	dynamics = Metropolis(β)
	
	# Simulation parameters, initial 10_000 steps
	steps = 100_000 * N; save_dt = N; save_state = true
	config = SimConfig(steps, seed, save_dt, save_state)
	
	# simulate
	sim_res = simulate(model, dynamics, g, config)
	
	states = get_states(sim_res.history)
	time_steps = get_times(sim_res.history)

	data = states_to_int_matrix(states)
	shuffled_data = copy(data)

	states_plot = zeros(N*G, length(time_steps))
	for t in 1:length(time_steps)
		for (node, i) in enumerate(1:G:N*G)
		 # 	states_plot[i, t] = states[t][node, 1]
			# states_plot[i+1, t] = states[t][node, 2]
			# states_plot[i+2, t] = states[t][node, 3]
			for j in 1:G
				states_plot[i+(j-1), t] = states[t][node, j]
			end
		end
	end

	mk.heatmap(states_plot[:, end-1000:end]')
	# mk.heatmap(states_plot[:, 1:10:1000]')

	# states[1:100]
	# states_plot[:, 1:100]
end

# ╔═╡ f7aeb22f-5775-4b46-af6f-e240f13b1d4a
md"""
---
"""

# ╔═╡ fce4468d-8735-4d56-99ea-f2bb6cf06ba3
md"""
## Measuring polarization

Since for large $N$ it takes an enormous amount of time to simulate for long enough to get good enough estiamtes of IT quantities, we limit the study of these quantities to relatively small $N=20$.

However, to compute polarization this is not a problem and we can cranck $N$ up.
"""

# ╔═╡ 9188ae47-2de2-4710-8551-1522257e0c12
md"""
Polarization measure from "Why more social interactions lead to more polarization in societies":

let $O_{ij} = 2\frac{\text{\# \{shared opinions\}}}{G}-1$,

then, polarization is defined as $\psi=\text{Var}(O_{ij})$
"""

# ╔═╡ 82bf56b3-a400-4341-b1a0-27b1ab69a14a
function polarization(state)
	N, G = size(state)
	overlaps_list = []
	for i in 1:N
		sᵢ = @view state[i, :]
		for j in i+1:N
			sⱼ = @view state[j, :]
			overlap = 2 * (count(sᵢ .== sⱼ)) / G - 1
			push!(overlaps_list, overlap)
		end
	end

	return var(overlaps_list)
end

# ╔═╡ f7d49264-523f-4819-86ac-d1c17bb88b84
let
	seed = 1
	N = 50
	L = 0
	p = L / N

	println("Using:")
	@show N
	@show p
	
	rng = MersenneTwister(seed)
		
	# underlying network set-up
	k = 4; ϵ = 0.175;
	A = BitMatrix(adjacency_matrix(watts_strogatz(N, k, ϵ; rng=rng)))
	g = DynGraph(A)
	
	# model
	G = 3
	λ = random_lambda(N, L; rng)
	μ_α = 0.4; σ_α = 0.05;
	# α = get_alpha(rng, μ_α, σ_α, N)
	α = 0.4
	model = AttractionRepulsionSpinGlass(;G, alpha=α, lambda=λ, N)
	
	# dynamics
	# β = 3.4
	β = 2.7
	dynamics = Metropolis(β)
	
	# Simulation parameters, initial 10_000 steps
	steps = 100_000 * N; save_dt = N; save_state = true
	config = SimConfig(steps, seed, save_dt, save_state)
	
	# simulate
	sim_res = simulate(model, dynamics, g, config)
	
	states = get_states(sim_res.history)
	time_steps = get_times(sim_res.history)

	data = states_to_int_matrix(states)
	shuffled_data = copy(data)

	S = states[end]
	overlaps_matrix = zeros(N, N)
	overlaps_list = []
	for i in 1:N
		sᵢ = @view S[i, :]
		for j in i+1:N
			# if A[i,j]
				sⱼ = @view S[j, :]
				overlap = 2 * (count(sᵢ .== sⱼ)) / G - 1
				overlaps_matrix[i,j] = overlap
				overlaps_matrix[j,i] = overlaps_matrix[i,j]
				push!(overlaps_list, overlap)
			# end
		end
	end

	@show var(overlaps_list)

	fig = Figure(size  =(300, 300))
	ax = Axis(fig[1,1])
	mk.hist!(ax, overlaps_list)
	fig
end

# ╔═╡ b6ec5ae9-3344-4715-bf1f-68f33d4c0a1c
md"""
## Info-theoretic measures 
"""

# ╔═╡ bfd7127e-e805-45ec-908d-5af07e6f6a09
function transfer_entropy_pairs(Xpast::VecOrMatInt, Ypast::VecOrMatInt, Yfuture::VecOrMatInt)
   return mutual_information(Xpast, Yfuture; Z=Ypast)
end

# ╔═╡ 707a97b6-eb23-4b70-a1f1-a25a22388388
md"""
The function below implements Eq. (3) from:

["Quantifying dynamical high-order interdependencies from the O-information: an application to neural spiking dynamics"](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.595736/full)
"""

# ╔═╡ b109f765-2761-47f0-9ca2-45dfc711b249
function dynamical_o_information_pairs(Xpast::Matrix{Int}, Ypast::Vector{Int}, Yfuture::Vector{Int})
    K = size(Xpast, 2)  # number of source variables
    term1 = (1 - K) * transfer_entropy_pairs(Xpast, Ypast, Yfuture)
    term2 = sum(transfer_entropy_pairs(Xpast[:, setdiff(1:K, j)], Ypast, Yfuture) for j in 1:K)
    return term1 + term2
end

# ╔═╡ 4e5b4b95-4c76-4f79-9234-6a377c353b2b
md"""
The function below implements Eq. (2) from:

["Synergistic Signatures of Group Mechanisms in Higher-Order Systems"](https://doi.org/10.1103/PhysRevLett.134.137401)
"""

# ╔═╡ 4f19f4ec-d6e1-43de-b3b7-fce0ba1eb0fb
function total_dynamical_o_information_pairs(Xt::Matrix{Int}, Xt1::Matrix{Int})
    R, N = size(Xt)
    total = 0.0
    for target in 1:N
        src_idx = setdiff(1:N, target)
        Xpast = Xt[:, src_idx]
        Ypast = Xt[:, target]
        Yfuture = Xt1[:, target]
        total += dynamical_o_information_pairs(Xpast, Ypast, Yfuture)
    end
    return total
end

# ╔═╡ eb83b43e-4c97-480f-b417-f274c5c20dc6
md"""
!!! important
	When collecting data, we immediately compute the difference between the total O-info using the raw data and the total O-info using the shuffled data.
	
	See function below to see how we do this more precisely.
"""

# ╔═╡ 9371b625-09d2-401c-a69e-6095dd4a11f5
function dOinfo3_from_replicas(replicas_t0, replicas_t1, g)

	triangle_list = triangles(g)
	data_t0 = states_to_int_matrix(replicas_t0)
	data_t1 = states_to_int_matrix(replicas_t1)
	
	shuffled_data_t0 = copy(data_t0)
	shuffled_data_t1 = copy(data_t1)
	for j in axes(shuffled_data_t0, 2)
		shuffle!(view(shuffled_data_t0, :, j))
		shuffle!(view(shuffled_data_t1, :, j))
	end

	# compute total dyamical O-information
	dOinfo3 = []
	for (triangle_idx, triangle) in enumerate(triangle_list)
		dOinfo_raw = total_dynamical_o_information_pairs(
			data_t0[:, triangle],
			data_t1[:, triangle]
		)
		dOinfo_shuffle = total_dynamical_o_information_pairs(
			shuffled_data_t0[:, triangle],
			shuffled_data_t1[:, triangle]
		)
		push!(dOinfo3, dOinfo_raw - dOinfo_shuffle)
	end

	return mean(dOinfo3)
end

# ╔═╡ bddc50ac-7f25-47a5-878b-54d9f8f1c29b
md"""
## Simulation functions
"""

# ╔═╡ 4217e388-aa00-4b11-9c5c-f86129906977
function polarization_experiment_simulate(
    model::AbstractModel,
    dynamics::AbstractDynamics,
    g::AbstractDynGraph,
    config::SimConfig
)
    rng = MersenneTwister(config.seed)

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

# ╔═╡ 4a3491ef-0cf0-4046-bfe1-495ab638d996
md"""
!!! note
	Below is the function that does all the work.

	1. It first implements the first step (homophilous simulation for $10000$ sweeps) and saves the polarization at this stage.
	2. Then implements step 2, creating idential replicas after turning $L$ elements from heterophilous to homophilous. From these replicas, it saves the total dynamical O-info at the time of recording of the replicas, as well as the polarization.
"""

# ╔═╡ db70704b-2847-4357-a20b-96051881ec36
function polarization_experiment_replica_collect(N, L, starting_seed; n_replicas=100_000, t_record_step=5, max_t_record=100)

	seed = starting_seed
	rng = MersenneTwister(seed)
	
	# ----- fixed parameters in step 1 ----------
	k = 4; ϵ = 0.175;
	β = 3.4 # 3.4 # 2.7
	G = 3
	μ_α = 0.4; σ_α = 0.05;
	
	# dynamics
	dynamics = Metropolis(β)

	# underlying network set-up
	A = BitMatrix(adjacency_matrix(watts_strogatz(N, k, ϵ; rng=rng)))
	g = DynGraph(A)

	# initial model (all individuals homophilous)
	λ = fill(Int8(-1), N)
	α = get_alpha(rng, μ_α, σ_α, N)
	model_step_1 = AttractionRepulsionSpinGlass(;G, alpha=α, lambda=λ, N)
	
	# ----------------------------------------

	# ----- Begin step 1 ----------
	## Simulate polarization
	# Simulation parameters, initial 10_000 steps
	steps = 10_000 * N; save_dt = N; save_state = true
	config_step_1 = SimConfig(steps, seed, save_dt, save_state)
	
	# simulate
	sim_res_step_1 = simulate(model_step_1, dynamics, g, config_step_1)
	states_step_1 = get_states(sim_res_step_1.history)
	
	# time_steps_po = get_times(sim_res.history)
	last_state_step_1 = states_step_1[end]

	# measure polarization after step 1
	homophily_ψ = mean(polarization.(states_step_1[end-100:end]))
	
	# ----------------------------------------

	# ----- Begin step 2 ----------
	# set up replicas collect
	## add L heterophilous elements
	new_λ = random_lambda(N, L; rng)
	
	# initialize replicas dictionary
	replicas_dict = Dict(
		"t0" => Dict(), # stores snapshots at t_record
		"t1" => Dict(), # stores snapshots at t_record + 1
	)
	T = typeof(last_state_step_1)
	for t_record in 1:t_record_step:max_t_record-1
		replicas_dict["t0"][t_record] = Vector{T}(undef, n_replicas)
		replicas_dict["t1"][t_record] = Vector{T}(undef, n_replicas)
	end

	for r in 1:n_replicas
		seed = starting_seed + r
		# create identical model for step 2, but updated λ
		# model_step_2 = AttractionRepulsionSpinGlass(;G, alpha=α, lambda=new_λ, N)
		model_step_2 = deepcopy(model_step_1)
		# initialize with same state as last state from step 1
		set_state!(model_step_2, g, last_state_step_1)
		model_step_2.lambda = new_λ # update lambda
		# update simulation configuration (decrease no steps)
		config_step_2 = SimConfig(max_t_record * N, seed, save_dt, save_state)
		# simulate new model
		sim_res = polarization_experiment_simulate(model_step_2, dynamics, g, config_step_2)

		# record states
		for t_record in 1:t_record_step:max_t_record-1
			# store snapshot at t0
			replica_t0 = sim_res.history.snapshots[t_record].state
			replicas_dict["t0"][t_record][r] = replica_t0

			# store snapshot at t1
			replica_t1 = sim_res.history.snapshots[t_record + 1].state
			replicas_dict["t1"][t_record][r] = replica_t1
		end
	end

	dOinfo3_at_t_record = zeros(length(1:t_record_step:max_t_record-1))
	perturbation_ψ_at_t_record = zeros(length(1:t_record_step:max_t_record-1))
	
	@progress for (i, t_record) in enumerate(1:t_record_step:max_t_record-1)

		replicas_t0 = replicas_dict["t0"][t_record]
		replicas_t1 = replicas_dict["t1"][t_record]
		
		dOinfo3 = dOinfo3_from_replicas(replicas_t0, replicas_t1, g)
		dOinfo3_at_t_record[i] = dOinfo3

		perturbation_ψ = mean(polarization.(replicas_t0[end-100:end]))
		perturbation_ψ_at_t_record[i] = perturbation_ψ
	end

	@info "last seed: $seed"
	
	return dOinfo3_at_t_record, homophily_ψ, perturbation_ψ_at_t_record
end

# ╔═╡ 23972a22-69b0-4965-a912-52454c18694a
md"""
Note, the function above returns:

- total dynamical o-info $d\Omega_3^{\mathrm{tot}}$: `dOinfo3_at_t_record`
- polarization in step $1$ (first $10000$ sweeps): `homophily_ψ`
- polarization after perturbation at t record: `perturbation_ψ_at_t_record`
"""

# ╔═╡ f6974316-848e-49f9-9e7f-e3a1791fd975
md"""
## Collect data
"""

# ╔═╡ b10f512b-da96-4e61-862d-0c63b914d170
md"""
!!! important
	To replicate our resuts exaclty use:

```julia
starting_seed = 1
N = 20
L_range = [0, 1, 2, 3, 5, 10, 15, 20]

n_replicas=100_000
t_record_step=5
max_t_record=100

# for the for loop use:
starting_seed:100_000:starting_seed+100_000*50
```

While to save data, just change `save_data_folder`
"""

# ╔═╡ 4b344e63-d65f-4403-be09-3c6881b631bd
md"""
!!! warning
	Cell below collects the data. By default it uses small values of `n_replicas` and `max_t_record`.
	This way you can do some of your own testing and then run the full experiment to replicate our results.
	
	**Note:** use at least `n_replicas = 50000` so that the tot dynamical O-info estimation is not too biased (generally, less than 50000 can give a strong synergy bias).
"""

# ╔═╡ a9f9dbc3-3464-4d1e-a821-23c4f58bf91d
let
	println("Cell to collect data")
	
	starting_seed = 1
	N = 20
	L_range = [0, 1, 5, 20] # [0, 1, 2, 3, 5, 10, 15, 20]

	n_replicas=1_000 # 100_000
	t_record_step=5
	max_t_record=10 # 100

	# save_data_folder = "/disrupting_polarization/perturbation_experiment/"

	counter = 1
	@progress for L in L_range
		
		starting_seed = counter
		
		for seed in starting_seed:100_000:starting_seed+100_000*50
			filename = "N$(N)_beta3.4_seed$(seed)_L$(L).jld2"
			
			res = polarization_experiment_replica_collect(
				N, L, seed;
				# optional parameters (n_replicas 100_000 takes a while!)
				n_replicas=n_replicas,
				t_record_step=t_record_step,
				max_t_record=max_t_record
			)
			
			# ----- save data
			
			# save_object(
			# 	save_data_folder * filename,
			# 	res
			# )
			
			counter += 1
		end
	end
end

# ╔═╡ Cell order:
# ╟─0fb40e5c-4ad5-11f1-905e-8fedd15a49b4
# ╟─91d3521e-e9a1-46ef-b121-31023097b286
# ╟─7943a7ef-e474-412b-b7d7-1a759b5d4416
# ╠═3331b66c-f4ff-467e-a2fd-54a3ecc54a75
# ╠═2de4ce6c-03ad-45b2-b911-3e009607e500
# ╠═68650560-587f-4f86-9d40-b966b60e2e71
# ╟─114fab7e-a45d-47a7-9f86-cfbdd199ceba
# ╟─3bb4ea65-79ac-4c6f-991d-9f16d7634af0
# ╟─9fa3dac8-dee8-4eea-aa1c-749c8a60fe8f
# ╟─6b2bf870-1954-4092-9e28-3947f1a7b5d4
# ╟─f7aeb22f-5775-4b46-af6f-e240f13b1d4a
# ╟─fce4468d-8735-4d56-99ea-f2bb6cf06ba3
# ╟─9188ae47-2de2-4710-8551-1522257e0c12
# ╟─82bf56b3-a400-4341-b1a0-27b1ab69a14a
# ╟─f7d49264-523f-4819-86ac-d1c17bb88b84
# ╟─b6ec5ae9-3344-4715-bf1f-68f33d4c0a1c
# ╟─bfd7127e-e805-45ec-908d-5af07e6f6a09
# ╟─707a97b6-eb23-4b70-a1f1-a25a22388388
# ╟─b109f765-2761-47f0-9ca2-45dfc711b249
# ╟─4e5b4b95-4c76-4f79-9234-6a377c353b2b
# ╟─4f19f4ec-d6e1-43de-b3b7-fce0ba1eb0fb
# ╟─eb83b43e-4c97-480f-b417-f274c5c20dc6
# ╟─9371b625-09d2-401c-a69e-6095dd4a11f5
# ╟─bddc50ac-7f25-47a5-878b-54d9f8f1c29b
# ╟─4217e388-aa00-4b11-9c5c-f86129906977
# ╟─4a3491ef-0cf0-4046-bfe1-495ab638d996
# ╟─db70704b-2847-4357-a20b-96051881ec36
# ╟─23972a22-69b0-4965-a912-52454c18694a
# ╟─f6974316-848e-49f9-9e7f-e3a1791fd975
# ╟─b10f512b-da96-4e61-862d-0c63b914d170
# ╟─4b344e63-d65f-4403-be09-3c6881b631bd
# ╟─a9f9dbc3-3464-4d1e-a821-23c4f58bf91d
