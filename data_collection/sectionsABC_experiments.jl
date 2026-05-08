### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 4605d981-e601-4e31-bac9-a74b694e957a
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()

    include(joinpath(@__DIR__, "..", "src", "HeterophilySynergy.jl"))
	println("HeterophilySynergy\nPS: do not run this cell twice!")
end

# ╔═╡ 4dfd7819-1848-4937-97a1-cdd087f45452
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
	# using LaTeXStrings
	# using CairoMakie
	# using GraphMakie
	# using NetworkLayout

	# extra
	using BenchmarkTools
	using ProgressLogging
	using DelimitedFiles
	using TerminalLoggers
	using Logging

	# saving files
	using FileIO
	using JLD2
end

# ╔═╡ ac04e022-4adc-11f1-a3ba-bdce3c59a914
md"""
---

## Contact Details

**Author:** Enrico Caprioglio

**Email:** ec627@sussex.ac.uk

---
"""

# ╔═╡ 1ee3edd2-e769-4cba-9390-d5c7bba69bc1
PlutoUI.TableOfContents()

# ╔═╡ 0c57ae47-ffdc-44a6-a4d1-ebd36ee5aef1
md"""
## Symmetric Systems (Section B)
"""

# ╔═╡ 33cdec0a-a39f-4d39-bbcc-aab6134936dc
md"""
### Varying temperature

Filename convention:

```julia
"N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_alpha$(c.α)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_beta_0_1_40.jld2"

if c.node_type == heterophilous
	filename = "heterophilous_" * filename
elseif c.node_type == homophilous
	filename = "homophilous_" * filename
end
```

since we are sweeping through:
```julia
β_range = 0.0:1.0:40.0
```
"""

# ╔═╡ 9ce111bb-f365-4f5c-b04f-244706afa931
md"""
To keep things clean, here we set up the experiment using a custom struct:

```julia
Base.@kwdef struct BetaSweepExperiment{R<:AbstractRange}
	# sweep
	β_range::R = 0.0:1.0:40.0
	no_tests::Int = 10
	starting_seed::Int = 1

	# select homophilous or heterophilous
	node_type::NodeType = heterophilous

	# network
	N::Int = 10
	G::Int = 3
	k::Int = 5
	ϵ::Float64 = 0.2

	# model
	α::Float64 = 0.4

	# dynamics
	local_energy_only::Bool = false

	# replica params
	n_replicas::Int = 10_000
	t_record::Int = 100
end

λvec(config::BetaSweepExperiment) =
fill(Int8(config.node_type == heterophilous ? 1 : -1), config.N)
```
"""

# ╔═╡ 34409a83-c871-4702-ac2a-748d00f849e9
begin
	println("This cell defines the BetaSweepExperiment struct")
	
	@enum NodeType homophilous heterophilous

	Base.@kwdef struct BetaSweepExperiment{R<:AbstractRange}
	    # sweep
	    β_range::R = 0.0:1.0:40.0
	    no_tests::Int = 10
		starting_seed::Int = 1
	
	    # case
	    node_type::NodeType = heterophilous
	
	    # network
	    N::Int = 10
	    G::Int = 3
	    k::Int = 5
	    ϵ::Float64 = 0.2
	
	    # model
	    α::Float64 = 0.4
	
	    # dynamics
	    local_energy_only::Bool = false
	
	    # replica params
	    n_replicas::Int = 10_000
	    t_record::Int = 100
	end

	λvec(config::BetaSweepExperiment) =
    fill(Int8(config.node_type == heterophilous ? 1 : -1), config.N)

	nothing
end

# ╔═╡ 0885bac4-eda5-4f60-ba30-d985f1bc7915
function mk_filename_varying_beta(config)

	c = config
	filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_alpha$(c.α)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_beta_0_1_40.jld2"
	
	if c.node_type == heterophilous
		filename = "heterophilous_" * filename
	elseif c.node_type == homophilous
		filename = "homophilous_" * filename
	end
	
	return filename
end

# ╔═╡ 71a64eeb-f553-49a1-a205-1b74078eaa9b
md"""
Take a look at the output in the cell above.

For each key in the dictionary (`"mean_u1", "mean_u2", "mean_Oinfo3"`), we have another dictionary for each value of $\beta$ used in the experiment.

For each value of $\beta$, we then have the results for each replica ensemble.
"""

# ╔═╡ b31980c4-1026-4b2f-8d18-2455e4e423f0
md"""
!!! warning "Attention:"
	Cell below was used to collect data.

	When using global update schemes it takes a while to run.
"""

# ╔═╡ 8667df8a-f0ec-4dfd-afe1-2c72a13d7e10
let
	println("Collect data cell, uncomment to run:\n")

	# # ----- set experiment parameters
	# c = BetaSweepExperiment(;
	# 	β_range = 0.0:1.0:40.0,
	# 	no_tests = 10,
	# 	starting_seed = 311_100,
	# 	node_type = homophilous,
	# 	N = 30,
	# 	G = 3,
	# 	k = 5,
	# 	ϵ = 0.2,
	# 	α = 0.4,
	# 	local_energy_only = false,
	# 	n_replicas = 10_000,
	# 	t_record = 50
	# )

	# λ = λvec(c)

	# # create filename
	# filename = mk_filename_varying_beta(c)
	# @show filename

	# # ------------------------------

	# store_results = Dict(
	# 	"mean_Oinfo3" => Dict{Real, Vector}(),
	# 	"mean_u1" => Dict{Real, Vector}(),
	# 	"mean_u2" => Dict{Real, Vector}(),
	# )
	
	# start_t = time()
	# seed = c.starting_seed
	
	# @inbounds @progress for (j, β) in enumerate(c.β_range)
		
	# 	store_results["mean_Oinfo3"][β] = zeros(c.no_tests)
	# 	store_results["mean_u1"][β] = zeros(c.no_tests)
	# 	store_results["mean_u2"][β] = zeros(c.no_tests)

	# 	for test in 1:c.no_tests

	# 		rng = MersenneTwister(seed)
			
	# 		# create graph using Graphs.jl
	# 		graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
	# 		A = BitMatrix(adjacency_matrix(graph))
		
	# 		# create composite type DynGraph
	# 		g = DynGraph(A)
		
	# 		# define model
	# 		model_stress = () -> AttractionRepulsionSpinGlass(
	# 			G=c.G, 
	# 			alpha=c.α,
	# 			lambda=λ,
	# 			N=c.N
	# 		)
		
	# 		# select dynamics params
	# 		dynamics = Gibbs(β, c.local_energy_only)
			
	# 		sim_config = SimConfig(
	# 			steps=c.t_record * c.N,
	# 			seed = seed,
	# 			save_dt = c.N,
	# 			save_state = true
	# 		)
		
	# 		rconfig = ReplicaConfig(
	# 		    n_replicas = c.n_replicas,
	# 		    starting_seed = seed,
	# 		    t_record = c.t_record,
	# 		    sim_config = sim_config
	# 		)
			
	# 		replicas = collect_replicas(model_stress, dynamics, g, rconfig)
	# 		data = states_to_int_matrix(replicas)

	# 		# compute IT quantities
	# 		ensemble_res = Oinfo3_components(g, data)

	# 		# store IT quantities
	# 		store_results["mean_Oinfo3"][β][test] = mean(ensemble_res.Ω)
	# 		store_results["mean_u1"][β][test] = mean(ensemble_res.u1)
	# 		store_results["mean_u2"][β][test]  = mean(ensemble_res.u2)
		
	# 		seed += c.n_replicas
	# 	end
	# end

	# println("time: $(time() - start_t)\n")
	# println("Final seed: ", seed)

	## uncomment to save results
	# folderpath = "varying_beta/"
	
	# save_object(
	# 	folderpath * filename,
	# 	(store_results=store_results, config=c)
	# )

	# store_results
end

# ╔═╡ 88482d55-665d-4bb4-b4fe-3d0414c79348
md"""
### Varying $\alpha$
"""

# ╔═╡ 4fdbd631-dc2c-43f1-88aa-b86b1cfdc4f3
md"""
Here, we do the same thing as above, define file convention and a struct `AlphaSweepExperiment` for the experiment.
"""

# ╔═╡ fa99dee7-b96c-452f-bad7-84d9a837a69f
md"""
Filename convention here:
```julia
filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_alpha.jld2"

if c.node_type == heterophilous
	filename = "heterophilous_" * filename
elseif c.node_type == homophilous
	filename = "homophilous_" * filename
end
```
"""

# ╔═╡ 2449509c-59a0-4ddc-b4ee-d0d26d0a0815
function mk_filename_varying_alpha(config)

	c = config
	filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_alpha.jld2"
	
	if c.node_type == heterophilous
		filename = "heterophilous_" * filename
	elseif c.node_type == homophilous
		filename = "homophilous_" * filename
	end
	
	return filename
end

# ╔═╡ d9757572-566f-4d38-b3a0-02ab88f76029
begin
	println("This cell defines the AlphaSweepExperiment struct")
	
	Base.@kwdef struct AlphaSweepExperiment{R<:AbstractRange}
		# sweep
		α_range::R = 0.0:1.0:40.0
		no_tests::Int = 10
		starting_seed::Int = 1
	
		# select homophilous or heterophilous
		node_type::NodeType = heterophilous
	
		# network
		N::Int = 10
		G::Int = 3
		k::Int = 5
		ϵ::Float64 = 0.2
	
		# model
		β::Float64 = 20.0
	
		# dynamics
		local_energy_only::Bool = true
	
		# replica params
		n_replicas::Int = 10_000
		t_record::Int = 100
	end
	
	λvec(config::AlphaSweepExperiment) =
	fill(Int8(config.node_type == heterophilous ? 1 : -1), config.N)

	nothing
end

# ╔═╡ 916b674a-a9bf-4b0a-815c-6daf58aa6f61
let
	println("Just an example usage (no save and using too few replicas):\n")

	# ----- set experiment parameters
	c = BetaSweepExperiment(;
		β_range = 0.0:5.0:40.0,
		no_tests = 2,
		starting_seed = 10_000_000,
		node_type = heterophilous,
		N = 30,
		G = 3,
		k = 5,
		ϵ = 0.2,
		α = 0.4,
		local_energy_only = true,
		n_replicas = 1_000,
		t_record = 50
	)

	λ = λvec(c)

	# create filename
	filename = mk_filename_varying_beta(c)
	@show filename

	# ------------------------------

	store_results = Dict(
		"mean_Oinfo3" => Dict{Real, Vector}(),
		"mean_u1" => Dict{Real, Vector}(),
		"mean_u2" => Dict{Real, Vector}(),
	)
	
	start_t = time()
	seed = c.starting_seed
	
	@inbounds @progress for (j, β) in enumerate(c.β_range)
		
		store_results["mean_Oinfo3"][β] = zeros(c.no_tests)
		store_results["mean_u1"][β] = zeros(c.no_tests)
		store_results["mean_u2"][β] = zeros(c.no_tests)

		for test in 1:c.no_tests

			rng = MersenneTwister(seed)
			
			# create graph using Graphs.jl
			graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
			A = BitMatrix(adjacency_matrix(graph))
		
			# create composite type DynGraph
			g = DynGraph(A)
		
			# define model
			model = () -> AttractionRepulsionSpinGlass(
				G=c.G, 
				alpha=c.α,
				lambda=λ,
				N=c.N
			)
		
			# select dynamics params
			dynamics = Gibbs(β, c.local_energy_only)
			
			sim_config = SimConfig(
				steps=c.t_record * c.N,
				seed = seed,
				save_dt = c.N,
				save_state = true
			)
		
			rconfig = ReplicaConfig(
			    n_replicas = c.n_replicas,
			    starting_seed = seed,
			    t_record = c.t_record,
			    sim_config = sim_config
			)
			
			replicas = collect_replicas(model, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)

			# compute IT quantities
			ensemble_res = Oinfo3_components(g, data)

			# store IT quantities
			store_results["mean_Oinfo3"][β][test] = mean(ensemble_res.Ω)
			store_results["mean_u1"][β][test] = mean(ensemble_res.u1)
			store_results["mean_u2"][β][test]  = mean(ensemble_res.u2)
		
			seed += c.n_replicas
		end
	end

	println("time: $(time() - start_t)\n")
	println("Final seed: ", seed)

	store_results
end

# ╔═╡ adcd82ca-f7d7-409e-a7c6-40e7cc63b2b9
let
	println("Collect data cell, uncomment to run:\n")

	# for N in [10, 20, 30, 40, 50]
	# 	# ----- set experiment parameters
	# 	c = AlphaSweepExperiment(;
	# 		α_range = 0.0:0.05:1.0,
	# 		no_tests = 10,
	# 		starting_seed = 1,
	# 		node_type = homophilous,
	# 		N = N,
	# 		G = 3,
	# 		k = 5,
	# 		ϵ = 0.2,
	# 		β = 20.0,
	# 		local_energy_only = false,
	# 		n_replicas = 10_000,
	# 		t_record = 100
	# 	)
	
	# 	λ = λvec(c)
	
	# 	# create filename
	# 	filename = mk_filename_varying_alpha(c)
	# 	@show filename
	
	# 	# ------------------------------
	
	# 	store_results = Dict(
	# 		"mean_Oinfo3" => Dict{Real, Vector}(),
	# 		"mean_u1" => Dict{Real, Vector}(),
	# 		"mean_u2" => Dict{Real, Vector}(),
	# 	)
		
	# 	start_t = time()
	# 	seed = c.starting_seed
		
	# 	@inbounds @progress for (j, α) in enumerate(c.α_range)
			
	# 		store_results["mean_Oinfo3"][α] = zeros(c.no_tests)
	# 		store_results["mean_u1"][α] = zeros(c.no_tests)
	# 		store_results["mean_u2"][α] = zeros(c.no_tests)
	
	# 		for test in 1:c.no_tests
	
	# 			rng = MersenneTwister(seed)
				
	# 			# create graph using Graphs.jl
	# 			graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
	# 			A = BitMatrix(adjacency_matrix(graph))
			
	# 			# create composite type DynGraph
	# 			g = DynGraph(A)
			
	# 			# define model
	# 			model_stress = () -> AttractionRepulsionSpinGlass(
	# 				G=c.G, 
	# 				alpha=α,
	# 				lambda=λ,
	# 				N=c.N
	# 			)
			
	# 			# select dynamics params
	# 			dynamics = Gibbs(c.β, c.local_energy_only)
				
	# 			sim_config = SimConfig(
	# 				steps=c.t_record * c.N,
	# 				seed = seed,
	# 				save_dt = c.N,
	# 				save_state = true
	# 			)
			
	# 			rconfig = ReplicaConfig(
	# 			    n_replicas = c.n_replicas,
	# 			    starting_seed = seed,
	# 			    t_record = c.t_record,
	# 			    sim_config = sim_config
	# 			)
				
	# 			replicas = collect_replicas(model_stress, dynamics, g, rconfig)
	# 			data = states_to_int_matrix(replicas)
	
	# 			# compute IT quantities
	# 			ensemble_res = Oinfo3_components(g, data)
	
	# 			# store IT quantities
	# 			store_results["mean_Oinfo3"][α][test] = mean(ensemble_res.Ω)
	# 			store_results["mean_u1"][α][test] = mean(ensemble_res.u1)
	# 			store_results["mean_u2"][α][test]  = mean(ensemble_res.u2)
			
	# 			seed += c.n_replicas
	# 		end
	# 	end
	
	# 	println("time: $(time() - start_t)\n")
	# 	println("Final seed: ", seed)
	
	# 	## uncomment to save results
	# 	folderpath = "/Users/ec627/Documents/Data/self-organized-synergy/grappa_analysis/varying_alpha/"
		
	# 	save_object(
	# 		folderpath * filename,
	# 		(store_results=store_results, config=c)
	# 	)
	
	# 	# store_results
	# end
end

# ╔═╡ bca285dd-b4dd-4734-b4cb-851808127a0a
md"""
### Varying `t_record`
"""

# ╔═╡ 11cabaa7-ddce-4060-a6b8-2ffd539df412
md"""
Filename convention here:
```julia
filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_beta$(c.β)_alpha$(c.α)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_t_record.jld2"

if c.node_type == heterophilous
	filename = "heterophilous_" * filename
elseif c.node_type == homophilous
	filename = "homophilous_" * filename
end

```
"""

# ╔═╡ f9c497e9-dec7-464f-bf1f-121adbd00f7c
function mk_filename_varying_t_record(config)

	c = config
	filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_beta$(c.β)_alpha$(c.α)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_t_record.jld2"
	
	if c.node_type == heterophilous
		filename = "heterophilous_" * filename
	elseif c.node_type == homophilous
		filename = "homophilous_" * filename
	end
	
	return filename
end

# ╔═╡ 3d660037-c5dd-4428-9436-2a0f56b632b3
let
	println("Collect data cell, uncomment to run:\n")

	## ----- set experiment parameters
	# c = tRecordSweepExperiment(;
	# 	t_record_range = collect(1:1:50),
	# 	# append!(collect(1:1:50), collect(100:50:1000))
	# 	no_tests = 5,
	# 	starting_seed = 5_110_000,
	# 	node_type = homophilous, # heterophilous,
	# 	N = 30,
	# 	G = 3,
	# 	k = 5,
	# 	ϵ = 0.2,
	# 	β = 20.0,
	# 	α = 0.4,
	# 	local_energy_only = false,
	# 	n_replicas = 10_000,
	# )

	# λ = λvec(c)

	# # create filename
	# filename = mk_filename_varying_t_record(c)
	# @show filename

	# # ------------------------------

	# store_results = Dict(
	# 	"mean_Oinfo3" => Dict{Real, Vector}(),
	# 	"mean_u1" => Dict{Real, Vector}(),
	# 	"mean_u2" => Dict{Real, Vector}(),
	# )
	
	# start_t = time()
	# seed = c.starting_seed
	
	# @inbounds @progress for (j, t_record) in enumerate(c.t_record_range)
		
	# 	store_results["mean_Oinfo3"][t_record] = zeros(c.no_tests)
	# 	store_results["mean_u1"][t_record] = zeros(c.no_tests)
	# 	store_results["mean_u2"][t_record] = zeros(c.no_tests)

	# 	for test in 1:c.no_tests

	# 		rng = MersenneTwister(seed)
			
	# 		# create graph using Graphs.jl
	# 		graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
	# 		A = BitMatrix(adjacency_matrix(graph))
		
	# 		# create composite type DynGraph
	# 		g = DynGraph(A)
		
	# 		# define model
	# 		model_stress = () -> AttractionRepulsionSpinGlass(
	# 			G=c.G, 
	# 			alpha=c.α,
	# 			lambda=λ,
	# 			N=c.N
	# 		)
		
	# 		# select dynamics params
	# 		dynamics = Gibbs(c.β, c.local_energy_only)
			
	# 		sim_config = SimConfig(
	# 			steps = t_record * c.N,
	# 			seed = seed,
	# 			save_dt = c.N,
	# 			save_state = true
	# 		)
		
	# 		rconfig = ReplicaConfig(
	# 		    n_replicas = c.n_replicas,
	# 		    starting_seed = seed,
	# 		    t_record = t_record,
	# 		    sim_config = sim_config
	# 		)
			
	# 		replicas = collect_replicas(model_stress, dynamics, g, rconfig)
	# 		data = states_to_int_matrix(replicas)

	# 		# compute IT quantities
	# 		ensemble_res = Oinfo3_components(g, data)

	# 		# store IT quantities
	# 		store_results["mean_Oinfo3"][t_record][test] = mean(ensemble_res.Ω)
	# 		store_results["mean_u1"][t_record][test] = mean(ensemble_res.u1)
	# 		store_results["mean_u2"][t_record][test]  = mean(ensemble_res.u2)
		
	# 		seed += c.n_replicas
	# 	end
	# end

	# println("time: $(time() - start_t)\n")
	# println("Final seed: ", seed)

	# ## uncomment to save results
	# folderpath = "/Users/ec627/Documents/Data/self-organized-synergy/grappa_analysis/varying_t_record/"
	
	# # save_object(
	# # 	folderpath * filename,
	# # 	(store_results=store_results, config=c)
	# # )

	# store_results
end

# ╔═╡ cb61b3c9-001f-4ab7-94ca-0b1d4d643a0f
md"""
## Asymmetric systems (section C)
"""

# ╔═╡ 36759ea7-595d-4b31-8a8c-7665daebdffb
md"""
### Triangle Analysis

Here we collect replicas to study the coupling structure.

!!! note
	For the triangle analysis we don't require as many replicas.
	Hence why this is separate from the robustness checks of IT quantities (which follows right below)
"""

# ╔═╡ 1397d0e3-3f5e-485a-a005-ba158b524f84
md"""
Sweep number of heterophilous elements.
"""

# ╔═╡ f7514b7d-1b3a-49d5-b9f4-1e98c115ee25
md"""
Filename convention here:
```julia
filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_meanalpha$(c.μ_α)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_triangle_analysis_varying_L.jld2"
```
"""

# ╔═╡ fefe018c-641f-4f65-93a4-20d72a0bc21d
function mk_filename_varying_L(config)
	c = config
	filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_meanalpha$(c.μ_α)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_triangle_analysis_varying_L.jld2"
	
	return filename
end

# ╔═╡ 260ba10c-2b56-43f1-9afd-533588e02275
begin
	println("This cell defines the noHeterophilousSweepExperiment struct")

	Base.@kwdef struct noHeterophilousSweepExperiment
		
	    no_tests::Int = 10
		starting_seed::Int = 1
	
	    # network
	    N::Int = 10
	    G::Int = 3
	    k::Int = 5
	    ϵ::Float64 = 0.2
	
	    # model
	    μ_α::Float64 = 0.4
		σ_α::Float64 = 0.05
		β::Float64 = 20.0
	
	    # dynamics
	    local_energy_only::Bool = true
	
	    # replica params
	    n_replicas::Int = 10_000
	    t_record::Int = 100
	end

	Lvec(config::noHeterophilousSweepExperiment) = 0:1:config.N

	nothing
end

# ╔═╡ cd2ae50a-ee35-4821-a62d-fc350af567bd
md"""
Note, we require a couple of extra functions here not included in `HeterophilySynergy.jl`.
"""

# ╔═╡ 267512f6-98ae-4cce-a944-4f72c692bb11
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

# ╔═╡ 43331cc3-70aa-4570-8b2e-ad5754c4be0d
function couplings_matrix(S::Matrix{Int8}, A::BitMatrix)
	
	N, G = size(S)
	J = zeros(N, N)
	
	for i in 1:N
		sᵢ = @view S[i, :]
		for j in i+1:N
			if A[i,j]
				sⱼ = @view S[j, :]
				J[i,j] = sign(dot(sᵢ, sⱼ))
				J[j,i] = J[i,j]
			end
		end
	end

	return J
end

# ╔═╡ d5221050-387a-4d59-a7f2-64575d7fad3d
function get_nₖ_across_replicas(data, g::AbstractDynGraph)

	triangles_list = triangles(g)
	no_trianlges = length(triangles_list)
	
	nₖ_counts = Dict(
		0 => [],
		1 => [],
		2 => [],
		3 => []
	)
	
	for step in 1:size(data, 1)
		S = state_convert(data[step, :], Matrix{Int8}; G=3)
		J = couplings_matrix(S, g.A)
		Δₖ = []
		for (i,j,k) in triangles_list
			 push!(Δₖ, count(x -> x < 0, [J[i,j], J[i,k], J[j,k]]))
		end
		for k in 0:3
			push!(nₖ_counts[k], count(x -> x == k, Δₖ) / no_trianlges)
		end
	end

	return nₖ_counts
end

# ╔═╡ 37180725-f017-4c5d-8b69-19943a094938
let
	println("Collect data cell:\n")

	# ----- set experiment parameters
	c = noHeterophilousSweepExperiment(;
		no_tests = 100,
		starting_seed = 1,
		
		N = 50,
		G = 3,
		k = 5,
		ϵ = 0.2,
	
		μ_α = 0.75,
		β = 10.0,
	
		local_energy_only = true,
		n_replicas = 10, # 10_000,
		t_record = 100
	)

	L_range = Lvec(c)

	# create filename
	filename = mk_filename_varying_L(c)
	@show filename

	# ------------------------------

	store_results = Dict{Int, Dict}()
	
	# start_t = time()
	seed = c.starting_seed
	
	@inbounds @progress for (j, L) in enumerate(L_range)

		store_results[L] = Dict(
			"mean_n₀" => [],
			"mean_n₁" => [],
			"mean_n₂" => [],
			"mean_n₃" => [],
			# "std_n₀" => [],
			# "std_n₁" => [],
			# "std_n₂" => [],
			# "std_n₃" => [],
		)

		for test in 1:c.no_tests
	
			rng = MersenneTwister(seed)
			
			# create graph using Graphs.jl
			graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
			A = BitMatrix(adjacency_matrix(graph))
			# store_results["A"][L][test] = A
		
			# create composite type DynGraph
			g = DynGraph(A)
		
			# define model
			λ = random_lambda(c.N, L; rng = rng)
			# store_results["λ"][L][test] = λ
			α_vec = get_alpha(rng, c.μ_α, c.σ_α, c.N)
			
			model_fn = () -> AttractionRepulsionSpinGlass(
				G=c.G, 
				alpha=α_vec,
				lambda=λ,
				N=c.N
			)
		
			# select dynamics params
			dynamics = Gibbs(c.β, c.local_energy_only)
			
			sim_config = SimConfig(
				steps=c.t_record * c.N,
				seed = seed,
				save_dt = c.N,
				save_state = true
			)
		
			rconfig = ReplicaConfig(
			    n_replicas = c.n_replicas,
			    starting_seed = seed,
			    t_record = c.t_record,
			    sim_config = sim_config
			)
			
			replicas = collect_replicas(model_fn, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)

			nₖ_across_replicas = get_nₖ_across_replicas(data, g)
			push!(store_results[L]["mean_n₀"], mean(nₖ_across_replicas[0]))
			push!(store_results[L]["mean_n₁"], mean(nₖ_across_replicas[1]))
			push!(store_results[L]["mean_n₂"], mean(nₖ_across_replicas[2]))
			push!(store_results[L]["mean_n₃"], mean(nₖ_across_replicas[3]))
		
			seed += c.n_replicas
		end
	end

	# println("time: $(time() - start_t)\n")
	println("Final seed: ", seed)

	## uncomment to save results
	folderpath = "/Users/ec627/Documents/Data/self-organized-synergy/grappa_analysis/triangle_analysis/"
	
	# save_object(
	# 	folderpath * filename,
	# 	(store_results=store_results, config=c)
	# )

	store_results
end

# ╔═╡ 44209fd9-cb85-4605-a4a8-580323812b4d
md"""
### Robustness analysis IT measures
"""

# ╔═╡ 9eca7247-804e-41e5-b2a8-b3d1a7a6ab01
md"""
Filename convention:

```julia
filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_alpha$(c.α)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_L.jld2"
```

since we are sweeping through:
```julia
L = 0:1:N
```
"""

# ╔═╡ 257eefa9-0c9c-4128-84f2-308b3d8f953a
function mk_filename_varying_L_robustness(config)

	c = config
	filename = "N$(c.N)_G$(c.G)_k$(c.k)_epsilon$(c.ϵ)_meanalpha$(c.μ_α)_beta$(c.β)_local$(c.local_energy_only)_n_replicas$(c.n_replicas)_t_record$(c.t_record)_no_tests$(c.no_tests)_starting_seed$(c.starting_seed)_Oinfo_varying_L.jld2"
	
	return filename
end

# ╔═╡ 723ef286-30c1-450f-a48c-8f416fec4909
begin
	println("This cell defines the noHeterophilousSweepExperimentRobustness struct")

	Base.@kwdef struct noHeterophilousSweepExperimentRobustness
		
	    no_tests::Int = 10
		starting_seed::Int = 1
	
	    # network
	    N::Int = 10
	    G::Int = 3
	    k::Int = 5
	    ϵ::Float64 = 0.2
	
	    # model
	    μ_α::Float64 = 0.4
		σ_α::Float64 = 0.05
		β::Float64 = 20.0
	
	    # dynamics
	    local_energy_only::Bool = true
	
	    # replica params
	    n_replicas::Int = 10_000
	    t_record::Int = 100
	end

	LvecRobustness(config::noHeterophilousSweepExperimentRobustness) = 0:1:config.N

	nothing
end

# ╔═╡ 7f01b776-7258-4839-8f62-d627b495cbdf
function triplet_IT_components(g::DynGraph, data::Matrix{Int})

	N = nv(g)
	triangle_list = triangles(g)
	no_triangles = length(triangle_list)

	# get marginal entropies (precompute these)
	Hᵢ = zeros(N) # [Grappa.entropy(data[:, i]) for i in 1:N]
	Hᵢⱼ = zeros(N, N)
	for i in 1:N
		Hᵢ[i] = entropy(data[:, i])
		for j in Iterators.filter(>(i), neighbors(g, i))
			Hᵢⱼ[i,j] = entropy(data[:, [i,j]])
			Hᵢⱼ[j,i] = Hᵢⱼ[i,j]
		end
	end

	Ω_list = zeros(no_triangles)
	u1_list = zeros(no_triangles)
	u2_list = zeros(no_triangles)
    MI_matrix = zeros(N, N)
    CMI_list = [zeros(3) for _ in 1:no_triangles]
	H3_list = zeros(no_triangles)
	
	@inbounds for (triangle_idx, triangle) in enumerate(triangle_list)

		Iᵢⱼ = zeros(3)
		Iᵢⱼ₋ₖ = zeros(3)
		Hᵢⱼₖ = entropy(data[:, triangle])
		H3_list[triangle_idx] = Hᵢⱼₖ
		
		for (in_triangle_idx, k) in enumerate(Iterators.reverse(triangle))
			
			i, j = setdiff(triangle, k)
			
			## MI
			Iᵢⱼ[in_triangle_idx] = Hᵢ[i] + Hᵢ[j] - Hᵢⱼ[i,j]
            MI_matrix[i,j] = Iᵢⱼ[in_triangle_idx]
            MI_matrix[j,i] = MI_matrix[i,j]
			
			## CONDITIONAL MI (conditioning on element k)
			Iᵢⱼ₋ₖ[in_triangle_idx] = Hᵢⱼ[i,k] + Hᵢⱼ[j,k] - Hᵢ[k] - Hᵢⱼₖ
            CMI_list[triangle_idx][in_triangle_idx] = Iᵢⱼ₋ₖ[in_triangle_idx]
			
		end

		u1 = mean(Iᵢⱼ)
		u2 = mean(Iᵢⱼ₋ₖ)
		
		Ω_list[triangle_idx] = u1 - u2
		u1_list[triangle_idx] = u1
		u2_list[triangle_idx] = u2
	end

	return (
	Ω = Ω_list, Hᵢ=Hᵢ, Hᵢⱼ=Hᵢⱼ, u1=u1_list, u2=u2_list, MI = MI_matrix, CMI = CMI_list, H3=H3_list
	)
end

# ╔═╡ 169091da-2f26-4f92-aaba-a6b8f8223d83
let
	println("Collect data cell:\n")

	# ----- set experiment parameters
	# c = noHeterophilousSweepExperimentRobustness(;
	# 	no_tests = 2,
	# 	starting_seed = 1,
		
	# 	N = 20,
	# 	G = 3,
	# 	k = 5,
	# 	ϵ = 0.2,
	
	# 	μ_α = 0.75,
	# 	β = 10.0,
	
	# 	local_energy_only = true,
	# 	n_replicas = 100, # 10_000,
	# 	t_record = 50
	# )

	# L_range = LvecRobustness(c)

	# # create filename
	# filename = mk_filename_varying_L_robustness(c)
	# @show filename

	# # ------------------------------

	# store_results = Dict(
	# 	"mean_Oinfo3" => Dict{Int, Vector}(),
	# 	"mean_u1" => Dict{Int, Vector}(),
	# 	"mean_u2" => Dict{Int, Vector}(),
	# 	"A" => Dict{Int, Vector{AbstractArray}}(),
	# 	"λ" => Dict{Int, Vector{Vector}}(),
	# 	"MI" => Dict{Int, Vector{AbstractArray}}(),
	# 	"CMI" => Dict{Int, Dict{Int, Vector{Vector{Float64}}}}(),
	# 	"H3" => Dict{Int, Dict{Int, Vector}}(),
	# 	"Hᵢⱼ" => Dict{Int, Vector{AbstractArray}}(),
	# 	"Hᵢ" => Dict{Int, Vector{Vector}}(),
	# )
	
	# start_t = time()
	# seed = c.starting_seed
	
	# @inbounds @progress for (j, L) in enumerate(L_range)
		
	# 	store_results["mean_Oinfo3"][L] = zeros(c.no_tests)
	# 	store_results["mean_u1"][L] = zeros(c.no_tests)
	# 	store_results["mean_u2"][L] = zeros(c.no_tests)
	# 	store_results["A"][L] = [zeros(c.N, c.N) for _ in 1:c.no_tests]
	# 	store_results["λ"][L] = [zeros(c.N) for _ in 1:c.no_tests]
	# 	store_results["MI"][L] = [zeros(c.N, c.N) for _ in 1:c.no_tests]
	# 	store_results["CMI"][L] = Dict{Int, Vector{Vector{Float64}}}()
	# 	store_results["H3"][L] = Dict{Int, Vector}()
	# 	store_results["Hᵢⱼ"][L] = [zeros(c.N, c.N) for _ in 1:c.no_tests]
	# 	store_results["Hᵢ"][L] = [zeros(c.N) for _ in 1:c.no_tests]

	# 	for test in 1:c.no_tests

	# 		rng = MersenneTwister(seed)
			
	# 		# create graph using Graphs.jl
	# 		graph = watts_strogatz(c.N, c.k, c.ϵ; rng=rng)
	# 		A = BitMatrix(adjacency_matrix(graph))
	# 		store_results["A"][L][test] = A
		
	# 		# create composite type DynGraph
	# 		g = DynGraph(A)
		
	# 		# define model
	# 		λ = random_lambda(c.N, L; rng = rng)
	# 		store_results["λ"][L][test] = λ
	# 		α_vec = get_alpha(rng, c.μ_α, c.σ_α, c.N)
			
	# 		model_fn = () -> AttractionRepulsionSpinGlass(
	# 			G=c.G, 
	# 			alpha=α_vec,
	# 			lambda=λ,
	# 			N=c.N
	# 		)
		
	# 		# select dynamics params
	# 		dynamics = Gibbs(c.β, c.local_energy_only)
			
	# 		sim_config = SimConfig(
	# 			steps=c.t_record * c.N,
	# 			seed = seed,
	# 			save_dt = c.N,
	# 			save_state = true
	# 		)
		
	# 		rconfig = ReplicaConfig(
	# 		    n_replicas = c.n_replicas,
	# 		    starting_seed = seed,
	# 		    t_record = c.t_record,
	# 		    sim_config = sim_config
	# 		)
			
	# 		replicas = collect_replicas(model_fn, dynamics, g, rconfig)
	# 		data = states_to_int_matrix(replicas)

	# 		# compute IT quantities
	# 		ensemble_res = triplet_IT_components(g, data)
	# 		MI = ensemble_res.MI
	# 		CMI = ensemble_res.CMI
	# 		H3 = ensemble_res.H3
	# 		Hᵢⱼ = ensemble_res.Hᵢⱼ
	# 		Hᵢ = ensemble_res.Hᵢ

	# 		# store IT quantities
	# 		store_results["mean_Oinfo3"][L][test] = mean(ensemble_res.Ω)
	# 		store_results["mean_u1"][L][test] = mean(ensemble_res.u1)
	# 		store_results["mean_u2"][L][test]  = mean(ensemble_res.u2)
	# 		store_results["MI"][L][test] = MI
	# 		store_results["CMI"][L][test] = CMI
	# 		store_results["H3"][L][test] = H3
	# 		store_results["Hᵢⱼ"][L][test] = Hᵢⱼ
	# 		store_results["Hᵢ"][L][test] = Hᵢ
		
	# 		seed += c.n_replicas
	# 	end
	# end

	# println("time: $(time() - start_t)\n")
	# println("Final seed: ", seed)

	# ## uncomment to save results
	# folderpath = "robustness_experiments/"
	
	# # save_object(
	# # 	folderpath * "withH3_" * filename,
	# # 	(store_results=store_results, config=c)
	# # )

	# store_results
end

# ╔═╡ Cell order:
# ╟─ac04e022-4adc-11f1-a3ba-bdce3c59a914
# ╟─4605d981-e601-4e31-bac9-a74b694e957a
# ╟─4dfd7819-1848-4937-97a1-cdd087f45452
# ╠═1ee3edd2-e769-4cba-9390-d5c7bba69bc1
# ╟─0c57ae47-ffdc-44a6-a4d1-ebd36ee5aef1
# ╟─33cdec0a-a39f-4d39-bbcc-aab6134936dc
# ╟─0885bac4-eda5-4f60-ba30-d985f1bc7915
# ╟─9ce111bb-f365-4f5c-b04f-244706afa931
# ╟─34409a83-c871-4702-ac2a-748d00f849e9
# ╟─916b674a-a9bf-4b0a-815c-6daf58aa6f61
# ╟─71a64eeb-f553-49a1-a205-1b74078eaa9b
# ╟─b31980c4-1026-4b2f-8d18-2455e4e423f0
# ╟─8667df8a-f0ec-4dfd-afe1-2c72a13d7e10
# ╟─88482d55-665d-4bb4-b4fe-3d0414c79348
# ╟─4fdbd631-dc2c-43f1-88aa-b86b1cfdc4f3
# ╟─fa99dee7-b96c-452f-bad7-84d9a837a69f
# ╟─2449509c-59a0-4ddc-b4ee-d0d26d0a0815
# ╟─d9757572-566f-4d38-b3a0-02ab88f76029
# ╟─adcd82ca-f7d7-409e-a7c6-40e7cc63b2b9
# ╟─bca285dd-b4dd-4734-b4cb-851808127a0a
# ╟─11cabaa7-ddce-4060-a6b8-2ffd539df412
# ╟─f9c497e9-dec7-464f-bf1f-121adbd00f7c
# ╟─3d660037-c5dd-4428-9436-2a0f56b632b3
# ╟─cb61b3c9-001f-4ab7-94ca-0b1d4d643a0f
# ╟─36759ea7-595d-4b31-8a8c-7665daebdffb
# ╟─1397d0e3-3f5e-485a-a005-ba158b524f84
# ╟─f7514b7d-1b3a-49d5-b9f4-1e98c115ee25
# ╟─fefe018c-641f-4f65-93a4-20d72a0bc21d
# ╟─260ba10c-2b56-43f1-9afd-533588e02275
# ╟─cd2ae50a-ee35-4821-a62d-fc350af567bd
# ╟─267512f6-98ae-4cce-a944-4f72c692bb11
# ╟─43331cc3-70aa-4570-8b2e-ad5754c4be0d
# ╟─d5221050-387a-4d59-a7f2-64575d7fad3d
# ╟─37180725-f017-4c5d-8b69-19943a094938
# ╟─44209fd9-cb85-4605-a4a8-580323812b4d
# ╟─9eca7247-804e-41e5-b2a8-b3d1a7a6ab01
# ╟─257eefa9-0c9c-4128-84f2-308b3d8f953a
# ╟─723ef286-30c1-450f-a48c-8f416fec4909
# ╟─7f01b776-7258-4839-8f62-d627b495cbdf
# ╟─169091da-2f26-4f92-aaba-a6b8f8223d83
