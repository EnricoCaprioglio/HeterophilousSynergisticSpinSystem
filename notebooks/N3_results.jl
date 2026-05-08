### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 07ef8f99-f396-4e83-a76d-cf4e746ebbdf
begin
	using Pkg
	
	using LinearAlgebra
	using StatsBase
	using PlutoUI
	
	# saving and loading files
	using FileIO
	using JLD2

	# plotting
	using LaTeXStrings
	# using CairoMakie
	using GLMakie
	using GraphMakie
	using NetworkLayout
	using Logging
	using ProgressLogging
end

# ╔═╡ dc39ebe7-458b-46ff-883c-df7d70efca12
begin
	Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()

    include(joinpath(@__DIR__, "..", "src", "HeterophilySynergy.jl"))
	println("HeterophilySynergy\nPS: do not run this cell twice!")
end

# ╔═╡ 17758d85-8bf8-4bb2-9576-ffa52267ebfc
using Graphs

# ╔═╡ e91d69b1-6fb4-48b1-9edb-7a132a8cb988
using Random

# ╔═╡ 97f55482-4a7f-11f1-95be-95b2729f7c4f
md"""
---

## Contact Details

**Author:** Enrico Caprioglio

**Email:** ec627@sussex.ac.uk

---
"""

# ╔═╡ 4375c013-ebcc-4fe4-884b-7276df877468
md"""
# $N=3$ results
"""

# ╔═╡ 363ea244-4a46-424d-affc-6b349eecb965
md"""
Note, here we compute everything since it is fast.

No separate data collection required.
"""

# ╔═╡ bbed48a9-9687-4b46-916c-df15562f1ce3
PlutoUI.TableOfContents()

# ╔═╡ 5b26be7f-5841-4a66-b566-cea9671ecf1c
const mk = GLMakie

# ╔═╡ 101a233f-a4d9-4f1f-87d7-97cc6f514d0e
let	
	using Colors
	# parameters (odd t)
	G = 3; t = Int((G-1)/2)
	if G == 3
		_ticks = ([0, t, t+1, G], [L"0", L"h", L"h+1", L"G"])
	else
		_ticks = ([0, t-1, t, t+1, G], [L"0", L"h-1", L"h", L"h+1", L"G"])
	end
	
	purple_size = Int(30)
	pink_size = Int(30)
	orange_size = Int(30)
	if G == 3
		grey_size = Int(30)
	else
		grey_size = Int(20)
	end
	
	azimuth_val=.2π
	elevation_val=0.10π
	perspectiveness_val=0.1

	# main code
	condition1(d12, d13, d23) = sum([d12, d13, d23]) % 2 == 0
	condition2(d12, d13, d23, G) = sum([d12, d13, d23]) ≤ 2*G
	condition3(d12, d13, d23) = begin
		A = d12 ≤ d13 + d23
		B = d13 ≤ d12 + d23
		C = d23 ≤ d12 + d13
		return all([A,B,C])
	end
	conditions(d12, d13, d23, G) = all(
		[condition1(d12, d13, d23), condition2(d12, d13, d23, G), condition3(d12, d13, d23)]
	)

	distances_state_space = [[i,j,k] for i in 1:G for j in 1:G for k in 1:G]
	
	x = Float32[]; y = Float32[]; z = Float32[]
	colors = []
	sizes = []
	antibalanced_xyz = []
	balanced_xyz = []
		
	for i in 0:G
		for j in 0:G
			for k in 0:G
				if conditions(i, j, k, G)
					
					push!(x, Float32(i))
					push!(y, Float32(j))
					push!(z, Float32(k))
					
					if i in (t, t+1) && j in (t, t+1) && k in (t, t+1)
						push!(colors, :purple)
						push!(sizes, purple_size)
						push!(antibalanced_xyz, [i,j,k])
					
					# homophily Δ0, Δ2
					elseif (i==0 && j==0 && k==0) || ((i,j,k) in ((0,G,G), (G,0,G), (G,G,0)))
						push!(colors, :orange)
						push!(sizes, orange_size)
						push!(balanced_xyz, [i,j,k])
						
					# nothign
					else
						push!(colors, RGBA(.2,.2,.2,.2))
						push!(sizes, grey_size)
					end
				end
			end
		end
	end

	fig = Figure(fontsize=20)
	
	ax = Axis3(
		fig[1,1];
	    xlabel = L"d_{12}", ylabel = L"d_{13}", zlabel = L"d_{23}",
	    azimuth=azimuth_val,
		elevation=elevation_val,
		perspectiveness=perspectiveness_val,
		xticks = _ticks, # [0, t-1, t, t+1, G],
		yticks = _ticks, # [0, t-1, t, t+1, G],
		zticks = _ticks, # [0, t-1, t, t+1, G]
	)

	# project lines to the 0 planes:
	for p in antibalanced_xyz
	    i, j, k = p
	    # to x = 0 plane: (i,j,k) -> (0,j,k)
	    mk.lines!(ax, [i, 0], [j, j], [k, k];
	              color = RGBA(.2,.2,.2,.2), linestyle = :solid, linewidth = .8)
	    # to y = 0 plane: (i,j,k) -> (i,0,k)
	    mk.lines!(ax, [i, i], [j, 0], [k, k];
	              color = RGBA(.2,.2,.2,.2), linestyle = :solid, linewidth = .8)
	    # to z = 0 plane: (i,j,k) -> (i,j,0)
	    mk.lines!(ax, [i, i], [j, j], [k, 0];
	              color = RGBA(.2,.2,.2,.2), linestyle = :solid, linewidth = .8)
	end

	mk.xlims!(ax, 0, G); mk.ylims!(ax, 0, G); mk.zlims!(ax, 0, G)
	mk.scatter!(ax, x, y, z, color = colors, markersize = sizes)

	# dummy plots for legend (just easier)
	mk.scatter!(
		ax, [G+1f0], [G+1f0], [G+1f0], color = :purple, markersize = purple_size,
		label = L"d_{ij}\in\{h,h+1\}\;\forall{i,j}"
	)
	mk.scatter!(
		ax, [G+1f0], [G+1f0], [G+1f0], color = :orange, markersize = orange_size,
		label = L"d_{ij}\in\{0,G\}\;\forall{i,j}"
	)
	mk.axislegend(ax, position=:lt)

	folderpath = "/Users/ec627/Documents/Sussex/thesis_master/figures_inkscape/"
	save(folderpath * "distance_state_space.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ 318e6579-8726-489e-b0b8-a1f140a62128
md"""
## Figure 2
"""

# ╔═╡ 52801a9c-52c2-457d-9ffb-e4e492d5d3fd
md"""
## Figure 4
"""

# ╔═╡ 922b8c0d-949c-4f83-b637-6946249a1016
md"""
### Functions
"""

# ╔═╡ 38b99c56-6988-4aee-8a12-755b04a4f368
md"""
Note, this function implements Eq. (S1) from the paper (from the supplementary).
"""

# ╔═╡ c767fa4b-9e07-4e6d-8c21-eb423fa98b20
# Eq. (S1) from the paper
function get_degeneracy(d12, d13, d23, G)::Int
	
	get_a(d12, d13, d23)::Int = (d12 + d13 - d23) / 2
	get_b(d12, d13, d23)::Int = (d12 - d13 + d23) / 2
	get_c(d12, d13, d23)::Int = (- d12 + d13 + d23) / 2
	get_t(d12, d13, d23)::Int = G - (d12 + d13 + d23) / 2

	a = get_a(d12, d13, d23)
	b = get_b(d12, d13, d23)
	c = get_c(d12, d13, d23)
	t = get_t(d12, d13, d23)

	if length(unique([d12, d13, d23])) == 1
		multipl_factor = 1
	elseif length(unique([d12, d13, d23])) == 2
		multipl_factor = 3
	elseif length(unique([d12, d13, d23])) == 3
		multipl_factor = 6
	end

	if G < 21
		return multipl_factor*2^G*(factorial(G) / (factorial(a)*factorial(b)*factorial(c)*factorial(t)))
	else
		return multipl_factor*2^G*(factorial(big(G)) / (factorial(big(a))*factorial(big(b))*factorial(big(c))*factorial(big(t))))
	end
end

# ╔═╡ 65c6aeb6-fe5a-4eb6-b126-2721e5f2549d
function get_distances(α, λ, G)

	# some checks
	G ≥ 3 || throw(ArgumentError("G must be ≥ 3"))
    isodd(G) || throw(ArgumentError("G must be odd"))
    h::Int = (G - 1) / 2
	
	# Normalize λ into a key (this is different convention from the rest of the code, however it makes it slightly easier to use for this specific case)
    λkey = if λ isa Symbol
        λ
    elseif λ isa AbstractString
        Symbol(λ)
    elseif λ isa NTuple{3,<:Integer}
        all(==(-1), λ) ? :minuses : :other
        all(==(1), λ) ? :pluses : :other
    else
        :other
    end
    (λkey == :minuses || λkey == :pluses) || throw(ArgumentError("For λ, please use either :minuses or :pluses here (not λ = $λ)"))

	parity = isodd(h) ? :odd : :even

	if α in (1//4, 1//2, 3//4)
		α += 1 / floatmax(Float64)
    end

	# {Tuple{Symbol,Symbol}, Vector{Tuple{Real,NTuple{3,Int}}}}
	# below are the analytical results
	rules = Dict(
        (:minuses, :any) => [
            (1/2, (0, G, G)),
            (Inf,  (0, 0, 0)),
        ],
        (:pluses, :odd) => [
            (1/4, (h,   h,   h-1)),
            (1/2, (h,   h,   h+1)),
            (Inf,  (h+1, h+1, h+1)),
        ],
        (:pluses, :even) => [
            (1/2, (h,   h,   h  )),
            (3/4, (h,   h+1, h+1)),
            (Inf,  (h+1, h+1, h+2)),
        ]
    )

    rkey = (λkey, λkey == :minuses ? :any : parity)
    table = rules[rkey]

    dist = nothing
    for (ub, d) in table
        if α < ub
            dist = d
            break
        end
    end
    dist === nothing && error("No rule matched, please check table 1 in the paper")

	return sort(collect(dist))
end

# ╔═╡ 62535be9-55e3-4815-a646-3dc67ffca71f
md"""
with the distances $\delta^\star$ and $|M(\delta^\star)|$ we can compute info-theoretic quantities
"""

# ╔═╡ 35588d8c-4c08-4d63-9414-694f925f1739
function zero_temperature_IT(α, λ, G)
	
	d12, d13, d23 = get_distances(α, λ, G)
	Hsystem = log2(get_degeneracy(d12, d13, d23, G))

	d_multipl = countmap([d12, d13, d23])

	# compute H(d)
	Hr(d_mult) = begin
		out = 0
		for d in keys(d_mult)
			out += - d_mult[d] / 3 * log2(d_mult[d] / 3)
		end
		return out
	end

	# compute E [log₂ binom(G, d)] 
	exp_log_G_D(d_mult, G) = begin
		out = 0
		for d in keys(d_mult)
			out += d_mult[d] / 3 * log2(binomial(G, d))
		end
		return out
	end

	Hpair = Hr(d_multipl) + G + exp_log_G_D(d_multipl, G)

	output = (
	Ω = Hsystem + 3*G - 3*Hpair,
	u1 = 2*G - Hpair,
	u2 = 2*Hpair - G - Hsystem
	)

	return output
end

# ╔═╡ 7b39d6a7-897b-4586-b386-ef09d0c554ac
md"""
below is a way to compute the Boltzmann probabilities (I am sure there are better ways).
"""

# ╔═╡ 01596671-7292-4865-a1fb-e53e342e1d9b
function boltzmann_probabilities(model::AttractionRepulsionSpinGlass, g::DynGraph, β::Float64)

	N = nv(g)
	
	# get spin state space (Integers, not BitVector)
	int_spin_state_space = [
		bit_to_int(s, model.G) for s in spin_state_space(model.G)
	]
	# get system state space (Integers, not BitVector)
	int_system_state_space = collect(
		Iterators.product(fill(int_spin_state_space, N)...)
	)

	# compute global energy of each state
	nᵢ, nⱼ, nₖ = size(int_system_state_space)
	global_energies = zeros(size(int_system_state_space))
	for i in 1:nᵢ
		for j in 1:nⱼ
			for k in 1:nₖ
				model.S = state_convert(Int.([i,j,k]), Matrix{Int8}; G=model.G)
				global_energies[i,j,k] = global_energy(model, g)
			end
		end
	end

	boltz_factors = map(x -> exp(-β*x), global_energies)
	probabilities = boltz_factors ./ sum(boltz_factors)

	## 	SINGLE SPIN PROBABILITIES
	# for each spin, we compute the sum of all the probabilities in which spin i is in some state (8 possible states)
	# same as e.g., `[sum(probabilities[:,:,i]) for i in 1:8]` for spin 3
	p1(P) = vec(sum(P; dims=(2,3)))
	p2(P) = vec(sum(P; dims=(1,3)))
	p3(P) = vec(sum(P; dims=(1,2)))

	## MARGINAL PROBABILITIES (one spin removed)
	# similarly, but here we keep fixed two spins, and sum over the third
	# same as e.g., `sum([probabilities[:, :, i] for i in 1:8])`
	p12(P) = dropdims(sum(P; dims=3); dims=3)
	p13(P) = dropdims(sum(P; dims=2); dims=2)
	p23(P) = dropdims(sum(P; dims=1); dims=1)

	function get_entropy(pvec::AbstractVector{Float64})
		return _entropy_from_probs(pvec)
	end
	function get_entropy(pmat::AbstractArray)
		@assert isapprox(sum(pmat), 1) "probs do not sum to 1"
		pvec = reshape(pmat, length(pmat))
		return _entropy_from_probs(pvec)
	end

	## COMPUTE ENTROPIES
	# each spin
	p₁, p₂, p₃ = p1(probabilities), p2(probabilities), p3(probabilities)
	@assert isapprox(sum(p₁), 1) "probs do not sum to 1"
	@assert isapprox(sum(p₂), 1) "probs do not sum to 1"
	@assert isapprox(sum(p₃), 1) "probs do not sum to 1"
	# spin entropy
	H₁, H₂, H₃ = get_entropy(p₁), get_entropy(p₂), get_entropy(p₃)
	
	# each marginal
	p₁₂, p₁₃, p₂₃ = p12(probabilities), p13(probabilities), p23(probabilities)
	@assert isapprox(sum(p₁₂), 1) "probs do not sum to 1"
	@assert isapprox(sum(p₁₃), 1) "probs do not sum to 1"
	@assert isapprox(sum(p₂₃), 1) "probs do not sum to 1"
	# each marginal entropy
	H₁₂, H₁₃, H₂₃ = get_entropy(p₁₂), get_entropy(p₁₃), get_entropy(p₂₃)

	# SYSTEM ENTROPY
	@assert isapprox(sum(probabilities), 1) "probs do not sum to 1"
	# H₁₂₃ = get_entropy(reshape(probabilities, 2^(N * model.G)))
	H₁₂₃ = get_entropy(probabilities)
	
	## MUTUAL INFORMATION
	I₁₂ = H₁ + H₂ - H₁₂
	I₁₃ = H₁ + H₃ - H₁₃
	I₂₃ = H₂ + H₃ - H₂₃

	# CONDITIONAL MI
	I₁₂_₃ = H₁₃ + H₂₃ - H₃ - H₁₂₃
	I₁₃_₂ = H₁₂ + H₂₃ - H₂ - H₁₂₃
	I₂₃_₁ = H₁₂ + H₁₃ - H₁ - H₁₂₃

	Ω = (H₁ + H₂ + H₃) - (H₁₂ + H₁₃ + H₂₃) + H₁₂₃

	output = (
	Hᵢ = [H₁, H₂, H₃], 				# spin entropies
	H₁₂₃ = H₁₂₃, 					# sys entropy
	H₋ᵢ = [H₂₃, H₁₃, H₁₂], 			# entropies one spin removed
	
	Iᵢⱼ = [I₁₂, I₁₃, I₂₃], 			# mutual information
	Iᵢⱼ₋ₖ = [I₁₂_₃, I₁₃_₂, I₂₃_₁], 	# conditional MI

	Ω = Ω 							# O-info
	)
end

# ╔═╡ f7c0f414-b400-4136-a032-9e2069dbbe2c
md"""
### Examples
"""

# ╔═╡ 75dae93b-0f35-4fba-87d8-0a5d623b556c
let
	println("Just some examples\n")
	G = 3
	@show get_degeneracy(0, 0, 0, G)
	@show get_degeneracy(G, G, 0, G)

	println()
	@show zero_temperature_IT(.4, :pluses, 3).Ω

	g = DynGraph(BitMatrix([
		0 1 1;
		1 0 1;
		1 1 0
	]))
	β = 20.0
	model = AttractionRepulsionSpinGlass(G=3, alpha=.4, lambda=Int8.([1,1,1]), N = 3)
	@show boltzmann_probabilities(model, g, β).Ω
end

# ╔═╡ 9b726379-1607-45c1-9fd1-21485cf27b58
function plot_heterophilous!(axs, α_range, β, G, N, g)

	λ = fill(Int8(1), 3)

	boltz_results = Dict(
		"Ω" => [],
		"u1" => [],
		"u2" => []
	)
	analytical_results = Dict(
		"Ω" => [],
		"u1" => [],
		"u2" => []
	)

	label = L"\text{heterophilous}"
	
	for α in α_range
		
		# boltzmann finite temperature
		model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
		res = boltzmann_probabilities(model, g, β)
		push!(boltz_results["Ω"], res.Ω)
		push!(boltz_results["u1"], mean(res.Iᵢⱼ))
		push!(boltz_results["u2"], mean(res.Iᵢⱼ₋ₖ))

		# analytical zero temperature
		analytical_res = zero_temperature_IT(α, :pluses, G)
		push!(analytical_results["u1"], analytical_res.u1)
		push!(analytical_results["u2"], analytical_res.u2)
		push!(analytical_results["Ω"], analytical_res.Ω)
	end
	
	mk.scatter!(axs[1], α_range, boltz_results["u1"], color = :purple, label = label)
	mk.scatter!(axs[2], α_range, boltz_results["u2"], color = :purple, label = label)
	mk.scatter!(axs[3], α_range, boltz_results["Ω"], color = :purple, label = label)

	mk.lines!(axs[1], α_range, analytical_results["u1"], color = :purple)
	mk.lines!(axs[2], α_range, analytical_results["u2"], color = :purple)
	mk.lines!(axs[3], α_range, analytical_results["Ω"], color = :purple)

	axs[1].xticks = [0, 0.5, 1]

	return axs
end

# ╔═╡ 0e577872-09f0-4d19-b3f2-19a573e347c8
function plot_homophilous!(axs, α_range, β, G, N, g)

	λ = fill(Int8(-1), 3)

	boltz_results = Dict(
		"Ω" => [],
		"u1" => [],
		"u2" => []
	)
	analytical_results = Dict(
		"Ω" => [],
		"u1" => [],
		"u2" => []
	)

	label = L"\text{homophilous}"
	
	for α in α_range
		
		# boltzmann finite temperature
		model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
		res = boltzmann_probabilities(model, g, β)
		push!(boltz_results["Ω"], res.Ω)
		push!(boltz_results["u1"], mean(res.Iᵢⱼ))
		push!(boltz_results["u2"], mean(res.Iᵢⱼ₋ₖ))

		# analytical zero temperature
		analytical_res = zero_temperature_IT(α, :minuses, G)
		push!(analytical_results["u1"], analytical_res.u1)
		push!(analytical_results["u2"], analytical_res.u2)
		push!(analytical_results["Ω"], analytical_res.Ω)
	end
	
	mk.scatter!(axs[1], α_range, boltz_results["u1"], color = :darkorange, label=label)
	mk.scatter!(axs[2], α_range, boltz_results["u2"], color = :darkorange, label=label)
	mk.scatter!(axs[3], α_range, boltz_results["Ω"], color = :darkorange, label=label)

	mk.lines!(axs[1], α_range, analytical_results["u1"], color = :darkorange)
	mk.lines!(axs[2], α_range, analytical_results["u2"], color = :darkorange)
	mk.lines!(axs[3], α_range, analytical_results["Ω"], color = :darkorange)

	axs[1].xticks = [0, 0.5, 1]

	return axs
end

# ╔═╡ 397a2f66-7a95-4feb-997b-c89e0f3213d9
let
	N = 3
	g = DynGraph(BitMatrix([
		0 1 1;
		1 0 1;
		1 1 0
	]))
	β = 20.0
	α_range = 0.0:0.01:1

	fig = Figure(fontsize = 18, size = (650, 400))
	ylabels = [L"u_1", L"u_2", L"\Omega", L"u_1", L"u_2", L"\Omega"]
	axs = [Axis(fig[j, i]) for j in 1:2 for i in 1:3]

	for (ax, label) in zip(axs, ylabels)
		ax.ylabel = label
		ax.xtickformat = x -> latexstring.(round.(x, digits=3))
		ax.ytickformat = x -> latexstring.(round.(x, digits=3))
	end

	
	plot_heterophilous!(axs[1:3], α_range, β, 3, N, g)
	plot_homophilous!(axs[1:3], α_range, β, 3, N, g)
	plot_heterophilous!(axs[4:6], α_range, β, 5, N, g)
	plot_homophilous!(axs[4:6], α_range, β, 5, N, g)

	mk.Label(fig[3, 1:3], L"\alpha")
	mk.Legend(fig[0, 1:3], axs[1], framevisible = false, labelsize=18, orientation = :horizontal)

	rowgap!(fig.layout, 3, 0)

	panel_labels = ["A", "B", "C"]
	for i in 1:3
		Label(fig[1, i, TopLeft()], panel_labels[i];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (5, 0, 0, -12.5),
	        tellwidth = false, tellheight = false,
	    )
	end
	panel_labels = ["D", "E", "F"]
	for i in 1:3
		Label(fig[2, i, TopLeft()], panel_labels[i];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (5, 0, 0, -12.5),
	        tellwidth = false, tellheight = false,
	    )
	end
		
	# save(folderpath * "N3_exact.png", fig, px_per_unit = 400/96)
	fig
end

# ╔═╡ 960e80a9-792d-4201-bc6a-b0e168a52f50
md"""
## Figure S4
"""

# ╔═╡ 89e50715-9dc4-45c0-a920-21f67d6832b3
let
	G_range = 3:2:21
	
	fig = Figure(fontsize = 20, size = (650, 225))
	# ylabels = [L"u_1", L"u_2", L"\Omega", L"u_1", L"u_2", L"\Omega"]
	axs = [Axis(fig[1, i], ylabel = L"\Omega", xlabel = L"G") for i in 1:3]

	for ax in axs
		# ax.xtickformat = x -> latexstring.(round.(x, digits=1))
		ax.ytickformat = x -> latexstring.(round.(x, digits=3))
	end

	# homophilous system
	α_range = [0.25, 0.75]
	for α in α_range
		x = Int.([])
		y = []
		for G in G_range
			# if G % 4 == mod_no
				push!(x, Int(G))
				push!(y, zero_temperature_IT(α, :minuses, G).Ω)
			# end
		end
		mk.scatterlines!(axs[1], Int.(x), y, label = L"\alpha = %$(α)")
		axs[1].xticks = ([5, 10, 15, 20], [L"5", L"10", L"15", L"20"])
		axs[1].title = L"\lambda_i=-1"
	end

	# heterophilous system G=3 mod 4
	mod_no = 3
	α_range = [0.2, 0.4, 0.8]
	for α in α_range
		x = Int.([])
		y = []
		for G in G_range
			if G % 4 == mod_no
				push!(x, Int(G))
				push!(y, zero_temperature_IT(α, :pluses, G).Ω)
			end
		end
		mk.scatterlines!(axs[2], Int.(x), y, label = L"\alpha = %$(α)")
		axs[2].xticks = ([3, 7, 11, 15, 19], [L"3", L"7", L"11", L"15", L"19"])
		axs[2].title = L"\lambda_i=1,\; G = 3\;(\mathrm{mod}\;4)"
		axs[2].titlesize = 16
	end

	# heterophilous system G=1 mod 4 (G>3)
	mod_no = 1
	α_range = [0.2, 0.6, 0.9]
	for α in α_range
		x = Int.([])
		y = []
		for G in G_range
			if G % 4 == mod_no
				push!(x, Int(G))
				push!(y, zero_temperature_IT(α, :pluses, G).Ω)
			end
		end
		mk.scatterlines!(axs[3], Int.(x), y, label = L"\alpha = %$(α)")
		axs[3].xticks = ([5, 9, 13, 17,21], [L"5", L"9", L"13", L"17",L"21"])
		axs[3].title = L"\lambda_i=1,\; G = 1\;(\mathrm{mod}\;4)"
		axs[3].titlesize = 16
	end
	# axislegend(axs[3])

	for i in 1:3
		pos = :rc
		if i == 1
			pos = :rb
		end
		leg = axislegend(axs[i], position=pos)
		leg.labelsize = 16
		leg.patchsize = (12, 12)
		leg.rowgap = 2
		leg.padding = (3, 3, 3, 3)
	end
	
	axs[2].ylabel = ""
	axs[3].ylabel = ""

	panel_labels = ["A", "B", "C"]
	for i in 1:3
		Label(fig[1, i, TopLeft()], panel_labels[i];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (0, 0, 0, -10),
	        tellwidth = false, tellheight = false,
	    )
	end

	# save(folderpath * "N3_increasing_G.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ e9318f89-085b-4260-85da-7b9d59ca49c0
md"""
## Figure S3
"""

# ╔═╡ b9969be9-8831-4dae-9f50-a40203257a57
let
	starting_seed = 42
	no_tests = 10
	n_replicas_range = [10,100,1_000,10_000,100_000]

	# set network and dynamics params
	N = 3
	G = 3
	λ = Int8.([1,1,1])
	α = 0.4
	local_energy_only = false
	β = 3.4
	t_record = 100

	# create graph using Graphs.jl
	graph = complete_graph(N)
	A = BitMatrix(adjacency_matrix(graph))

	# create composite type DynGraph
	g = DynGraph(A)

	# define model
	model = () -> AttractionRepulsionSpinGlass(
		G=G, 
		alpha=α,
		lambda=λ,
		N=N
	)

	# select dynamics params
	dynamics = Gibbs(β, local_energy_only)

	# start simulations
	seed = starting_seed
	res = zeros(length(n_replicas_range), no_tests)
	@progress for (i, n_replicas) in enumerate(n_replicas_range)
		for test in 1:no_tests
			
			rng = MersenneTwister(seed)

			sim_config = SimConfig(
				steps=t_record * N,
				seed = seed,
				save_dt = N,
				save_state = true
			)
					
			rconfig = ReplicaConfig(
				n_replicas = n_replicas,
				starting_seed = seed,
				t_record = t_record,
				sim_config = sim_config
			)
			
			replicas = collect_replicas(model, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)
		
			# compute IT quantities
			ensemble_res = Oinfo3_components(g, data)
			res[i, test] = ensemble_res.Ω[1]

			seed += n_replicas
		end
	end

	model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
	exact_Ω = boltzmann_probabilities(model, g, β).Ω
	
	fig = Figure(size = (650, 250), fontsize = 24)
	ax = Axis(fig[1,1], xscale = log10, ylabel = L"\Omega", xlabel = L"R")
	ax.title = L"\text{Glauber global:}\;\alpha=%$(α),\;\beta=%$(β),\;\lambda=%$(Int.(λ))"
	ax.xticks = ([10,100,1_000,10_000,100_000], [L"10^1",L"10^2",L"10^3",L"10^4",L"10^5"])
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	mk.band!(ax, n_replicas_range,
		minimum(res, dims=2)[:], maximum(res, dims=2)[:]
	)
	mk.scatterlines!(ax, n_replicas_range, mean(res, dims=2)[:])
	mk.lines!(ax, n_replicas_range, repeat([exact_Ω], length(n_replicas_range)), color=:red)

	println("Final Ω: ", mean(res, dims=2)[end])
	println("Exact Ω: ", exact_Ω)

	# save(folderpath * "convergence_test1.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ 26207d82-c289-4035-8e99-80f7fccb38a6
let
	starting_seed = 42
	no_tests = 10
	n_replicas_range = [10,100,1_000,10_000,100_000]

	# set network and dynamics params
	N = 3
	G = 3
	λ = Int8.([1,1,1])
	α = 0.4
	local_energy_only = false
	β = 10.0
	t_record = 100

	# create graph using Graphs.jl
	graph = complete_graph(N)
	A = BitMatrix(adjacency_matrix(graph))

	# create composite type DynGraph
	g = DynGraph(A)

	# define model
	model = () -> AttractionRepulsionSpinGlass(
		G=G, 
		alpha=α,
		lambda=λ,
		N=N
	)

	# select dynamics params
	dynamics = Gibbs(β, local_energy_only)

	# start simulations
	seed = starting_seed
	res = zeros(length(n_replicas_range), no_tests)
	@progress for (i, n_replicas) in enumerate(n_replicas_range)
		for test in 1:no_tests
			
			rng = MersenneTwister(seed)

			sim_config = SimConfig(
				steps=t_record * N,
				seed = seed,
				save_dt = N,
				save_state = true
			)
					
			rconfig = ReplicaConfig(
				n_replicas = n_replicas,
				starting_seed = seed,
				t_record = t_record,
				sim_config = sim_config
			)
			
			replicas = collect_replicas(model, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)
		
			# compute IT quantities
			ensemble_res = Oinfo3_components(g, data)
			res[i, test] = ensemble_res.Ω[1]

			seed += n_replicas
		end
	end

	model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
	exact_Ω = boltzmann_probabilities(model, g, β).Ω
	
	fig = Figure(size = (650, 250), fontsize = 24)
	ax = Axis(fig[1,1], xscale = log10, ylabel = L"\Omega", xlabel = L"R")
	ax.title = L"\text{Glauber global:}\;\alpha=%$(α),\;\beta=%$(β),\;\lambda=%$(Int.(λ))"
	ax.xticks = ([10,100,1_000,10_000,100_000], [L"10^1",L"10^2",L"10^3",L"10^4",L"10^5"])
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	mk.band!(ax, n_replicas_range,
		minimum(res, dims=2)[:], maximum(res, dims=2)[:]
	)
	mk.scatterlines!(ax, n_replicas_range, mean(res, dims=2)[:])
	mk.lines!(ax, n_replicas_range, repeat([exact_Ω], length(n_replicas_range)), color=:red)

	println("Final Ω: ", mean(res, dims=2)[end])
	println("Exact Ω: ", exact_Ω)

	# save(folderpath * "convergence_test2.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ faff830b-c5db-46df-9e0d-58dbaa752259
let
	starting_seed = 42
	no_tests = 10
	n_replicas_range = [10,100,1_000,10_000,100_000]

	# set network and dynamics params
	N = 3
	G = 3
	λ = Int8.([-1,-1,-1])
	α = 0.8
	local_energy_only = false
	β = 3.4
	t_record = 100

	# create graph using Graphs.jl
	graph = complete_graph(N)
	A = BitMatrix(adjacency_matrix(graph))

	# create composite type DynGraph
	g = DynGraph(A)

	# define model
	model = () -> AttractionRepulsionSpinGlass(
		G=G, 
		alpha=α,
		lambda=λ,
		N=N
	)

	# select dynamics params
	dynamics = Gibbs(β, local_energy_only)

	# start simulations
	seed = starting_seed
	res = zeros(length(n_replicas_range), no_tests)
	@progress for (i, n_replicas) in enumerate(n_replicas_range)
		for test in 1:no_tests
			
			rng = MersenneTwister(seed)

			sim_config = SimConfig(
				steps=t_record * N,
				seed = seed,
				save_dt = N,
				save_state = true
			)
					
			rconfig = ReplicaConfig(
				n_replicas = n_replicas,
				starting_seed = seed,
				t_record = t_record,
				sim_config = sim_config
			)
			
			replicas = collect_replicas(model, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)
		
			# compute IT quantities
			ensemble_res = Oinfo3_components(g, data)
			res[i, test] = ensemble_res.Ω[1]

			seed += n_replicas
		end
	end

	model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
	exact_Ω = boltzmann_probabilities(model, g, β).Ω
	
	fig = Figure(size = (650, 250), fontsize = 24)
	ax = Axis(fig[1,1], xscale = log10, ylabel = L"\Omega", xlabel = L"R")
	ax.title = L"\text{Glauber global:}\;\alpha=%$(α),\;\beta=%$(β),\;\lambda=%$(Int.(λ))"
	ax.xticks = ([10,100,1_000,10_000,100_000], [L"10^1",L"10^2",L"10^3",L"10^4",L"10^5"])
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	mk.band!(ax, n_replicas_range,
		minimum(res, dims=2)[:], maximum(res, dims=2)[:]
	)
	mk.scatterlines!(ax, n_replicas_range, mean(res, dims=2)[:])
	mk.lines!(ax, n_replicas_range, repeat([exact_Ω], length(n_replicas_range)), color=:red)

	println("Final Ω: ", mean(res, dims=2)[end])
	println("Exact Ω: ", exact_Ω)

	# save(folderpath * "convergence_test3.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ 3301410e-393f-40c3-bd8b-7eaf643b129c
let
	starting_seed = 42
	no_tests = 10
	n_replicas_range = [10,100,1_000,10_000,100_000]

	# set network and dynamics params
	N = 3
	G = 3
	λ = Int8.([-1,-1,-1])
	α = 0.2
	local_energy_only = false
	β = 3.4
	t_record = 100

	# create graph using Graphs.jl
	graph = complete_graph(N)
	A = BitMatrix(adjacency_matrix(graph))

	# create composite type DynGraph
	g = DynGraph(A)

	# define model
	model = () -> AttractionRepulsionSpinGlass(
		G=G, 
		alpha=α,
		lambda=λ,
		N=N
	)

	# select dynamics params
	dynamics = Gibbs(β, local_energy_only)

	# start simulations
	seed = starting_seed
	res = zeros(length(n_replicas_range), no_tests)
	@progress for (i, n_replicas) in enumerate(n_replicas_range)
		for test in 1:no_tests
			
			rng = MersenneTwister(seed)

			sim_config = SimConfig(
				steps=t_record * N,
				seed = seed,
				save_dt = N,
				save_state = true
			)
					
			rconfig = ReplicaConfig(
				n_replicas = n_replicas,
				starting_seed = seed,
				t_record = t_record,
				sim_config = sim_config
			)
			
			replicas = collect_replicas(model, dynamics, g, rconfig)
			data = states_to_int_matrix(replicas)
		
			# compute IT quantities
			ensemble_res = Oinfo3_components(g, data)
			res[i, test] = ensemble_res.Ω[1]

			seed += n_replicas
		end
	end

	model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N = N)
	exact_Ω = boltzmann_probabilities(model, g, β).Ω
	
	fig = Figure(size = (650, 250), fontsize = 24)
	ax = Axis(fig[1,1], xscale = log10, ylabel = L"\Omega", xlabel = L"R")
	ax.title = L"\text{Glauber global:}\;\alpha=%$(α),\;\beta=%$(β),\;\lambda=%$(Int.(λ))"
	ax.xticks = ([10,100,1_000,10_000,100_000], [L"10^1",L"10^2",L"10^3",L"10^4",L"10^5"])
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	mk.band!(ax, n_replicas_range,
		minimum(res, dims=2)[:], maximum(res, dims=2)[:]
	)
	mk.scatterlines!(ax, n_replicas_range, mean(res, dims=2)[:])
	mk.lines!(ax, n_replicas_range, repeat([exact_Ω], length(n_replicas_range)), color=:red)

	println("Final Ω: ", mean(res, dims=2)[end])
	println("Exact Ω: ", exact_Ω)

	# save(folderpath * "convergence_test4.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ Cell order:
# ╟─97f55482-4a7f-11f1-95be-95b2729f7c4f
# ╟─4375c013-ebcc-4fe4-884b-7276df877468
# ╟─363ea244-4a46-424d-affc-6b349eecb965
# ╟─dc39ebe7-458b-46ff-883c-df7d70efca12
# ╟─07ef8f99-f396-4e83-a76d-cf4e746ebbdf
# ╠═bbed48a9-9687-4b46-916c-df15562f1ce3
# ╠═5b26be7f-5841-4a66-b566-cea9671ecf1c
# ╟─318e6579-8726-489e-b0b8-a1f140a62128
# ╟─101a233f-a4d9-4f1f-87d7-97cc6f514d0e
# ╟─52801a9c-52c2-457d-9ffb-e4e492d5d3fd
# ╟─922b8c0d-949c-4f83-b637-6946249a1016
# ╟─38b99c56-6988-4aee-8a12-755b04a4f368
# ╟─c767fa4b-9e07-4e6d-8c21-eb423fa98b20
# ╟─65c6aeb6-fe5a-4eb6-b126-2721e5f2549d
# ╟─62535be9-55e3-4815-a646-3dc67ffca71f
# ╟─35588d8c-4c08-4d63-9414-694f925f1739
# ╟─7b39d6a7-897b-4586-b386-ef09d0c554ac
# ╟─01596671-7292-4865-a1fb-e53e342e1d9b
# ╟─f7c0f414-b400-4136-a032-9e2069dbbe2c
# ╟─75dae93b-0f35-4fba-87d8-0a5d623b556c
# ╟─9b726379-1607-45c1-9fd1-21485cf27b58
# ╟─0e577872-09f0-4d19-b3f2-19a573e347c8
# ╟─397a2f66-7a95-4feb-997b-c89e0f3213d9
# ╟─960e80a9-792d-4201-bc6a-b0e168a52f50
# ╟─89e50715-9dc4-45c0-a920-21f67d6832b3
# ╟─e9318f89-085b-4260-85da-7b9d59ca49c0
# ╠═17758d85-8bf8-4bb2-9576-ffa52267ebfc
# ╠═e91d69b1-6fb4-48b1-9edb-7a132a8cb988
# ╟─b9969be9-8831-4dae-9f50-a40203257a57
# ╟─26207d82-c289-4035-8e99-80f7fccb38a6
# ╟─faff830b-c5db-46df-9e0d-58dbaa752259
# ╟─3301410e-393f-40c3-bd8b-7eaf643b129c
