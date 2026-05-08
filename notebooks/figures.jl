### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 609ffe4a-dc33-48c8-86cd-dcf670365396
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
end

# ╔═╡ 96057253-077e-40fb-8cf1-a4e20e9e424c
begin
	Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()

    include(joinpath(@__DIR__, "..", "src", "HeterophilySynergy.jl"))
	println("HeterophilySynergy\nPS: do not run this cell twice!")
end

# ╔═╡ f1d34693-c035-4c9b-867e-81030c35306d
using Logging

# ╔═╡ d9819fdc-4a5e-11f1-131a-638b931665ac
md"""
---

## Contact Details

**Author:** Enrico Caprioglio

**Email:** ec627@sussex.ac.uk

---
"""

# ╔═╡ b08200a9-5409-4efc-825d-82d9404c7661
PlutoUI.TableOfContents()

# ╔═╡ 01f6a9ac-b4dc-4f0d-a8c1-702a24487c63
const mk = GLMakie

# ╔═╡ 0f03a51d-ea78-4766-aa0a-ee83248feba6
let
	using Graphs

	# ----- Functions -----
	# we need a couple of extra functions here:
	
	function compute_J(A::BitMatrix, S::Matrix{Int8})
		N, G = size(S)
		J = fill(Int8(0), N, N)
		for i in 1:N-1, j in i+1:N
			A[i,j] || continue
			overlap = count(S[i,k] == S[j,k] for k in 1:G)
			value = overlap > G/2 ? Int8(1) : Int8(-1)
			J[i,j] = value; J[j,i] = value
		end
		return J
	end

	function get_triangle_stats(J::Matrix{Int8})
		N = size(J, 1)
		out = Int8[]
		for i in 1:N-2
			for j in i+1:N-1
				J[i,j] == 0 && continue
				for k in j+1:N
					(J[j,k] != 0 && J[k,i] != 0) || continue
					push!(out, count(x -> x < 0, (J[i,j], J[j,k], J[k,i])))
				end
			end
		end
		return out
	end

	function fig3_get_data(N, G, β, λ, α_range; steps, seed, no_tests)
		store_n = zeros(4, length(α_range))
		for test in 1:no_tests
			graph = watts_strogatz(N, Int(ceil(N/10)), 0.2; seed=seed+test)
			A = BitMatrix(adjacency_matrix(graph))
			g = DynGraph(A)
			for (i, α) in enumerate(α_range)
				model = AttractionRepulsionSpinGlass(G=G, alpha=α, lambda=λ, N=N)
				res   = simulate(model, Gibbs(β, false), g,
				                 SimConfig(steps=steps, seed=seed+test, save_dt=steps, save_state=false))
				J = compute_J(A, res.model.S)
				dm = get_triangle_stats(J)
				no_Δ = length(dm);
				no_Δ == 0 && continue
				store_n[1,i] += count(==(0), dm) / no_Δ / no_tests
				store_n[2,i] += count(==(1), dm) / no_Δ / no_tests
				store_n[3,i] += count(==(2), dm) / no_Δ / no_tests
				store_n[4,i] += count(==(3), dm) / no_Δ / no_tests
			end
		end
		return store_n
	end

	# ----- Set parameters -----
	N = 50; seed = 1; steps = 1_000; no_tests = 20
	α_range = 0.0:0.025:1.0

	param_config = [
		(G=15, β=30.0, λ=fill(Int8(-1), N), title=L"\lambda_i=-1,\;G=15"),
		(G=15, β=30.0, λ=fill(Int8( 1), N), title=L"\lambda_i=+1,\;G=15"),
		(G=17, β=30.0, λ=fill(Int8( 1), N), title=L"\lambda_i=+1,\;G=17"),
	]

	fig = Figure(fontsize=20, size=(750, 250))
	axs = [Axis(fig[1, j]) for j in 1:3]

	for (j, cfg) in enumerate(param_config)
		store_n = fig3_get_data(N, cfg.G, cfg.β, cfg.λ, α_range; steps, seed, no_tests)
		ax = axs[j]
		mk.scatterlines!(ax, collect(α_range), store_n[1,:], label=L"n_0")
		mk.scatterlines!(ax, collect(α_range), store_n[2,:], label=L"n_1")
		mk.scatterlines!(ax, collect(α_range), store_n[3,:], label=L"n_2")
		mk.scatterlines!(ax, collect(α_range), store_n[4,:], label=L"n_3")
		ax.title = cfg.title
		ax.xticks = ([0, 1/4, 1/2, 3/4, 1])
		ax.xtickformat = x -> latexstring.(round.(x, digits=3))
		ax.ytickformat = x -> latexstring.(round.(x, digits=3))
		mk.ylims!(ax, 0, 1)
	end

	hideydecorations!.(axs[2:3]; label=false, grid=false)
	Label(fig[end+1, :], L"\alpha", valign=:top)
	Legend(fig[1, 0], axs[1])

	# folder = ""
	# filename = "triangles_large_N_example.png"
	# save(folder * filename, fig, px_per_unit = 400/96)


	fig
end

# ╔═╡ bf128101-9592-49a9-a398-ebd170d4f405
folderpath = joinpath(@__DIR__)[1:end-9] * "data/" # this just removes "notebook/"

# ╔═╡ be4aa52f-f145-41d7-8db2-53e5bf49282a
md"""
## Figure 2
see `notebooks/N3_results.jl`
"""

# ╔═╡ 71fa844e-0cb2-4eeb-ab66-62b3663b863e
# varying_alpha

# ╔═╡ 0b904fff-353f-4792-ae35-414304bd0d9d
md"""
## Figure 3
"""

# ╔═╡ d65cda47-4a97-4a96-8179-29a603ad02bf
md"""
## Figure 4
see `notebooks/N3_results.jl`
"""

# ╔═╡ 245d1034-b94f-44cc-84d1-b86d2e6e4b25
md"""
## Figure 5
"""

# ╔═╡ efcc3926-7c1b-421d-a340-61f1e026679e
md"""
For this figure, data was collected in `/data_collection/sectionsABC_experiments.jl`.
"""

# ╔═╡ 819f6a4b-21c9-4db1-aca9-8fb5eb7874df
function plot_res_var_alpha(ax, node_type, N, measure)

	measures = ["mean_u1", "mean_u2", "mean_Oinfo3"]
	node_types = ["homophilous", "heterophilous"]
	α_range = 0:0.05:1
	N_range = [10, 20, 30, 40, 50]
	
	cols = Dict(
		"homophilous" => Dict(
			10 => colorant"#B55400",
			30 => colorant"#E06D00",
			50 => colorant"#FF8C00"
		),
		"heterophilous" => Dict(
			10 => colorant"#3F007D",
			30 => colorant"#6A00A8",
			50 => colorant"#9C6ADE"
		),
	)

	label = L"N=%$(N)"
	
	res = with_logger(NullLogger()) do
		load_object(folderpath*"/varying_alpha/$(node_type)_N$(N)_G3_k5_epsilon0.2_beta20.0_localfalse_n_replicas10000_t_record100_no_tests10_starting_seed1_Oinfo_varying_alpha.jld2")
	end

	ys = zeros(length(α_range))
	yerr = zeros(length(α_range))
	
	for (j, α) in enumerate(α_range)
		ys[j] = mean(res.store_results[measure][α])
		yerr[j] = std(res.store_results[measure][α])
	end

	# mk.band!(ax, α_range, ys .- yerr, ys .+ yerr, color=cols[node_type][N], alpha=0.5)
	
	pl = mk.scatterlines!(ax, α_range, ys, color=cols[node_type][N], markersize = 8)
	mk.errorbars!(ax, α_range, ys, yerr, color=cols[node_type][N])

	return ax, pl, label
end

# ╔═╡ 8c44f1bd-fa00-4fa8-96dd-67e2b468aa35
let
	fig = Figure(size = (650,225), fontsize = 18)
	axs = [Axis(fig[1:4,i]) for i in 1:3]
	ylabels = [
		L"\langle u_1\rangle", L"\langle u_2\rangle", L"\langle\Omega_3\rangle"
	]
	for (i, ax) in enumerate(axs)
		ax.xtickformat = x -> latexstring.(round.(x, digits=3))
		ax.ytickformat = x -> latexstring.(round.(x, digits=3))
		ax.ylabel = ylabels[i]
	end

	measures = ["mean_u1", "mean_u2", "mean_Oinfo3"]
	node_types = ["homophilous", "heterophilous"]
	α_range = 0:0.05:1
	N_range = [10, 30, 50]

	het_handles = Any[]
	het_labels  = Any[]
	hom_handles = Any[]
	hom_labels  = Any[]

	for N in N_range
		for node_type in node_types
			for measure in measures
				
				if measure == "mean_u1"
					ax_idx = 1
				elseif measure == "mean_u2"
					ax_idx = 2
				elseif measure == "mean_Oinfo3"
					ax_idx = 3
				end

				ax, pl, label = plot_res_var_alpha(
					axs[ax_idx], node_type, N, measure
				)

				if node_type == "heterophilous" && measure == "mean_u1"
		            push!(het_handles, pl)
		            push!(het_labels,  label)
				elseif node_type == "homophilous" && measure == "mean_u1"
		            push!(hom_handles, pl)
		            push!(hom_labels,  label)
		        end
			end
		end
	end

	padding = (3.0f0, 3.0f0, 3.0f0, 3.0f0)
	mk.Legend(
		fig[1:2,4],
		hom_handles, hom_labels, L"\text{homophilous}";
		labelsize=12, titlesize=12,
		padding = padding,
		rowgap = .2f0,
	    titlegap = 2.0f0,
	    margin = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
		framevisible = true
	)

	leg1 = mk.Legend(
		fig[3:4,4],
		het_handles, het_labels, L"\text{heterophilous}";
		labelsize=12, titlesize=12,
		padding = padding,
		rowgap = .2f0,
	    titlegap = 2.0f0,
	    margin = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
		framevisible = true
	)
	mk.Label(fig[5, 1:3], L"\alpha")
	rowgap!(fig.layout, 4, 0)

	panel_labels = ["A", "B", "C"]
	for i in 1:3
		Label(fig[1, i, TopLeft()], panel_labels[i];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (10, 3, 3, -20),
	        tellwidth = false, tellheight = false,
	    )
	end
	
	# save(folderpath * "N10_30_50_numerical.png", fig, px_per_unit = 400/96)

	fig
end

# ╔═╡ 7edbd94c-f774-45b7-beab-010c09210862
md"""
## Figure 6
"""

# ╔═╡ 79853191-7e2d-4481-9253-b239f5b5706e
md"""
For this figure, data was collected in `/data_collection/sectionsABC_experiments.jl`.
"""

# ╔═╡ 471cb091-d126-4657-a373-b0ea134894e8
function plot_Ω_u1_u2!(ax, μ_α, β; baseline = true)
	
	res = with_logger(NullLogger()) do
		load_object(folderpath * "/robustness_experiments/N50_G3_k5_epsilon0.2_meanalpha$(μ_α)_beta$(β)_localtrue_n_replicas10000_t_record100_no_tests20_starting_seed1_Oinfo_varying_L.jld2")
	end

	L_range = 0:1:res.config.N
	measures = ["mean_Oinfo3", "mean_u1", "mean_u2"]

	cols = [colorant"#D55E00", colorant"#3F007D", colorant"#56B4E9"]
	
	ys = zeros(length(measures), length(L_range))
	max_ys = zeros(length(measures), length(L_range))
	min_ys = zeros(length(measures), length(L_range))
	yerr = zeros(length(measures), length(L_range))
	for (i, measure) in enumerate(measures)
		for L in L_range
			ys[i, L+1] = mean(res.store_results[measure][L])
			max_ys[i, L+1] = maximum(res.store_results[measure][L])
			min_ys[i, L+1] = minimum(res.store_results[measure][L])
			
			yerr[i, L+1] = std(res.store_results[measure][L])
		end
	end

	# get values for zero heterophilous elements
	ys_0 = [ys[1, 1], ys[2, 1], ys[3, 1]]
	
	ax.xtickformat = x -> latexstring.(round.(x, digits=3))
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	plots = []
	for (i, measure) in enumerate(measures)

		x = L_range ./ res.config.N
		y = []
		if baseline
			y = ys[i, :] ./ ys_0[i]
		else
			y = ys[i, :]
		end

		y_err = []
		if baseline

			SE_yp = yerr[i, :] ./ sqrt(10)
			SE_y0 = yerr[i, 1] ./ sqrt(10)
			
			y_err = abs.(ys[i, :] ./ ys_0[i]) .* sqrt.( (SE_yp ./ ys[i, :]).^2 .+ (SE_y0 ./ ys_0[i]).^2 )
		else
			y_err = yerr[i, :]
		end
		# y_err = yerr[i, :]
		
		
		max_y = max_ys[i, :]
		min_y = min_ys[i, :]
		
		# mk.band!(ax, x, y .- y_err, y .+ y_err, alpha = 0.5)
		pl = mk.scatter!(ax, x, y, color = cols[i], markersize = 5)
		push!(plots, pl)
		
		mk.errorbars!(ax, x, y, y_err, color = cols[i])
	end

	if baseline
		println("baseline Ω = $(ys_0[1])")
		println("baseline u1 = $(ys_0[2])")
		println("baseline u2 = $(ys_0[3])")
	end
	return ax, plots
end

# ╔═╡ 56792996-9e37-4688-afa3-33e5b664d142
function plot_triangles!(ax, μ_α, β)
	
	res = with_logger(NullLogger()) do
		load_object(folderpath * "triangle_analysis/N50_G3_k5_epsilon0.2_meanalpha$(μ_α)_beta$(β)_localtrue_n_replicas10_t_record100_no_tests100_starting_seed1_triangle_analysis_varying_L.jld2")
	end

	L_range = 0:1:50

	keys = ["mean_n₀", "mean_n₁", "mean_n₂", "mean_n₃"]
	ys = zeros(4, length(L_range))
	ys_err = zeros(4, length(L_range))
	for L in L_range
		for i in 1:4
			ys[i, L+1] = mean(res.store_results[L][keys[i]])
			ys_err[i, L+1] = std(res.store_results[L][keys[i]])
		end
	end

	# labels = [L"\langle n_0\rangle",L"\langle n_1\rangle",L"\langle n_2\rangle",L"\langle n_3\rangle"]
	labels = [L"n_0",L"n_1",L"n_2",L"n_3"]

	ax.xtickformat = x -> latexstring.(round.(x, digits=3))
	ax.ytickformat = x -> latexstring.(round.(x, digits=3))

	plots = []
	for i in 1:4
		pl = mk.scatter!(ax, L_range ./ 50, ys[i, :], label = labels[i], markersize = 7)
		mk.errorbars!(ax, L_range ./ 50, ys[i, :], ys_err[i, :])
		push!(plots, pl)
	end
	# mk.axislegend(ax, orientation=:horizontal)
	
	return ax, plots, labels
end

# ╔═╡ 37ced568-122b-42e8-85f6-ccde9f2c9a4e
let
	μ_α_range = [0.15, 0.4, 0.75]
	β = 20.0

	fig = Figure(size = (650, 500), fontsize = 18)

	baseline = false

	if baseline
		labels_measures = [
			L"\langle\hat{\Omega}_3\rangle", L"\langle\hat{u}_1\rangle", L"\langle\hat{u}_2\rangle"
		]
	else
		labels_measures = [
			L"\langle{\Omega_3}\rangle", L"\langle{u_1}\rangle", L"\langle{u_2}\rangle"
		]
	end
	if baseline
		ylabel_measures = L"\langle{\hat{X}(p)}\rangle"
	else
		ylabel_measures = L"\text{bits}"
	end
	
	plots_triangles=0; labels_triangles=0; plots_measures=0
	panel_labels = ["A","B","C","D","E","F"]

	for (i, μ_α) in enumerate(μ_α_range)

		if i == 3
			xlabel = L"p"
		else
			xlabel = ""
		end
		ax1 = Axis(fig[i,1], ylabel = ylabel_measures, xlabel=xlabel)
		ax1, plots_measures = plot_Ω_u1_u2!(ax1, μ_α, β; baseline=baseline)

		Label(fig[i, 1, TopLeft()], panel_labels[2i-1];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (10, 3, 3, -20),
	        tellwidth = false, tellheight = false,
	    )

		ax2 = Axis(fig[i,2], ylabel = L"n_k", xlabel=xlabel)
		ax2, plots_triangles, labels_triangles = plot_triangles!(ax2, μ_α, β)
		ax2.yticks = [0, 0.4, 0.8]
		mk.ylims!(ax2, -0.05, 1.05)

		Label(fig[i, 2, TopLeft()], panel_labels[2i];
	        font = :bold, fontsize = 22,
	        halign = :left, valign = :top,
	        padding = (10, 3, 3, -20),
	        tellwidth = false, tellheight = false,
	    )

	end

	padding = (5f0, 5f0, 5f0, 5f0)
	margin = (3f0, 3f0, 3f0, 3f0)
	mk.Legend(
		fig[0,1],
		plots_measures,
		labels_measures,
		labelsize=18,
		orientation=:horizontal,
		patchlabelgap = 0,
		markersize = 20,
		padding = padding,
		margin = margin,
		colgap=3,
	)
	mk.Legend(
		fig[0,2],
		plots_triangles,
		labels_triangles,
		markersize = 20,
		labelsize=18,
		orientation=:horizontal,
		padding = padding,
		patchlabelgap = 0,
		margin = margin,
		colgap=0
	)

	α_labels = [L"\mu_\alpha = 0.15", L"\mu_\alpha = 0.4", L"\mu_\alpha = 0.75"]
	for i in 1:3
		for j in 1:2
			mk.Label(
			    fig[i, j, TopRight()],
			    α_labels[i];
			    padding = (0, 20, 0, 0),
			    halign = :right, valign = :top,
			    tellwidth = false, tellheight = false,
			)
		end
	end
	
	# save(folderpath * "015_04_075_robustness_checks.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ ad56916f-6d9b-4812-9ab0-43f12da5d9ef
md"""
## Figure 7
"""

# ╔═╡ a6fccc6b-c7b8-4501-8b3a-141229b65e0c
md"""
For this figure, data was collected in `data_collection/polarization_experiment.jl`
"""

# ╔═╡ 87035a29-6090-47b3-9a93-16a0de2acc8d
function parse_run_filename(filename::AbstractString)
    m = match(r"^N(\d+)_beta([0-9]+(?:\.[0-9]+)?)_seed(\d+)_L(\d+)\.jld2$", filename)

    N = parse(Int, m.captures[1])
    beta = parse(Float64, m.captures[2])
    seed = parse(Int, m.captures[3])
    L = parse(Int, m.captures[4])

    return (N=N, beta=beta, seed=seed, L=L)
end

# ╔═╡ 598596a5-9795-4a36-ac02-6280ac5e6c1c
let
	N = 20
	max_t_record = 100
	t_record_step = 5

	# ----- plot set up ----------
	fig = Figure(fontsize = 28, size = (650, 600))
	ax1 = Axis(fig[1:2,1], xlabel = "", ylabel = L"\psi")
	ax2 = Axis(fig[3,1], xlabel = L"t_b", ylabel = L"\langle {d\Omega_3^\mathrm{tot}} \rangle")
	ax1.xtickformat = x -> latexstring.(round.(x, digits=3))
	ax1.ytickformat = x -> latexstring.(round.(x, digits=3))
	ax2.xtickformat = x -> latexstring.(round.(x, digits=3))
	ax2.ytickformat = x -> latexstring.(round.(x, digits=3))
	cols = [:red, :blue, :orange, :green, :purple]
	panel_labels = ["A","B", "C", "D"] # ,"C","D","E","F"]

	Label(fig[1:2, 1, TopLeft()], panel_labels[1];
		font = :bold, fontsize = 32,
		halign = :left, valign = :top,
		padding = (20, 0, 0, -20),
		tellwidth = false, tellheight = false,
	)
	Label(fig[3, 1, TopLeft()], panel_labels[2];
		font = :bold, fontsize = 32,
		halign = :left, valign = :top,
		padding = (20, 0, 0, -20),
		tellwidth = false, tellheight = false,
	)
	Label(fig[1, 2, TopLeft()], panel_labels[3];
		font = :bold, fontsize = 32,
		halign = :left, valign = :top,
		padding = (20, 0, 0, -20),
		tellwidth = false, tellheight = false,
	)
	Label(fig[2, 2, TopLeft()], panel_labels[4];
		font = :bold, fontsize = 32,
		halign = :left, valign = :top,
		padding = (20, 0, 0, -20),
		tellwidth = false, tellheight = false,
	)


	# ----- load data ----------

	overlaps_list0 = load_object(folderpath*"disrupting_polarization/data_for_histograms/N20_L0_seed42_data.jld2")
	overlaps_list1 = load_object(folderpath*"disrupting_polarization/data_for_histograms/N20_L20_seed42_data.jld2")

	hist_x = sort(unique(overlaps_list0)) # [-1, -1/3, 1/3, 1]
	hist_xticks = (hist_x, [L"-1", L"-1/3", L"1/3", L"1"])
	hist_xlabel = L"O_{ij}"
	hist_ylabel = L"%"
	axhist1 = Axis(fig[1,2], xticks = hist_xticks, xlabel = hist_xlabel, title = L"p=0", xlabelsize = 16, xticklabelsize=12, yticklabelsize=12, ylabel=hist_ylabel, titlesize=24)
	axhist2 = Axis(fig[2,2], xticks = hist_xticks, xlabel = hist_xlabel, title = L"p=1", xlabelsize = 16, xticklabelsize=12, yticklabelsize=12, ylabel=hist_ylabel, titlesize=24)

	cm1 = countmap(overlaps_list0)
	cm2 = countmap(overlaps_list1)
	hist1_y = [get(cm1, x, 0) for x in hist_x]
	hist1_y /= sum(hist1_y)
	hist2_y = [get(cm2, x, 0) for x in hist_x] 
	hist2_y /= sum(hist2_y)
	barplot!(axhist1, hist_x, hist1_y; width = 2/3, gap = 0.08, strokewidth = 0.5)
	barplot!(axhist2, hist_x, hist2_y; width = 2/3, gap = 0.08, strokewidth = 0.5)
	axhist1.ytickformat = x -> latexstring.(round.(x, digits=3))
	axhist2.ytickformat = x -> latexstring.(round.(x, digits=3))
	
	folder = folderpath*"disrupting_polarization/perturbation_experiment/"

	for (Lidx, L) in enumerate([0, 5, 10, 15, 20])
		filenames = readdir(folder)
		
		desired_filenames = []
		for filename in filenames
			if filename != ".DS_Store"
				params = parse_run_filename(filename)
				if params.N == N && params.L == L
					push!(desired_filenames, filename)
				end
			end
		end

		# ----- plot data ----------
		
		no_seeds = length(desired_filenames)
		dΩ = zeros(length(1:t_record_step:max_t_record), no_seeds)
		ψ = zeros(length(0:t_record_step:max_t_record), no_seeds)
		
		for (i, filename) in enumerate(desired_filenames)
			res = load_object(folder * filename)
			dΩ[:, i] = res[1]
			ψ_homophily = res[2]
			ψ[2:end, i] = res[3]
			ψ[1, i] = ψ_homophily # adds starting ψ at perturbation
		end

		
		x = collect(0:t_record_step:max_t_record)[1:11]

		p = L / N
		dΩ_mean = mean(dΩ, dims=2)[1:11]
		dΩ_std = std(dΩ, dims=2)[1:11]

		ψ_mean = mean(ψ, dims=2)[1:11]
		ψ_std = std(ψ, dims=2)[1:11]

		mk.scatterlines!(
			ax2, x[1:end], dΩ_mean, label = L"p = %$(p)",
			color=cols[Lidx]
		)
		
		mk.scatterlines!(
			ax1, x[1:end], ψ_mean,
			color=cols[Lidx]
		)
	end

	# ----- plot adjustments ----------
	mk.Legend(fig[3, 2], ax2, labelsize=24)
	ax1.xticklabelsvisible = false	

	gl = fig.layout

	colsize!(gl, 1, Relative(0.72))
	colsize!(gl, 2, Relative(0.28))
	
	rowsize!(gl, 1, Relative(0.2))
	rowsize!(gl, 2, Relative(0.2))
	rowsize!(gl, 3, Relative(0.6))

	# save(folderpath * "polarization-perturbation.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ cf786b17-d31e-435f-a98b-524b9a3d3f79
md"""
# Supplementary Information
"""

# ╔═╡ a93b9d70-8c63-494e-852c-7acbde72867c
md"""
## Figure S1
"""

# ╔═╡ 5ed7b870-b6c7-417b-a34a-a76ce3a8ce0f
md"""
For this figure, data was collected in `/data_collection/sectionsABC_experiments.jl`.
"""

# ╔═╡ 81633ebe-1add-4c84-9131-9ba16783ee66
let
	files = [
		folderpath*"varying_t_record/heterophilous_N30_G3_k5_epsilon0.2_beta5.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed1000000_Oinfo_varying_t_record.jld2", # heterophilous beta = 5
		folderpath*"varying_t_record/heterophilous_N30_G3_k5_epsilon0.2_beta10.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed2000000_Oinfo_varying_t_record.jld2", # heterophilous beta = 10.0
		folderpath*"varying_t_record/heterophilous_N30_G3_k5_epsilon0.2_beta20.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed3000000_Oinfo_varying_t_record.jld2", # heterophilous beta = 20.0
		folderpath*"varying_t_record/homophilous_N30_G3_k5_epsilon0.2_beta5.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed5010000_Oinfo_varying_t_record.jld2", # homophilous beta = 5.0
		folderpath*"varying_t_record/homophilous_N30_G3_k5_epsilon0.2_beta10.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed5000000_Oinfo_varying_t_record.jld2", # homophilous beta = 10.0
		folderpath*"varying_t_record/homophilous_N30_G3_k5_epsilon0.2_beta20.0_alpha0.4_localfalse_n_replicas10000_no_tests5_starting_seed5110000_Oinfo_varying_t_record.jld2", # homophilous beta = 20.0
	]

	fig = Figure(size = (650,550), fontsize = 20, figure_padding = (10, 10, 10, 30))
	
	
	axs = [Axis(fig[i,j], ylabel = L"\langle\Omega_3\rangle", xlabel = L"t_b") for j in 1:2 for i in 1:3]
	
	function plot_var_t(fig, ax, file)
		res = with_logger(NullLogger()) do
			load_object(file)
		end
		t_record_range = res.config.t_record_range[1:50]
	
		ys = []
		ys_err = []
		for t_record in t_record_range
			# append!(ys, res.store_results["mean_Oinfo3"][t_record])
			# or mean_Oinfo3, mean_u1, mean_u2
			push!(ys, mean(res.store_results["mean_Oinfo3"][t_record]))
			push!(ys_err, std(res.store_results["mean_Oinfo3"][t_record]))
		end
		# mk.scatter!(ax, repeat(t_record_range, inner=res.config.no_tests), ys)
		mk.scatter!(ax, t_record_range, ys)
		mk.errorbars!(ax, t_record_range, ys, ys_err)
		return fig, ax
	end
	
	for (i, file) in enumerate(files)
		plot_var_t(fig, axs[i], file)
		axs[i].ylabel = ""
		axs[i].xlabel = ""
		axs[i].xtickformat = x -> latexstring.(round.(x, digits=3))
		axs[i].ytickformat = x -> latexstring.(round.(x, digits=3))
	end

	β_labels = [L"\beta = 5", L"\beta = 10", L"\beta = 20"]
	for i in 1:3
		mk.Label(
			fig[i, 1, TopRight()],
			β_labels[i];
			padding = (0, 5, 0, 5),
			halign = :right, valign = :top,
			tellwidth = false, tellheight = false,
		)
		mk.Label(
			fig[i, 2, BottomRight()],
			β_labels[i];
			padding = (0, 5, 0, -30),
			halign = :right, valign = :top,
			tellwidth = false, tellheight = false,
		)
	end

	panel_labels = ["A","B","C","D","E","F"]
	function add_panel_label!(fig_panel, panel_label)
		Label(fig_panel, panel_label;
			font = :bold, fontsize = 24,
			halign = :left, valign = :top,
			padding = (-5, 3, 3, -30),
			tellwidth = false, tellheight = false,
		)	
	end
	add_panel_label!(fig[1, 1, TopLeft()], panel_labels[1])
	add_panel_label!(fig[1, 2, TopLeft()], panel_labels[2])
	add_panel_label!(fig[2, 1, TopLeft()], panel_labels[3])
	add_panel_label!(fig[2, 2, TopLeft()], panel_labels[4])
	add_panel_label!(fig[3, 1, TopLeft()], panel_labels[5])
	add_panel_label!(fig[3, 2, TopLeft()], panel_labels[6])

	Label(fig[4, 1:2], L"t_b", fontsize = 24)
	Label(fig[1:3, 0], L"\langle\Omega_3\rangle",fontsize = 24)
		
	# save(folderpath * "N30_sensitivity_t_b.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ 83f3e3b0-e20f-4f1f-816d-0d28ed1fdf5a
md"""
## Figure S2
"""

# ╔═╡ a57c6255-c9e4-44cf-a5fb-68ac251a29e2
md"""
For this figure, data was collected in `/data_collection/sectionsABC_experiments.jl`.
"""

# ╔═╡ a22bf2af-3da5-4d9f-b7fa-4e5aa29eba89
let
	files = [
		folderpath*"varying_beta/heterophilous_N10_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed100100_Oinfo_varying_beta_0_1_40.jld2", # heterophilous N = 10
		folderpath*"varying_beta/heterophilous_N20_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed101100_Oinfo_varying_beta_0_1_40.jld2", # heterophilous N = 20
		folderpath*"varying_beta/heterophilous_N30_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed10000_Oinfo_varying_beta_0_1_40.jld2", # heterophilous N = 30
		folderpath*"varying_beta/homophilous_N10_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed211100_Oinfo_varying_beta_0_1_40.jld2", # homophilous N = 10
		folderpath*"varying_beta/homophilous_N20_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed111100_Oinfo_varying_beta_0_1_40.jld2", # homophilous N = 20
		folderpath*"varying_beta/homophilous_N30_G3_k5_epsilon0.2_alpha0.4_localfalse_n_replicas10000_t_record50_no_tests10_starting_seed311100_Oinfo_varying_beta_0_1_40.jld2", # homophilous N = 30
	]
	
	fig = Figure(size = (650,550), fontsize = 20, figure_padding = (10, 10, 10, 30))
	axs = [Axis(fig[i,j], ylabel = L"\langle\Omega_3\rangle", xlabel = L"\beta") for j in 1:2 for i in 1:3]
	
	function plot_var_t(fig, ax, file)
		res = with_logger(NullLogger()) do
			load_object(file)
		end
		β_range = res.config.β_range
	
		ys = []
		ys_err = []
		for β in β_range
			push!(ys, mean(res.store_results["mean_Oinfo3"][β]))
			push!(ys_err, std(res.store_results["mean_Oinfo3"][β]))
		end
		
		mk.scatter!(ax, β_range, ys)
		mk.errorbars!(ax, β_range, ys, ys_err)
		return fig, ax
	end
	
	for (i, file) in enumerate(files)
		plot_var_t(fig, axs[i], file)
		axs[i].ylabel = ""
		axs[i].xlabel = ""
		axs[i].xtickformat = x -> latexstring.(round.(x, digits=3))
		axs[i].ytickformat = x -> latexstring.(round.(x, digits=3))
	end

	β_labels = [L"N = 10", L"N = 20", L"N = 30"]
	for i in 1:3
		mk.Label(
			fig[i, 1, TopRight()],
			β_labels[i];
			padding = (0, 5, 0, 5),
			halign = :right, valign = :top,
			tellwidth = false, tellheight = false,
		)
		mk.Label(
			fig[i, 2, BottomRight()],
			β_labels[i];
			padding = (0, 5, 0, -30),
			halign = :right, valign = :top,
			tellwidth = false, tellheight = false,
		)
	end

	panel_labels = ["A","B","C","D","E","F"]
	function add_panel_label!(fig_panel, panel_label)
		Label(fig_panel, panel_label;
			font = :bold, fontsize = 24,
			halign = :left, valign = :top,
			padding = (-5, 3, 3, -30),
			tellwidth = false, tellheight = false,
		)	
	end
	add_panel_label!(fig[1, 1, TopLeft()], panel_labels[1])
	add_panel_label!(fig[1, 2, TopLeft()], panel_labels[2])
	add_panel_label!(fig[2, 1, TopLeft()], panel_labels[3])
	add_panel_label!(fig[2, 2, TopLeft()], panel_labels[4])
	add_panel_label!(fig[3, 1, TopLeft()], panel_labels[5])
	add_panel_label!(fig[3, 2, TopLeft()], panel_labels[6])

	Label(fig[4, 1:2], L"\beta", fontsize = 24)
	Label(fig[1:3, 0], L"\langle\Omega_3\rangle",fontsize = 24)
		
	# save(folderpath * "sensitivity_beta.png", fig, px_per_unit = 400/96)
	
	fig
end

# ╔═╡ 1d01cd59-35ff-4be6-bd13-2ccf46ab88a4
md"""
## Figure S3
"""

# ╔═╡ 087fd71b-6528-4a7f-b409-618da30d99a6
md"""
see `notebooks/N3_results.jl`
"""

# ╔═╡ 7972a7d3-c5cf-4783-9725-6dfe4d0001e3
md"""
## Figure S4
"""

# ╔═╡ bdf7200f-a7c9-4aef-baff-292a40c2a01f
md"""
see `notebooks/N3_results.jl`
"""

# ╔═╡ e9401527-4c5c-411b-b500-74332cae29c2
md"""
## Figure S5
"""

# ╔═╡ 7305a13c-68c0-4951-b3b5-b33b65ae5073
md"""
For this figure, data was collected in `data_collection/polarization_experiment.jl`
"""

# ╔═╡ cb42a155-51ea-42d9-8958-a571d7cb7e26
let
	N = 20
	max_t_record = 100
	t_record_step = 5

	# ----- plot set up ----------
	fig = Figure(fontsize = 24, size = (650, 600))
	
	axs = [
		Axis(fig[1,1]), Axis(fig[1,2]),
		Axis(fig[2,1]), Axis(fig[2,2]),
		Axis(fig[3,1])
	]

	for ax in axs
		ax.xtickformat = x -> latexstring.(round.(x, digits=3))
		ax.ytickformat = x -> latexstring.(round.(x, digits=3))
	end
	
	cols = [:red, :blue, :orange, :green, :purple]
	
	folder = folderpath*"disrupting_polarization/perturbation_experiment/"

	for (Lidx, L) in enumerate([0, 5, 10, 15, 20])
		filenames = readdir(folder)
		
		desired_filenames = []
		for filename in filenames
			if filename != ".DS_Store"
				params = parse_run_filename(filename)
				if params.N == N && params.L == L
					push!(desired_filenames, filename)
				end
			end
		end

		# ----- plot data ----------
		
		no_seeds = length(desired_filenames)
		dΩ = zeros(length(1:t_record_step:max_t_record), no_seeds)
		ψ = zeros(length(0:t_record_step:max_t_record), no_seeds)
		
		for (i, filename) in enumerate(desired_filenames)
			res = load_object(folder * filename)
			dΩ[:, i] = res[1]
			ψ_homophily = res[2]
			ψ[2:end, i] = res[3]
			ψ[1, i] = ψ_homophily # adds starting ψ at perturbation
		end

		
		x = collect(0:t_record_step:max_t_record)[1:11]

		p = L / N
		dΩ_mean = mean(dΩ, dims=2)[1:11]
		dΩ_std = std(dΩ, dims=2)[1:11]

		for i in 1:size(dΩ, 2)
			mk.lines!(axs[Lidx], x, dΩ[1:11, i], color = cols[Lidx], alpha=0.3)
		end
		axs[Lidx].title = L"p = %$(p)"
		axs[Lidx].ylabel = L"\langle d\Omega_3^\mathrm{tot} \rangle"
		axs[Lidx].xlabel = L"t_b"
	end

	# ----- plot adjustments ----------
	panel_labels = ["A","B", "C", "D", "E"]
	function add_panel_label!(fig_panel, panel_label)
		Label(fig_panel, panel_label;
	        font = :bold, fontsize = 28,
	        halign = :left, valign = :top,
	        padding = (30, 3, 3, -15),
	        tellwidth = false, tellheight = false,
	    )
	end
	
	add_panel_label!(fig[1,1,TopLeft()], panel_labels[1])
	add_panel_label!(fig[1,2,TopLeft()], panel_labels[2])
	add_panel_label!(fig[2,1,TopLeft()], panel_labels[3])
	add_panel_label!(fig[2,2,TopLeft()], panel_labels[4])
	add_panel_label!(fig[3,1,TopLeft()], panel_labels[5])

	# save(folderpath * "polarization-perturbation-extra.png", fig, px_per_unit = 400/96)
	
	
	fig
end

# ╔═╡ Cell order:
# ╟─d9819fdc-4a5e-11f1-131a-638b931665ac
# ╟─96057253-077e-40fb-8cf1-a4e20e9e424c
# ╟─609ffe4a-dc33-48c8-86cd-dcf670365396
# ╠═b08200a9-5409-4efc-825d-82d9404c7661
# ╠═f1d34693-c035-4c9b-867e-81030c35306d
# ╠═01f6a9ac-b4dc-4f0d-a8c1-702a24487c63
# ╠═bf128101-9592-49a9-a398-ebd170d4f405
# ╟─be4aa52f-f145-41d7-8db2-53e5bf49282a
# ╠═71fa844e-0cb2-4eeb-ab66-62b3663b863e
# ╟─0b904fff-353f-4792-ae35-414304bd0d9d
# ╟─0f03a51d-ea78-4766-aa0a-ee83248feba6
# ╟─d65cda47-4a97-4a96-8179-29a603ad02bf
# ╟─245d1034-b94f-44cc-84d1-b86d2e6e4b25
# ╟─efcc3926-7c1b-421d-a340-61f1e026679e
# ╟─819f6a4b-21c9-4db1-aca9-8fb5eb7874df
# ╟─8c44f1bd-fa00-4fa8-96dd-67e2b468aa35
# ╟─7edbd94c-f774-45b7-beab-010c09210862
# ╟─79853191-7e2d-4481-9253-b239f5b5706e
# ╟─471cb091-d126-4657-a373-b0ea134894e8
# ╟─56792996-9e37-4688-afa3-33e5b664d142
# ╟─37ced568-122b-42e8-85f6-ccde9f2c9a4e
# ╟─ad56916f-6d9b-4812-9ab0-43f12da5d9ef
# ╟─a6fccc6b-c7b8-4501-8b3a-141229b65e0c
# ╟─87035a29-6090-47b3-9a93-16a0de2acc8d
# ╟─598596a5-9795-4a36-ac02-6280ac5e6c1c
# ╟─cf786b17-d31e-435f-a98b-524b9a3d3f79
# ╟─a93b9d70-8c63-494e-852c-7acbde72867c
# ╟─5ed7b870-b6c7-417b-a34a-a76ce3a8ce0f
# ╟─81633ebe-1add-4c84-9131-9ba16783ee66
# ╟─83f3e3b0-e20f-4f1f-816d-0d28ed1fdf5a
# ╟─a57c6255-c9e4-44cf-a5fb-68ac251a29e2
# ╟─a22bf2af-3da5-4d9f-b7fa-4e5aa29eba89
# ╟─1d01cd59-35ff-4be6-bd13-2ccf46ab88a4
# ╟─087fd71b-6528-4a7f-b409-618da30d99a6
# ╟─7972a7d3-c5cf-4783-9725-6dfe4d0001e3
# ╟─bdf7200f-a7c9-4aef-baff-292a40c2a01f
# ╟─e9401527-4c5c-411b-b500-74332cae29c2
# ╟─7305a13c-68c0-4951-b3b5-b33b65ae5073
# ╟─cb42a155-51ea-42d9-8958-a571d7cb7e26
