# ----- Information-theoretic measures -----
#
# All functions operate on Matrix{Int} (samples × nodes) or Vector{Int}.
# Convention: rows = samples/time steps, columns = variables/nodes.

const VecOrMatInt = Union{Vector{Int}, Matrix{Int}}

# ----- Conditional entropy -----

"""
    conditional_entropy(X, Y) -> Float64

Compute H(X|Y) = H(X,Y) - H(Y) in bits.
"""
function conditional_entropy(X::VecOrMatInt, Y::VecOrMatInt)
    XY = _hcat_data(X, Y)
    return entropy(XY) - entropy(Y)
end

# ----- Mutual information -----

"""
    mutual_information(X, Y; Z=nothing) -> Float64

Compute I(X;Y) in bits. If `Z` is provided, computes I(X;Y|Z).
"""
function mutual_information(X::VecOrMatInt, Y::VecOrMatInt;
                            Z::Union{Nothing, VecOrMatInt}=nothing)
    if Z === nothing
        return entropy(X) + entropy(Y) - entropy(_hcat_data(X, Y))
    else
        # I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        XZ = _hcat_data(X, Z)
        YZ = _hcat_data(Y, Z)
        XYZ = _hcat_data(X, Y, Z)
        return entropy(XZ) + entropy(YZ) - entropy(XYZ) - entropy(Z)
    end
end

# ----- Transfer entropy -----

"""
    transfer_entropy(X, Y; τ=1) -> Float64

Compute transfer entropy TE(X→Y) = I(Xₜ ; Y_{t+τ} | Yₜ) in bits.
"""
function transfer_entropy(X::VecOrMatInt, Y::VecOrMatInt; τ::Int=1)
    @assert τ >= 1 "τ must be ≥ 1"
    Y_future = _lag_forward(Y, τ)
    Y_past = _lag_backward(Y, τ)
    X_past = _lag_backward(X, τ)
    return mutual_information(X_past, Y_future; Z=Y_past)
end

"""
    total_transfer_entropy(X::Matrix{Int}; τ=1) -> Float64

Sum of all pairwise transfer entropies TE(Xᵢ→Xⱼ) for i ≠ j.
"""
function total_transfer_entropy(X::Matrix{Int}; τ::Int=1)
    _, N = size(X)
    tte = 0.0
    for i in 1:N, j in 1:N
        i == j && continue
        tte += transfer_entropy(X[:, i], X[:, j]; τ=τ)
    end
    return tte
end

# ----- High-order measures -----

"""
    total_correlation(X::Matrix{Int}) -> Float64

TC(X) = Σᵢ H(Xᵢ) - H(X) in bits. Measures total statistical dependence.
"""
function total_correlation(X::Matrix{Int})
    _, N = size(X)
    Hᵢ = [entropy(X[:, i]) for i in 1:N]
    H_joint = entropy(X)
    return sum(Hᵢ) - H_joint
end

"""
    dual_total_correlation(X::Matrix{Int}) -> Float64

DTC(X) = H(X) - Σᵢ H(Xᵢ|X₋ᵢ) in bits. Also called binding information.
"""
function dual_total_correlation(X::Matrix{Int})
    _, N = size(X)
    H_joint = entropy(X)
    # H(Xᵢ|X₋ᵢ) = H(X) - H(X₋ᵢ)
    residuals = [H_joint - entropy(X[:, setdiff(1:N, i)]) for i in 1:N]
    return H_joint - sum(residuals)
end

"""
    o_information(X::Matrix{Int}) -> Float64

O-information Ω(X) = TC(X) - DTC(X) in bits.
Ω > 0 → redundancy-dominated, Ω < 0 → synergy-dominated.
"""
function o_information(X::Matrix{Int})
    return total_correlation(X) - dual_total_correlation(X)
end

"""
    s_information(X::Matrix{Int}) -> Float64

S-information S(X) = TC(X) + DTC(X) in bits.
"""
function s_information(X::Matrix{Int})
    return total_correlation(X) + dual_total_correlation(X)
end

"""
    ho_summary(X::Matrix{Int}) -> NamedTuple{(:TC, :DTC, :Ω, :S)}

Compute all high-order measures at once (avoids redundant entropy computations).
"""
function ho_summary(X::Matrix{Int})
    _, N = size(X)

    H_joint = entropy(X)
    Hᵢ = [entropy(X[:, i]) for i in 1:N]
    H₋ᵢ = [entropy(X[:, setdiff(1:N, i)]) for i in 1:N]

    TC = sum(Hᵢ) - H_joint
    # H(Xᵢ|X₋ᵢ) = H_joint - H₋ᵢ
    residuals = [H_joint - H₋ᵢ[i] for i in 1:N]
    DTC = H_joint - sum(residuals)

    Ω = TC - DTC
    S = TC + DTC
    return (TC=TC, DTC=DTC, Ω=Ω, S=S)
end

# ----- Dynamical high-order measures -----

"""
    dynamical_o_information(X::VecOrMatInt, Y::VecOrMatInt; τ=1) -> Float64

Dynamical O-information from sources X to target Y.
Measures whether information transfer is redundancy- or synergy-dominated.
"""
function dynamical_o_information(X::VecOrMatInt, Y::VecOrMatInt; τ::Int=1)
    N = X isa Vector{Int} ? 1 : size(X, 2)
    term1 = (1 - N) * transfer_entropy(X, Y; τ=τ)
    term2 = sum(transfer_entropy(X[:, setdiff(1:N, j)], Y; τ=τ) for j in 1:N)
    return term1 + term2
end

"""
    total_dynamical_o_information(X::Matrix{Int}; τ=1) -> Float64

Total dynamical O-information: sum over all target nodes.
"""
function total_dynamical_o_information(X::Matrix{Int}; τ::Int=1)
    _, N = size(X)
    total = 0.0
    for j in 1:N
        sources = X[:, setdiff(1:N, j)]
        target = X[:, j]
        total += dynamical_o_information(sources, target; τ=τ)
    end
    return total
end

# ----- High-order measures over all triangles -----
function Oinfo3_components(g::DynGraph, data::Matrix{Int})

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
	
	@inbounds for (triangle_idx, triangle) in enumerate(triangle_list)

		Iᵢⱼ = zeros(3)
		Iᵢⱼ₋ₖ = zeros(3)
		Hᵢⱼₖ = entropy(data[:, triangle])
		
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

	return (Ω = Ω_list, Hᵢ=Hᵢ, Hᵢⱼ=Hᵢⱼ, u1=u1_list, u2=u2_list, MI = MI_matrix, CMI = CMI_list)
end

# ----- Test data generators -----

"""
    generate_xor(samples, N; cardinality=2, rng=Random.default_rng()) -> Matrix{Int}

Generate N-bit XOR test data: N-1 independent variables plus a parity variable.
Canonical example of pure synergy (Ω < 0).
"""
function generate_xor(samples::Int, N::Int; cardinality::Int=2,
                      rng::AbstractRNG=Random.default_rng())
    states = Matrix{Int}(undef, samples, N)
    for t in 1:samples
        bits = rand(rng, 0:cardinality-1, N-1)
        parity = sum(bits) % cardinality
        states[t, 1:N-1] = bits
        states[t, N] = parity
    end
    return states
end

# ----- Helpers -----

"""Concatenate data columns, handling Vector and Matrix inputs."""
function _hcat_data(args...)
    cols = []
    for x in args
        if x isa Vector
            push!(cols, reshape(x, :, 1))
        else
            push!(cols, x)
        end
    end
    return hcat(cols...)
end

"""Lag forward: X[(1+τ):end, :]"""
function _lag_forward(X::VecOrMatInt, τ::Int)
    return X isa Vector ? X[(1+τ):end] : X[(1+τ):end, :]
end

"""Lag backward: X[1:end-τ, :]"""
function _lag_backward(X::VecOrMatInt, τ::Int)
    return X isa Vector ? X[1:end-τ] : X[1:end-τ, :]
end
