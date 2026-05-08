# ----- Probability estimation and entropy -----
#
# All functions here operate on Matrix{Int} (samples × variables) or Vector{Int}.
# This is the frequency-based (plug-in) estimator.
# Other estimators (e.g., shrinkage, KSG for continuous) can be added
# by defining new estimator types.

# ----- Simple countmap (this is just to avoid StatsBase dependency) -----

"""
    _countmap(x::AbstractVector) -> Dict

Count occurrences of each unique element in `x`.
"""
function _countmap(x::AbstractVector)
    counts = Dict{eltype(x), Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return counts
end

# ----- Probability estimation -----

"""
    probabilities(X::Vector{Int}) -> Vector{Float64}

Estimate the empirical probability distribution of a single variable.
Returns only the probability values (not the keys).
"""
function probabilities(X::Vector{Int})
    n = length(X)
    counts = _countmap(X)
    return [v / n for v in values(counts)]
end

"""
    probabilities(X::Matrix{Int}) -> Vector{Float64}

Estimate the empirical joint probability distribution over all columns.
Each row is a sample, each column is a variable.
Returns only the probability values.
"""
function probabilities(X::Matrix{Int})
    n = size(X, 1)
    # use tuples of rows as joint keys
    counts = Dict{Vector{Int}, Int}()
    for i in 1:n
        row = X[i, :]
        counts[row] = get(counts, row, 0) + 1
    end
    return [v / n for v in values(counts)]
end

# ----- Shannon entropy -----

"""
    entropy(X) -> Float64

Compute the Shannon entropy H(X) in bits from data.
`X` can be `Vector{Int}` (single variable) or `Matrix{Int}` (joint, samples × variables).
"""
entropy(X::Union{Vector{Int}, Matrix{Int}}) = _entropy_from_probs(probabilities(X))

"""
    _entropy_from_probs(p::Vector{Float64}) -> Float64

Compute Shannon entropy in bits from a probability vector.
"""
function _entropy_from_probs(p::AbstractVector{Float64})
    H = 0.0
    for pᵢ in p
        if pᵢ > 0
            H -= pᵢ * log2(pᵢ)
        end
    end
    return H
end
