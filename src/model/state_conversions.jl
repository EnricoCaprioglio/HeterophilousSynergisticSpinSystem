# ----- Spin state conversions -----
# Three representations:
#   Matrix{Int8}  — entries in {-1, +1}, shape (N, G). Used for computation.
#   BitMatrix     — Bool matrix (true = +1, false = -1), shape (N, G). Memory-efficient storage.
#   Vector{Int}   — integer-encoded states in 1:2^G, length N. Used for info theory.
#
# Note: G is the spin dimension. Converting FROM Vector{Int} requires knowing G.

# ----- bit/int helpers -----

"""Convert a single BitVector of length G to an integer in 1:2^G (Julia 1-indexed)."""
bit_to_int(s::BitVector, G::Int) = dot(s, 1 .<< (G-1:-1:0)) + 1

"""Convert an integer in 1:2^G to a BitVector of length G."""
int_to_bit(s::Int, G::Int) = BitVector(reverse(digits(s - 1, base=2, pad=G)))

"""Return all 2^G spin states as BitVectors."""
spin_state_space(G::Int) = [int_to_bit(s, G) for s in 1:2^G]

# ----- System state conversions -----

# identity
_state_convert(S::Matrix{Int8}, ::Type{Matrix{Int8}}; G=nothing) = S
_state_convert(S::BitMatrix, ::Type{BitMatrix}; G=nothing) = S
_state_convert(S::Vector{Int}, ::Type{Vector{Int}}; G=nothing) = S

# BitMatrix <-> Matrix{Int8}
_state_convert(S::BitMatrix, ::Type{Matrix{Int8}}; G=nothing) = ifelse.(S, Int8(1), Int8(-1))
_state_convert(S::Matrix{Int8}, ::Type{BitMatrix}; G=nothing) = S .== 1

# BitMatrix -> Vector{Int}
function _state_convert(S::BitMatrix, ::Type{Vector{Int}}; G=nothing)
    N, Gc = size(S)
    return [bit_to_int(BitVector(S[i, :]), Gc) for i in 1:N]
end

# Vector{Int} -> BitMatrix
function _state_convert(S::Vector{Int}, ::Type{BitMatrix}; G)
    @assert G !== nothing "G is required when converting from Vector{Int}"
    bitvecs = [int_to_bit(s, G) for s in S]
    return BitMatrix(reduce(hcat, bitvecs)')
end

# ----- combined conversions -----
_state_convert(S::Matrix{Int8}, ::Type{Vector{Int}}; G=nothing) =
    _state_convert(_state_convert(S, BitMatrix), Vector{Int})
_state_convert(S::Vector{Int}, ::Type{Matrix{Int8}}; G) =
    _state_convert(_state_convert(S, BitMatrix; G=G), Matrix{Int8})

"""
    state_convert(S, ::Type{T}; G=nothing) -> T

Convert between system state encodings:
- `Matrix{Int8}` — entries in `{-1, +1}`, shape (N, G)
- `BitMatrix` — Bool matrix, `true` = spin +1
- `Vector{Int}` — integer-encoded states in `1:2^G`

The keyword `G` is required when converting FROM `Vector{Int}`.
"""
state_convert(S, ::Type{T}; G=nothing) where {T} = _state_convert(S, T; G=G)

"""
    states_to_int_matrix(states::Vector{<:Matrix{Int8}}) -> Matrix{Int}

Convert a vector of state snapshots (each `Matrix{Int8}` of shape N×G) into
a `Matrix{Int}` of shape (T, N) with integer-encoded states. Used for info theory analysis.
"""
function states_to_int_matrix(states::Vector{<:Matrix{Int8}})
    int_vecs = [state_convert(s, Vector{Int}) for s in states]
    return reduce(hcat, int_vecs)' |> Matrix{Int}
end

"""
    states_to_int_matrix(states::Vector{BitMatrix}) -> Matrix{Int}

Convert a vector of BitMatrix snapshots into a `Matrix{Int}` of shape (T, N).
"""
function states_to_int_matrix(states::Vector{BitMatrix})
    int_vecs = [state_convert(s, Vector{Int}) for s in states]
    return reduce(hcat, int_vecs)' |> Matrix{Int}
end
