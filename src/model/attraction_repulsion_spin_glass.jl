# ----- AttractionRepulsionSpinGlass model -----

"""
    AttractionRepulsionSpinGlass <: AbstractDiscreteModel

Adaptive vector spin glass with attraction/repulsion dynamics.

Each of N nodes has a G-dimensional spin vector sᵢ ∈ {-1, +1}^G (G must be odd).
Couplings are adaptive: Jᵢⱼ = Aᵢⱼ · sign(sᵢ · sⱼ).

Local energy of node i:
    Eᵢ = λᵢ · [(αᵢ/G) · Σ_{j:O>0} Oᵢⱼ + ((1-αᵢ)/G) · Σ_{j:O<0} |Oᵢⱼ|]

where Oᵢⱼ = sᵢ · sⱼ is the overlap (dot product) between spins.

# Fields
- `G::Int` — spin dimension (must be odd)
- `alpha` — mixing parameter in [0,1], scalar or per-node vector
- `lambda::Vector{Int8}` — homophily (-1) / heterophily (+1) per node
- `S::Matrix{Int8}` — state matrix (N × G), entries in {-1, +1}
"""
mutable struct AttractionRepulsionSpinGlass <: AbstractDiscreteModel
    G::Int
    alpha::Union{Float64, Vector{Float64}}
    lambda::Vector{Int8}
    S::Matrix{Int8}
end

"""
    AttractionRepulsionSpinGlass(; G, alpha=0.5, lambda, N)

Construct an AttractionRepulsionSpinGlass with uninitialized state.
Call `init_state!` before running a simulation.
"""
function AttractionRepulsionSpinGlass(; G::Int, alpha::Union{Float64, Vector{Float64}}=0.5,
                                        lambda::Vector{Int8}, N::Int)
    @assert isodd(G) "G must be odd"
    S = Matrix{Int8}(undef, N, G)
    return AttractionRepulsionSpinGlass(G, alpha, lambda, S)
end

# ----- Interface: state management -----

"""Initialize random spin state on the graph."""
function init_state!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph;
                     rng::AbstractRNG=Random.default_rng())
    N = nv(g)
    for i in 1:N, k in 1:model.G
        model.S[i, k] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    return model
end

"""Return a copy of the current state."""
get_state(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph) = copy(model.S)

"""Restore a previously saved state."""
function set_state!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, state::Matrix{Int8})
    copyto!(model.S, state)
    return model
end

# ----- Interface: energy -----

"""
    local_energy(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int)

Compute the local energy of node i (temperature-independent).
"""
function local_energy(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int)
    S = model.S
    G = model.G
    alpha_i = model.alpha isa Vector ? model.alpha[i] : model.alpha
    lambda_i = model.lambda[i]

    pos = 0
    neg = 0
    sᵢ = @view S[i, :]

    @inbounds for j in neighbors(g, i)
        sⱼ = @view S[j, :]
        Oᵢⱼ = sᵢ' * sⱼ  # overlap (dot product)
        if Oᵢⱼ > 0
            pos += Oᵢⱼ
        else
            neg += Oᵢⱼ
        end
    end

    T = (alpha_i / G) * pos + ((1 - alpha_i) / G) * abs(neg)
    return lambda_i * T
end

"""Compute the global energy H = Σᵢ local_energy(i)."""
function global_energy(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph)
    return sum(local_energy(model, g, i) for i in 1:nv(g))
end

"""Compute the global energy contribution from node i changes: H = local_energy(i) + Σⱼ local_energy(j).
This is used to compute ΔH for dynamics implementation (faster than computing ΔH from the global energy)."""
function global_energy_contribution_neighbours(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int)
    energy_contribution = local_energy(model, g, i)
    @inbounds for j in neighbors(g, i)
       energy_contribution += local_energy(model, g, j)
    end
    return energy_contribution
end

# ----- Interface: Metropolis proposal -----

"""Propose a local move: flip one random component of node i's spin."""
function propose!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int;
                  rng::AbstractRNG=Random.default_rng())
    k = rand(rng, 1:model.G)
    old_val = model.S[i, k]
    model.S[i, k] = -old_val
    return (component=k, old_val=old_val)
end

"""Revert a proposed move."""
function revert!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int, proposal)
    model.S[i, proposal.component] = proposal.old_val
    return model
end

# ----- Interface: Gibbs sampling -----

"""
    gibbs_sample!(model::AttractionRepulsionSpinGlass, g, i; local_only, beta, rng) -> Bool

Sample one component of node i's spin from its conditional distribution.
Returns whether the state changed.
"""
function gibbs_sample!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph, i::Int;
                       local_only::Bool=true, beta::Float64=1.0,
                       rng::AbstractRNG=Random.default_rng())
    k = rand(rng, 1:model.G)
    old = model.S[i, k]

    # energy with s_{i,k} = +1
    model.S[i, k] = Int8(1)
    H_plus = local_only ? local_energy(model, g, i) : global_energy_contribution_neighbours(model, g, i)

    # energy with s_{i,k} = -1
    model.S[i, k] = Int8(-1)
    H_minus = local_only ? local_energy(model, g, i) : global_energy_contribution_neighbours(model, g, i)

    # P(s=+1 | rest) ∝ exp(-β H_plus)
    p_plus = 1.0 / (1.0 + exp(-beta * (H_minus - H_plus)))
    new_val = rand(rng) < p_plus ? Int8(1) : Int8(-1)
    model.S[i, k] = new_val

    return new_val != old
end

# ----- Convenience: sync with graph properties -----

"""Copy model state into node properties on the graph (`:state` key)."""
function sync_to_graph!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph)
    for i in 1:nv(g)
        set_node_prop!(g, i, :state, model.S[i, :])
    end
    return nothing
end

"""Copy node properties (`:state` key) back into model state."""
function sync_from_graph!(model::AttractionRepulsionSpinGlass, g::AbstractDynGraph)
    for i in 1:nv(g)
        model.S[i, :] = get_node_prop(g, i, :state)
    end
    return nothing
end

# TODO
# add function to synch the couplings as well.
# This shuld be an edge property, or a graph property with the whole coupling matrix. Or maybe both?
# then using this, we can add some analysis functions to compute balance and antibalance, triangles etc
# these functions are easier to implement if we have a matrix I guess, so will probably just add graph properties

# ----- Helpers -----

"""
    random_lambda(N, L; rng) -> Vector{Int8}

Create a random λ vector: L heterophilous nodes (+1), N-L homophilous (-1).
"""
function random_lambda(N::Int, L::Int; rng::AbstractRNG=Random.default_rng())
    @assert 0 <= L <= N "L must be in [0, N]"
    v = fill(Int8(-1), N)
    idx = randperm(rng, N)[1:L]
    v[idx] .= Int8(1)
    return v
end

# TODO
# add a function to generate a random_alpha as well.
# This can either be a normally distributed α vector (with μ_α and σ_α) or a uniform distribution