# ----- DynGraph: core graph data structure -----

"""
    DynGraph <: AbstractDynGraph

A mutable graph with node and edge properties.
Adjacency matrix is stored as a `BitMatrix`  (undirected, symmetric, squared).

- `A::BitMatrix` — symmetric adjacency matrix
- `node_props::Dict{Int, Dict{Symbol, Any}}` — per-node properties
- `edge_props::Dict{Tuple{Int,Int}, Dict{Symbol, Any}}` — per-edge properties (keys are sorted tuples)
- `neighbors_list::Vector{Vector{Int}}` —  neighbor lists for each node (precomputed)
"""
mutable struct DynGraph <: AbstractDynGraph
    A::BitMatrix
    node_props::Dict{Int, Dict{Symbol, Any}}
    edge_props::Dict{Tuple{Int,Int}, Dict{Symbol, Any}}
    neighbors_list::Vector{Vector{Int}}
end

# ----- Constructors -----

"""
    DynGraph(N::Int)

Create an empty graph with `N` nodes and no edges.
"""
function DynGraph(N::Int)
    # just a matrix of falses
    A = falses(N, N)
    # initialize node and edge properties
    node_props = Dict{Int, Dict{Symbol, Any}}(i => Dict{Symbol, Any}() for i in 1:N)
    edge_props = Dict{Tuple{Int,Int}, Dict{Symbol, Any}}()
    neighbors_list = [Int[] for _ in 1:N]
    return DynGraph(A, node_props, edge_props, neighbors_list)
end

"""
    DynGraph(A::BitMatrix)

Create a graph from an existing adjacency matrix.
`A` must be symmetric with zero diagonal.
"""
function DynGraph(A::BitMatrix)
    N = size(A, 1)
    @assert size(A, 1) == size(A, 2) "adjacency matrix must be square"
    @assert A == A' "adjacency matrix must be symmetric (undirected graph)"
    @assert all(A[i,i] == false for i in 1:N) "adjacency matrix must have zero diagonal"

    node_props = Dict{Int, Dict{Symbol, Any}}(i => Dict{Symbol, Any}() for i in 1:N)
    edge_props = Dict{Tuple{Int,Int}, Dict{Symbol, Any}}()

    # initialise edge property dicts for existing edges
    for i in 1:N, j in (i+1):N
        if A[i, j]
            edge_props[(i, j)] = Dict{Symbol, Any}()
        end
    end

    neighbors_list = _build_neighbors(A, N)
    return DynGraph(A, node_props, edge_props, neighbors_list)
end

"""Build neighbor lists from adjacency matrix."""
_build_neighbors(A::BitMatrix, N::Int) = [findall(A[i, :]) for i in 1:N]

# ----- Get basic graph properties -----

"""Number of nodes."""
nv(g::DynGraph) = size(g.A, 1)

"""Number of edges."""
ne(g::DynGraph) = count(g.A) ÷ 2

"""Check whether edge (i, j) exists."""
has_edge(g::DynGraph, i::Int, j::Int) = g.A[i, j]

"""Return neighbours of node `i`."""
neighbors(g::DynGraph, i::Int) = g.neighbors_list[i]

"""Degree of node `i`."""
degree(g::DynGraph, i::Int) = count(g.A[i, :])

"""Degree vector for all nodes."""
degree(g::DynGraph) = [degree(g, i) for i in 1:nv(g)]

"""Returns triangle list"""
function triangles(g::DynGraph)
	N = nv(g)
	triangle_list = Vector{Vector{Int}}()
	for i in 1:N
		for j in Iterators.filter(>(i), neighbors(g, i))
			for k in Iterators.filter(>(j), neighbors(g, j))
				@inbounds if g.A[i,k]
					push!(triangle_list, [i,j,k])
				end
			end
		end
	end
	return triangle_list
end

# ----- Mutate the graph -----

# for undirected graphs, convention is to use small index first in the tuple to order the edges
# to avoid any issues, we define a simple util to never make the mistake of using the largest index first
"""Canonical edge key: sorted tuple (min, max)."""
_edge_key(i::Int, j::Int) = i < j ? (i, j) : (j, i)

"""
    add_edge!(g::DynGraph, i::Int, j::Int)

Add an undirected edge between nodes `i` and `j`.
"""
function add_edge!(g::DynGraph, i::Int, j::Int)
    @assert i != j "no self-loops in the graph. If you want these in the model, define it in the model itself using a node property"
    # add edge to network
    g.A[i, j] = true
    g.A[j, i] = true
    # update neighbor lists
    if j ∉ g.neighbors_list[i]
        push!(g.neighbors_list[i], j)
        sort!(g.neighbors_list[i])
        push!(g.neighbors_list[j], i)
        sort!(g.neighbors_list[j])
    end
    # initiate new edge property
    key = _edge_key(i, j)
    if !haskey(g.edge_props, key)
        g.edge_props[key] = Dict{Symbol, Any}()
    end
    return g
end

"""
    remove_edge!(g::DynGraph, i::Int, j::Int)

Remove the undirected edge between nodes `i` and `j`.
"""
function remove_edge!(g::DynGraph, i::Int, j::Int)
    # remove edge
    g.A[i, j] = false
    g.A[j, i] = false
    # update neighbor lists
    filter!(!=(j), g.neighbors_list[i])
    filter!(!=(i), g.neighbors_list[j])
    # remove property
    delete!(g.edge_props, _edge_key(i, j))
    return g
end

# ----- Node properties -----
# properties must be symbols. For example :weight, or :color or :nickname
# best to use the function below to not have issues to assign a new property, just in case. This should be more robust.
"""
    set_node_prop!(g::DynGraph, i::Int, key::Symbol, val)

Set property `key` to `val` for node `i`.
"""
set_node_prop!(g::DynGraph, i::Int, key::Symbol, val) = (g.node_props[i][key] = val)

# use the function below if you don't like typing square brackets really.
"""
    get_node_prop(g::DynGraph, i::Int, key::Symbol)

Get property `key` of node `i`.
"""
get_node_prop(g::DynGraph, i::Int, key::Symbol) = g.node_props[i][key]

"""
    get_node_prop(g::DynGraph, i::Int)

Get all properties of node `i`.
"""
get_node_prop(g::DynGraph, i::Int) = g.node_props[i]

# ----- Edge properties -----

"""
    set_edge_prop!(g::DynGraph, i::Int, j::Int, key::Symbol, val)

Set property `key` to `val` for edge (i, j).
"""
function set_edge_prop!(g::DynGraph, i::Int, j::Int, key::Symbol, val)
    k = _edge_key(i, j)
    @assert haskey(g.edge_props, k) "edge ($i, $j) does not exist"
    g.edge_props[k][key] = val
end

"""
    get_edge_prop(g::DynGraph, i::Int, j::Int, key::Symbol)

Get property `key` of edge (i, j).
"""
get_edge_prop(g::DynGraph, i::Int, j::Int, key::Symbol) = g.edge_props[_edge_key(i, j)][key]

"""
    get_edge_prop(g::DynGraph, i::Int, j::Int)

Get all properties of edge (i, j).
"""
get_edge_prop(g::DynGraph, i::Int, j::Int) = g.edge_props[_edge_key(i, j)]

# ----- Shortcuts (graph-tool style) -----

"""Shortcut for `get_node_prop`. Usage: `np(g, i, :state)` or `np(g, i)`."""
np(g::DynGraph, args...) = get_node_prop(g, args...)

"""Shortcut for `get_edge_prop`. Usage: `ep(g, i, j, :weight)` or `ep(g, i, j)`."""
ep(g::DynGraph, args...) = get_edge_prop(g, args...)

# TODOS
# - add and remove nodes (not sure whether I should actually do this. In the models I have in mind, it is easier to start already with the total number of nodes you may want, and then add a simple property like :status => active/inactive)