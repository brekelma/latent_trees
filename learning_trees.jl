using MATLAB

include("ipopt.jl")
include("utils.jl")

#function greedy_combine{T<:Real}(edges::Array{Tuple{T,T},1})
#function greedy_combine{T<:Tuple}(edges::Array{T,1})
function recursive_grouping{T <: Real}(samples::Array{T, 2}; cl::Bool = false)
	if cl
		cl_matrix = dict2array(chow_liu(mi_bin(samples))) # dictionary
		for i in internal_nodes(cl_matrix)
			continue
		end
	else	
		adj_mat, edge_distances = mxcall(:clrg, 2, matlab_samples(samples), 0)
	end
	return adj_mat, edge_distances
end

function internal_nodes{T<:Real}(cl_matrix::Array{T,2})
	internal = []
	for i=1:size(cl_matrix)[1]
		if sum([1 for j=(i+1):size(cl_matrix)[2] if cl_matrix[i,j] > 0.0]) > 1.0
			println("internal node: ", i)
			append!(internal, i)
		end
	end
	return internal
end

function chow_liu{T <: Real}(mi::Array{T,2}; min::Bool = false)
	edges = Dict{T, Tuple}()
	cl_tree = Dict{Tuple, T}()
	mis = []
	for i =1:size(mi)[2]
		for j =i+1:size(mi)[2]
			append!(mis, [min ? mi[i,j] : -1*mi[i,j]])
			edges[length(mis)] = sort_tuple((i,j))
		end
	end 
	for ind in sortperm(mis)
		candidate = edges[ind]
		i = candidate[1]
		j = candidate[2]
		cl_edges = collect(keys(cl_tree))
		if !connected(i,j, cl_edges, size(mi)[2])
		#all([!(sort_tuple((i,k)) in cl_edges && sort_tuple((j,k)) in cl_edges) for k = 1:size(mi)[2]])
			cl_tree[sort_tuple((i,j))] = -1*mis[ind]
		else
			#println("Cycle from edge ", i, ", ", j, " in ", cl_edges)
		end
	end
	return cl_tree
end

function generate_random_tree(dim::Int64, obs::Int64)
	# ensure all leaves are observed
	# hiddens at least degree 3, numbered obs+1:dim 
end

function connected(i::Int64, j::Int64, edges::Array{Tuple, 1}, dim::Int64)
	neighbors = [k for k=1:dim if sort_tuple((i,k)) in edges]
	#println("trying edge ", i, ", ",j, ", edges : ", edges)
	#println("neighbors: ", neighbors)
	visited = zeros(Bool, dim)
	while !isempty(neighbors)
		if j in neighbors
			return true
		end
		neighbors = [k for k=1:dim for n in neighbors if sort_tuple((n,k)) in edges && !visited[k]]
		visited = [ visited[k] || k in neighbors for k=1:dim]
	end
	return false
end


function threshold_params{T<:Tuple, S<:Real}(dict::Dict{T,S}; threshold::Float64 = 0.0, num_nodes::Int64 = 0)
	edges = collect(keys(dict))
	edges = [e for e in edges if length(e) >= 2] 
	nodes = Array{Int64, 1}()
	if threshold > 0.0
		thresh_edges = [e for e in edges if abs(dict[e]) > abs(threshold)]
		nodes = [v for e in thresh_edges for v in e]
		nodes = !isempty(nodes) ? unique(nodes) : nodes 
	else
		num_nodes = num_nodes > 0 ? num_nodes : 3
		sorted = edges[sortperm([-abs(dict[e]) for e in edges])]
		for e in sorted
			if length(nodes) >= num_nodes
				break
			else
				append!(nodes, [v for v in e if !(v in nodes)])
			end
		end
	end
	return nodes
end

function distance_algebra(nodes::Array{Int64, 1}, corrs::Array{Float64, 2}; calculated = Array{Float64, 2}(), addl_hidden = 0, threshold = 0.05, distances = false, max_dist = 4.6)
	# nodes = list of indices
	# corrs 
	# BE CLEVER ABOUT ON THE FLY CORRELATION CALCULATIONS
	max_val = 10.0^5
	adjacency = Array{Tuple, 1}()
	comparison_order = 3 # shouldnt change
	# INCORPORATE RECURSION (pre calculated distances of given size)

	if !distances
		dist = -log.(abs.(corrs))
	else
		dist = corrs
	end
	total_vars = size(corrs)[1]
	
	#calculated=Array{Float64, 2}()
	if isempty(calculated)
		calculated = zeros(size(corrs))
		#calculated = zeros(size(corrs)[1]+num_hidden, size(corrs)[2]+num_hidden)
	end 
	if length(nodes) < 3 
		error("Need 3 nodes to do distance algebra")
	end
	phi = Dict{Tuple, Float64}()
	nodes = sort(nodes)
	permlist = permutations(nodes, comparison_order, asymmetric=true)
	#println("Correlations")
	#pprint2d(corrs)
	#println()
	#perms = Dict{Tuple, Float64}()
	triplets = 0
	calcs = 0
	siblings = Array{Tuple, 1}()
	parents = Dict{Int64, Array{Int64, 1}}()
	for p in permlist
		if length(unique(p)) != comparison_order
			continue
		end
		tup = tuple([sort_tuple(p[1:2])...;p[3]]...)
		if !haskey(phi, tup)
			
			phi[tup] = max_val
	#end
			# IMPLEMENT OTHERS FOR HIDDEN NODES !!!! 
			(i,j,k) = tup
			if calculated[i, j] == 0
				calculated[i,j] = dist[i,j]
				calculated[j,i] = dist[j,i]
				calcs += 1
			end
			if calculated[i, k] == 0
				calculated[i,k] = dist[i,k]
				calculated[k,i] = dist[k,i]
				calcs += 1
			end
			if calculated[j, k] == 0
				calculated[j,k] = dist[j,k]
				calculated[k,j] = dist[k,j]
				calcs += 1
			end
			phi[(i,j,k)] = calculated[i,k] - calculated[j,k]
			#phi[(j,i,k)] = calculated[j,k] - calculated[i,k]
			triplets += 1
		end
	end
	considered = Array{Tuple, 1}()
	for (i,j,k) in keys(phi)
		if !((i,j) in considered)
			k_phis = [phi[(i,j,kk)] for kk in nodes if ((i,j,kk) in keys(phi) && calculated[i,kk]<=max_dist && calculated[j,kk]<=max_dist)|| ((j, i, kk) in keys(phi) && calculated[i,kk]<=max_dist && calculated[j,kk]<=max_dist)]
			#println("diff phis ", maximum(k_phis) - minimum(k_phis), " # k's ", length(k_phis))
			ks = [kk for kk in nodes if ((i,j,kk) in keys(phi) && calculated[i,kk]<=max_dist && calculated[j,kk]<=max_dist)|| ((j, i, kk) in keys(phi) && calculated[i,kk]<=max_dist && calculated[j,kk]<=max_dist)]
			#println("tuple : ", (i,j), " | phi :", [round(phi[(i,j,k)],3) for k in ks],  " other nodes:, ", ks)
			#println("distances used : ", [(calculated[i,k], calculated[j,k]) for k in ks])
			#println("all distances : ", [(calculated[i,k], calculated[j,k]) for k in nodes])
			if !isempty(k_phis) && maximum(k_phis) - minimum(k_phis) < threshold 
				if (phi[(i,j,k)] <= calculated[i,j] + threshold) && (phi[(i,j,k)] >= calculated[i,j] - threshold)
					# declare parent / child
					append!(adjacency, [(j,i)])
					if !haskey(parents, j)
						parents[j] = []
					end
					append!(parents[j], [i])
				elseif (phi[(i,j,k)] <= -calculated[i,j] + threshold) && (phi[(i,j,k)] >= -calculated[i,j] - threshold)
					# declare parent / child
					append!(adjacency, [(i,j)])
					if !haskey(parents, i)
						parents[i] = []
					end
					append!(parents[i], [j])
				elseif length(k_phis) > 1
					#println("siblings ", (i,j), " from key ", (i,j,k))
					append!(siblings, [(i,j)])
				end
			end
			println()
			append!(considered, [(i,j)])
		end
	end

	hidden_children = Array{Tuple,1}()
	siblings = unique(siblings)
	#(why not make this a dictionary => nodes or edge tuples...need tuples to calc distances)
	if !isempty(siblings)
		for tup in siblings
			addl_sib = [k for k in nodes if !(k in tup) && (sort_tuple((tup[1], k)) in siblings && sort_tuple((tup[2], k)) in siblings)]
			#println("tuple in siblings ", tup, " addl sibs: ", addl_sib)
			sib_tup = sort_tuple(tuple([tup[1:2]...;addl_sib...]...))

			if !(sib_tup in hidden_children)
				append!(hidden_children, [sib_tup])
			end
		end
	end
	# hidden dist
	hidden_dists = zeros(total_vars)
	
	#addl_hidden = 0
	hidden_edges = Dict{Int64, Array{Tuple, 1}}()
	for child_tup in hidden_children
		hidden_dists = zeros(total_vars + addl_hidden)

		d_ih = 0
		for i in child_tup
			d_ih = 0
			for j in child_tup
				if i != j
					#println("Searching tuple ", tuple([sort_tuple((i,j))...; 3]...))
					k_ij = [phi[tuple([sort_tuple((i,j))...; k]...)] for k in nodes if tuple([sort_tuple((i,j))...; k]...) in keys(phi)]
					d_ih += calculated[i,j] + 1.0/length(k_ij)*sum(k_ij)
					#println("i: ", i, " j: ", j, " k_ij: ", [tuple([sort_tuple((i,j))...; k]...) for k in nodes if tuple([sort_tuple((i,j))...; k]...) in keys(phi)])
				end
			end
			d_ih = 1/(2*length(child_tup)-1)*d_ih
			hidden_dists[i] = d_ih
		end
		for i in nodes 
			if !(i in child_tup)
				if i <= total_vars # Needs to be observed...
					d_ih = 1/(length(child_tup))*sum(calculated[i,k] - hidden_dists[k] for k in child_tup)
				else
					d_ih = 0.0
					println("not calculated for ", i, " child tup ", child_tup)
					#need to implement
				end
			end
			hidden_dists[i] = d_ih
		end
		#println("calculated size ", size(calculated), " hidden dist ", size(transpose(hidden_dists)))
		calculated = [calculated ; transpose(hidden_dists)]
		#println("appending to hidden_dists ", child_tup)
		append!(hidden_dists, [0])
		calculated = [calculated hidden_dists]
		addl_hidden +=1
		hidden_edges[total_vars+addl_hidden] = [(i, total_vars+addl_hidden) for i in child_tup]
	end
	println("siblings: ", siblings)
	println("hidden children ", hidden_children)
	return hidden_edges, calculated
end


permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partial_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    if order == 0
        return [tuple(partial_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partial_perm) > 0
                if partial_perm[end] >= item
                    continue
                end
            end
            perm = permutations(vcat(partial_perm, item), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end

function greedy_combine{T<:Tuple, S<:Real}(dict::Dict{T, S}; order::Int64=3, connect_all::Bool= true)
	# cliques of 3 
	edges = collect(keys(dict))
	prev_dim = maximum([i for theta in edges for i in theta])
	
	edges = [e for e in edges if length(e) >= 2] 
	new_dict = dict
	new_trees = Dict{Int64, Array{Int64, 1}}()
	new_edges = edges
	used_nodes = [] 
	obs = prev_dim
	hidden_links = []
	hidden_edges = []
	cliques = find_cliques(edges)


	if haskey(cliques, order+1)
		println("Higher Order Cliques ", cliques[order+1])
		println()
	end
	non_clique_interactions = setdiff([e for e in edges if length(e)==order], cliques[order])
	if !isempty(non_clique_interactions)
		println("NOTE: Three-body Interaction(s) not a clique : ", non_clique_interactions)
		append!(cliques[order], non_clique_interactions)
		println("Adding to cliques of order ", order)
	end
	# TO DO: alternative way of handling multiple cliques with given node?  max 3 bdoy / what about only edges? 
	for three_clique in cliques[order]
	#for three_clique in used_cliques
		reused = intersect([i for i in three_clique], used_nodes)
		for i in reused
			# # allow for now, add edges and hope to prune in recursive RISE
			#println("Warning: One node is in 2 different 3 cliques") 
			append!(hidden_links, [j for j=(obs)+1:prev_dim if (i,j) in new_edges])
			# println(cliques[3])
			# return false
		end
		if connect_all
			a = [i for tup in cliques[order] for i in setdiff(tup, reused) if !isempty(reused)]
			append!(new_edges, [sort_tuple((i,j)) for i in a, j in hidden_links if !(sort_tuple((i,j)) in new_edges)])
		end

		append!(used_nodes, [i for i in three_clique])
		# deleting edges is useful / necessary
		new_edges = [e for e in new_edges if length(intersect(e, three_clique))<2]
		
		#!(e[1] in three_clique && e[2] in three_clique && (length(e) < 3 || e[3] in three_clique))]
		for k in keys(new_dict)
			if !(e[1] in three_clique && e[2] in three_clique)
				delete!(new_dict, k)
			end
		end
		prev_dim += 1

		new_trees[prev_dim] = [i for i in three_clique]
		# add tree edges 
		#append!(new_edges, [tuple([prev_dim]...)])
		append!(new_edges, [(i, prev_dim) for i in three_clique])
		# TRIANGULATE multiple parents of clique nodes	
		append!(hidden_edges, [sort_tuple((i,prev_dim)) for i in hidden_links if (i != prev_dim && !(sort_tuple((i,prev_dim)) in hidden_edges))])
		#append!(new_edges, [sort_tuple((i,prev_dim)) for i in hidden_links if (i != prev_dim && !(sort_tuple((i,prev_dim)) in new_edges))])
		println()
		println("Adding 3-clique ", three_clique, ".  Three body exists? ", sort_tuple(three_clique) in edges)
	end		
	return new_edges, new_trees, hidden_edges
	# determine parent / child / sibling using correlations?
end

#function add_three_body{T<:Tuple, S<:Real}(dict::Dict{T, S}; combine_cliques::Bool = true)
#	edges = collect(keys(dict))
#end
	
#function find_cliques{T<:Real}(edges::Array{Tuple{T,T},1})
function find_cliques{T<:Tuple}(edges::Array{T ,1})
	k = 3
	cliques = Dict{Int64, Array{Tuple, 1}}()
	cliques[2] = [e for e in edges if length(e) == 2] 
	k_cliques = []
	while !isempty(cliques[k-1])
		cliques[k] = []
		for i = 1:length(cliques[k-1])
			for j = i:length(cliques[k-1])
				vertex_diff = symdiff(cliques[k-1][i], cliques[k-1][j])
				if length(vertex_diff) == 2
					if (vertex_diff[1], vertex_diff[2]) in edges || (vertex_diff[2], vertex_diff[1]) in edges
						append!(cliques[k], [tuple(union(cliques[k-1][i], cliques[k-1][j])...)])  
					end
				end	
			end
		end
		cliques[k]= unique([tuple(sort!([i for i in tup])...) for tup in cliques[k]])
		k = k + 1
	end
	return cliques
end

# function sort_tuple(t::Tuple)
# 	return tuple(sort!([i for i in t])...)
# end