#using PyPlot
include("mrf.jl")
include("math.jl")
using PlotRecipes
using Graphs
using GraphPlot
using LightGraphs, SimpleWeightedGraphs
using Colors
using Compose
using Combinatorics



function prune_params{T <: Real}(params::Dict{Tuple, T}; tol::Float64 = .01, prune_field::Bool = false)
	for k in keys(params)
		if abs(params[k]) < tol && !(prune_field && length(k)==1)
			delete!(params, k)
		end
	end
	return params
end


function prune_params{T <: Real, S <: Real}(params::Dict{Tuple, T}, true_params::Dict{Tuple, S}; prune_field::Bool = false)
	min_coupling = minimum([true_params[k] for k in keys(true_params) if length(k) >= 2])
	for k in keys(params)
		if abs(params[k]) < min_coupling/2 && !(prune_field && length(k)==1)
			delete!(params, k)
		end
	end
	return params
end

function sort_params{T <: Real}(params::Dict{Tuple, T})
	keyz = collect([keys(params)...])
	for j in maximum([length(k) for k in keyz]):-1:2
		keyz = keyz[sortperm([length(i)<j ? 0 : i[j] for i in keyz])]
	end
	keyz = keyz[sortperm([length(i) for i in keyz])]
	new_keys = keyz[sortperm([i[1] for i in keyz])]
	return new_keys
end

function dict2array{T<:Real}(dict::Dict{Tuple, T}; skip_higher::Bool = true, absolute_value::Bool = false)
	dim = maximum([i for theta in keys(dict) for i in theta])
	arr = zeros(dim, dim)
	for k in keys(dict)
		if length(k) == 2
			arr[k[1], k[2]] = absolute_value ? abs(dict[k]) : dict[k]
			arr[k[2], k[1]] = absolute_value ? abs(dict[k]) : dict[k]
		elseif length(k) > 1 
			if skip_higher
				#println("Warning : skipping higher order interactions in dictionary to array conversion")
			else
				for i=1:length(k)
					for j = i+1:length(k)
						arr[k[i], k[j]] += absolute_value ? abs(dict[k]) : dict[k]
						arr[k[j], k[i]] += absolute_value ? abs(dict[k]) : dict[k]
					end
				end
			end
		end
	end
	return arr
end

function array2dict{T <: Real}(dict::Dict{Any, T})
	return dict
end
function array2dict{T <: Real}(array::Array{T, 2})
	dict = Dict{Tuple, T}()
	for i=1:size(array)[1]	
		for j=1:size(array)[2]
			if i == j
				tup = (i,)
			else
				tup = sort_tuple((i,j))
			end
			if haskey(dict, tup)
				if dict[tup] != array[i,j]
					error("Array of reconstruction not symmetric")
				end
			else
				dict[tup] = array[i,j]
			end
		end
	end
	return dict
end

function check_structure(adj1::Array{Bool, 2}, adj2::Array{Float64, 2}, obs::Int64)
	if size(adj1)!=size(adj2)
		return false
	end
	for perm in permutations([obs+1:size(adj1)[1]...])
		p = append!([1:obs...], perm)
		if _check_structure(adj1[p, p], adj2)
			return true
		end
	end
	return false
end
	
function _check_structure(adj1::Array{Bool, 2}, adj2::Array{Float64, 2})
	for i=1:size(adj1)[1]
		for j=1:size(adj1)[2]
			if (abs(adj1[i,j])>0 && adj2[i,j] ==0) || (adj1[i,j]==0 && abs(adj2[i,j])>0)
				return false
			end
		end
	end
	return true
end

# function siblings{T <: Real}(dict::Dict{Tuple,T})
# 	sibling = Array{Tuple, 1}()
# 	edges = collect(keys(dict))
# 	nodes = unique([i for param in edges for i in param])
# 	for i in nodes
# 		println(i, " : ", [k for k in setdiff(nodes,i) for j in setdiff(nodes, i) if (sort_tuple((i,j)) in edges && sort_tuple((j,k)) in edges && !(sort_tuple((i,k)) in edges))])
# 		append!(sibling, tuple([k for k in setdiff(nodes,i) for j in setdiff(nodes, i) if sort_tuple((i,j)) in edges && sort_tuple((j,k)) in edges && !(sort_tuple((i,k)) in edges)]...))
# 	end
# end

function matlab_samples{T <: Real}(samples::Array{T, 2})
	matlab_samples = Array{T,2}()
	for k = 1:size(samples)[1]
		samps = repmat(samples[k, 2:end], 1, samples[k,1]) #hcat([samples[k, 2:end] for i=1:samples[k,1]]...)
		matlab_samples = isempty(matlab_samples) ? samps : hcat(matlab_samples, samps)
	end
	return matlab_samples
end

function random_init_multi(dim::Int64, order::Int64; range = [-1, 1], min_abs = 0.0)
    params = Dict{Tuple, Float64}()
    for current_spin = 1:dim
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        
        for p = 1:order
            nodal_keys = Array{Tuple{},1}()
            neighbours = [i for i=1:dim if i!=current_spin]
            if p == 1
                nodal_keys = [(current_spin,)]
            else
                perm = permutations(neighbours, p - 1)
                if length(perm) > 0
                    nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                end
            end

            for index = 1:length(nodal_keys)
                key = tuple(sort([i for i in nodal_keys[index]])...)
                if !haskey(params, key)
                    randnum = rand()[1]*(range[end]-range[1])+range[1]  
                    while abs(randnum) <  min_abs
                        randnum = rand()[1]*(range[end]-range[1])+range[1]  
                    end
                    params[key] = randnum
                end
            end
        end
    end
    return params
end



function random_init_tree_3(d::Int, order::Int; field = true, range = [-1,1], seed = 0)
	tup = Dict{Tuple, Float64}()
	tups = Array{Tuple, 1}()
	#for i = 1:order
		#append!(mat, Any[i:d])
	#end
	for i=1:d+1
		if field
			append!(tups, [(i,)])
		end
		if i != d+1
			append!(tups, [(i, d+1)])
		end
	end	
	for t in unique(tups) #product(mat...)
		tup[t] = rand()[1]*(range[end]-range[1])+range[1] # seed
	end 
	return tup
end

function random_init_dense(d::Int, order::Int; field = true, range = [-1,1], seed = 0, min_abs = .1)
	tup = Dict{Tuple, Float64}()
	tups = Array{Tuple, 1}()
	#for i = 1:order
		#append!(mat, Any[i:d])
	#end
	for i=1:d+1
		if field
			append!(tups, [(i,)])
		end
		for j=i+1:d+1
			append!(tups, [(i,j)])
		end
	end	
	for t in unique(tups) #product(mat...)
		rando = rand()[1]*(range[end]-range[1])+range[1] # seed = 
		while abs(rando) < min_abs
			rando = rand()[1]*(range[end]-range[1])+range[1]
		end
		tup[t] = rando 
	end 
	return tup
end

function random_init_p(d::Int, order::Int; field = true, range = [0,1], seed = 0)
	tup = Dict{Tuple, Float64}()
	tups = Array{Tuple, 1}()
	#for i = 1:order
		#append!(mat, Any[i:d])
	#end
	for i=1:d
		if field
			append!(tups, [(i,)])
		end
		for j=(1+i):d
			if order >=2
				append!(tups, [sort_tuple((i,j))])
			end
			for k=(1+j):d
				if order >= 3
					append!(tups, [sort_tuple((i,j,k))])
				end
			end
		end

	end	
	for t in unique(tups) #product(mat...)
		tup[t] = rand()[1]*(range[end]-range[1])+range[1] # seed
	end 
	return tup
end

function sort_tuple(t::Tuple)
	return tuple(sort!([i for i in t])...)
end

function params_to_dict(final_params::Array{Any,1})
	learned = Dict{Tuple, Array}()
	for i = 1:length(final_params)
		if !haskey(learned, final_params[i][1])
			learned[final_params[i][1]]= Array{Float64,1}()
		end 
		append!(learned[final_params[i][1]], final_params[i][2])
	end
	return learned
end

#function plot_param_runs(run_dict::Dict{Tuple,Array{Any,1}}, param_values::Array{Any, 1} = [], param_name::String=""; title::String="", reverse = false)
#	plot_param_runs(run_dict, float(param_values), param_name, title = title, reverse = reverse)
#end


function plot_param_runs(run_dict::Dict{Tuple,Array}, param_values::Array{Float64, 1} = [], param_name::String=""; title::String="", reverse = false, orders = [1,2])
	#fig_size = 
	if isempty(param_values)
		x_lbl = "random initialization #"
		k = collect(keys(run_dict))[1]
		param_values = [i for i=1:length(run_dict[k])]
		inits = true
	else
		inits = false
		x_lbl = string("param value ", param_name)
	end
	mrk = 2
	fnt = 10
	
	key = [k for k in keys(run_dict)]
	key_per_dim = fill(Array{Tuple, 1}(), (length(orders),)) #Array{Array, 1}() #)#
	lengths = zeros(length(orders))
	println("key ", typeof(key), " ", length(key))
	for i= 1:length(orders)
		if length(key_per_dim) < i
			append!(key_per_dim, [])
		end
		println("orders ", orders)
		println("k ", [k for k in key])
		println("len k ", [length(k) for k in key])
		append!(key_per_dim[i], [k for k in keys(run_dict) if length(k)==orders[i]])
		#idx_ones = [k for k in key if length(k)==orders[1]]
		#idx_twos = [k for k in key if length(k)==orders[2]]
	end
	println("len key_per_dim ", length(key_per_dim))
	println("length 1 ", key_per_dim[1])
	println("length 2 ", key_per_dim[2])
	lengths = [length(key_per_dim[i]) for i in 1:length(key_per_dim)]	
	#ones = length(idx_ones)
	#twos = length(idx_twos)
	#println("GRID ones: ", ones, " twos: ", twos)
	#l = @layout [ grid(ones,1) grid(twos,1) ]
	l = @layout [ hcat([grid(lengths[i], 1) for i =1:length(lengths)]...)]
	series = [[run_dict[j] for j in key_per_dim[i]] for i in 1:length(key_per_dim)]
	#series_ones = [run_dict[i] for i in idx_ones]
	#series_twos = [run_dict[i] for i in idx_twos]
	# arg sort
	# if sortit
	# 	s1 = sortperm(series_ones)
	# 	s2 = sortperm(series_twos)
	# 	series_ones = series_ones[s1]
	# 	series_two = series_twos[s2]
	# 	param_values1 = param_value[s1]
	# 	param_values2 = param_value[s2]
	# end
	scatters = fill(scatter(), (length(key_per_dim))) #Array{Any, 1}()
	for i=1:length(key_per_dim)
		scatters[i] = scatter(param_values, vcat(inits ? [sort(j) for j in series[i]] : series[i]) , layout = grid(lengths[i], 1), title = hcat([string(j[1], ", ") for j in key_per_dim[i]]...), markersize = mrk)#, titlefont = font(fnt)))
		#left = scatter( param_values, vcat(inits ? [sort(i) for i in series_ones] : series_ones) , layout = grid(ones, 1), title = hcat([string(i[1], ", ") for i in idx_ones]...), markersize = mrk)#, titlefont = font(fnt))
		#right = scatter( param_values, vcat(inits ? [sort(i) for i in series_twos] : series_twos) , layout = grid(twos, 1), title = hcat([string("(", i[1], ", ", i[2],")") for i in idx_twos]...), markersize = mrk)#, titlefont = font(fnt))
	end
	println(length(scatters), typeof(scatters[1]))
	#title =  string("Learned ", reverse ? "P" : "Q", " Params by ", inits ? "Init" : string("Parameter ", param_name))
	PlotRecipes.scatter(scatters, layout=l, legend=false, reuse = false, size = (20, 20))
	#working: lay = @layout [ a b ]
	#working: PlotRecipes.scatter(left, right, layout=lay, legend=false, reuse = false, size = (20, 20))
	
	#PyPlot.suptitle(string("Learned ", reverse ? "P" : "Q", " Params by ", inits ? "Init" : string("Parameter ", param_name)))
	#PyPlot.savefig("param_runs.pdf")
	savefig(string("param_runs_", title, ".pdf"))
	#display(plt)
	#plot([ hcat(run_dict[i] for i in keys(run_dict)) ], layout = l, title = [ hcat([i for i in keys(run_dict)]) ]  )
end

#function plot_param_variance(learned_params::Dict{Tuple,Array})
#	plot_param_variance(learned_params)
#end

function plot_param_variance(learned_params::Dict{Real, Dict{Tuple,Array{Any, 1}}})
	nsamples = collect(keys(learned_params))
 	x_params = collect(keys(learned_params[nsamples[1]]))
 	println("x params ", x_params)
 	println("num_samples ", typeof(nsamples), nsamples)
	for n in keys(learned_params)
		# plot each?
		#series = vcat[learned_params[n]
	end

end


# function plot_sample_runs{T <: Real, S <: Real}(sample_kls::Dict{T, Array{Any, 1}}, param_value::Array{S, 1}, param_name::String = ""; slacks::Any = [])
# 	labels = collect(keys(sample_kls))
# 	sample_kls = [sample_kls[i] for i in keys(sample_kls)]
# 	println(size(sample_kls), [size(sample_kls[i]) for i =1:length(sample_kls)])
# 	if isempty(param_name)
# 		xlab = "run" 
# 	else 
# 		xlab = string("param value ", param_name)
# 	end
# 	if isempty(slacks)
# 		plot(param_value, [kl for kl in sample_kls], label=[string(i, " samples") for i in labels], title = string("KL Divergence by Sample / Parameter ", param_name), xlabel = xlab, legend = true, reuse = false, show = true)
# 	else
# 		plot(param_value, [kl for kl in sample_kls], label=[string(i, " samples") for i in labels], title = string("KL Divergence by Sample / Parameter ", param_name), xlabel = xlab, legend = true, reuse = false, show = true)
# 	end
# 	plot(param_value, slacks)#label=[string(i, " samples") for i in labels], title = string("KL Divergence by Sample / Parameter ", param_name), xlabel = xlab, legend = true, reuse = false, show = true)
# 	savefig("sample_runs.pdf")
# end


# let sample_kls be a dictionary
function plot_sample_runs{T <: Real, S <: Real}(sample_kls::Dict{T, Array{Any, 1}}, param_value::Array{S, 1}, param_name::String = "")
	#pyplot()
	labels = collect(keys(sample_kls))
	sample_kls = [sample_kls[i] for i in keys(sample_kls)]
	println(size(sample_kls), [size(sample_kls[i]) for i =1:length(sample_kls)])
	if isempty(param_name)
		xlab = "run" 
	else 
		xlab = string("param value ", param_name)
	end
	#xlabel = xlab, #", param_name
	println("try plot", " labels ", labels, " sample_kls ", length(sample_kls))
	PlotRecipes.plot(param_value, [kl for kl in sample_kls], label=[string(string(i), " samples") for i in labels], title = string("KL Divergence by Sample / Parameter "), legend = true, reuse = false)
	savefig("sample_runs.pdf")
end

function display_factor(m::MRF, name::String="learned_graph"; field::Bool = false, observed::Int64=0)
	_display_factor(m.params, m.dim, name, field=field, observed = observed)
end
function display_factor(m::FactorGraph, name::String="learned_graph"; field::Bool = false, observed::Int64=0)
	_display_factor(m.terms, m.varible_count, name, field=field, observed = observed)
end
function display_factor{T <: Real }(params::Dict{Tuple, T}, name::String="learned_graph"; field::Bool = false, observed::Int64=0)
	dim = maximum([i for theta in keys(params) for i in theta])
	_display_factor(params, dim, name, field=field, observed = observed)
end
function _display_factor{T <: Real}(params::Dict{Tuple, T}, dim::Int64, name::String; field::Bool = false, observed::Int64=0, disp_field::Bool =false)
	nodelabels = [1.0*1:dim...]
	g = SimpleWeightedGraph(dim) #simple_graph(m.dim, is_directed = false)
	v = collect(LightGraphs.vertices(g))
	#wedges = Array{Any, 1}()
	for coupling in keys(params)
		sort_tuple(coupling)
		if length(coupling) == 1
			if disp_field
				nodelabels[coupling[1]] = params[coupling]
			end
		elseif length(coupling) == 2
			LightGraphs.add_edge!(g, v[coupling[1]], v[coupling[2]], params[coupling])
			#append!(wedges, [we])
		else
			factor_edge = false
			str_name = "factor_"
			for i =1:length(coupling)
				str_name = string(str_name, string(coupling[i]))
			end
			LightGraphs.add_vertex!(g)#, str_name)
			v_ind = nv(g)
			for i =1:length(coupling)
				if factor_edge
					LightGraphs.add_edge!(g, v[coupling[i]], v_ind)#, params[coupling])
				else
					LightGraphs.add_edge!(g, v[coupling[i]], v_ind) #params[coupling])
				end
			end
			if factor_edge
				append!(nodelabels, v_ind)
			else
				append!(nodelabels, params[coupling])
			end
		end
	end
	node_fill = [1:dim...]
	#println("node labels ", length(nodelabels), nodelabels)
	#println("OBSERVED ", obs)
	c1 = [201, 201, 201]
	c2 = [250, 235, 215]
	c3 = [255, 255, 0]
	c1 = RGB(c1[1]/255, c1[2]/255, c1[3]/255) #parse(Colorant, "blue")
	c2 = RGB(c2[1]/255, c2[2]/255, c2[3]/255) #parse(Colorant, "yellow")
	c3 = RGB(c3[1]/255, c3[2]/255, c3[3]/255) 
	node_fill = [i <= observed ? c1 : (i <= dim ? c2 : c3 ) for i =1:length(nodelabels)]
	#println(node_fill)
	weights = zeros(ne(g))
	i = 0
	for wedge in LightGraphs.edges(g)
		i = i+1
		weights[i] = round(weight(wedge), 2)
		#weights[edge_index(wedge.edge, g)] = wedge.weight
	end #nodefillc = node_fill,
	nodelabels = disp_field ? [round(n, 3) for n in nodelabels] : [n % 1 == 0 ? floor(Int, n) : round(n, 3) for n in nodelabels]
	#random_layout, circular_layout, spring_layout, stressmajorize_layout, shell_layout, spectral_layout
	l = length(weights) < length(nodelabels)-1 ? shell_layout : spring_layout
	Compose.draw(PDF(string(name, ".pdf"), 10cm, 10cm), gplot(g, layout = l, nodelabel = nodelabels,  edgelabel = weights, nodefillc = node_fill))  #
	#gplot(g, nodefillc = node_fill, nodelabel = nodelabels, edgelabel = weights)
end



function vis_mrf(m::MRF)
	source = Array{Int64,1}()
	dest = Array{Int64,1}()
	weights = Array{Float64,1}()
	nodes = Array{Any,1}()
	nodew = Array{Any,1}()

	obs = size(m.samples[1])[2] - 1 - (isa(m, mrf) ? 0 : length(m.hsupport[1]))
	
	_keys = [i for i in keys(m.params)]
	sorted = _keys[sortperm([j[1] for j in _keys])]
	sorted = sorted[sortperm([length(j) for j in sorted])]
	for coupling in sorted
		println(coupling)
		if length(coupling) == 1
			append!(nodes, coupling[1])
			append!(nodew, m.params[coupling])
		elseif length(coupling) == 2
			append!(source, coupling[1])
			append!(dest, coupling[2])
			append!(weights, m.params[coupling])
		else
			new_node = length(nodes)+1
			append!(nodes, new_node)
			append!(nodew, round(m.params[coupling], 2))
			for i in coupling
				append!(source, i)
				append!(dest, new_node)
			end
		end
	end
	println("weights ", weights)
	#fontsize, nodeshape, and nodesize 
	graphplot(source, dest, weights, names=nodes, curves=false, root = :left, node_weights = nodew, nodeshape = :circle, nodesize = 6, fontsize = 14, 
		l = (4, cgrad()), method=:tree, m =[n > obs ? :yellow : :steelblue for n in nodes])
end

function read_params(fn::String; rand_init = false, range = [-1, 1], field = true, field_range = [], min_abs = 0, delim="\t", datarow = 2)
    df = CSV.read(fn; delim = delim, header=0, datarow = datarow, types = [String, Float64], allowmissing =:none)
	splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
	params = Dict{Tuple, Float64}()
	
	for r=1:size(df)[1]
		if rand_init
			if field || length(splits[r])>1
				if field && !isempty(field_range)
					randnum = rand()[1]*(field_range[end]-field_range[1])+field_range[1]
				else
					randnum = rand()[1]*(range[end]-range[1])+range[1]
				end
				if length(splits[r]) > 1 || isempty(field_range)
					while abs(randnum) <  min_abs
						randnum = rand()[1]*(range[end]-range[1])+range[1]	
					end
				end
				params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = randnum
			end
		else
			#println("adding param ", df[r,2])
			params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
		end
	end
	return params
end

function print_params_latex{T <: Real}(dict::Dict{Tuple, T})
	for k in sort_params(dict)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & ", round(dict[k],3), " \\\\ ")
		#println("\t ", round(p_params[k],3))#" \\\\ ")
	end
end

function print_params{T <: Real}(dict::Dict{Tuple, T})
	_keys = [i for i in keys(dict)]
	sorted = _keys[sortperm([j[1] for j in _keys])]
	sorted = sorted[sortperm([length(j) for j in sorted])]
	for i in sorted
		println(i, "\t", round(dict[i],3))
	end
	println()
end

function pprint2d{T <: Real}(arr::Array{T, 2}; latex = true, rounding = 4)
	for i = 1:length(arr[:,1])
		if i > 1
			print("\n")
		end
		for j = 1:length(arr[1,:])
			print(round(arr[i,j], rounding), "\t")
			if latex && j != length(arr[1,:])
				print("& ")
			end
		end
	end
	if latex
		println(" \\\\")
	else
		println()
	end
end


function print_stats{T <: Real}(samples::Array{T, 2})
	num_conf = size(samples)[1]
	d = size(samples)[2]-1 # NOT ROBUST
	num_samp = sum(samples[k,1] for k=1:num_conf)
	#mean = [sum(samples[k,1]/num_samp*samples[k,1+i] for k=1:num_conf) for i=1:d]
	#corr = [sum(samples[k,1]/num_samp*samples[k,1+i]*samples[k,1+j] for k=1:num_conf) for i=1:d, j=1:d]
	#cov = [corr[i,j] - mean[i]mean[j] for i=1:d, j=1:d]

	mean = means(samples)
	cov = covs(samples)
	corr = corrs(samples)
	rho = corrs(samples, pearson = true)
	_, pearl_rhos, _ = pearl_sandwich_full_cov(samples)

	println("mean spins ")
	println(mean)
	
	println("multiplicative correlation ")
	pprint2d(corr)
	println()

	# println("covariances")
	# pprint2d(cov)
	# println()

	println("reduced correlation")
	pprint2d(rho)

	#println("pearl corr")
	#for i in keys(pearl_rhos)
	#	println(i, " : ", pearl_rhos[i])
	#end
end

