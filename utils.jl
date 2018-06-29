#using PyPlot
include("mrf.jl")
include("math.jl")
using PlotRecipes
#using Graphs
using GraphPlot
using LightGraphs, SimpleWeightedGraphs
using Colors
using Compose

function matlab_samples{T <: Real}(samples::Array{T, 2})
	matlab_samples = Array{T,2}()
	for k = 1:size(samples)[1]
		println(samples[k,1])
		samps = hcat([samples[k, 2:end] for i=1:samples[k,1]]...)
		println("samps ", size(samps), " matlab ", size(matlab_samples))
		matlab_samples = isempty(matlab_samples) ? samps : hcat(matlab_samples, samps)
	end
	println("total size : ", size(matlab_samples))
	return matlab_samples
end

function random_init_tree_3(d::Int, order::Int; field = true, range = [0,1], seed = 0)
	tup = Dict{Tuple, Float64}()
	tups = Array{Tuple, 1}()
	#for i = 1:order
		#append!(mat, Any[i:d])
	#end
	for i=1:d
		if field
			append!(tups, [(i,)])
		end
		if i != 4
			append!(tups, [(i, 4)])
		end
	end	
	for t in unique(tups) #product(mat...)
		tup[t] = rand()[1]*range[end]+range[1] # seed
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
		tup[t] = rand()[1]*range[end]+range[1] # seed
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


function plot_param_runs(run_dict::Dict{Tuple,Array{Any,1}}, param_values::Array{Float64, 1} = [], param_name::String=""; title::String="", reverse = false)
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
	idx_ones = [k for k in key if length(k)==1]
	idx_twos = [k for k in key if length(k)==2]
	ones = length(idx_ones)
	twos = length(idx_twos)
	
	l = @layout [ grid(ones,1) grid(twos,1) ]
	series_ones = [run_dict[i] for i in idx_ones]
	series_twos = [run_dict[i] for i in idx_twos]
	# arg sort
	# if sortit
	# 	s1 = sortperm(series_ones)
	# 	s2 = sortperm(series_twos)
	# 	series_ones = series_ones[s1]
	# 	series_two = series_twos[s2]
	# 	param_values1 = param_value[s1]
	# 	param_values2 = param_value[s2]
	# end

	left = scatter( param_values, vcat(inits ? [sort(i) for i in series_ones] : series_ones) , layout = grid(ones, 1), title = hcat([string(i[1], ", ") for i in idx_ones]...), markersize = mrk, titlefont = font(fnt))
	right = scatter( param_values, vcat(inits ? [sort(i) for i in series_twos] : series_twos) , layout = grid(twos, 1), title = hcat([string("(", i[1], ", ", i[2],")") for i in idx_twos]...), markersize = mrk, titlefont = font(fnt))
	lay = @layout [ a b ]
	#title =  string("Learned ", reverse ? "P" : "Q", " Params by ", inits ? "Init" : string("Parameter ", param_name))
	scatter(left, right, layout=lay, legend=false, reuse = false)
	#PyPlot.suptitle(string("Learned ", reverse ? "P" : "Q", " Params by ", inits ? "Init" : string("Parameter ", param_name)))
	#PyPlot.savefig("param_runs.pdf")
	savefig(string("param_runs", title, ".pdf"))
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
	plot(param_value, [kl for kl in sample_kls], label=[string(string(i), " samples") for i in labels], title = string("KL Divergence by Sample / Parameter "), legend = true, reuse = false)
	savefig("sample_runs.pdf")
end

function display_factor(m::MRF)
	_display_factor(m.params, m.dim)
end
function display_factor(m::FactorGraph)
	_display_factor(m.terms, m.variable_count)
end
function display_factor{T <: Real }(params::Dict{Tuple, T})
	dim = maximum([i for theta in keys(params) for i in theta])
	_display_factor(params, dim)
end
function _display_factor{T <: Real}(params::Dict{Tuple, T}, dim::Int64)

	nodelabels = zeros(dim)
	g = SimpleWeightedGraph(dim) #simple_graph(m.dim, is_directed = false)
	v = collect(LightGraphs.vertices(g))
	#wedges = Array{Any, 1}()
	for coupling in keys(params)
		
		sort_tuple(coupling)
		if length(coupling) == 1
			nodelabels[coupling[1]] = params[coupling]
		elseif length(coupling) == 2
			LightGraphs.add_edge!(g, v[coupling[1]], v[coupling[2]], params[coupling])
			#append!(wedges, [we])
		else
			str_name = "factor_"
			for i =1:length(coupling)
				str_name = string(str_name, string(coupling[i]))
			end
			LightGraphs.add_vertex!(g)#, str_name)
			v_ind = nv(g)
			for i =1:length(coupling)
				LightGraphs.add_edge!(g, v[coupling[i]], v_ind, params[coupling])
			end
			append!(nodelabels, [params[coupling]])
		end
	end
	node_fill = []
	for i=1:length(nodelabels)
		if i < dim
			node_fill[i] = append!(node_fill, [distinguishable_colors(3)[1]])
		else
			node_fill[i] = append!(node_fill, [distinguishable_colors(3)[2]])
		end
	end
	weights = zeros(ne(g))
	i = 0
	for wedge in LightGraphs.edges(g)
		i = i+1
		weights[i] = round(weight(wedge), 2)
		#weights[edge_index(wedge.edge, g)] = wedge.weight
	end #nodefillc = node_fill,
	nodelabels = [round(n, 2) for n in nodelabels]
	draw(PNG("learned_graph.png", 8cm, 8cm),gplot(g,  nodelabel = nodelabels, edgelabel = weights)) 
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


function read_params(fn::String; rand_init = false, range = [-1, 1], delim="\t", datarow = 2)
    df = CSV.read(fn; delim = delim, header=0, datarow = datarow, types = [String, Float64], allowmissing =:none)
	splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
	params = Dict{Tuple, Float64}()
	for r=1:size(df)[1]
		if rand_init
			params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = rand()[1]*range[end]+range[1]
		else
			params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
		end
	end
	return params
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

function pprint2d{T <: Real}(arr::Array{T, 2})
	for i = 1:length(arr[:,1])
		if i > 1
			print("\n")
		end
		for j = 1:length(arr[1,:])
			print(round(arr[i,j], 4), "\t")
		end
	end
	println()
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
	
	println("multiplicative correlations ")
	pprint2d(corr)
	println()

	println("covariances")
	pprint2d(cov)
	println()

	println("corr coeff")
	pprint2d(rho)

	#println("pearl corr")
	#for i in keys(pearl_rhos)
	#	println(i, " : ", pearl_rhos[i])
	#end
end

