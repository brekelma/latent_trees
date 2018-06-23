using CSV
using PlotRecipes
#import PyPlot
include("mrf.jl")
include("math.jl")


function random_init_q(d::Int, order::Int; field = true, range = 1, seed = 0)
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
		tup[t] = rand(range)[1] # seed
	end 
	return tup
end

function random_init_p(d::Int, order::Int; field = true, range = 1, seed = 0)
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
		tup[t] = rand(range)[1] # seed
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


function plot_param_runs(run_dict::Dict{Tuple,Array{Any,1}}, param_values::Array{Any, 1} = []; param_name::String="", title::String="", reverse = false)
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
	labels = collect(keys(sample_kls))
	sample_kls = [sample_kls[i] for i in keys(sample_kls)]
	println(size(sample_kls), [size(sample_kls[i]) for i =1:length(sample_kls)])
	if isempty(param_name)
		xlab = "run" 
	else 
		xlab = string("param value ", param_name)
	end
	#xlabel = xlab, #", param_name
	println("try plot")
	plot(param_value, [kl for kl in sample_kls], label=[string(string(i), " samples") for i in labels], title = string("KL Divergence by Sample / Parameter "), legend = true, reuse = false)
	
	savefig("sample_runs.pdf")
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


function read_params(fn::String, delim="\t")
	df = CSV.read(fn; delim = "\t", header=0, types = [String, Float64], nullable = false)
	splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
	params = Dict{Tuple, Float64}()
	for r=1:size(df)[1]
		params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
	#println("Key: ", tuple(parse(Int64, split(df[r,1],',', keep = false)[i]) for i=1:edge_orders[r]))
	end
end

# PARAMS only... also add rand / copy instructions to the csv?
function read_params(fn::String, delim="\t")
	df = CSV.read(fn; delim = delim, header=0, types = [String, Float64], nullable = false)
	return df
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

	println("pearl corr")
	for i in keys(pearl_rhos)
		println(i, " : ", pearl_rhos[i])
	end
end

