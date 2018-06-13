using CSV
using PlotRecipes

function display_model(m::MRF)
	source = Array{Int64, 1}()
	dest = Array{Int64, 1}()
	weights = Array{Float64, 1}()
	nodes = Array{Any, 1}()
	for coupling in keys(m.params)
		if length(coupling) == 1
			println(typeof(coupling[1]))
			append!(nodes, coupling[1])
		elseif length(coupling) == 2
			append!(source, coupling[1])
			append!(dest, coupling[2])
			append!(weights, m.params[coupling])
		else
			new_node = round(m.params[coupling], 2)
			append!(nodes, new_node)
			for i in coupling
				append!(source, i)
				append!(dest, new_node)
			end
		end
	end
	graphplot(source, dest, names=nodes, method=:tree, m =[length(split(n,'.', keep = false)) > 1 ? :yellow : :steelblue for n in nodes])
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
		println(i, "\t", dict[i])
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
end

function print_stats{T <: Real}(samples::Array{T, 2})
	num_conf = size(samples)[1]
	d = size(samples)[2]-1 # NOT ROBUST
	num_samp = sum(samples[k,1] for k=1:num_conf)
	mean = [sum(samples[k,1]/num_samp*samples[k,1+i] for k=1:num_conf) for i=1:d]
	corr = [sum(samples[k,1]/num_samp*samples[k,1+i]*samples[k,1+j] for k=1:num_conf) for i=1:d, j=1:d]
	println("mean spins ")
	println(mean)
	
	println("correlations ")
	pprint2d(corr)
	println()
end
