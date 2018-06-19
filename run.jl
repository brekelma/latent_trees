using GraphicalModelLearning
using CSV

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
num_samp = 10000000
vary_samples = [100000]
tol = .00001
verbose = false
ipopt = true
q_field = true 
random_p = false

# Vary 3 body strength?
vary_param = true
varied = (1,2,3)


runs = 10
range3body = 2
# Vary inits to test convexity? 
vary_inits = true #false
# KL(q||p)? best 3 body for given tree
reverse = false

samples = Array{Real,2}
kls = Array{Float64, 1}()
final_params = Array{Any, 1}()

#if vary_3_body
interactions = Array{Float64, 1}()
triangle_test = Array{Any, 1}()
corr_test = Array{Any, 1}()
#end

function read_params(fn::String = "example.csv")#, fn_q::String = "q.csv"; verbose = false, ipopt = true)
	df = CSV.read(fn; delim = "\t", header=0, types = [String, Float64], nullable = false)
	splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
	param = Dict{Tuple, Float64}()
	for r=1:size(df)[1]
		param[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
		#println("Key: ", tuple(parse(Int64, split(df[r,1],',', keep = false)[i]) for i=1:edge_orders[r]))
	end
	return param
end

params = read_params("example.csv")
min_param = params[varied]

sample_kls = Array{Any, 1}()

for s = 1:length(vary_samples)
	qq = Dict{Any, Any}()
	q_params = Dict{Any, Any}()
	
	if !random_p
		params = read_params("example.csv")
		min_param = params[varied]
	end
	interactions = Array{Float64, 1}()
	triangle_test = Array{Any, 1}()
	corr_test = Array{Any, 1}()
	
	for z = 1:runs
		if z > 1 && vary_param
			params[varied] = range3body / runs * z + min3
			#println("run ", z, ": ",params[(1,2,3)])
		end

		d =  maximum([i for theta in keys(params) for i in theta])
		order = maximum([length(i) for i in keys(params)])
		if random_p
			params = random_init_p(d, order, field = q_field)
			println("random p init ", [k for k in keys(params)])
		end
		if s == 1
			append!(interactions, params[varied])
		end

		if vary_inits
			#if z == 1
		#		q_params = random_init(d+1, order, field = q_field)
		#	else
		#		for i in keys(q_params)
	#				q_params[i] = q_params[i]*-1
	#			end
	#		end
			# back to normal
			q_params = random_init_q(d+1, order, field = q_field)
		else
			q_params = read_params("q.csv")
		end


		# hack for hidden

		#println([k for k in keys(params)], d)
		if reverse
			dh = maximum([i for theta in keys(q_params) for i in theta])
			order = maximum([length(i) for i in keys(q_params)])
			model = FactorGraph(order, dh, :spin, q_params)
		else
			model = FactorGraph(order, d, :spin, params)
		end	
		samples = sample(model, num_samp)
		num_conf = size(samples)[1]

		p = mrf(params, samples)
		a= [hcat(samples, fill(h_i, size(samples)[1])) for h_i in [-1,1]]
		#println(size(a),typeof(a))
		q = hmrf(q_params, samples, [-1,1])

		if verbose
			print_params(params)
			print_stats(samples)
			println("q params initialization")
			print_params(q.params)
			println()
			println("Corr Test for P model")
		end

		if !reverse
			recoverable, correlations, marginal = pearl_sandwich(p)
			#min_corr, correlations = pearl_corr_test(p)
		end

		kl = 0
		if ipopt 
			if reverse
				kl = min_kl(q, p)
			else
				kl = min_kl(p, q)
			end
		else
			kl = min_kl_manual(p, q, verbose = verbose)
		end
		append!(kls, kl)
		
		append!(triangle_test, recoverable)
		append!(corr_test, [Array([correlations[1,2], correlations[2,3], correlations[1,3]])])
		#append!(triangle_test, min_corr)
		#append!(corr_test, correlations)

		if reverse
			append!(final_params, p.params)
		else
			append!(final_params, q.params)
		end
	end
	append!(sample_kls, kls)
end
#vis_mrf(q)

println()
println()
println("3 Body Couplings")
println(interactions)
println()

if vary_inits
	learned = params_to_dict(final_params)
	println()
	println("parameter variance")
	for i in keys(learned)
		println(i, " : ", var(learned[i]))
	end
end
println("objective")
println("mean: ", round(mean(kls), 4), "    variance: ", var(kls))
println("kl min ", minimum(kls), " kl max ", maximum(kls))
println()
println("Pearl Test")
println([(kls[i], triangle_test[i] && prod(corr_test[i])>0, sum(corr_test[i][k] > 0 for k=1:length(corr_test[i])))  for i=1:length(triangle_test)])

println()
println("p stats")
print_stats(samples)


plot_sample_runs(sample_kls, interactions, vary_samples)
plot_param_runs(learned)
