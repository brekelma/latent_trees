using CSV
using GraphicalModelLearning

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")

vary_samples = [1000000]
tol = .00001
verbose = false
ipopt = true
field = true

# Vary 3 body strength?
vary_param = false
varied = (1,2,3)
rand_init = false

runs = 5#10000
range_param = 4
# Vary inits to test convexity? 
vary_inits = false
# KL(q||p)? best 3 body for given tree
reverse = true

samples = Array{Real,2}()
final_params = Array{Any, 1}()
init_params = Array{Any, 1}()

kls = Array{Float64, 1}()
#if vary_3_body
interactions = Array{Float64, 1}()
triangle_test = Array{Any, 1}()
#triangle_slack = Array{Any, 1}()
triangle_slack = Array{Array{Any,1}, 1}()
corr_test = Array{Any, 1}()
#end
z =0
p = mrf()
q = mrf()
params = read_params("example.csv")
min_param = params[varied]

q_params = Dict{Tuple, Float64}()
sample_kls = Dict{Real, Array{Any, 1}}()
learned_params = Dict{Real, Dict{Tuple, Array{Any, 1}}}()
inits = Dict{Real, Dict{Tuple, Array{Any, 1}}}()

for s = 1:length(vary_samples)
	q_params = Dict{Tuple, Float64}()
	num_samp = vary_samples[s]

	if !(vary_inits && reverse)
		params = read_params("example.csv")
		min_param = params[varied]
	end
	kls = Array{Float64, 1}()
	interactions = Array{Float64, 1}()
	triangle_test = Array{Any, 1}()
	corr_test = Array{Any, 1}()
	
	for z = 1:runs
		if z > 1 && vary_param
			params[varied] = range_param / runs * z + min_param
			#println("run ", z, ": ",params[(1,2,3)])
		end

		d =  maximum([i for theta in keys(params) for i in theta])
		order = maximum([length(i) for i in keys(params)])

		if reverse && vary_inits
			params = random_init_p(d, order, field = field)
			append!(init_params, params)
			#inits[vary_samples[s]] = params_to_dict(params)
		end

		if vary_inits && !reverse
			#if z == 1
		#		q_params = random_init(d+1, order, field = field)
		#	else
		#		for i in keys(q_params)
	#				q_params[i] = q_params[i]*-1
	#			end
	#		end
			# back to normal
			q_params = random_init_tree_3(d+1, order, field = field)
			append!(init_params, q_params)
			#inits[vary_samples[s]] = params_to_dict(q_params)
		elseif vary_param && reverse
			#q_params = read_params("q.csv"; rand_init = rand_init, field = true)
			q_params = read_params("example.csv")
			if z== 1
				min_param = q_params[varied]
			end
			q_params[varied] = range_param / runs * z + min_param
		else
			#println("constat q init")
			q_params = read_params("q.csv", rand_init = rand_init, field = true)
		end

		if vary_param && !reverse
			append!(interactions, params[varied])
		elseif vary_param && reverse
			append!(interactions, q_params[varied])
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
			recoverable, correlations, marginal, slack, slack_l = pearl_sandwich(p)
			#min_corr, correlations = pearl_corr_test(p)
		end

		kl = 0
		if ipopt 
			if reverse
				kl = min_kl(q, p, verbose = verbose)
				#alt_kl = min_kl(p, q, verbose = verbose)
			else
				kl = min_kl(p, q, verbose = verbose)
			end
		else
			kl = min_kl_manual(p, q, verbose = verbose)
		end
		append!(kls, kl)
		if !reverse
			append!(triangle_test, recoverable)
			append!(corr_test, [Array([correlations[1,2], correlations[2,3], correlations[1,3]])])
			#append!(triangle_slack, slack)
			append!(triangle_slack, [slack_l])
		end
		#append!(triangle_test, min_corr)
		#append!(corr_test, correlations)

		if reverse
			append!(final_params, p.params)
		else
			append!(final_params, q.params)
			if z % 100 ==0
				println(q.params)
			end
		end
		if kl < .02
			break
		end
	end
	
	sample_kls[vary_samples[s]] = kls
	learned_params[vary_samples[s]] = params_to_dict(final_params) #[params_to_dict(final_param) for final_param in final_params]
	inits[vary_samples[s]] = params_to_dict(init_params)
end
#vis_mrf(q)

println("AFTER ", z , ", runs ")
if vary_param
	println("3 Body Couplings")
	println(interactions)
	if reverse
		for k in keys(q.params)
			println(k, ": ", q.params[k])
		end
	end
end
#println(length(learned_params))
init = Dict{Tuple, Array{Any, 1}}
learned = Dict{Tuple,Array{Any,1}}
#if vary_inits
for s in 1:length(learned_params)
	learned = learned_params[vary_samples[s]]
	if vary_inits
		init = inits[vary_samples[s]]
	end
	#println()
	#println("size learned ", length(learned))
	#println("parameter variance")
	for i in keys(learned)
		println("learned ", i, "variance")
		#println(learned[i])
		#println(i, " : ", var(learned[i]))
	end
end
println("objective")
println("mean: ", round(mean(kls), 4), "    variance: ", var(kls))
println("kl min ", minimum(kls), " kl max ", maximum(kls))
println()
if !reverse
	#sum(corr_test[i][k] > 0 for k=1:length(corr_test[i]))) 
	println("Pearl Test")
	for i = 1:length(triangle_test)
		println((kls[i], triangle_test[i] && prod(corr_test[i])>0, round(triangle_slack[i],3)))
	end
end

println()
#println("p stats")
#
println(typeof(sample_kls))
#plot_sample_runs(sample_kls, interactions, "")#; slacks = triangle_slack)
#plot_param_variance(learned_params)
if vary_param
	plot_param_runs(learned, interactions, "")
else
	println()
	#plot_param_runs(learned, Array{Float64,1}(), "")
end
if vary_inits
	plot_param_runs(learned, Array{Float64,1}(), "")
end


println()
print_stats(samples)
println()
println(samples)
println()
println("marginals")
m = marginals(samples)
for k in keys(m)
	println(k, " : ", m[k])
end


println("*** Learned Q Corrs ***")
dh = maximum([i for theta in keys(q_params) for i in theta])
order = maximum([length(i) for i in keys(q_params)])
modelq = FactorGraph(order, dh, :spin, q_params)
samples_q = sample(modelq, vary_samples[end])
print_stats(samples_q)

println(collect(keys(q.params)))
# reverse = true, q tree params => learned 3 body (no field)... check vs. closed form
if reverse &&  maximum([i for theta in keys(q.params) for i in theta]) >= 4
	mm = false
	println("mathematica ", mm)
	test_reverse_closed_form(q.params, p.params, mm)
end


