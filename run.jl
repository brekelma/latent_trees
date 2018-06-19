using GraphicalModelLearning
using CSV

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
num_samp = 10000000
tol = .00001
verbose = true
ipopt = true
q_field = true 

# Vary 3 body strength?
vary_3_body = true
runs = 10
range3body = 4
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
println(typeof(params))
min3 = params[(1,2,3)]
qq = Dict{Any, Any}()
q_params = Dict{Any, Any}()
for z = 1:runs
	if z > 1 && vary_3_body
		params[(1,2,3)] = range3body / runs * z + min3
		#println("run ", z, ": ",params[(1,2,3)])
	end

	d =  maximum([i for theta in keys(params) for i in theta])
	order = maximum([length(i) for i in keys(params)])
	params = random_init_p(d, order, field = q_field)
	println([k for k in keys(params)])
	append!(interactions, params[(1,2,3)])

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
	append!(corr_test, [Array([correlations[(1,2)], correlations[(2,3)], correlations[(1,3)]])])
	#append!(triangle_test, min_corr)
	#append!(corr_test, correlations)

	if reverse
		append!(final_params, p.params)
	else
		append!(final_params, q.params)
	end
end

#plot_model(q)

if vary_3_body
	println()
	println()
	println("3 Body Couplings")
	println(interactions)
	println()
	println()
	println("KL Divergences")
	println(kls)
	println()
	println()
	println("KL(=0?) vs Pearl Test")
	println([(kls[i], triangle_test[i] && prod(corr_test[i])>0, sum(corr_test[i][k] > 0 for k=1:length(corr_test[i]))) for i=1:length(triangle_test)])
	#println([(kls[i], triangle_test[i]>0 && prod(corr_test[i])>0) for i=1:length(triangle_test)])
	#println([sum(convert(Int64, corr_test[i]>0)) for i=1:length(corr_test)])

elseif vary_inits
	println("final params size ", size(final_params), " type ", typeof(final_params), " 1 element ", typeof(final_params[1]))
	
	learned = Dict{Tuple, Array}()
	for i = 1:length(final_params)
		if !haskey(learned, final_params[i][1])
			learned[final_params[i][1]]= Array{Float64,1}()
		end 
		append!(learned[final_params[i][1]], final_params[i][2])
	end
	println()
	println("variances")
	ppp = true
	for i in keys(learned)
		println(i, " : ", var(learned[i]))
	end
	println("objective")
	println("mean: ", round(mean(kls), 4), "    variance: ", var(kls))
	println("kl min ", minimum(kls), " kl max ", maximum(kls))
	println()
	println("Pearl Test")
	println([(kls[i], triangle_test[i] && prod(corr_test[i])>0, sum(corr_test[i][k] > 0 for k=1:length(corr_test[i])))  for i=1:length(triangle_test)])
end
println()
println("p stats")
print_stats(samples)
