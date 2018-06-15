using GraphicalModelLearning
using CSV
include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
num_samp = 100000
tol = .00001

df = CSV.read("example.csv"; delim = "\t", header=0, types = [String, Float64], nullable = false)
splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
params = Dict{Tuple, Float64}()
for r=1:size(df)[1]
	params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
	#println("Key: ", tuple(parse(Int64, split(df[r,1],',', keep = false)[i]) for i=1:edge_orders[r]))
end

df = CSV.read("q.csv"; delim = "\t", header=0, types = [String, Float64], nullable = false)
splits = [split(df[r,1],',', keep = false) for r=1:size(df)[1]]
q_params = Dict{Tuple, Float64}()
for r=1:size(df)[1]
	q_params[tuple([parse(Int64, splits[r][i]) for i=1:length(splits[r])]...)] = df[r,2]
	#println("Key: ", tuple(parse(Int64, split(df[r,1],',', keep = false)[i]) for i=1:edge_orders[r]))
end



d =  maximum([i for theta in keys(params) for i in theta])
order = maximum([length(i) for i in keys(params)])

model = FactorGraph(order, d, :spin, params)
samples = sample(model, num_samp)
num_conf = size(samples)[1]
println(typeof(samples), typeof(model))

print_params(params)
print_stats(samples)

#println()
#print_pearl_corrs(samples)



p = mrf(params, samples)
a= [hcat(samples, fill(h_i, size(samples)[1])) for h_i in [-1,1]]
#println(size(a),typeof(a))
q = hmrf(q_params, samples, [-1,1])

println("q params initialization")
print_params(q.params)
println()

println("Corr Test for P_true")
pearl_corr_test(p)

run_manual = false
verbose = false
if run_manual 
	kl_history = zeros(0)
	# replace with gradient condition? #size(kl_history)[1] <= 1 || 
	while size(kl_history)[1] <= 1 || abs(kl_history[end] - kl_history[end-1]) > tol 
		#println("Entering while")
		append!(kl_history, kl_empirical(p,q, base=2))
		#println("gradient ascent")
		kl_gradient_ascent(q, step = .01, reversed=false)
		if verbose && size(kl_history)[1]%100== 0
			println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
		end
	end
	println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
	println()
	println("Final Params")
	print_params(q.params)
end

kl = emp_lld(p) - max_lld(q)
println("Final KL divergence: bits ", ln_convert(kl,2), " nats: ", kl)
#display_model(q)

println("")
println("Optimized Q Corr Test?")
pearl_corr_test(p)

