using GraphicalModelLearning
using CSV
include("utils.jl")
include("mrf.jl")
num_samp = 100000
tol = .000001

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


p = mrf(params, samples)
a= [hcat(samples, fill(h_i, size(samples)[1])) for h_i in [-1,1]]
#println(size(a),typeof(a))
q = hmrf(q_params, samples, [-1,1])

println("q params initialization")
print_params(q.params)

println(q.samples, size(q.samples[1]))

kl_history = zeros(0)
# replace with gradient condition? #size(kl_history)[1] <= 1 || 
while size(kl_history)[1] <= 1 || abs(kl_history[end] - kl_history[end-1]) > tol 
	#println("Entering while")
	append!(kl_history, kl_empirical(p,q, base=2))
	#println("gradient ascent")
	kl_gradient_ascent(q, reversed=false)
	if size(kl_history)[1]%100== 0
		println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
	end
end
println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
println()
println("Final Params")
print_params(q.params)

display_model(q)