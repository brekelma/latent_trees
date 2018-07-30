using CSV
using GraphicalModelLearning
using MATLAB
using PyPlot

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
include("learning_trees.jl")

PyPlot.ioff()

runs = 10
fn = "extended_tree_321.csv"
fn = "example_zerocorr.csv"
fn = "example_1236.csv"

srand(floor(Int, 100*(time() % 1)));
obs = 6 # total = 9
learn_order = 3
num_samp = 10000000
# rand init and range for values
rand_init = false
range_params = [-1, 1]
field_range = [-.5, .5]
# magnetic field or no?
field = true
# homogenous overrides rand_init
rise_reg = 0.0
min_abs = .1

samples = Array{Int64,2}()
full_samples = samples

adj_dict = Dict{Tuple, Real}()


dim = maximum([i for coup in keys(read_params(fn; rand_init = rand_init, range = range_params, field = field)) for i in coup])
s = "log_rise"

if s == "log_rise"
	mRISE = logRISE(rise_reg, true)
elseif s == "multi_rise"
	mRISE = multiRISE(rise_reg, true, learn_order)
elseif s == "rple"
	mRISE = RPLE(rise_reg, true, learn_order)
end

#for r = 1:runs
tree_params = read_params(fn; rand_init = rand_init, range = range_params, field = field, field_range = field_range, min_abs = min_abs)
tree_adj = dict2array(tree_params)

model = FactorGraph(maximum([length(i) for i in keys(tree_params)]), 
		maximum([i for theta in keys(tree_params) for i in theta]), 
		:spin, tree_params)

full_samples = sample(model, num_samp)
samples = full_samples[:,1:(obs+1)]

p = mrf(tree_params, full_samples)
q = learn(samples, mRISE, NLP()) # returns Factor Graph


if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
	q = FactorGraph(learn_order, obs, :spin, array2dict(q)) 
end
# q .terms == p.params
q_mrf = mrf(q.terms, samples)
#kl =  kl_empirical(p, q_mrf)


#hess, indices = likelihood_hessian(q_mrf)
println("P Params")
for k in sort_params(tree_params)
	for i in 1:length(k)
		print(i != length(k) ? string(k[i], ", ") : k[i])
	end
	println("\t & ", round(tree_params[k],3), " \\\\ ")
	#println("\t ", round(tree_params[k],3))#" \\\\ ")
end
keyz = collect([keys(q.terms)...])
# keyz = keyz[sortperm([length(i)==1 ? 0 : i[2] for i in keyz])]
# keyz = keyz[sortperm([i[1] for i in keyz])]
key2 = keyz[sortperm([-abs(q.terms[i]) for i in keyz])]
println("RISE Learned by Magnitude")
for k in key2
	for i in 1:length(k)
		print(i != length(k) ? string(k[i], ", ") : k[i])
	end
	println("\t & \t", round(q.terms[k],3), " \\\\ ")
end


#q_hmrf = hmrf(random_init_dense(3,3), samples)
#println("Learning Tree Model by Maximum Likelihood")
#kl = min_kl(p, q_hmrf, verbose = false)
# println()
# println("Params learned by Maximum likelihood ( Tree ) : ")
# if q.params[(q.dim,)]*tree_params[(q.dim,)] < 0
# 	switch_sign(q, q.dim)
# end
# for k in sort_params(q.params)
# 	for i in 1:length(k)
# 		print(i != length(k) ? string(k[i], ", ") : k[i])
# 	end
# 	println("\t & ", round(q.params[k],3), " \\\\ ")
# 	#println(k, " : ", q.params[k], " \t  true param: ", p.params[k])
# end
# println()

# println("Params learned by Maximum likelihood ( All Couplings ) : ")
# if q_hd.params[(q_hd.dim,)]*tree_params[(q_hd.dim,)] < 0
# 	switch_sign(q_hd, q_hd.dim)
# end
# for k in sort_params(q_hd.params)
# 	for i in 1:length(k)
# 		print(i != length(k) ? string(k[i], ", ") : k[i])
# 	end
# 	println("\t & ", round(q_hd.params[k],3), " \\\\ ")
# 	#println(k, " : ", q.params[k], " \t  true param: ", p.params[k])
# end





println(" ***** Thresholding ***** ")
println()
#println("NODES for 0.1 threshold")
#tnodes = threshold_params(q.terms; threshold=0.1)
#fournodes = threshold_params(q.terms; num_nodes = 4)
#println(threshold_params(q.terms; threshold=0.1))
println()
corr = Array{Float64, 2}()
dists = Array{Float64, 2}()
hidden_edges = Dict{Int64, Array{Tuple, 1}}()
nodes = 4
addl_obs = 0
while nodes + addl_obs <= size(samples)[2]-1
	inodes = threshold_params(q.terms; num_nodes = nodes + addl_obs)
	if length(inodes) > nodes + addl_obs
		deleteat!(inodes, length(inodes))
	end
	# should get inclusion of hidden nodes, previous for free... just union with hiddens
	inodes = union(inodes, keys(hidden_edges))
	println("Dist Alg Nodes: ", inodes)
	if isempty(corr)
		corr = corrs(samples, pearson = true)
	end
	#a = distance_algebra(tnodes, corr; distances = false)
	edge_dict, dists = distance_algebra(inodes, corr; calculated = dists, distances = false, addl_hidden = isempty(dists) ? 0 : size(corr)[1])
	println()
	if !isempty(edge_dict)
		new_hidden = []
		for k in keys(edge_dict)
			println("Hidden added: ", k, " edges: ", edge_dict)
			if !haskey(hidden_edges, k)
				hidden_edges[k] = Array{Tuple, 1}()
				append!(new_hidden,[k])
			end
			append!(hidden_edges[k], edge_dict[k])
		end
		if isempty(new_hidden)
			addl_obs += 1
		end
	else
		addl_obs += 1
	end

	println("hidden edges added? ", hidden_edges)
end



println()
pprint2d(corr)