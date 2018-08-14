using CSV
using GraphicalModelLearning
#using MATLAB
#using PyPlot

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
#include("learning_trees.jl")

#PyPlot.ioff()

runs = 10
fn = "extended_tree_321.csv"
fn = "example_zerocorr.csv"
fn = "example_1236.csv"

srand(floor(Int, 100*(time() % 1)));
obs = 6 # total = 9
learn_order = 2
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
s = "multi_rise"

if s == "log_rise"
	learner = logRISE(rise_reg, true)
elseif s == "multi_rise"
	learner = multiRISE(rise_reg, true, learn_order)
elseif s == "rple"
	learner = RPLE(rise_reg, true, learn_order)
end

#for r = 1:runs
tree_params = read_params(fn; rand_init = rand_init, range = range_params, field = field, field_range = field_range, min_abs = min_abs)
tree_adj = dict2array(tree_params)

model = FactorGraph(maximum([length(i) for i in keys(tree_params)]), 
		maximum([i for theta in keys(tree_params) for i in theta]), 
		:spin, tree_params)

full_samples = GraphicalModelLearning.sample(model, num_samp)
samples = full_samples[:,1:(obs+1)]

p = mrf(tree_params, samples)

#if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
#	q = FactorGraph(learn_order, obs, :spin, array2dict(q)) 
#end
# q .terms == p.params
#q_mrf = mrf(q.terms, samples)
#kl =  kl_empirical(p, q_mrf)

#3 and 1, see if it learns tree from 51-49 init


function param_distance{T<:Real, S<:Real}(params1::Dict{Tuple, T}, params2::Dict{Tuple,S}; norm = 2)
	dist = 0
	if isempty(params2) || !isempty(setdiff(collect(keys(params1)), collect(keys(params2))))
		return NaN
	end
	for i in keys(params1)
		if i in keys(params2)
			if norm == 2 
				dist += (params1[i]-params2[i])^2
			else #if norm == "max" || norm == "infinity"
				dist = (params1[i]-params2[i]) > dist ? (params1[i]-params2[i]) : dist
			end
		end
	end
	return isa(Int64, norm) ? dist^(1/norm) : dist
end



obs =3
max_hidden = 1
n_hidden = 1
init_order = 3
my_params = random_init_multi(obs+n_hidden, init_order)
prev_params = my_params
#true_params = read_params("extended_tree_321.csv")
true_params = read_params("four_vars.csv")
model = FactorGraph(maximum([length(i) for i in keys(true_params)]), 
		maximum([i for theta in keys(true_params) for i in theta]), 
		:spin, true_params)
my_samples = GraphicalModelLearning.sample(model, num_samp)
obs_samples = my_samples[:,1:obs+1]

# ANYTHING ELSE TO GET HIDDEN CL 
init = "cl" #

#init_model = FactorGraph(maximum([length(i) for i in keys(my_params))]), 
#		maximum([i for theta in keys(my_params) for i in theta]), 
#		:spin, my_params)
#samps = GraphicalModelLearning.sample(init_model, num_samp)


println("True Params")
for i in sort_params(true_params)
	println("\t", i, " => ", true_params[i])
end
println()

myp = chow_liu(mi_bin(obs_samples))
myp = chow_liu(mi_bin(my_samples))
#println("CL keys ", collect(keys(myp)))
if init == "cl"
	mi_init = rand(obs+n_hidden)
	mi_init = rand(obs+n_hidden)*rand(obs+n_hidden).'
	my_params = chow_liu(mi_init)
	#full_obs = mrf(random_init_p(obs, learn_order; range = [-1,1]), obs_samps)
	#kl = min_kl(mrf(true_params, obs_samples), full_obs, verbose = false)

	# starting point is CL tree on observed
	for i in keys(my_params)
		#if 
		my_params[i] = rand()[1]*(range_params[end]-range_params[1])+range_params[1]
		#else # initializing parameters to full observed model, with uniform hidden...
		#	my_params[i] = full_obs.params[i]
		#end
	end
elseif init == "perturb"
	added = false
	my_params = deepcopy(true_params)
	for i in sort_params(my_params)
		if !added
			#added = true
			my_params[i] = true_params[i] + randn()[1]*min_abs/5 #rand()[1]*()+range_params[1]	
			println(i, ": ", my_params[i])
		else
			my_params[i] = true_params[i]
		end
	end
end
perturbed = deepcopy(my_params)
display_factor(myp, "ground_truth_tree", field = field, observed = obs)
println("CL True ", collect(keys(myp)))
println("CL Init ", collect(keys(my_params)))
clint = collect(keys(my_params))
display_factor(my_params, "initial_CL", field = false, observed = obs)

#(need to sample hidden, from model with no params for it...)
#println("my params INIT ", my_params)
# doesn't make sense
#my_params = random_init_tree_3(3,2)

# initialize with random CL tree?  



iters = 100
tol = .001

learn_all_nodes = false
printall = false
just_hidden = false


#println("initial samples")
#pprint2d(my_samples, latex = false)
kls = Array{Float64,1}()
update_dist = Array{Float64,1}()
true_dist = Array{Float64, 1}()
iterstops = Array{Float64,1}()

for n_hidden = 1:max_hidden
	println("**** ADDING HIDDEN NODE ", obs+n_hidden, " ******")
	if n_hidden > 1
		if init != "perturb"
			rand_params = random_init_multi(obs+n_hidden, init_order)
			for k in keys(rand_params)
				if obs+n_hidden in k
					my_params[k] = rand_params[k]
				end 
			end
		end
	end
end
println("New keys ", collect(keys(my_params)))
iter = 0

while iter < iters && ( iter <=5 || (!(param_distance(my_params, prev_params) < tol) && iter < iters))
	
	h_edges = Array{Tuple,1}()
	other_edges = Array{Tuple,1}()
	prev_params = iter > 0 ? my_params : Dict{Tuple,Real}()

	if iter > 0
		println("params ", collect(keys(my_params)))
	end
	my_mrf = mrf(my_params, obs_samples)
	my_mrf.dim = maximum([ii for i in keys(my_params) for ii in i])
	my_samples = cond_sample_hidden(my_mrf, obs)
	#pprint2d(my_samples, latex = false)
	mi_mi = ln_convert(mi_bin(my_samples))
	cl_dict = chow_liu(mi_mi)
	println("CL keys ", collect(keys(cl_dict)))
	if iter % 5 == 0
		display_factor(cl_dict, string("chow_liu_EM_", length(kls)), field = field, observed = obs)
	end
	# dictionary of edges 
	append!(h_edges, [(i,j) for (i,j) in keys(cl_dict) if (i > obs) || (j > obs)])
	#h_edges = [tuple([i,j]...) for (i,j) in keys(cl_dict) if (i > obs) || (j > obs)]
	append!(other_edges, [(i,j) for (i,j) in keys(cl_dict) if !((i > obs) || (j > obs))])
	#other_edges = [k for k in keys(cl_dict) if !((k[1] > obs) || (k[2] > obs))]
	#other_edges = [(i,j) for (i,j) in keys(cl_dict) if !((i > obs) || (j > obs))]
	h_inds = unique([i+1 for tup in h_edges for i in tup])

	other_inds = isempty(other_edges) ? [] : unique([i+1 for tup in other_edges for i in tup])
	#println("h inds ", [(i,) for i in h_inds])
	# add fields
	hidden_learn = unique([i for tup in h_edges for i in tup if sum([i in e ? 1 : 0 for e in other_edges])==0])
	other_learn = isempty(other_edges) ? [] : unique([i for tup in other_edges for i in tup])

	# if any edge! 
	append!(h_edges, unique([(i,) for tup in h_edges for i in tup]))
	append!(other_edges, [(i,) for i in other_learn])
	h_all = union(h_edges, [sort_tuple((i,j)) for i in hidden_learn, j in other_learn if sort_tuple((i,j)) in other_edges])
	
	#append!(h_edges, [(i,) for i in h_learn])
	
	if just_hidden
		qh = GraphicalModelLearning.learn(my_samples, learner, NLP(), h_edges, node = obs+n_hidden)
	else
		qh = GraphicalModelLearning.learn(my_samples, learner, NLP(), h_edges, node = hidden_learn)
	end
	if full_rise
		q_all = GraphicalModelLearning.learn(my_samples, learner, NLP())
		q_all = isa(q_all, Array) ? array2dict(q_all) : q_all.terms
	end
	#println("Qh Terms")
	#for hi in sort_params(qh.terms)
#		println(hi, " : ", qh.terms[hi])
#	end


	qh = isa(qh, Array) ? array2dict(qh) : qh.terms
	#println("hidden params ", collect(keys(qh)))

	
	if !isempty(other_edges)
		other_edges = other_edges[randperm(length(other_edges))]
		qo = GraphicalModelLearning.learn(my_samples, learner, NLP(), other_edges, node = other_learn)
		qo = isa(qo, Array) ? array2dict(qo) : qo.terms
		#println("obs params ", collect(keys(qo)))
		println("**** other edges ", other_edges, " ****", collect(keys(qo)))
		my_params = merge(qh,qo)
		for (i, val) in intersect(qh, qo)
			println("DUPLICATE KEY ", i, "\t : ", qh[i], " , ", qo[i])
			my_params[i] = .5*(qh[i]+qo[i])
			my_params[i] = .5*(qh[i]+qo[i])
		end
	else
		my_params = qh
	end
	

	#println("qh edges : ", keys(qh), " qo edges : ", keys(qo))

	#end
	# given structure .... 
	# println("degree of hidden in Chow-Liu : ", length([h for h in h_edges if length(h)==2]))
	
	iter +=1
	pp = deepcopy(p)
	pp.params = Dict{Tuple, Float64}()
	#println("empty params ", isempty(pp.params), "size ", size(pp.samples)[1])
	my_hmrf = mrf(my_params, my_samples)
	kl = kl_empirical(pp, my_hmrf)
	println("KL divergence ", kl)
	append!(kls, [kl])

	param_dist = param_distance(my_params, prev_params)
	append!(update_dist, [param_dist])
	true_diff = param_distance(my_params, true_params)
	append!(true_dist, [true_diff])

	if printall 
		for i in sort_params(my_params)
			println("\t", i, " => ", my_params[i])
		end
	end
	
	just_hidden = !isnan(param_dist)
	if isnan(param_dist)
		keyz = collect([keys(q_all)...])
		# keyz = keyz[sortperm([length(i)==1 ? 0 : i[2] for i in keyz])]
		# keyz = keyz[sortperm([i[1] for i in keyz])]
		key2 = keyz[sortperm([-abs(q_all[i]) for i in keyz])]
		for k in key2
			for i in 1:length(k)
				print(i != length(k) ? string(k[i], ", ") : k[i])
			end
			println("\t & \t", round(q_all[k],3), " \\\\ ")
		end
	end

	end
	println("Param update : ", param_dist)
end
append!(iterstops, [length(kls)])
#end
println()
if init == "perturb"
	println("Init Params")
	for i in sort_params(perturbed)
		println(i, ": ", perturbed[i])
	end
end


println("Final Params")
for i in sort_params(my_params)
	println("\t", i, " => ", my_params[i])
end

println()

println("True Params")
for i in sort_params(true_params)
	println("\t", i, " => ", true_params[i])
end

#println("CL Init ", clint)
#clinit = collect(keys(my_params))


j = 1
for i =1:length(kls)
	if i > iterstops[j]
		println()
		println("Hidden variable : ", obs+j)
		j+=1
	end 
	println("Iter ", i, " \t KL : ", kls[i], " \t Param Improvement : ", update_dist[i], " \t Param Improvement : ", true_dist[i])
end
# put my_params into MRF
#model = FactorGraph(maximum([length(i) for i in keys(my_params )]), 
#		maximum([i for theta in keys(my_params) for i in theta]), 
#		:spin, my_params)

#my_samples = GraphicalModelLearning.sample(model, num_samp)
#q = learn(samples, learner, NLP()) # returns Factor Graph

# need some recursive screening on hidden neighbors in CL, use params to sample 