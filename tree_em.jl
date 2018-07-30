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




if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
	q = FactorGraph(learn_order, obs, :spin, array2dict(q)) 
end
# q .terms == p.params
q_mrf = mrf(q.terms, samples)
#kl =  kl_empirical(p, q_mrf)

#3 and 1, see if it learns tree from 51-49 init

obs = 3
true_params = read_params("four_vars.csv")
model = FactorGraph(maximum([length(i) for i in keys(true_params)]), 
		maximum([i for theta in keys(true_params) for i in theta]), 
		:spin, true_params)
samps = sample(model, num_samp)

cl_tree =chow_liu(mi_bin(samps))
my_params = Dict{Tuple, Float64}() # initialize to correct keys = edges of chow liu, random init 
my_params = random_init_tree_3(3,2)
# initialize with random CL tree?  

function param_distance(params1::Dict{Tuple, Float64}, params2::Dict{Tuple,Float64})
	dist = 0
	for i in params[1]
		dist += (params[1]-params[2])^2
	end
	return sqrt(dist)
end
iters = 10
iter = 0
while iter < iters || param_distance(my_params, prev_params)
	println("params ", collect(keys(my_params)))
	println("my_params ", typeof(my_params))
	my_mrf = mrf(my_params, samps)
	println()
	println("mrf params ", typeof(my_mrf.params))
	println("cond sample hidden")
	my_samples = cond_sample_hidden(my_mrf, obs)
	println("my samples size ", size(my_samples))
	mi_mi = ln_convert(mi_bin(my_samples))
	cl_dict = chow_liu(mi_mi)
	# dictionary of edges 
	h_edges = [(i,j) for (i,j) in keys(cl_dict) if (i > obs) || (j > obs)]
	other_edges = [(i,j) for (i,j) in keys(cl_dict) if !(i > obs) || (j > obs)]
	h_inds = unique([i+1 for tup in h_edges for i in tup])
	learn(samples[:,h_inds], learner, NLP(), h_edges)

	other_inds = unique([i+1 for tup in other_edges for i in tup])
	learn(samples[:, other_inds], learner, NLP(), other_edges)
	# given structure .... 
	println("degree of hidden in Chow-Liu : ", length(h_edges))
	display_factor(cl_dict, string("random_init_chow_liu_EM"), field = field, observed = obs)
	println("done ")
end


# put my_params into MRF
model = FactorGraph(maximum([length(i) for i in keys(my_params )]), 
		maximum([i for theta in keys(my_params) for i in theta]), 
		:spin, my_params)

my_samples = sample(model, num_samp)
q = learn(samples, mRISE, NLP()) # returns Factor Graph

# need some recursive screening on hidden neighbors in CL, use params to sample 