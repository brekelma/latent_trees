using CSV
using GraphicalModelLearning
using MATLAB

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")

fn = "extended_tree.csv"
obs = 6 # total = 9
learn_order = 3
num_samp = 1000000
rand_init = true
field = true
range_params = [-2, 2]
samples = Array{Real,2}()
full_samples = samples
p = mrf()
q = mrf()
adj_mat = []
edge_dists = []
adj_mat_cl = []
adj_dict = Dict{Tuple, Real}()
adj_cl = Dict{Tuple, Real}()
orig_dists = []
base_params = Array{Any, 1}()
learned_params = Array{Any, 1}()
homogenous = Array{Float64, 1}()
runs = 1
vary_homo = 1
min_homo = 0.1

mRISE = multiRISE(.4, true, learn_order)

for r = 1:runs
	tree_params = read_params(fn; rand_init = rand_init, range = range_params, field = field)
	if vary_homo > 0
		for i in keys(tree_params)
			if field || length(i) != 1
				tree_params[i] = min_homo + vary_homo / runs * (r-1)
			else
				tree_params[i] = 0
			end
		end
	end 

	model = FactorGraph(maximum([length(i) for i in keys(tree_params)]), 
			maximum([i for theta in keys(tree_params) for i in theta]), 
			:spin, tree_params)

	full_samples = sample(model, num_samp)
	samples = full_samples[:,1:(obs+1)]
	println("Observed : ", obs, " out of Total : ", maximum([i for theta in keys(tree_params) for i in theta]))

	p = mrf(tree_params, samples)
	q = learn(samples, mRISE, NLP()) # returns Factor Graph
	# q .terms == p.params

	q_stats = Array{Float64, 1}()
	partition = 0
	for k = 1:size(samples)[1]
		log_evidence = sum([sum(samples[k, param[i]+1]*q.terms[param] for i=1:length(param)) for param in collect(keys(q.terms))])
		append!(q_stats, log_evidence)
		partition = partition + exp(log_evidence)
	end

	kl = sum(samples[k,1]/num_samp*(log(samples[k,1]/num_samp) - q_stats[k] + log(partition)) for k = 1:size(samples)[1])
	#msamples = mxarray(matlab_samples(samples))
	println("RISE achieved KL : ", ln_convert(kl))
	msamples = matlab_samples(samples)

	#mat"$adj_mat, $edge_distances = run_clrg($msamples)"
	adj_mat, edge_dists, adj_mat_cl, orig_dists = mxcall(:run_clrg, 4, msamples, [.01, .99])
	#eval_string("[adj_mat, edge_distances] = run_clrg(msamples)")
	#adj_mat = jarray(adj_mat)
	#edge_distances = jarray(edge_distances)
	println("adj mat")
	println(adj_mat)
	println("edges")
	println(full(edge_dists))
	nz = findnz(adj_mat)
	adj_dict = Dict{Tuple, Real}()
	for i=1:length(nz[1])
		tup = sort_tuple((nz[1][i], nz[2][i]))
		if !haskey(adj_dict,tup)
			adj_dict[tup] = edge_dists[tup[1], tup[2]]
		end
	end
	nz = findnz(adj_mat_cl)
	adj_cl = Dict{Tuple, Real}()
	for i=1:length(nz[1])
		tup = sort_tuple((nz[1][i], nz[2][i]))
		if !haskey(adj_cl,tup)
			adj_cl[tup] = orig_dists[tup[1], tup[2]]
		end
	end
	println()
	println("Q terms")

	q.terms = prune_params(q.terms, tol = .009)
	q = learn(samples, mRISE, NLP(), collect(keys(q.terms)))
	for i in keys(q.terms)
		println(i, " : ", q.terms[i])
	end
	println("P terms")
	for i in keys(p.params)
		println(i, " : ", p.params[i])
	end
	println(typeof(p.params), typeof(q.terms))
	append!(base_params, p.params)
	append!(learned_params, q.terms)
	append!(homogenous, tree_params[collect(keys(tree_params))[1]])
	# what to record from anima?
end

base_by_coupling = params_to_dict(base_params)
learned_by_coupling = params_to_dict(learned_params)

display_factor(p, "orig_tree")
display_factor(q, "learned_graph")
display_factor(adj_dict, "anima_tree")
display_factor(adj_cl, "chow_liu")

print_stats(samples)

print_stats(full_samples)


for k in keys(learned_by_coupling)
	println(k, " : ", [(homogenous[i], learned_by_coupling[k][i]) for i = 1:length(learned_by_coupling[k])])
end
print("trying sparse cut")
sparsest_cut(q.terms)

#plot_param_runs(learned_by_coupling, homogenous, "", title="homogenous_rise", orders = [2,3])

# type FactorGraph{T <: Real}
#     order::Int
#     varible_count::Int
#     alphabet::Symbol
#     terms::Dict{Tuple,T} # TODO, would be nice to have a stronger tuple type here
#     variable_names::Nullable{Vector{String}}
#     FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
# end
