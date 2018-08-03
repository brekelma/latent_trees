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
tree_type = "321"
fn = string("extended_tree_", tree_type, ".csv")
#fn = "example.csv"


srand(floor(Int, 100*(time() % 1)));
obs = 6 # total = 9
learn_order = 6
num_samp = 1000000
# rand init and range for values
rand_init = false
range_params = [-1, 1]
field_range = [-.5, .5]
# magnetic field or no?
field = true
# homogenous overrides rand_init
vary_homo = 0
min_homo = .2
log_rise = false
h_support = [-1,1]
example_distibutions = false
use_threshold = false
plots = false
eps = 10.0^-6
min_abs_param = .2
run_anima = false
cl = false
calc_param_var = false

samples = Array{Int64,2}()
full_samples = samples
p = mrf()
q = mrf()
r = mrf()
adj_mat = []
edge_dists = []
adj_mat_cl = []
adj_dict = Dict{Tuple, Real}()
adj_cl = Dict{Tuple, Real}()
mi_me = Array{Float64, 2}()
msamples = Array{Float64, 2}()
orig_dists = []
base_params = Array{Any, 1}()
learned_params = Array{Any, 1}()
homogenous = Array{Float64, 1}()
three_bodies = Dict{Tuple, Array{Float64, 1}}()
three_body_counts = Array{Int64, 1}()
three_body_gt = Array{Int64, 1}()
max3 = Array{Float64, 1}()
kls = Array{Float64, 1}()
anima_struct = 0
anima_num = 0
thresholds = Array{Float64, 1}()
num_cliques = Dict{Int64, Array{Int64, 1}}()
gt_cliques = Dict{Int64, Array{Int64, 1}}()
gt_edges = Dict{Tuple, Array{Bool, 1}}()
top_three_body = Dict{Tuple, Array{Bool, 1}}()
top_four_body = Dict{Tuple, Array{Bool, 1}}()
dim = maximum([i for coup in keys(read_params(fn; rand_init = rand_init, range = range_params, field = field)) for i in coup])
for i=1:dim
	num_cliques[i] = zeros(runs)
	gt_cliques[i]= zeros(runs)
end


mis = Array{Float64,2}()
anima_wrong_struct = Array{Dict{Tuple, Float64}, 1}()
anima_wrong_num = Array{Dict{Tuple, Float64}, 1}()
three_body_p = Array{Dict{Tuple, Float64}, 1}()
three_body_q = Array{Dict{Tuple, Float64}, 1}()
all_models = Array{Dict{Tuple, Float64}, 1}()
max_cl_degree = Array{Float64, 1}()
max_corrs = Array{Float64,1}()
avg_corrs = Array{Float64,1}()	
min_cl_edge = Array{Float64, 1}()
distance = Array{Float64, 2}()
degree_cl_param = Array{Int64, 1}()
all_params = Dict{Tuple, Array{Float64, 1}}()

if tree_type == "321"
	ground_truth_cliques = [(1,2,3), (4,5,6)]
	ground_truth_edges = [(1,2), (1,3), (2,3), (4,5)]
elseif tree_type == "33"
	ground_truth_cliques = [(1,2,3), (4,5,6)]
	ground_truth_edges = [ ]
elseif tree_type == "42"
	ground_truth_cliques = [(1,2,3,4), (1,2,3), (1,2,4), (2,3,4), (1,3,4)]
elseif tree_type == "43"
	ground_truth_cliques = [(1,2,3,4), (1,2,3), (1,2,4), (2,3,4), (1,3,4), (4,5,6)]
end

rise_reg = 0.0
if log_rise
	mRISE = logRISE(rise_reg, true)
else
	mRISE = multiRISE(rise_reg, true, learn_order)
end

full_clique = 0 
for r = 1:runs
	println()
	println("run ", r)
	tree_params = read_params(fn; rand_init = rand_init, range = range_params, field = field, field_range = field_range, min_abs = min_abs_param)
	tree_adj = dict2array(tree_params)
	#siblings(tree_params)
	append!(all_models, [tree_params])

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

	p = mrf(tree_params, full_samples)
	q = learn(samples, mRISE, NLP()) # returns Factor Graph
	# q= array2dict(q)
	
	if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
		q = FactorGraph(learn_order, obs, :spin, array2dict(q)) 
	end
	# q .terms == p.params
	q_mrf = mrf(q.terms, samples)
	kl =  kl_empirical(p,q_mrf)
	#msamples = mxarray(matlab_samples(samples))
	println("RISE achieved KL : ", ln_convert(kl))

	if run_anima
		msamples = matlab_samples(samples)

		#mat"$adj_mat, $edge_distances = run_clrg($msamples)"
		adj_mat, edge_dists, adj_mat_cl, mis, mi_me, distance = mxcall(:run_clrg, 6, msamples, [.051, .949])
		
		param_adj_mat,  param_edge_dist = mxcall(:CLRG, 2, distance, 1, num_samp, [.051, .949], dict2array(chow_liu(dict2array(q.terms))))
		#eval_string("[adj_mat, edge_distances] = run_clrg(msamples)")
		#adj_mat = jarray(adj_mat)
		#edge_distances = jarray(edge_distances)
		#println("adj mat")
		#println(adj_mat)
		#println("edges")
		println("Anima MI")
		pprint2d(ln_convert(mis))
		println("My Matlab MI")
		pprint2d(ln_convert(mi_me))
		println("")
		edge_corrs = 1./exp(full(edge_dists))
		#orig_corrs = 1./exp(full(orig_dists))
		
		nz = findnz(adj_mat)
		adj_dict = Dict{Tuple, Real}()
		adj_anima_mi = Dict{Tuple, Real}()
		for i=1:length(nz[1])
			tup = sort_tuple((nz[1][i], nz[2][i]))
			if !haskey(adj_dict,tup)
				adj_dict[tup] = edge_corrs[tup[1], tup[2]] == 1 ? 0 : edge_corrs[tup[1], tup[2]] 
				#adj_anima_mi[tup]= ln_convert(mi_me[tup[1], tup[2]])#adj_dict[tup] = edge_dists[tup[1], tup[2]]
			end
		end
		nz = findnz(adj_mat_cl)
		adj_cl = Dict{Tuple, Real}()
		adj_cl_distances = Dict{Tuple, Real}()
		for i=1:length(nz[1])
			tup = sort_tuple((nz[1][i], nz[2][i]))
			if !haskey(adj_cl,tup)
				adj_cl[tup] = mis[tup[1], tup[2]]
			end
			if !haskey(adj_cl_distances,tup)
				adj_cl_distances[tup] = edge_corrs[tup[1], tup[2]]
			end
		end
	else
		#distance = mxcall(: )
	end
	#max_degree_cl = maximum([sum(1*(abs.(full(adj_mat_cl)[k,i])>eps) for k=1:size(full(adj_mat_cl))[1]) for i=1:size(full(adj_mat_cl))[2]])
	
	#println()
	
	mi_mi = ln_convert(mi_bin(samples))
	cl_dict = chow_liu(mi_mi)
	cl_adj = dict2array(cl_dict)
	cl_param = dict2array(chow_liu(dict2array(q.terms)))
	max_degree_cl = maximum([sum(1*(abs.(cl_adj[k,i])>eps) for k=1:size(cl_adj)[1]) for i=1:size(cl_adj)[2]])
	min_cl_mi = minimum([values(cl_dict)...])
	deg_cl_param = maximum([sum(1*(abs.(cl_param[k,i])>eps) for k=1:size(cl_param)[1]) for i=1:size(cl_param)[2]])
	append!(max_cl_degree, [max_degree_cl])
	append!(min_cl_edge, [min_cl_mi])
	append!(degree_cl_param, [deg_cl_param])
	println("MAX DEGREE CL ", max_degree_cl)
	println("Min CL Edge ", min_cl_mi)
	println("Degree CL Params ", deg_cl_param)

	corr = corrs(samples, pearson = true)
	max_corr = maximum(abs.([corr[i,j] for i=1:size(corr)[1], j=1:size(corr)[2] if i!=j]))
	avg_corr = mean(abs.([corr[i,j] for i=1:size(corr)[1], j=1:size(corr)[2] if i!=j]))
	println("MAX CORR ", max_corr)
	append!(max_corrs, [max_corr])
	append!(avg_corrs, [avg_corr])
	#println("Rise Reg ", rise_reg)
	#println("Initial Q terms")
	#for i in keys(q.terms)
	#	println(i, " : ", q.terms[i])
	#end

	# not quite true... appropriate threshold is when learning same graph
	#thres = minimum([abs(tree_params[k]) for k in keys(tree_params)])/2
	if use_threshold
		thres = minimum([abs(tree_params[k]) for k in keys(tree_params) if length(k)==2])/2
		#thres = .01
		append!(thresholds, thres)
		thres = 0.0
		append!(kls, ln_convert(kl))

		# println("First Q terms: threshold = ", thres)
		# keyz = collect([keys(q.terms)...])
		# keyz = keyz[sortperm([-abs(q.terms[i]) for i in keyz])]
		# for i in keyz
		# 	println(i, " : ", q.terms[i])
		# end

		q.terms = prune_params(q.terms, tol = thres)
		
		q = learn(samples, mRISE, NLP(), collect(keys(q.terms)))
		if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
			q = FactorGraph(learn_order, obs, :spin, array2dict(q)) 
		end


		#q = mrf(array2dict(q), samples)
		println("Re-run Q terms: threshold = ", thres)
	else
		println("RISE Learned Parameters")
	end
	keyz = collect([keys(q.terms)...])
	# keyz = keyz[sortperm([length(i)==1 ? 0 : i[2] for i in keyz])]
	# keyz = keyz[sortperm([i[1] for i in keyz])]
	key2 = keyz[sortperm([-abs(q.terms[i]) for i in keyz])]
	for k in sort_params(q.terms)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t \t ", round(q.terms[k],3))
		if !haskey(all_params, k)
			all_params[k] = []
		end
		append!(all_params[k], [q.terms[k]])
	end
	println("P terms")

	for k in sort_params(p.params)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & ", round(p.params[k],3), " \\\\ ")
	end

	println()

	println("RISE Learned by Magnitude")
	for k in key2
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & \t", round(q.terms[k],3), " \\\\ ")
	end
	println()
	sortededges = [k for k in key2 if length(k) == 2][1:length(ground_truth_edges)]
	for i in sortededges
		if !haskey(gt_edges, i)
			gt_edges[i] = []
		end
		append!(gt_edges[i], i in sortededges)
	end
	if tree_type == "321" && ((1,2) in sortededges && (1,3) in sortededges && (2,3) in sortededges)
		full_clique += 1
	elseif tree_type == "321"

	end
	top3 = [k for k in key2 if length(k) == 3][1]
	println("top 3 body ", top3, " ", q.terms[top3]) 
	if !haskey(top_three_body, top3)
		top_three_body[top3] = []
	end
	append!(top_three_body[top3], true)
	top4 = [k for k in key2 if length(k) == 4][1]
	println("top 4 body ", top4, " ", q.terms[top4]) 
	if !haskey(top_four_body, top4)
		top_four_body[top4] = []
	end
	append!(top_four_body[top4], true)
	# AFTER pruning (was previously before)
	some_3_body = false
	for k in keys(q.terms)
		if length(k)==3 #&& q.terms[k] > thres
			if !haskey(three_bodies, k) 
				three_bodies[k] = []
			end
			append!(three_bodies[k], q.terms[k])
			some_3_body = true
		end
		if k in ground_truth_cliques
			append!(three_body_p, [tree_params])
			append!(three_body_q, [q.terms])
		end
	end
	#if some_3_body
	#	display_factor(prune_params(q.terms, tol = thres), string("three_body_", r), field= field, observed = obs)
	#end

	if !log_rise
		three_body = [i for i in keys(q.terms) if length(i)==3]
		append!(three_body_counts, length(three_body))
		append!(three_body_gt, length([i for i in three_body if i in ground_truth_cliques]))
		max3body = isempty(three_body) ? 0 : maximum([abs(q.terms[k]) for k in three_body])
		append!(max3, max3body)
	end
	
	cliques= find_cliques(collect(keys(q.terms)))
	#println("Cliques")
	#println(cliques)
	for k in keys(cliques)
		num_cliques[k][r]= length(cliques[k])
		if k >= 3 
			if !isempty(cliques[k])
				for ind =1:length(cliques[k])
					for tup in ground_truth_cliques
						if all([i in tup for i in cliques[k][ind]])
						 	gt_cliques[k][r] += 1
						end
					end
				end
			end
		end
	end

	println(" ***** Thresholding ***** ")
	println()
	println("NODES for 0.1 threshold")
	tnodes = threshold_params(q.terms; threshold=0.1)
	println(threshold_params(q.terms; threshold=0.1))
	println()
	println("NODES for num nodes = 4")
	println(threshold_params(q.terms; num_nodes = 4))

	a = distance_algebra(tnodes, corrs; distances = false)

	# r = learn(full_samples, mRISE, NLP(), collect(keys(adj_dict)))
	# if isa(r, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
	# 	r = FactorGraph(learn_order, obs, :spin, array2dict(r)) 
	# end
	# println("R terms: ")
	# for i in keys(r.terms)
	# 	println(i, " : ", r.terms[i])
	# end
	if run_anima
		if size(adj_mat) == size(tree_adj)
			anima_num += 1
			if check_structure(full(adj_mat), tree_adj, obs)
				println("Anima CORRECT")
				anima_struct += 1
			else
				println("*** Correct # hidden, incorrect structure ***")
				println("Anima")
				pprint2d(full(adj_mat))
				println("Tree")
				pprint2d(tree_adj)
				append!(anima_wrong_struct, [tree_params])
			end
		else
			println("Anima learns ", size(adj_mat)[1], " variables vs. ", size(tree_adj)[1], " true")
			# PRINTS IF SIZE(ANIMA) > SIZE TREE

			#if size(adj_mat)[1] > size(tree_adj)[1]
			#	display_factor(p, string("orig_tree_corrs_", r), field=field, observed = obs)
			#	display_factor(adj_dict, string("anima_", r), field=field, observed = obs)
			#end
			append!(anima_wrong_num, [tree_params])
		end
		if size(param_adj_mat) == size(tree_adj)
			anima_num += 1
			if check_structure(full(param_adj_mat), tree_adj, obs)
				println("Param CL CORRECT")
				param_struct += 1
			else
				println("*** Correct # hidden, incorrect structure ***")
				println("Param CL")
				pprint2d(full(param_adj_mat))
				println("Tree")
				pprint2d(tree_adj)
				#append!(anima_wrong_struct, [tree_params])
			end
		else
			println("Anima learns ", size(param_adj_mat)[1], " variables vs. ", size(tree_adj)[1], " true")
			# PRINTS IF SIZE(ANIMA) > SIZE TREE

			#if size(adj_mat)[1] > size(tree_adj)[1]
			#	display_factor(p, string("orig_tree_corrs_", r), field=field, observed = obs)
			#	display_factor(adj_dict, string("anima_", r), field=field, observed = obs)
			#end
			#append!(anima_wrong_num, [tree_params])
		end


	end
	
	
	#new_edges, new_trees, hidden_edges = greedy_combine(q.terms, connect_all = false)
	# --- THROUGH 354 --- 
	# new_hiddens = collect(keys(new_trees))
	# new_models = Dict{Int64, MRF}()
	# new_params = Dict{Tuple, Float64}()
	# for i in new_edges 
	# 	# initialize to zero
	# 	j = sort_tuple(i)
	# 	#if !haskey(new_params, j)
	# 	#	new_params[j] = Array{Float64,1}()
	# 	if j in collect(keys(q.terms))
	# 		new_params[j] = q.terms[j]
	# 	end
	# 	#end
	# end


	# for h in keys(new_trees)
	# 	# max kl
	# 	nodes = new_trees[h]
	# 	inds = [1 ; nodes]
	# 	ph = mrf()
	# 	new_samples = zeros(Int64, 2^length(nodes), 1+length(nodes))
	# 	for k = 1:(2^length(nodes))
	# 		node_spins = zeros(Int64, length(nodes))
	# 		digits!(node_spins, k-1, 2) #digits(k, 2)
	# 		node_spins = [2*n - 1 for n in node_spins]
	# 		new_samples[k,1] = sum(full_samples[k,1] for k = 1:size(full_samples)[1] if all([full_samples[k, 1+nodes[i]]==node_spins[i] for i=1:length(nodes)]))
	# 		new_samples[k,2:end] = node_spins
	# 	end

	# 	ph.samples = [new_samples]
	# 	ph.dim = length(nodes)
	# 	qh_params = Dict{Tuple, Float64}()
	# 	subtree_fields = false
	# 	# NOTE: qh_params indexed based on index in nodes (e.g. 1:3 + 4th hidden)
	# 	for i =1:length(nodes) 
	# 		hh = length(nodes) + 1
	# 		if subtree_fields
	# 			# OR KEEP IT THERE AS UNLEARNABLE PARAMETER
	# 			qh_params[(i,)] = 0 #rand()
	# 		end
	# 		if i == 1
	# 			# HAVE TO ADD MAGNETIC FIELD (unless this gets done within new edges (doesn't make sense))
	# 			qh_params[(hh,)] = 0 #rand()
	# 		end
	# 		qh_params[(i, hh)] = rand() # hidden variable is index 4
	# 	end
		
	# 	qh = hmrf(qh_params, new_samples, h_support)
	# 	qh.dim = length(nodes) + 1
	# 	qh.order = 2

	# 	# println("params before")
	# 	# for k in keys(qh.params)
	# 	#  	_k = tuple([i <= length(nodes) ? nodes[i] : h for i in k]...)
	# 	# 	println(_k, ": ", qh.params[k])
	# 	# end

	# 	println("Hidden node ", h, " with clique ", nodes)
	# 	kl = min_kl(ph, qh, verbose = false)
	# 	new_models[h] = qh
		
	# 	#println("learned q hidden")
	# 	qhk = collect(keys(qh.params))
	# 	for k in qhk 
	# 	 	tup = sort_tuple(tuple([i <= length(nodes) ? nodes[i] : h for i in k]...))
	# 	 	if haskey(new_params, tup)
	# 	 		error("Already added key for ", tup, " hidden ", h)
	# 	 	else
	# 	 		new_params[tup] = qh.params[k]
	# 	 	end
	# 	 	#if !haskey(new_params, tup)
	# 	 	#	new_params[tup] = []
	# 	 	#end
	# 	 	#ppend!(new_params[tup], qh.params[k])
	# 	 	qh.params[tup] = qh.params[k]
	# 	 	delete!(qh.params, k)
	# 		#println(tup, ": ", qh.params[k])
	# 	end
	# 	# now have learned (c1,h), (c2, h), (c3, h) model parameters
	# 	# use this to sample h for c1, c2, c3 given by original samples
	# end
	# println("New Params")
	# q_params = Dict{Tuple, Float64}()
	# for k in keys(new_params)
	# 	# NOT USED
	# 	println(k, " : ", new_params[k])
	# 	q_params[k] = mean(new_params[k]) 
	# end
	# # sample from new hidden nodes
	# println("new hiddens ", new_hiddens)
	# hidden_configs = [digits!(zeros(Int64, length(new_hiddens)), k, 2) for k=0:2^length(new_hiddens)-1] 
	# println("hidden configs ", size(hidden_configs))
	# hidden_configs = hidden_configs.*2-1
	# #for k in 0:2^length(new_hiddens)
	# #	hidden_val = zeros(length(new_hiddens)) 
	# #	hidden_configs, digits!(zeros(length(new_hiddens)), digits(k, 2))
	# #end
	# new_hidden_samples = zeros(Int64, size(full_samples)[1]*length(hidden_configs), size(full_samples)[2]+length(new_hiddens))#Array{Int64, 2}()
	# new_hmrf_samples = Array{Array{Real, 2}, 1}()
	# for i = 1:length(hidden_configs)
	# 	append!(new_hmrf_samples, [zeros(Int64, size(full_samples)[1], size(full_samples)[2]+length(new_hiddens))])
	# end
	# #new_hmrf_samples = zeros(Int64, size(full_samples)[1], size(full_samples)[2]+length(new_hiddens), length(hidden_configs))
	# counts = Dict{Int64, Array{Int64,1}}()
	# for k = 1:size(full_samples)[1]
	# 	samps = zeros(Int64, full_samples[k,1], length(new_hiddens))
	# 	for h_ind = 1:length(new_hiddens) # also keys(new_models)
	# 		w = cond_prob_hidden(new_models[new_hiddens[h_ind]], new_hiddens[h_ind], full_samples[k, :], h_support) # returns Array of length = length(h_support) = + 1 prob / -1 prob
	# 		samps[:, h_ind] = StatsBase.sample(h_support, Weights(w), full_samples[k,1], replace = true)	
	# 		#counts[h] = [sum([samps[kk]==-1 for kk=1:length(samps)]), sum([samps[kk]==1 for kk=1:length(samps)])]
	# 	end
	# 	for kh=1:length(hidden_configs)
	# 		freq = sum(samps[kk, :] == (hidden_configs[kh]) for kk = 1:size(samps)[1])
	# 		new_hidden_samples[(k-1)*length(hidden_configs) + kh, :] = [freq ; full_samples[k,2:end] ; hidden_configs[kh]]
	# 		# sample from new_models[h]
	# 		new_hmrf_samples[kh][k, :] = [freq ; full_samples[k,2:end] ; hidden_configs[kh]]
	# 	end
	# end
	# # qq = mrf(q_params, )
	# # KL better than before?
	# # PRUNE 
	# # RUN RISE AGAIN 
	# println("new_hidden_samples size ", size(new_hidden_samples), " n samps ", sum(new_hidden_samples[:,1]))
	# #println("new hmrf ", size(new_hmrf_samples), " n samps ", sum(sum(new_hmrf_samples[:, 1, j] for j =1:size(new_hmrf_samples)[3])))
	# # sum new hidden samples over observed
	# q_tree = hmrf(new_params, new_hmrf_samples) 
	# kl = emp_lld(p) - emp_lld(q_tree)

	# println("New KL ", ln_convert(kl))

	append!(base_params, p.params)
	append!(learned_params, q.terms)
	if vary_homo > 0
		append!(homogenous, tree_params[collect(keys(tree_params))[1]])
	end

	#clmi = chow_liu(mi_mi)
	if run_anima
		cldist = chow_liu(distance, min = true)
		if !isempty(setdiff(collect(keys(cl_dict)), collect(keys(cldist))))
			display_factor(cl_dict, string("chow_liu_mutual_information_",r), field = field, observed = obs)
			display_factor(cldist, string("chow_liu_distances_", r), field = field, observed = obs)
		end
		if cl
			display_factor(cl_dict, string("chow_liu_",r), field = field, observed = obs)
			display_factor(p, string("orig_tree_", r), field=field, observed = obs)
			display_factor(adj_cl_distances, string("chow_liu_corrs_",r), field = field, observed = obs)
			display_factor(chow_liu(dict2array(q.terms)), string("chow_liu_params_",r), field = field, observed = obs)
		end
	end
	# what to record from anima?
	display_factor(chow_liu(dict2array(q.terms)), string("chow_liu_params_",r), field = field, observed = obs)
	println("Full Corrs")
	full_corr = corrs(full_samples, pearson = true)
	pprint2d(full_corr, latex = true)
end

base_by_coupling = params_to_dict(base_params)
learned_by_coupling = params_to_dict(learned_params)


pcorr= Dict{Tuple, Float64}()
corr = corrs(full_samples)
println(size(corr))
for k in keys(p.params)
	if length(k) == 2
		pcorr[k] = corr[k[1], k[2]]
	end 
end


display_factor(p, "orig_tree", field=field, observed = obs)
display_factor(pcorr, "orig_tree_corrs", field=field, observed = obs)


qcorr= Dict{Tuple, Float64}()
corr = corrs(samples, pearson = true)
for k in keys(q.terms)
	if length(k) == 2
		qcorr[k] = corr[k[1], k[2]]
	end 
end
println("my corr")
pprint2d(corr)
display_factor(q, "learned_graph", field= field, observed = obs)
#display_factor(qcorr, "learned_graph_corrs", field= field, observed = obs)
if run_anima
	display_factor(adj_dict, "anima_tree", field=field, observed = obs)#should match
	#display_factor(chow_liu(mis), "chow_liu_WrongMI", field = field, observed = obs)
end


println()
println("mi")
mi_mi =ln_convert(mi_bin(samples))
pprint2d(mi_mi)

if run_anima
	display_factor(chow_liu(mi_mi), "chow_liu_my_correct_MI", field = field, observed = obs)
	display_factor(chow_liu(distance, min = true), "chow_liu_distances", field = field, observed = obs)
end
display_factor(chow_liu(dict2array(q.terms)), string("chow_liu_params"), field = field, observed = obs)

f = PyPlot.figure("Ground Truth Edges Learned");
key_sort = collect(keys(gt_edges))
percent = sortperm([-sum(gt_edges[k])/runs for k in key_sort])
key_sort = key_sort[percent]
PyPlot.plt[:bar]([1:length(key_sort)...], [sum(gt_edges[k])/runs for k in key_sort]);
PyPlot.title(string("Edge Learned by RISE"));
PyPlot.xticks([1:length(key_sort)...], key_sort)
#PyPlot.xlabel("Edge", key_sort);
PyPlot.savefig(string("learned_edges_", tree_type, ".pdf"));

println("full clique learned ", full_clique, " out of ", runs)


ffff = PyPlot.figure("Three Body Learned");
key_sort = collect(keys(top_three_body))
percent = sortperm([-sum(top_three_body[k])/runs for k in key_sort])
key_sort = key_sort[percent]
PyPlot.plt[:bar]([1:length(key_sort)...], [sum(top_three_body[k])/runs for k in key_sort]);
PyPlot.title(string("Top 3-Body Learned by RISE"));
PyPlot.xticks([1:length(key_sort)...], key_sort)
#PyPlot.xlabel("Edge", key_sort);
PyPlot.savefig(string("learned_3_body_", tree_type, ".pdf"));

f = PyPlot.figure("Four Body Learned");
key_sort = collect(keys(top_four_body))
percent = sortperm([-sum(top_four_body[k])/runs for k in key_sort])
key_sort = key_sort[percent]
PyPlot.plt[:bar]([1:length(key_sort)...], [sum(top_four_body[k])/runs for k in key_sort]);
PyPlot.title(string("Top 4-Body Learned by RISE"));
PyPlot.xticks([1:length(key_sort)...], key_sort)
#PyPlot.xlabel("Edge", key_sort);
PyPlot.savefig(string("learned_4_body_", tree_type, ".pdf"));


#println("sample stats")
#print_stats(samples)
if run_anima
	println("edge distances")
	pprint2d(exp.(-distance))

	println()
	println("MI Me Matlab")
	pprint2d(ln_convert(mi_me))

	println()
	println("Chow Liu Adjacency")
	pprint2d(full(adj_mat_cl))
end








#rintln("MI Info Measures")
#mi_im = ln_convert(mi_bin_im(msamples))
#pprint2d(mi_im)
#print_stats(full_samples)

println("Chow Liu (mine)")
cl_dict = chow_liu(mi_mi)
pprint2d(dict2array(cl_dict))

#println(size(learned_by_coupling))
if vary_homo > 0
	for k in keys(learned_by_coupling)
		println(k, " : ", [(homogenous[i], learned_by_coupling[k][i]) for i = 1:length(learned_by_coupling[k])])
	end
end

if calc_param_var
	for k in keys(all_params)
		learned_values = all_params[k]
		v = var(learned_values)
		println("Parameter Variance ", k, " : ", v)	
		thresh = [l for l in learned_values if abs(l) > v]
		println("Pruned ", length(learned_values)-length(thresh), " out of ", length(learned_values))
	end
end
# println("KLs")
# println(kls)
# println()
# println("MAXIMUM 3 BODY")
# println(max3)
# println()
# println()
# println("three bodies")
# println(three_bodies)


# println("kl length ", length(kls), kls)
# println("cliques 2: ", length(num_cliques[2])," ", num_cliques[2])
# println("cliques 3: ", length(num_cliques[3])," ", num_cliques[3])
# println("cliques 4: ", length(num_cliques[4])," ", num_cliques[4])
# println("gt cliques: ", length(gt_cliques[3])," ", gt_cliques)
# println("3 body counts: ", length(three_body_counts), " ", three_body_counts)
# println("3 body gt: ", length(three_body_gt), " ", three_body_gt)
# println("max 3 body: ", length(max3), " ", max3)

# println()
if plots
	asc_sort = sortperm(kls)
	desc_sort = sortperm(-kls)
	sorter = asc_sort
	kls = kls[sorter]
	num_cliques[2] = num_cliques[2][sorter]
	num_cliques[3] = num_cliques[3][sorter]
	num_cliques[4] = num_cliques[4][sorter]
	gt_cliques[3] = gt_cliques[3][sorter]
	three_body_counts = three_body_counts[sorter]
	three_body_gt = three_body_gt[sorter]
	max3 = max3[sorter]
	thresholds = thresholds[sorter]

	bins = 10
	f =PyPlot.figure("KLs by Random Init");
	PyPlot.plt[:hist](kls, bins);
	PyPlot.title(string("KLs by Random Init ", tree_type));
	PyPlot.xlabel("Random Init (sorted by KL)");
	PyPlot.savefig(string("kl_hist_", tree_type, ".pdf"));


	f = PyPlot.figure("Max 3 Body by Random Init");
	PyPlot.plt[:hist](max3, bins);
	PyPlot.title(string("Max 3 Body by Random Init ", tree_type));
	PyPlot.xlabel("Random Init (sorted by KL)");
	PyPlot.savefig(string("max3body_", tree_type, ".pdf"));


	f = PyPlot.figure("Cliques Learned by Size KL");
	PyPlot.plt[:scatter]([1:runs...], sort(num_cliques[2]), label="# edges"); # kls, 
	PyPlot.plt[:scatter]([1:runs...], sort(num_cliques[3]), label="# 3 cliques");
	PyPlot.plt[:scatter]([1:runs...], sort(num_cliques[4]), label="# 4 cliques");
	PyPlot.legend();
	PyPlot.title(string("Cliques by Size ", tree_type));
	PyPlot.xlabel("Random Init (sorted by KL)");
	PyPlot.savefig(string("cliques_by_size_kl_", tree_type, ".pdf"));


	# resort for threshold
	for i in keys(num_cliques)
		asc_sort = sortperm(num_cliques[i])
		asc_sort2 = sortperm(gt_cliques[i])
		num_cliques[i] = num_cliques[i][asc_sort]
		gt_cliques[i] = gt_cliques[i][asc_sort2]
	end

	f = PyPlot.figure("Cliques Learned by Size Thres");
	PyPlot.plt[:scatter](thresholds, num_cliques[2], label="# edges");
	PyPlot.plt[:scatter](thresholds, num_cliques[3], label="# 3 cliques");
	PyPlot.plt[:scatter](thresholds, num_cliques[4], label="# 4 cliques");
	PyPlot.legend();
	PyPlot.xticks([]);
	PyPlot.title(string("Cliques by Size ", tree_type));
	PyPlot.xlabel("Threshold");
	PyPlot.savefig(string("cliques_by_size_thr_", tree_type, ".pdf"));

	f = PyPlot.figure("Threshold vs KL");
	PyPlot.plt[:scatter](kls, thresholds, label="3-Cliques");
	PyPlot.title(string("Threshold vs KL ", tree_type));
	PyPlot.xlabel("Random Init (sorted by KL)");
	PyPlot.savefig(string("threshold_kl_", tree_type, ".pdf"));



	f = PyPlot.figure("Matching Ground Truth");
	PyPlot.plt[:scatter]([1:runs...], gt_cliques[3], label="3-Cliques");
	PyPlot.plt[:scatter]([1:runs...], three_body_gt, label="3-Body");
	PyPlot.legend();
	PyPlot.xticks([]);
	PyPlot.title(string("Cliques by Size ", tree_type));
	PyPlot.xlabel("Random Init (sorted by KL)");
	PyPlot.savefig(string("match_gt_", tree_type, ".pdf"));
end


if run_anima
	println("Anima algo :")
	println("correct structure ", anima_struct, " out of ", runs)
	println("correct num hidden ", anima_num, " out of ", runs)
end

if example_distibutions
	println("Anima Wrong Struct ", length(anima_wrong_struct))
	println()
	for pp in 1:minimum([5, length(anima_wrong_struct)])
		println("P params: ")
		for k in keys(anima_wrong_struct[pp])
			for i in 1:length(k)
				print(i != length(k) ? string(k[i], ", ") : k[i])
			end
			println("\t", anima_wrong_struct[pp][k])
		end

	end
	println()
	println("Anima Wrong Num Hidden ", length(anima_wrong_num))
	println()
	for pp in 1:minimum([5, length(anima_wrong_num)])
		println("P params: ")
		for k in keys(anima_wrong_num[pp])
			for i in 1:length(k)
				print(i != length(k) ? string(k[i], ", ") : k[i])
			end
			println("\t", anima_wrong_num[pp][k])
		end
		println()
	end

	println()
	println("Three Body Learned ", length(three_body_p))
	println()
	for pp in 1:minimum([5, length(three_body_p)])
		println("P params: ")
		for k in keys(three_body_p[pp])
			println(k, " : ", three_body_p[pp][k])
		end
		println("Q params: ")
		for k in keys(three_body_q[pp])
			for i in 1:length(k)
				print(i != length(k) ? string(k[i], ", ") : k[i])
			end
			println("\t", three_body_q[pp][k])
		end
		println()
	end
end
println("max CL Degrees")
println(max_cl_degree)
println("max? min CL edge")
println(min_cl_edge)
println("max corrs")
println(max_corrs)
println("max degree CL_Param")
println(degree_cl_param)

println("max 3 body")
println(max3)
#println(maximum(max3))
#plot_param_runs(learned_by_coupling, homogenous, "", title="homogenous_rise", orders = [2,3])
#sparsest_cut(q.terms)

# type FactorGraph{T <: Real}
#     order::Int
#     varible_count::Int
#     alphabet::Symbol
#     terms::Dict{Tuple,T} # TODO, would be nice to have a stronger tuple type here
#     variable_names::Nullable{Vector{String}}
#     FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
# end
