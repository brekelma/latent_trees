using CSV
using GraphicalModelLearning
using MATLAB
using PyPlot

include("utils.jl")
include("mrf.jl")
include("ipopt.jl")
#include("hiddenGML.jl")
PyPlot.ioff()

fn = "four_vars.csv"

runs = 5
rand_init = false
rise = true
all_rise = false
sparsity_reg = 0.0
entropy_reg = 0.0
num_samp = 10000000
obs = 3
learn_order = 2
range_ = [-1, 1]
field_ = [-.5, .5]
min_abs = .3
_qh_ = false
print_hessian = false


learner =  rise ? hRISE(sparsity_reg, true, learn_order, 1, entropy_reg) : hRPLE(sparsity_reg, true, learn_order, 1)
learner =  all_rise ? hRISEAll(sparsity_reg, true, learn_order, 1, entropy_reg) : learner
#learner = hRISE(0.0, true, learn_order, 1)
degen_eigens = Array{Float64,2}()
for r=1:runs
	p_params = read_params(fn; rand_init = rand_init, range = range_, field_range = field_, min_abs = min_abs)
	pp_params = p_params

	println("P Params")
	for k in sort_params(p_params)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & ", round(p_params[k],3), " \\\\ ")
		#println("\t ", round(p_params[k],3))#" \\\\ ")
	end
	

	model = FactorGraph(maximum([length(i) for i in keys(p_params)]), 
		maximum([i for theta in keys(p_params) for i in theta]), 
		:spin, p_params)

	full_samples = GraphicalModelLearning.sample(model, num_samp)
	samples = full_samples[:,1:(obs+1)]

	p = mrf(p_params, full_samples)
	pp = mrf(pp_params, full_samples)

	# HIDDEN RISE / RPLE ... COMMENTED OUT TO AVOID hiddenGML import errors / conflict with GraphicalModelLearning

	#println("min KL Trivial p:p :", min_kl(p, pp))
	# if rise
	# 	q = learn(samples, learner, NLP())
	# 	#q, q_h = learn(samples, learner, NLP()) # returns Factor Graph
	# else
	# 	q = learn(samples, learner, NLP())
	# end


	# qs = Array{FactorGraph, 1}()
	# if isa(q, Array) #typeof(q) != GraphicalModelLearning.FactorGraph{Float64} && typeof(q) != GraphicalModelLearning.FactorGraph{Real}
	# 	if length(q) == 1
	# 		q = FactorGraph(learn_order, obs+1, :spin, array2dict(q)) 
	# 		append!(qs, [fg])
	# 	else
	# 		for i=1:length(q)
	# 			fg = FactorGraph(learn_order, obs+1,  :spin, q[i])
	# 			append!(qs, [fg])
	# 		end
	# 	end
	# else
	# 	append!(qs, [q])
	# end
	# for i = 1:length(qs)
	# 	println()
	# 	q_mrf = mrf(qs[i].terms, full_samples)
	# 	kl =  kl_empirical(p, q_mrf)
	# 	println("KL obs for spin ", i," := ", kl)

	# 	if i != length(qs) # NOTE : SKIPS HIDDEN ... error with sum over empty set !(hidden in key)
	# 		for k in sort_params(q_mrf.params)
	# 			for ii in 1:length(k)
	# 				print(ii != length(k) ? string(k[ii], ", ") : k[ii])
	# 			end
	# 			println("\t & ", round(q_mrf.params[k],3), " \\\\ ")
	# 			#println(k, " : ", q_mrf.params[k])
	# 		end
	# 		println("RPLE for spin ", i, " : ", hrple(q_mrf)[i])
	# 		println("RISE for spin ", i, " : ", hrise(q_mrf)[i])
	# 	end
	# end

	# if rise && _qh_
	# 	q_hmrf = mrf(q_h.terms, full_samples)
	# 	kl =  kl_empirical(p,q_hmrf)
	# 	println("KL hidden model ", kl)
	# end
	
	println()
	println()
	println("********************** Maximum Likelihood **********************")
	# max likelihood min KL (REWRITE & DISPLAY)
	#q_hmrf = hmrf(random_init_dense(3,2), samples, [-1,1])

	q_hmrf = hmrf(random_init_tree_3(obs,2), samples, [-1,1])
	println("Learning Tree Model by Maximum Likelihood")
	kl = min_kl(p, q_hmrf, verbose = false)
	println()

	if print_hessian
		println("Hessian (Tree)")
		for k in sort_params(q_hmrf.params)
			print(k," & \t ")
		end
	end
	println()
	hess, indices = likelihood_hessian(q_hmrf)
	firstone = false
	for k in sort_params(q_hmrf.params)#1:length(hess[:,1])
		i = findfirst(indices, k)
		if firstone #i > 1
			print("\n")
		end
		firstone = true
		print(indices[i], " & \t")
		for j = 1:length(hess[1,:])
			print(round(hess[i,j], 4), " & \t")
		end
	end
	println("Eigenvalues of hessian")
	eigs = eig(hess)
	println(eigs[1])

	q_hd = hmrf(random_init_dense(obs,2), samples, [-1,1])
	println("Learning Dense Hidden Model with Max Likelihood ")
	kl = min_kl(p, q_hd, verbose = false)
	println()

	if print_hessian
		println("Hessian (Dense model)")
	
		for k in sort_params(q_hd.params)
			print(k," & \t ")
		end
	end
	hess, indices = likelihood_hessian(q_hd)
	firstone = false
	if print_hessian
		for k in sort_params(q_hd.params)#1:length(hess[:,1])
			i = findfirst(indices, k)
			if firstone #i > 1
				print("\n")
			end
			firstone = true
			print(indices[i], " & \t")
			for j = 1:length(hess[1,:])
				print(round(hess[i,j], 4), " & \t")
			end
		end
	end
	println()
	
	eigs = eig(hess)
	degen_inds = find([abs(eigs[1][i]) < .0001 for i =1:length(eigs[1])])
	if print_hessian
		println("Eigenvalues of hessian")
		println(eigs[1])
		
		println("degen inds ", degen_inds)
		println()
		println("Eigenvectors")
		println("last 3")
		#println(eigs[2][:,end-2:end])
		println(eigs[2][:,degen_inds])
	end

	if isempty(degen_eigens)
		degen_eigens =  eigs[2][:,degen_inds]
	else
		degen_eigens = [degen_eigens  eigs[2][:,degen_inds]]
	end

	sorted_params = sort_params(q_hd.params)



	for k in sort_params(q_hd.params)#1:length(hess[:,1])
		i = findfirst(indices, k)
		if firstone #i > 1
			print("\n")
		end
		firstone = true
		print(indices[i], " & \t")
		for j = length(eigs[2][1,:])-2:length(eigs[2][1,:]) # 1:length(eigs[2][1,:])
			print(round(eigs[2][i,j], 4))
			if j == length(eigs[2][1,:])
				print(" \\\\ ")
			else
				print(" & \t")
			end
		end
	end

	println()
	println("ML params ")
	for k in sort_params(q_hd.params)#1:length(hess[:,1])
		println(k, " : ", q_hd.params[k])
	end

	ml_params = deepcopy(q_hd.params)
	old_kl = kl_empirical(p, q_hd)
	println("KL emp before adding hessian: ", old_kl)
	for e = 2:-1:0
		for mult = [-2, -1, -.5, .5, 1, 2]
			for k in keys(q_hd.params)
				row_k = findfirst(indices, k)
				q_hd.params[k] = ml_params[k] + mult*eigs[2][row_k, end-e]  
			end
			#println("updated params for innermost eigen ")
			#for k in sort_params(q_hd.params)#1:length(hess[:,1])
			#	println(k, " : ", q_hd.params[k])
			#end
			new_kl = kl_empirical(p, q_hd)
			println("KL eigen ", e+1, " * ", mult, ": ", new_kl)
		end
	end

	println()


	#pprint2d(eig(hess)[2])
	# println()
	# println("Learning Obseved Model ")
	# full_obs = mrf(random_init_p(3, learn_order; range = [-1,1]), samples)
	# kl = min_kl(q_hmrf, full_obs, verbose = false)
	# println()
	# println()
	# println("Best Order ", learn_order, " Observed Model : ")
	# for k in sort_params(full_obs.params)
	# 	for i in 1:length(k)
	# 		print(i != length(k) ? string(k[i], ", ") : k[i])
	# 	end
	# 	println("\t & ", round(full_obs.params[k],3), " \\\\ ")
	# 	#println(k, " : ", q_hmrf.params[k], " \t  true param: ", p.params[k])
	# end

	# println()

	# println("P Params")
	# for k in sort_params(p_params)
	# 	for i in 1:length(k)
	# 		print(i != length(k) ? string(k[i], ", ") : k[i])
	# 	end
	# 	println("\t & ", round(p_params[k],3), " \\\\ ")
	# 	#println("\t ", round(p_params[k],3))#" \\\\ ")
	# end
	# println()
	println("Params learned by Maximum likelihood ( Tree ) : ")
	println("q hmrf ", collect(keys(q_hmrf.params)))
	println("p ", collect(keys(p_params)))
	if q_hmrf.params[(q_hmrf.dim,)]*p_params[(q_hmrf.dim,)] < 0
		switch_sign(q_hmrf, q_hmrf.dim)
	end
	for k in sort_params(q_hmrf.params)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & ", round(q_hmrf.params[k],3), " \\\\ ")
		#println(k, " : ", q_hmrf.params[k], " \t  true param: ", p.params[k])
	end
	println()
	println("Params learned by Maximum likelihood ( All Couplings ) : ")
	if q_hd.params[(q_hd.dim,)]*p_params[(q_hd.dim,)] < 0
		switch_sign(q_hd, q_hd.dim)
	end
	for k in sort_params(q_hd.params)
		for i in 1:length(k)
			print(i != length(k) ? string(k[i], ", ") : k[i])
		end
		println("\t & ", round(q_hd.params[k],3), " \\\\ ")
		#println(k, " : ", q_hmrf.params[k], " \t  true param: ", p.params[k])
	end
	qmodel = FactorGraph(maximum([length(i) for i in keys(q_hmrf.params)]), 
		maximum([i for theta in keys(q_hmrf.params) for i in theta]), 
		:spin, q_hmrf.params)
	q_samples = GraphicalModelLearning.sample(qmodel, num_samp)
	println()
	#println("Maximum likelihood Samples vs True : size q_samples ", size(q_samples))
	my_kl = 0.0
	my_kl_2 = 0.0

	ev = evidence(q_hd)
	normalized = 0.0
	nrmlzd = 0.0
	nq = 0.0
	lp = log_partition(q_hd)

	qmodelhd = FactorGraph(maximum([length(i) for i in keys(q_hmrf.params)]), 
		maximum([i for theta in keys(q_hd.params) for i in theta]), 
		:spin, q_hd.params)
	q_samples = GraphicalModelLearning.sample(qmodelhd, num_samp)
	for k = 1:size(q_samples)[1]
		normalized += sum(ev[k])/exp(lp)
		nrmlzd += full_samples[k,1]/sum(full_samples[:,1])
		nq += q_samples[k,1]/sum(q_samples[:,1])
		#println(k, ": ", full_samples[k,2:end], "\t", full_samples[k,1], "\t \t ", q_samples[k,2:end], ": ", q_samples[k,1])
		
		#println("KL val ", full_samples[k,1]/sum(full_samples[:,1])*(log(full_samples[k,1]/sum(full_samples[:,1]))-log(q_samples[k,1]/sum(q_samples[:,1]))))
		my_kl += full_samples[k,1]/sum(full_samples[:,1])*(log(full_samples[k,1]/sum(full_samples[:,1]))-log(q_samples[k,1]/sum(q_samples[:,1])))
		
		my_kl_2 += full_samples[k,1]/sum(full_samples[:,1])*(log(full_samples[k,1]/sum(full_samples[:,1])) - log(ev[k][Int64(1+floor((full_samples[k,end]+1)/2))]) + lp)
	end

	# println()
	# println("My KL calculation ", ln_convert(my_kl, 2))
	# println("My (dist params) KL calc ", ln_convert(my_kl_2, 2))
	# println()
	

	# println("RPLE Objective for Maximum Likelihood")
	# rple_values = hrple(q_hmrf)
	# for i in keys(rple_values)
	# 	println("Node ", i, " : hRPLE : ", rple_values[i])
	# end
	# println()
	# println("hRISE for Maximum Likelihood")
	# rise_values = hrise(q_hmrf)
	# for i in keys(rise_values)
	# 	println(i)
	# 	println("Node ", i, " : hRISE : ", rise_values[i])
	# end


	# #println("Q learned params")
	# for i = 1:length(qs)
	# 	q = qs[i]
	# 	#if i == 1
	# 	my_kl = 0.0
	# 	if i != length(qs) || true
	# 		qmodel = FactorGraph(maximum([length(i) for i in keys(q.terms)]), 
	# 			maximum([i for theta in keys(q.terms) for i in theta]), 
	# 			:spin, q.terms)
	# 		q_samples = GraphicalModelLearning.sample(qmodel, num_samp)
	# 		println()
	# 		println("Q Model Node ", i, " (", size(q_samples)[1], " configs vs. full ", size(full_samples)[1],")")
	# 		#pprint2d(q_samples)
	# 		for k = 1:size(q_samples)[1]
	# 			if rise
	# 				println(full_samples[k,2:end], "\t", full_samples[k,1], "\t \t", q_samples[k,1])#, qh_samples[k,1])
	# 			else
	# 				println(full_samples[k,2:end], "\t", full_samples[k,1], "\t \t", q_samples[k,1])
	# 			end
	# 			my_kl += full_samples[k,1]/sum(full_samples[:,1])*(log(full_samples[k,1]/sum(full_samples[:,1]))-log(q_samples[k,1]/sum(q_samples[:,1])))
	# 		end


	# 		conf_dict = Dict{Tuple, Array{Int64, 1}}()
	# 		for k = 1:size(q_samples)[1]
	# 			tup = tuple(q_samples[k, 2:(2+obs-1)]...)
				
	# 			if !haskey(conf_dict, tup)
	# 				conf_dict[tup] = []
	# 			end
	# 			append!(conf_dict[tup], [q_samples[k,1]])
	# 		end

	# 		conditional_ent = 0
	# 		for key in keys(conf_dict)
	# 			k = conf_dict[key]
	# 			conditional_ent -= sum(k[i]/num_samp*log(k[i]/sum(k)) for i = 1:length(k))
	# 		end
	# 		println()
	# 		println("KL divergence for Q model node ", i, " : ", my_kl)
	# 		println("Conditional entropy of hidden node wrt ", i, ": ", conditional_ent)
	# 		println()
	# 	end
	# 	# for k in sort_params(q.terms)
	# 	# 	println(k, " : ", q.terms[k])
	# 	# end
	# end
	# println()
	# #println("samples")
	# #pprint2d(full_samples)

	# qmodel = FactorGraph(maximum([length(i) for i in keys(q.terms)]), 
	# 	maximum([i for theta in keys(q.terms) for i in theta]), 
	# 	:spin, q.terms)

	# q_samples = GraphicalModelLearning.sample(qmodel, num_samp)

	# #println("Observed samples")
	# #pprint2d(samples)

	# conf_dict = Dict{Tuple, Array{Int64, 1}}()
	# for k = 1:size(q_samples)[1]
	# 	tup = tuple(q_samples[k, 2:(2+obs-1)]...)
	# 	println(tup)
	# 	if !haskey(conf_dict, tup)
	# 		conf_dict[tup] = []
	# 	end
	# 	append!(conf_dict[tup], [q_samples[k,1]])
	# end

	# conditional_ent = 0
	# for key in keys(conf_dict)
	# 	k = conf_dict[key]
	# 	conditional_ent -= sum(k[i]/num_samp*log(k[i]/sum(k)) for i = 1:length(k))
	# end
	# println()
	# println("Calculated conditional entropy hidden : ", conditional_ent)
	# println()

	# if rise && _qh_
	# 	qhmodel = FactorGraph(maximum([length(i) for i in keys(q_h.terms)]), 
	# 	maximum([i for theta in keys(q_h.terms) for i in theta]), 
	# 	:spin, q_h.terms)

	# 	qh_samples = GraphicalModelLearning.sample(qhmodel, num_samp)
	# 	println()
	# 	println("Q hidden samples")
	# 	pprint2d(qh_samples)

	# 	conf_dict = Dict{Tuple, Array{Int64, 1}}()
	# 	for k = 1:size(qh_samples)[1]
	# 		tup = tuple(qh_samples[k, 2:(2+obs-1)]...)
	# 		if !haskey(conf_dict, tup)
	# 			conf_dict[tup] = []
	# 		end
	# 		append!(conf_dict[tup], [qh_samples[k,1]])
	# 	end

	# 	conditional_ent = 0
	# 	for key in keys(conf_dict)
	# 		k = conf_dict[key]
	# 		println("key ", key, " counts over hidden : ", k, " sum ", sum(k), " ", [log(k[i]/sum(k)) for i = 1:length(k)])
	# 		conditional_ent -= sum(k[i]/num_samp*log(k[i]/sum(k)) for i = 1:length(k))
	# 		println("ent contrib ", sum(k[i]/num_samp*log(k[i]/sum(k)) for i = 1:length(k)))
	# 	end
	# 	println()
	# 	println("calculated conditional entropy ", conditional_ent)
	# 	println()
	# end 

	# println()
	# println("Sample Counts  \t \t Original \t Obs \t Hidden")
	# for k = 1:size(q_samples)[1]
	# 	if rise
	# 		println(full_samples[k,1:end], "\t", full_samples[k,1], "\t", q_samples[k,1], "\t")#, qh_samples[k,1])
	# 	else
	# 		println(full_samples[k,1:end], "\t", full_samples[k,1], "\t", q_samples[k,1])
	# 	end
	# end
	# # i'd hope we could learn from 4 / 321 , then learn correlation with 5 (also observed)


	# display_factor(p_params, string("orig_4var", r), field = true, observed = obs)
end

println("Runs ", runs)
println("Degen Eigens Size: ", size(degen_eigens))
println("singular values: ", svd(degen_eigens)[2])