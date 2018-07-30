using Ipopt
using JuMP
using Compat
using MathProgBase
using Mosek 
using SCS

include("mrf.jl")
include("utils.jl")
#include("math.jl")
# CONVERT EVERYTHING TO MY TYPE?
# @compat abstract type GMLMethod end

# type NLP <: GMLMethod
#     solver::MathProgBase.AbstractMathProgSolver
# end
# # default values
# NLP() = NLP(IpoptSolver(print_level = 0))


function max_lld(m::MRF; verbose = true, constrain_triangle = false, constrain_corr = false, calc_hess = false) 
	opt = Model(solver = IpoptSolver(print_level = 0))

	index_map = Dict{Int64, Tuple}()
	n_keys = 0
	for k in keys(m.params)
		n_keys = n_keys+1
        index_map[n_keys] = k
	end

	@variable(opt, params[i=1:n_keys])
	for i in 1:n_keys
    	setvalue(params[i], m.params[index_map[i]])    	
	end
	num_conf = size(m.samples[1])[1]
	num_samp = sum(m.samples[1][:,1])
	println()
	#println("length of samples ", length(m.samples))
	#println()
	#JuMP.register(opt, :xent, 2, xent, autodiff=true)
	@NLobjective(opt, Max, sum(m.samples[1][k, 1]/num_samp*
            (log(sum(exp(
            	sum(prod(m.samples[h][k, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.samples)))
            -log(
            	sum(exp(
            	sum(prod(m.samples[h][kk, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.samples), kk=1:num_conf)))
			for k=1:num_conf))

	if constrain_triangle
		for i= 1:m.dim
			for j= i:m.dim
				l = setdiff([1,2,3],[i,j])[1]
				println("constraint on ", "i ", i, " j ", j, " k ", l)
				@constraint(opt, sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+i]*m.samples[1][l,1+j] - mu[i]*mu[j]) for l=1:num_conf) >= 
								sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+i]*m.samples[1][l,1+k] - mu[i]*mu[k]) for l=1:num_conf)*
								sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+j]*m.samples[1][l,1+k] - mu[j]*mu[k]) for l=1:num_conf))
			end
		end
	elseif constrain_corr
		@constraint(opt, sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+i]*m.samples[1][l,1+j] - mu[i]*mu[j]) for l=1:num_conf)*
								sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+i]*m.samples[1][l,1+k] - mu[i]*mu[k]) for l=1:num_conf)*
								sum(m.samples[1][l,1]/num_samp*(m.samples[1][l,1+j]*m.samples[1][l,1+k] - mu[j]*mu[k]) for l=1:num_conf) >= 0)
	end

	status = solve(opt)
	@assert status == :Optimal
	optimized = deepcopy(getvalue(params))
	#if verbose
#		println()
#		println("Optimized Parameters")
	if verbose
		println("Optimized Parameters")
	end
	for i in 1:n_keys
		m.params[index_map[i]] = optimized[i]
		if verbose
			println(index_map[i], ": ", round(m.params[index_map[i]],3))
		end
	end

	hess = fwddiff_hessian(m, optimized, index_map)
	
	if calc_hess
		println("IPOPT Hessian")
		ihess = ipopt_hessian(opt, m.params, index_map)
		# print with params?
		pprint2d(ihess)
		println()
		println("Eigenvalues ", eig(ihess)[1])
		println("Eigenvectors")
		pprint2d(eig(ihess)[2])
	end

	return getobjectivevalue(opt)#, hess
end


function likelihood_hessian(m::MRF)
	index_map = Dict{Int64, Tuple}()
	params = Array{Float64, 1}()
	n_keys = 0
	for k in keys(m.params)
		n_keys = n_keys+1
        index_map[n_keys] = k
        append!(params, [m.params[k]])
	end
	return fwddiff_hessian(m, params, index_map), index_map
end

function fwddiff_hessian{T<:Real}(m::MRF, params::Array{T, 1}, index_map::Dict{Int64, Tuple})
	num_conf = size(m.samples[1])[1]
	num_samp = sum(m.samples[1][:,1])
	n_keys = length(index_map)
	f(x::Vector) = sum(m.samples[1][k, 1]/num_samp*
            (log(sum(exp(
            	sum(prod(m.samples[h][k, 1+var] for var in index_map[i])
            		*x[i] for i = 1:n_keys)) for h = 1:length(m.samples)))
            -log(
            	sum(exp(
            	sum(prod(m.samples[h][kk, 1+var] for var in index_map[i])
            		*x[i] for i = 1:n_keys)) for h = 1:length(m.samples), kk=1:num_conf)))
			for k=1:num_conf)


	hess = ForwardDiff.hessian(f, params)
	return hess
end


function ipopt_hessian{T<:Real}(m::Model, params::Dict{Tuple, T}, index_map::Dict{Int64, Tuple})
	d = JuMP.NLPEvaluator(m)
    MathProgBase.initialize(d, [:Grad,:Jac,:Hess])
    #objval = MathProgBase.eval_f(d, nodal_reconstruction)
    n_params = length(index_map)

    H = zeros(n_params^2) #zeros(n_params^2)-1
    evalues = zeros(n_params)
    
    keys_map = collect(keys(index_map))
    keys_map = keys_map[sortperm([index_map[k] for k in keys_map])]
    
    my_hess = zeros(n_params, n_params)
    for i in keys(index_map)
        evalues[i] = params[index_map[i]]#+rand()[1]
    end
    structure = MathProgBase.hesslag_structure(d)
    MathProgBase.eval_hesslag(d, H, evalues, 1.0, zeros(n_params))
    
    for i = 1:length(structure[1])
        ind1 = structure[1][i]
        ind2 = structure[2][i]
        #ind1 = index_map[keys_map[structure[1][i]]]
        #ind2 = index_map[keys_map[structure[2][i]]]
        my_hess[ind1, ind2] = H[i]
        my_hess[ind2, ind1] = H[i]
    end
    return my_hess
    #pprint2d(my_hess, rounding = 10)
end


function leighton_rao{T <: Real}(edge_dict::Dict{Tuple, T}, nodes::Int=0)
	# weights = n x n matrix
	if nodes == 0
		nodes = maximum([i for theta in keys(edge_dict) for i in theta])
	end
	edges = collect(keys(edge_dict))

	weights = zeros(nodes, nodes)
	for i = 1:nodes
		for j = i:nodes
			weights[i,j] = sum([edge_dict[tup] for tup in edges if (i in tup && j in tup)])
		end
	end
	# for tup in edges
	# 	weights[tup[1], tup[2]] = edge_dict[tup]
	# 	weights[tup[2], tup[1]] = edge_dict[tup]
	# end
	#opt = Model(solver = SCSSolver())#
	opt = Model(solver = MosekSolver())
	#@variable(opt, points[i=1:nodes])
	@variable(opt, X[1:nodes,1:nodes], SDP) # n points in n dim space
	@NLobjective(opt, Min, sum(weights[i,j] * sum((X[k,i] - X[k,j])^2 for k =1:nodes) for i=1:nodes for j=1:nodes)) 
	@constraint(opt, normalized[i = 1:nodes, j = 1:nodes], sum((X[k,i] - X[k,j])^2 for k=1:nodes) == 1)
	@constraint(opt, triangle[i = 1:nodes, j = i:nodes, k = j:nodes], 
		sum((X[kk,i] - X[kk,j])^2 for kk=1:nodes) + sum((X[kk,j] - X[kk,k])^2 for kk=1:nodes) >= sum((X[kk,i] - X[kk,k])^2 for kk=1:nodes)) 
	status = solve(opt)
end



function sparsest_cut{T <: Real}(edge_dict::Dict{Tuple, T}, nodes::Int=0)
	# weights = n x n matrix
	if nodes == 0
		nodes = maximum([i for theta in keys(edge_dict) for i in theta])
	end
	edges = collect(keys(edge_dict))

	weights = zeros(nodes, nodes)
	for i = 1:nodes
		for j = i:nodes
			weights[i,j] = sum([edge_dict[tup] for tup in edges if (i in tup && j in tup)])
		end
	end
	# for tup in edges
	# 	weights[tup[1], tup[2]] = edge_dict[tup]
	# 	weights[tup[2], tup[1]] = edge_dict[tup]
	# end
	#opt = Model(solver = SCSSolver())#
	opt = Model(solver = MosekSolver())
	#@variable(opt, points[i=1:nodes])
	@variable(opt, X[1:nodes,1:nodes], SDP) # n points in n dim space
	@NLobjective(opt, Min, sum(weights[i,j] * sum((X[k,i] - X[k,j])^2 for k =1:nodes) for i=1:nodes for j=1:nodes)) 
	
	#exp1=@QuadConstraint(sum((X[k,2] - X[k,1])^2 for k=1:nodes)==1)
	#println(typeof(exp1))
	#l2con = Array{JuMP.GenericQuadConstraint, 2}(nodes, nodes)
	#for i=1:nodes
#		for j=1:nodes
	#		l2con[i,j] = @QuadConstraint(sum((X[k,i] - X[k,j])^2 for k=1:nodes)==1)
	#	end
	#end
	@constraint(opt, normalized[i = 1:nodes, j=1:nodes], norm(X[:,i] - X[:,j]) <= 1)
	#@constraint(opt, normalized[i = 1:nodes, j=1:nodes], l2con[i,j]==1)

	#@constraint(opt, normalized[i = 1:nodes, j = 1:nodes], sum((X[k,i] - X[k,j])^2 for k=1:nodes) == 1)
	#@constraint(opt, triangle[i = 1:nodes, j = i:nodes, k = j:nodes], 
	#	sum((X[kk,i] - X[kk,j])^2 for kk=1:nodes) + sum((X[kk,j] - X[kk,k])^2 for kk=1:nodes) >= sum((X[kk,i] - X[kk,k])^2 for kk=1:nodes)) 
	status = solve(opt)
end

