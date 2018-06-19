using Ipopt
using JuMP
using Compat
using MathProgBase

include("mrf.jl")
#include("math.jl")
# CONVERT EVERYTHING TO MY TYPE?
# @compat abstract type GMLMethod end

# type NLP <: GMLMethod
#     solver::MathProgBase.AbstractMathProgSolver
# end
# # default values
# NLP() = NLP(IpoptSolver(print_level = 0))


function max_lld(m::MRF; verbose = true, constrain_triangle = false, constrain_corr = false) 
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

	#JuMP.register(opt, :xent, 2, xent, autodiff=true)
	@NLobjective(opt, Max, sum(m.samples[1][k, 1]/num_samp*
            (log(sum(exp(
            	sum(prod(m.samples[h][k, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.samples)))
            -log(sum(
            	sum(exp(
            	sum(prod(m.samples[h][kk, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.samples)) for kk=1:num_conf)))
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
	optimized = deepcopy(getvalue(params))
	if verbose
		println()
		println("Optimized Parameters")
		for i in 1:n_keys
			m.params[index_map[i]] = optimized[i]
			println(index_map[i], ": ", round(m.params[index_map[i]],3))
		end
	end

	d = JuMP.NLPEvaluator(opt)
	MathProgBase.initialize(d, [:Grad])
	objval = MathProgBase.eval_f(d, optimized)
	return objval
end