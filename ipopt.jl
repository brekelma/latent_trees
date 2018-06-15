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


function max_lld(m::MRF) 
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
	num_samples = sum(m.samples[1][:,1])

	#JuMP.register(opt, :xent, 2, xent, autodiff=true)
	@NLobjective(opt, Max, sum(m.samples[1][k, 1]/num_samples*
            (log(sum(exp(
            	sum(prod(m.samples[h][k, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.hsupport)))
            -log(sum(
            	sum(exp(
            	sum(prod(m.samples[h][kk, 1+var] for var in index_map[i])
            		*params[i] for i = 1:n_keys)) for h = 1:length(m.hsupport)) for kk=1:num_conf)))
			for k=1:num_conf))

	status = solve(opt)
	optimized = deepcopy(getvalue(params))
	println("Optimized Parameters")
	for i in 1:n_keys
		m.params[index_map[i]] = optimized[i]
		println(index_map[i], ": ", round(m.params[index_map[i]],3))
	end
	d = JuMP.NLPEvaluator(opt)
	MathProgBase.initialize(d, [:Grad])
	objval = MathProgBase.eval_f(d, optimized)
	return objval
end