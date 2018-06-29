using Compat
using Graphs
include("math.jl")
export min_kl, min_kl_manual, emp_lld, dist_lld
# CONVERT EVERYTHING TO MY TYPE?
@compat abstract type MRF end


type mrf <: MRF
	params::Dict{Tuple, Float64}
	samples::Array{Array{Real,2},1} #Array{Real,2}
	dim::Int64
	order::Int64
	#base::Float64
	#stats::Dict{Tuple, Array{Float64,1}}
	mrf(params, samples) = new(params, [samples], maximum([i for theta in keys(params) for i in theta]), maximum([length(i) for i in keys(params)]))

end

# Add hidden configurations (Specify as Joint binary, e.g.)  
# [samples( configs x (dims + h_i) ), h_i] adds column of z = h_i to samples
type hmrf <: MRF
	params::Dict{Tuple, Float64}
	samples::Array{Array{Real,2},1} # extra dimension for hidden node(s)
	hsupport::Array{Real,1} #, String}
	dim::Int64
	order::Int64
	#stats::Dict{Tuple, Array{Float64,1}}


	# CHECK AGAIN
	hmrf(params, samples, hsupport) = new(params, [hcat(samples, fill(h_i, size(samples)[1])) for h_i in hsupport], hsupport, maximum([i for theta in keys(params) for i in theta]), maximum([length(i) for i in keys(params)]))
	#FULL CONSTRUCTOR
	#if ndims(samples, 2) && !isa(hsupport, String) ? new(params, permutedims([hcat(samples, fill(h_i, size(samples)[1])) for h_i in hsupport], [2,3,1]), hsupport) : new(params, permutedims([samples], [2,3,1]), hsupport)#, Dict{Tuple, Array{Float64,1}}())
	
	# NECESSARY to permute dims?
	#hmrf(params, samples, hsupport) = new(params, samples, hsupport, Dict{Tuple, Array{Float64,1}}())

	hmrf(params, samples) = new(params, [samples], "none", maximum([i for theta in keys(params) for i in theta]), maximum([length(i) for i in keys(params)]))
end


function evidence(m::MRF)
	configs = size(m.samples[1])[1]
	# return list over hidden variable values, each of which is evidence over k configs
	stats = [[exp(sum(m.params[coupling]*prod(m.samples[h][k, 1 + coupling[i]] for i=1:length(coupling)) for coupling in keys(m.params))) for h=1:length(m.samples)] for k=1:configs]
	# k x h matrix
	#ev = Array{Float64, 2}
	#for k = 1:length(stats)
#		for h = 1:length(stats[1])
		#	ev[k,h] = stats[k][h]
		#end
	#end
	return stats
end

# log sum exp taken over all k configs, h hidden states
function log_partition(m::MRF)
	#evidence_by_hidden = evidence(m)
	return logsumlog(evidence(m))
	#return logsumexp(evidence(m))
end

# marginalizes hidden variables automatically
function emp_lld(m::MRF)
	data_evidence = [sum(evidence(m)[k]) for k=1:size(m.samples[1])[1]] # sum over hidden states
	log_marginal = [log(data_evidence[k]) - log_partition(m) for k=1:length(data_evidence)]
	# dim 1 = configs, dim 2 = sample counts, dim 3 = necessary for hiddens
	return sum(m.samples[1][k, 1]/sum(m.samples[1][:,1])*log_marginal[k] for k=1:size(m.samples[1])[1])
end

function dist_lld(m::MRF, dist::Array{Float64, 1})
	data_evidence = sum(evidence(m), 2) # sum over hidden states
	log_marginal = log(data_evidence) - log_partition(m)
	return sum(dist[k]*log_marginal[k] for k=1:size(m.samples[1])[1])
end

function xent(p, samples)
	m = mrf(p, samples)
	return sum(emp_lld(m))
end


function kl_empirical(m1::MRF, m2::MRF; base::Real = exp(1))
	return ln_convert(sum(emp_lld(m1)) - sum(emp_lld(m2)), base)
end

# can get rid of dist??
function kl_dist(m1::MRF, m2::MRF, dist::Array{Float64, 1}; base::Real= exp(1))
	return ln_convert(dist_lld(m1, dist) - dist_lld(m2, dist), base)
end


function gradient_xent(m::MRF; reversed=false)
	grads = Dict{Tuple, Float64}()
	
	if isa(m, mrf)
		println("MRF with no hidden variables")
		reversed = true
	end
	# marginalizing over hiddens (unnecessary for reversed problem)
	
	if !reversed
		sample_exp = latent_expectation_samples(m)
	end # runs evidences twice
	evidences = evidence(m) # k x h
	dist = [sum(evidences[i]) for i=1:length(evidences)] / exp(log_partition(m))
	#println("DISTRIBUTION OVER DATA : ", dist, sum(dist))
	#sum(evidences, length(size(evidences))) / exp(log_partition(m))

	# gradient ( E_emp log model ) = E_emp - E_model (params)
	for coupling in keys(m.params)
		if reversed
			grads[coupling] = sum((m.samples[1][k, 1]/sum(m.samples[1][:,1] - dist[k])*prod(m.samples[1][1+var, 1] for var in coupling) for k=1:size(m.samples[1])[1]))
		else
			# samples hidden replaced by expectation
			# only these params: (m.hsupport[h]*conditionals[k, h])
			grads[coupling] = sum((sample_exp[k,1]/sum(sample_exp[:,1]) - dist[k])*prod(sample_exp[k, 1+var] for var in coupling) for k=1:size(m.samples[1])[1])
		end
	end
	return grads
end


function latent_expectation_samples(m::MRF)
		evidences = evidence(m)
		data_evidence = [sum(evidences[i]) for i=1:length(evidences)] # k x 1
		
		conditionals = evidences ./ data_evidence
		# Conditional Expectation, ONLY WORKS FOR BINARY.... 
		hiddens = [sum(conditionals[k][h]*m.hsupport[h] for h=1:length(m.hsupport)) for k=1:size(m.samples[1])[1]]
		sample_exp = hcat(m.samples[1][:,1:size(m.samples[1])[2]-length(m.hsupport)+1], hiddens)
		return sample_exp
end

function kl_gradient_ascent(m::MRF; step::Float64 = .001, reversed=false)
	for coupling in keys(m.params)
		m.params[coupling] = m.params[coupling] + step * gradient_xent(m)[coupling]
	end
end

function min_kl_manual(p::MRF, q::MRF; verbose = false)
	kl_history = zeros(0)
	# replace with gradient condition? #size(kl_history)[1] <= 1 || 
	while size(kl_history)[1] <= 1 || abs(kl_history[end] - kl_history[end-1]) > tol 
		#println("Entering while")
		append!(kl_history, kl_empirical(p,q, base=2))
		#println("gradient ascent")
		kl_gradient_ascent(q, step = .01, reversed=false)
		if verbose && size(kl_history)[1]%100== 0
			println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
		end
	end
	println(" * Iter ", size(kl_history)[1], "* KL = ", kl_history[end])
	println()
	println("Final Params")
	print_params(q.params)
	return kl_history[end]
end

function min_kl(p::MRF, q::MRF; verbose = true)
	kl_ = emp_lld(p) - max_lld(q, verbose = verbose)
	println("Final KL divergence (bits): ", ln_convert(kl_, 2))
	return ln_convert(kl_, 2)
end 



function pearl_sandwich(m::MRF)
	if !isa(m, mrf)
		samples = latent_expectation_samples(m)
	else
		samples = m.samples[1]
	end
	return pearl_sandwich_marginal(samples)
	#return pearl_sandwich_full_cov(samples)
end
