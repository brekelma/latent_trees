using Compat
include("math.jl")

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

function avg_lld(m::MRF, dist::Array{Float64, 1})
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
	return ln_convert(avg_lld(m1, dist) - avg_lld(m2, dist), base)
end


function gradient_xent(m::MRF; reversed=false)
	grads = Dict{Tuple, Float64}()
	
	if isa(m, mrf)
		println("MRF with no hidden variables")
		reversed = true
	end
	# marginalizing over hiddens (unnecessary for reversed problem)
	
	evidences = evidence(m) # k x h
	if !reversed
		data_evidence = [sum(evidences[i]) for i=1:length(evidences)] # k x 1
		
		conditionals = evidences ./ data_evidence
		# Conditional Expectation, ONLY WORKS FOR BINARY.... 
		hiddens = [sum(conditionals[k][h]*m.hsupport[h] for h=1:length(m.hsupport)) for k=1:size(m.samples[1])[1]]
		sample_exp = hcat(m.samples[1][:,1:size(m.samples[1])[2]-length(m.hsupport)+1], hiddens)
	end
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


function kl_gradient_ascent(m::MRF; step::Float64 = .001, reversed=false)
	for coupling in keys(m.params)
		m.params[coupling] = m.params[coupling] + step * gradient_xent(m)[coupling]
	end
end

function pearl_corr_test(m::MRF)
	# calc rho_ij - rho_jk*rho_ik
	tests = Dict{Tuple, Float64}()
	rho = corr(m.samples, pearson = true)
	for coupling in m.params
		if length(coupling) == 2
			i = coupling[1]
			j = coupling[2]
			k = setdiff([ii for ii=1:m.dim],[i,j])
			tests[coupling] = rho[i,j] - rho[i,k]*rho[j,k]
			println("ij: ", coupling, " k:", k, " test: ", tests[coupling])
		end
	end
	min = minimum([v for v in tests[k] for k in keys(tests)])
	println("Pearl 3 Body Reconstruction Test ", min > 0 ? "SUCCEEDS" : "FAILS")
	println("min value ", min)
	println("Correlations Test ", rho[i,j]*rho[i,k]*rho[j,k] > 0 ? "SUCCEEDS" : "FAILS")
end