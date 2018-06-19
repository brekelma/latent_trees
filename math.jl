function ln_convert(ln::Float64, base::Real=2.0)
	return log(base, exp(ln))
end

function logsumexp(x::AbstractArray{T}) where T<:Real
    S = typeof(exp(zero(T)))    # because of 0.4.0
    isempty(x) && return -S(Inf)
    u = maximum(x)
    abs(u) == Inf && return any(isnan, x) ? S(NaN) : u
    s = zero(S)
    for i = 1:length(x)
        @inbounds s += exp(x[i] - u)
    end
    log(s) + u
end

function logsumexp(x::Array{Array{T,N},N}) where N where T<:Real
	a = zeros(T, (length(x), length(x[1])))
	for i=1:length(x)
		a[i,:] = x[i]
	end
	
	return logsumexp(a)
	#sum(logsumexp(x[i]) for i in length(x))
end

function logsumlog(x::Array{Array{T,N},N}) where N where T<:Real
	a = zeros(T, (length(x), length(x[1])))
	for i=1:length(x)
		a[i,:] = [log(l) for l in x[i]]
	end
	
	return logsumexp(a)
	#sum(logsumexp(x[i]) for i in length(x))
end

function means{T <: Real}(samples::Array{T, 2})
	num_conf = size(samples)[1]
	d = size(samples)[2]-1 # NOT ROBUST to hiddens
	num_samp = sum(samples[k,1] for k=1:num_conf)
	mean = [sum(samples[k,1]/num_samp*samples[k,1+i] for k=1:num_conf) for i=1:d]
	return mean
end

function covs{T <: Real}(samples::Array{T, 2})
	num_conf = size(samples)[1]
	num_samp = sum(samples[k,1] for k=1:num_conf)
	mu = means(samples)
	d = length(mu)

	# p_ij - p_i p_j for all 
	cov = [sum(samples[k,1]/num_samp*(samples[k,1+i]*samples[k,1+j] - mu[i]*mu[j]) for k=1:num_conf) for i=1:d, j=1:d]

	return cov
end

function marginals{T <: Real}(samples::Array{T, 2})
	num_conf = size(samples)[1]
	d = size(samples)[2]-1
	num_samp = sum(samples[k,1] for k=1:num_conf)
	marginals = Dict{Tuple, Float64}()
	for i = 1:d
		marginals[(i,)]= sum(samples[k,1]/num_samp*(samples[k, 1+i]==1) for k = 1:num_conf) 
		for j = (i+1):d
			marginals[(i,j)] = sum(samples[k,1]/num_samp*(samples[k, 1+i]==1)*(samples[k, 1+j]==1) for k = 1:num_conf)
			for m = (j+1):d
				marginals[(i,j,m)] = sum(samples[k,1]/num_samp*(samples[k, 1+i]==1)*(samples[k, 1+j]==1)*(samples[k, 1+m]==1) for k = 1:num_conf)
			end
		end
	end
	return marginals
end

function corrs{T <: Real}(samples::Array{T, 2}; pearson = false)
	num_conf = size(samples)[1]
	num_samp = sum(samples[k,1] for k=1:num_conf)
	mu = means(samples)
	cov = covs(samples)
	d = length(mu)

	if pearson
		corr = [cov[i,j]/sqrt(cov[i,i]*cov[j,j]) for i=1:d, j=1:d]
	else
		corr = [sum(samples[k,1]/num_samp*samples[k,1+i]*samples[k,1+j] for k=1:num_conf) for i=1:d, j=1:d]
	end
	return corr
end

