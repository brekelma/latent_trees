using InformationMeasures

function mi_bin_im{T <: Real}(samples::Array{T, 2})
	mi = zeros(size(samples)[1], size(samples)[1])
	for i = 1:size(samples)[1]
		for j=i+1:size(samples)[1]
			mi[i,j] = log(2^(get_mutual_information(samples[i,:], samples[j,:])))
		end
	end
	mi = mi + transpose(mi)
	return mi
end



function ln_convert(ln::Float64, base::Real=2.0)
	return log(base, exp(ln))
end


function ln_convert(ln::Array, base::Real=2.0)
	return log.(base, exp.(ln))
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
	cov = [sum(samples[k,1]/num_samp*(samples[k,1+i]-mu[i])*(samples[k,1+j]-mu[j]) for k=1:num_conf) for i=1:d, j=1:d]
	#cov = [sum(samples[k,1]/num_samp*(samples[k,1+i]*samples[k,1+j]-mu[i]*mu[j]) for k=1:num_conf) for i=1:d, j=1:d]
	#println("diagonal covs ", cov[1,1], " ", cov[2,2], " ", cov[3,3])
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


function mi_bin{T <: Real}(samples::Array{T, 2})
	marg = marginals(samples)
	configs = size(samples)[1]
	ns = sum(samples[:,1])
	println("samples ", ns)
	mi = zeros(size(samples)[2]-1, size(samples)[2]-1)
	ps = zeros(2,2)
	pp = Dict{String, Float64}()
	ep = 10.0^-6
	for i = 1:size(samples)[2]-1
		for j = i+1:size(samples)[2]-1
			# mi[i,j] = marg[(i,j)]*log(marg[(i,j)]/((marg[(i,)])*(marg[(j,)])))
			# + (marg[(i,)] - marg[(i,j)])*log((marg[(i,)] - marg[(i,j)])/((1-marg[(i,)])*(marg[(j,)])))
			# + (marg[(j,)] - marg[(i,j)])*log((marg[(j,)] - marg[(i,j)])/((marg[(i,)])*(1-marg[(j,)])))
			# + (1-marg[(i,)]-marg[(j,)] + marg[(i,j)])*log((1-marg[(i,)]-marg[(j,)] + marg[(i,j)])/((1-marg[(i,)])*(1-marg[(j,)])))
			# println("MI (", i, ",", j, "):", mi[i,j])
			# pp["00"] = sum([samples[k,1]/ns*1*(samples[k,1+i] == -1 && samples[k,1+j]== -1) for k =1:configs])
			# pp["01"] = sum([samples[k,1]/ns*1*(samples[k,1+i] == -1 && samples[k,1+j]== 1) for k =1:configs])
			# pp["10"] = sum([samples[k,1]/ns*1*(samples[k,1+i] == 1 && samples[k,1+j]== -1) for k =1:configs])
			# pp["11"] = sum([samples[k,1]/ns*1*(samples[k,1+i] == 1 && samples[k,1+j]== 1) for k =1:configs])
			
			# #println("pp ", pp)
			# mi[i,j] = pp["00"]*(log(pp["00"]+eps) - log(1-marg[(i,)]+eps) - log(1-marg[(j,)])+eps)
			# 	+ pp["01"]*(log(pp["01"]+eps) - log(1-marg[(i,)]+eps) - log(marg[(j,)])+eps)
			# 	+ pp["10"]*(log(pp["10"]+eps) - log(marg[(i,)]+eps) - log(1-marg[(j,)])+eps)
			# 	+ pp["11"]*(log(pp["11"]+eps) - log(marg[(i,)]+eps) - log(marg[(j,)])+eps)
			# println("MI (", i, ",", j, "):", mi[i,j])
		# 	println("sum of values ", sum(values(pp)))
		# 	println("MI brute: ", ln_convert(mi[i,j]))
			ps = zeros(2,2)
			ps = [sum(samples[k,1]/ns*1*(samples[k,1+i]== (2*ii-3) && samples[k,1+j]== (2*jj-3)) for k=1:configs) for ii =1:2, jj =1:2]
			mi[i,j] = sum([ps[ii, jj]*log((ps[ii, jj]+ep)/(ep+(ii == 2 ? marg[(i,)] : 1- marg[(i,)])*(jj == 2 ? marg[(j,)] : 1- marg[(j,)]))) for ii =1:2, jj =1:2])# if ps[ii, jj] > 0])
		 	#println("MI (", i, ",", j, "):", mi[i,j])
		 	#println()
		 end
	end
	mi = mi + transpose(mi)
	#println("mi")
	#pprint2d(mi)
	return mi
end


function pearl_sandwich_full_cov{T <: Real}(samples::Array{T, 2})
	# calc rho_ij - rho_jk*rho_ik
	println("Pearl Test on Full (+/- 1) Correlations")
	rhos = Dict{Tuple, Float64}()
	marginal = marginals(samples)
	count = 0
	pass = true
	rhos = covs(samples)
	#for coupling in keys(m.params)
		#if length(coupling) == 2
	for count = 0:(size(samples)[2]-1)-1
		i = count % 3 + 1
		j = (count + 1) % 3 + 1
		k = (count + 2) % 3 + 1
		#ik = sort_tuple((i,k))
		#ij = sort_tuple((i,j))
		#jk = sort_tuple((j,k))
		rhos[i,i] = sqrt(rhos[i,i]) #rhos[(i,)]  = sqrt(marginal[(i,)]*(1-marginal[(i,)]))
		rhos[j,j] = sqrt(rhos[j,j])
		rhos[i,j] = rhos[i,j]/(rhos[i,i]*rhos[j,j])#marginal[ij] - marginal[(i,)]*marginal[(j,)]
		#println("i: ", i, " ", marginal[(i,)], " j: ",  j, " ", marginal[(j,)], " k:", k, " ", marginal[(k,)], " test: ", rhos[i,j])
		count = count + 1
	end
	#println("RHOS ", rhos)
	#println("mult corr ", corrs(samples; pearson=false))
	#println("CORRS ", corrs(samples; pearson=true))
	slack = 0
	for count = 0:(size(samples)[2]-1)-1
		i = count % 3 + 1
		j = (count + 1) % 3 + 1
		k = (count + 2) % 3 + 1
		ik = sort_tuple((i,k))
		ij = sort_tuple((i,j))
		jk = sort_tuple((j,k))
		ijk = sort_tuple((i,j,k))
		lower_bound = marginal[ik]*marginal[ij]/marginal[(i,)] #marginal[ik]*marginal[ij]/marginal[(i,)]
		triangle = rhos[j,j]*rhos[k,k]*(rhos[j,k]-rhos[i,j]*rhos[i,k]) #rhos[(j,)]*rhos[(k,)]*(rhos[jk]-rhos[ij]*rhos[ik])
		upper_bound = lower_bound + triangle
		#println("i: ", i, " j: ", j, " k:", k, " ij: ", ij, " triangle ", sqrt(rhos[j,j]*rhos[k,k]), " rho diff ", rhos[j,k]-rhos[i,j]*rhos[i,k])

		ind_fail = marginal[ijk] < lower_bound || marginal[ijk] > upper_bound
		if ind_fail
			pass = false
		end
		slack = maximum([lower_bound - marginal[ijk], marginal[ijk] - upper_bound, slack])
		println("lower ", lower_bound, " IJK : ", marginal[ijk], " upper : ", upper_bound, ind_fail ? " FAIL": " ")
	end
	println("off by ", slack)
	return pass, rhos, marginal, slack
	#min = minimum([v for k in keys(tests) for v in tests[k]])
	#println("Pearl 3 Body Reconstruction Test ", min > 0 ? "SUCCEEDS" : "FAILS")
	#println("Correlations Test ", rho[1,2]*rho[1,3]*rho[2,3] > 0 ? "SUCCEEDS" : "FAILS")
	#return min, [rho[1,2],rho[1,3],rho[2,3]]
end

function pearl_sandwich_marginal{T <: Real}(samples::Array{T, 2})
	#println("Pearl Test on Marginal ijk = 1")
	# calc rho_ij - rho_jk*rho_ik
	rhos = Dict{Tuple, Float64}()
	marginal = marginals(samples)
	count = 0
	pass = true
	slack = 0.0000000
	slack_l = Array{T, 1}()
	#rhos = covs(samples)
	#for coupling in keys(m.params)
		#if length(coupling) == 2
	for count = 0:(size(samples)[2]-1)-1
		i = count % 3 + 1
		j = (count + 1) % 3 + 1
		k = (count + 2) % 3 + 1
		ik = sort_tuple((i,k))
		ij = sort_tuple((i,j))
		jk = sort_tuple((j,k))
		rhos[(i,)] = sqrt(marginal[(i,)]*(1-marginal[(i,)]))
		rhos[(j,)] = sqrt(marginal[(j,)]*(1-marginal[(j,)]))
		rhos[ij] = marginal[ij] - marginal[(i,)]*marginal[(j,)]
		#println("i: ", i, " ", marginal[(i,)], " j: ",  j, " ", marginal[(j,)], " k:", k, " ", marginal[(k,)], " test: ", rhos[ij])
		count = count + 1
	end
	for count = 0:(size(samples)[2]-1)-1
		i = count % 3 + 1
		j = (count + 1) % 3 + 1
		k = (count + 2) % 3 + 1
		ik = sort_tuple((i,k))
		ij = sort_tuple((i,j))
		jk = sort_tuple((j,k))
		ijk = sort_tuple((i,j,k))
		lower_bound = marginal[ik]*marginal[ij]/marginal[(i,)] #marginal[ik]*marginal[ij]/marginal[(i,)]
		triangle = rhos[(j,)]*rhos[(k,)]*(rhos[jk]-rhos[ij]*rhos[ik]) #rhos[(j,)]*rhos[(k,)]*(rhos[jk]-rhos[ij]*rhos[ik])
		upper_bound = lower_bound + triangle
		#println("i: ", i, " j: ", j, " k:", k, " ij: ", ij, " triangle ", sqrt(rhos[(j,)]*rhos[(k,)]), " rho diff ", rhos[jk]-rhos[ij]*rhos[ik])

		ind_fail = (marginal[ijk] < lower_bound) || (marginal[ijk] > upper_bound)
		if ind_fail
			pass = false
		end
		append!(slack_l, maximum([lower_bound - marginal[ijk], marginal[ijk] - upper_bound]))
		slack = maximum([lower_bound - marginal[ijk], marginal[ijk] - upper_bound, slack])
		#println("lower ", lower_bound, " IJK : ", marginal[ijk], " upper : ", upper_bound, ind_fail ? " FAIL": " ")
	end
	return pass, rhos, marginal, slack, slack_l
end


function test_reverse_closed_form{T <: Real}(tree_params::Dict{Tuple, T}, three_params::Dict{Tuple, T}, mathematica::Bool=true)
	tree_params[(4,4)] = tree_params[(4,)]

	pm = zeros(8,4)
	pm[1,:] = [1,1,1,1]
	pm[2,:] = [1,1,-1,1]
	pm[3,:] = [1,-1,1,1]
	pm[4,:] = [1,-1,-1,1]
	pm[5,:] = [-1,1,1,1]
	pm[6,:] = [-1,1,-1,1]
	pm[7,:] = [-1,-1,1,1]
	pm[8,:] = [-1,-1,-1,1]

	coshes = [sum(tree_params[(i,4)]*pm[k,i] for i=1:4) for k=1:8]
	coshes = [log(cosh(c)) for c in coshes]
	#[log(cosh(q.params))]
	mm = Dict{Tuple, Float64}()
	my = Dict{Tuple, Float64}()
	diffs = Dict{Tuple, Float64}()
	my[(1,2)] = .5*(coshes[1]-coshes[4]+coshes[5]-coshes[6])
	my[(1,3)] = .25*(coshes[1]+coshes[3]-coshes[5]-coshes[7])
	my[(2,3)] = .5*(coshes[4]-coshes[5])
	my[(1,2,3)] = .5*(coshes[1]-coshes[8])
	
	mm[(1,2)] = .5*coshes[2]-.25*coshes[7]-.25*coshes[3]-.25*coshes[1]+.25*coshes[8]
	mm[(1,3)] = -.25*coshes[7] + .25*coshes[3] - .25*coshes[5]+ .25*coshes[8]
	mm[(2,3)] = .25*coshes[7] - .25*coshes[3] + .25*coshes[5] + .25*coshes[8]
	mm[(1,2,3)] = -.5*coshes[2] + .5*coshes[7]
	mm[(0,)] = -.25*(coshes[7] + coshes[3] + coshes[5] + coshes[1])

	if !mathematica
		diffs[(1,2)] = my[(1,2)]-three_params[(1,2)]
		diffs[(1,3)] = my[(1,3)]-three_params[(1,3)]
		diffs[(2,3)] = my[(2,3)]-three_params[(2,3)]
		diffs[(1,2,3)] = my[(1,2,3)]-three_params[(1,2,3)]
	else
		diffs[(1,2)] = mm[(1,2)]-three_params[(1,2)]
		diffs[(1,3)] = mm[(1,3)]-three_params[(1,3)]
		diffs[(2,3)] = mm[(2,3)]-three_params[(2,3)]
		diffs[(1,2,3)] = mm[(1,2,3)]-three_params[(1,2,3)]
	end

	println("Diffs")
	if mathematica 
		println("Zp - Zq ", mm[(0,)])
	end
	for k in keys(diffs)
		println(k, " : ", diffs[k])
	end
end
