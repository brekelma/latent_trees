using JuMP
using MathProgBase # for solver type
using Ipopt
using ForwardDiff
using Compat
using GraphicalModelLearning

type hRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
    hidden::Integer
    entropy_regularization::Real
end
# default values
hRISE() = hRISE(0.4, true, 2, 1, 0.0)

type hRPLE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
    hidden::Integer
end
# default values
hRPLE() = hRPLE(0.8, true, 2, 1)

type hRPLEall <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
    hidden::Integer
end
# default values
hRPLEall() = hRPLEall(0.8, true, 2, 1)


type hRISEAll <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
    hidden::Integer
    entropy_regularization::Real
end
# default values
hRISEAll() = hRISEAll(0.8, true, 2, 1, 0.0)

@compat abstract type GMLMethod end

type NLP <: GMLMethod
    solver::MathProgBase.AbstractMathProgSolver
end
# default values
NLP() = NLP(IpoptSolver(mu_init = .000001, tol = 10.0^-6, print_level=0))


permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partial_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    if order == 0
        return [tuple(partial_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partial_perm) > 0
                if partial_perm[end] >= item
                    continue
                end
            end
            perm = permutations(vcat(partial_perm, item), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end

function data_info{T <: Real}(samples::Array{T,2})
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 1
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end


function learn{T <: Real}(samples::Array{T,2}, formulation::hRPLE, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order
    hidden = formulation.hidden + num_spins 
    zero_init = false


    if formulation.hidden > 1
        error("Marginal hRPLE not implemented for more than 1 hidden node.")
    end
    reconstruction_list = Dict{Tuple,Array{Real, 1}}()
    hidden_stat = Dict{Tuple,Array{Real,1}}()
    index_maps = Dict{Int64, Dict{Tuple, Int64}}()
    # find hidden stats (for single hidden)
    current_spin = hidden
    index_maps[current_spin] = Dict{Tuple, Int64}()
    for p = 1:inter_order
        nodal_keys = Array{Tuple{},1}()
        neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
        if p == 1
            nodal_keys = [(current_spin,)]
        else
            perm = permutations(neighbours, p - 1)
            if length(perm) > 0
                nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
            end
        end

        for index = 1:length(nodal_keys)
            # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
            key = tuple(sort([i for i in nodal_keys[index]])...)
            hidden_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
            index_maps[current_spin][key] = length(hidden_stat)
        end
    end
    #total_hidden = copy(n_keys)

    rand_init = Dict{Tuple, Real}()
    # rand_init[(4,)] = -.268
    # rand_init[(3,)] = -.201
    # rand_init[(1,4)] = -.492
    # rand_init[(2,4)] = .524
    # rand_init[(2,)] = -.088
    # rand_init[(3,4)] = -.528
    # rand_init[(1,)] = -.243
    models = Array{Dict{Tuple, Real}, 1}()
    
    reconstruction = Dict{Tuple,Real}()
    for current_spin = 1:hidden
        index_maps = Dict{Int64, Dict{Tuple, Int64}}()
        # all optimizations include hidden parameters
        index_maps[current_spin] = copy(index_maps[hidden])

        append!(models, [Dict{Tuple, Real}()])
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        nodal_stat_minus = Dict{Tuple,Array{Real,1}}()
        
        for p = 1:inter_order
            nodal_keys = Array{Tuple{},1}()
            neighbours = [i for i=1:hidden if i!=current_spin]
            if p == 1
                nodal_keys = [(current_spin,)]
            else
                perm = permutations(neighbours, p - 1)
                if length(perm) > 0
                    nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                end
            end

            for index = 1:length(nodal_keys)
                key = tuple(sort([i for i in nodal_keys[index]])...)
                # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-

                nodal_stat[key] =  [ prod( i != hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                nodal_stat_minus[key] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
                #nodal_stat[nodal_keys[index]] =  [ prod( i != hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                #nodal_stat_minus[nodal_keys[index]] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
                if !haskey(index_maps[current_spin], key)
                    index_maps[current_spin][key] = length(nodal_stat) + (current_spin != hidden ? length(index_maps[hidden]) : 0 )
                end
            end
        end
        println("Params ", index_maps[current_spin])
        m = Model(solver = method.solver)
        @variable(m, x[1:length(index_maps[current_spin])])
        @variable(m, z[1:length(index_maps[current_spin])])

        #@variable(m, x[union(keys(nodal_stat), keys(hidden_stat))])
        #&variable(m, z[union(keys(nodal_stat), keys(hidden_stat))])
        for i in union(keys(nodal_stat), keys(hidden_stat))
            # need sort tuple 
            #@variable(m, x[union([tuple(sort([i for i in k])...) for k in keys(nodal_stat)], [tuple(sort([i for i in k])...) for k in keys(hidden_stat)])])
            if !haskey(rand_init, i)
                rand_init[i] = zero_init ? 0.0 : rand()[1]*(rand()[1] > 0.5 ? 1 : -1)
            end
            println("initializing", i, " to ", rand_init[i])
            setvalue(x[index_maps[current_spin][i]], rand_init[i])    
        end
        
        # only for current_spin in inter
        # sinh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat) if hidden in inter) 
        # cosh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) 

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[inter]*stat[k]*
        #         (hidden in inter ?  
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) - exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))/
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) + exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))
        #         : 1)
        #         for (inter, stat) = nodal_stat)    
        #         -log(exp(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) + exp(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        # ) 
        if current_spin == hidden
            # @NLobjective(m, Min, -sum((samples[k,1]/num_samples)*(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)*
            #     tanh(sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat)) 
            #     - log(2*cosh(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat)))) for k=1:num_conf)
            #     +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
            @NLobjective(m, Min, -sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter]]*stat[k] for (inter, stat) = nodal_stat)*
                tanh(sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)) 
                - log(2*cosh(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = nodal_stat)))) for k=1:num_conf)
                +lambda*sum(z[index_maps[current_spin][inter7]] for inter7 = keys(nodal_stat) if length(inter7)>1))
        else
            @NLobjective(m, Min,
            -sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
                + sum(x[index_maps[current_spin][inter4]]*stat4[k]*
                    tanh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
                -exp(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter1]]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)))
                -exp(-sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter5]]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[index_maps[current_spin][inter6]]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
                 for k=1:num_conf) +lambda*sum(z[index_maps[current_spin][inter7]] for inter7 = keys(nodal_stat) if length(inter7)>1))

            # @NLobjective(m, Min,
            # -sum((samples[k,1]/num_samples)*(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
            #     + sum(x[inter4]*stat4[k]*
            #         tanh(sum(x[i]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
            #     -exp(sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter1]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat)))
            #     -exp(-sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter5]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[inter6]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
            #      for k=1:num_conf) +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))

        end

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[int]*nodal_stat[int][k]*
        #         (hidden in int ?  
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) - exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))/
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) + exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))
        #         : 1)
        #         for int in keys(nodal_stat))    
        #         -log(exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat))) + exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat)))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[i] for i = keys(nodal_stat) if length(i)>1)
        # ) 


        for inter in union(keys(nodal_stat), keys(hidden_stat))
           @constraint(m, z[index_maps[current_spin][inter]] >=  x[index_maps[current_spin][inter]]) #z_plus
           @constraint(m, z[index_maps[current_spin][inter]] >=  -x[index_maps[current_spin][inter]]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        println("hRPLE Objective spin ", current_spin, " : ", getobjectivevalue(m))

        nodal_reconstruction = Dict{Tuple, Float64}()
        xx = getvalue(x)
        for k in union(keys(nodal_stat), keys(hidden_stat))
            nodal_reconstruction[k] = xx[index_maps[current_spin][k]]
        end

        for inter = union(keys(nodal_stat), keys(hidden_stat))
            key= tuple(sort([i for i in inter])...)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
            if !haskey(reconstruction_list, key)
                reconstruction_list[key]= []
            end
            append!(reconstruction_list[key], deepcopy(nodal_reconstruction[inter]))
            println(inter, ": ", reconstruction[inter])
            models[current_spin][inter] = deepcopy(nodal_reconstruction[inter])
        end

        d = JuMP.NLPEvaluator(m)
        MathProgBase.initialize(d, [:Grad,:Jac,:Hess])
        #objval = MathProgBase.eval_f(d, nodal_reconstruction)
        n_params = length(union(keys(nodal_stat), keys(hidden_stat)))

        H = zeros(n_params^2) #zeros(n_params^2)-1
        #H = zeros(n_params, n_params)
        evalues = zeros(n_params)
        
        println("Structure ", ", num params ", n_params, "size union ", size(union(keys(nodal_stat), keys(hidden_stat))), " nodal: ", length(nodal_reconstruction))
        
        keys_map = collect(keys(index_maps[current_spin]))
        keys_map = keys_map[sortperm([index_maps[current_spin][k] for k in keys_map])]
        println(keys_map)
        println("indices ", [index_maps[current_spin][k] for k in keys_map])
        
        my_hess = zeros(n_params, n_params)
        for i in union(keys(nodal_stat), keys(hidden_stat))
            evalues[index_maps[current_spin][i]] = nodal_reconstruction[i]#+rand()[1]
        end
        structure = MathProgBase.hesslag_structure(d)
        MathProgBase.eval_hesslag(d, H, evalues, 1.0, zeros(n_params))
        println("Hessian values ", H)
        for i = 1:length(structure[1])
            # getting correct tuples, now what?
            println("accessing ", structure[1][i], " ", structure[2][i], " : ", H[i])
            ind1 = index_maps[current_spin][keys_map[structure[1][i]]]
            ind2 = index_maps[current_spin][keys_map[structure[2][i]]]
            my_hess[ind1, ind2] = H[i]
            my_hess[ind2, ind1] = H[i]
        end
        println("Hessian for optimization of node ", current_spin, " : ")
        pprint2d(my_hess, rounding = 10)
        #println(reshape(H, n_params, n_params))
        #hessian[current_spin] = reshape(H, n_params, n_params)
        #println(eig(reshape(H, n_params, n_params)))
        #ForwardDiff
        f(x::Vector) = sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf);
        if current_spin == hidden
            f(x::Vector) = -sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter]]*stat[k] for (inter, stat) = nodal_stat)*
                tanh(sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)) 
                - log(2*cosh(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = nodal_stat)))) for k=1:num_conf)
            println("Hidden hessian")
        else
            println("Fwd diff hessian")
        end
        a = ForwardDiff.hessian(f, xx)
        # random other values?
        pprint2d(a, rounding = 10)
        println()
        println("Eigenvalues (jump)")
        println(eig(my_hess)[1])
        println("Eigenvalues (fwd diff))")
        println(eig(a)[1])
    end


    if formulation.symmetrization
        # reconstruction_list = Dict{Tuple,Vector{Real}}()
        # for (k,v) in reconstruction
        #     key = tuple(sort([i for i in k])...)
        #     if !haskey(reconstruction_list, key)
        #         reconstruction_list[key] = Vector{Real}()
        #     end
        #     push!(reconstruction_list[key], v)
        # end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            reconstruction[k] = mean(v)
        end
    end


    return models
    #return FactorGraph(inter_order, hidden, :spin, reconstruction) 
end

function calc_hessian{T <: Real}(samples::Array{T,2}, params::Dict{Tuple, Float64}, formulation::hRISE, hidden_stat::Dict{Tuple,Array{Real,1}}, nodal_stat::Dict{Tuple,Array{Real,1}})
    index_map = Dict{Tuple, Int64}()
    all_keys = Array{Tuple, 1}()
    n_params = 0
    x = params
    ns = sum(samples[:,1])
    for k in union(keys(hidden_stat), keys(nodal_stat))
        index_map[k] = length(index_map)+1
        append!(all_keys, [k])
        n_params = maximum([n_params ; [kk for kk in k]])
    end
    hidden = [tup[1] for tup in keys(hidden_stat) if length(tup)==1][1]
    hessian = zeros(n_params, n_params)
    println("Num params ", n_params)

    if length(hidden_stat) == length(nodal_stat)

        println("working on node ", hidden)
        for i = 1:length(index_map)
            for j = i:length(index_map)
                ki = all_keys[i]
                kj = all_keys[j]
                i_spins = (length(ki)==1 && ki[1] == hidden) ? 1 : prod(v for v in ki if v != hidden)
                j_spins = (length(kj)==1 && kj[1] == hidden)? 1 : prod(v for v in kj if v != hidden)
                dij = sum(samples[k,1]/ns*(i_spins*j_spins*(2*sinh(sum(x[ii]*s[k] for (ii, s)=hidden_stat)))^2/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))^3 - 1/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))) for k = 1:size(samples)[1])
                hessian[i,j] = dij
                hessian[j,i] = dij
                println("Keys ", ki, " ", kj, ": Hessian : ", dij)
            end
        end
    else
        node = [tup[1] for tup in keys(nodal_stat) if length(tup)==1][1]
        println("working on node ", node)
        for i = 1:length(index_map)
            for j = i:length(index_map)
                ki = all_keys[i]
                kj = all_keys[j]
                i_spins = (length(ki)==1 && ki[1] == hidden) ? 1 : prod(v for v in ki if v != hidden)
                j_spins = (length(kj)==1 && kj[1] == hidden)? 1 : prod(v for v in kj if v != hidden)
                if hidden in ki
                    if hidden in kj
                        dij = sum(samples[k,1]/ns*i_spins*j_spins*exp(-sum(x[ii]*s[k] for (ii, s)=hidden_stat if !(hidden in ii)))*
                            (-2*sinh(sum(x[ii]*s[k] for (ii, s)=hidden_stat if !(node in ii)))*sinh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))^2+2*cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat if !(node in ii)))*sinh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))^2/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))^3 ) for k = 1:size(samples)[1])
                    elseif node in kj
                        dij = 1
                    else

                    end
                elseif node in ki
                    if node in kj

                    elseif hidden

                    end
                else

                end    
                dij = sum(samples[k,1]/ns*(i_spins*j_spins*(2*sinh(sum(x[ii]*s[k] for (ii, s)=hidden_stat)))^2/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))^3 - 1/cosh(sum(x[ii]*s[k] for (ii, s)=hidden_stat))) for k = 1:size(samples)[1])
                

                hessian[i,j] = dij
                hessian[j,i] = dij
                println("Keys ", ki, " ", kj, ": Hessian : ", dij)
            end
        end

    end

    return hessian
end


function learn{T <: Real}(samples::Array{T,2}, formulation::hRISE, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    ent_reg = formulation.entropy_regularization
    inter_order = formulation.interaction_order
    hidden = formulation.hidden + num_spins 
    zero_init = false
    if formulation.hidden > 1
        error("Marginal hRISE not implemented for more than 1 hidden node.")
    end
    reconstruction_list = Dict{Tuple,Array{Real, 1}}()
    hidden_model = Dict{Tuple, Real}()
    hidden_stat = Dict{Tuple,Array{Real,1}}()

    index_maps = Dict{Int64, Dict{Tuple, Int64}}()
    # find hidden stats (for single hidden)
    current_spin = hidden
    index_maps[current_spin] = Dict{Tuple, Int64}()
    for p = 1:inter_order
        nodal_keys = Array{Tuple{},1}()
        neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
        if p == 1
            nodal_keys = [(current_spin,)]
        else
            perm = permutations(neighbours, p - 1)
            if length(perm) > 0
                nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
            end
        end

        for index = 1:length(nodal_keys)
            # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
            key = tuple(sort([i for i in nodal_keys[index]])...)
            hidden_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
            index_maps[current_spin][key] = length(hidden_stat)
        end
    end
    rand_init = Dict{Tuple, Real}()
    reconstruction = Dict{Tuple,Real}()
    # rand_init[(4,)] = -.268
    # rand_init[(3,)] = -.201
    # rand_init[(1,4)] = -.492
    # rand_init[(2,4)] = .524
    # rand_init[(2,)] = -.088
    # rand_init[(3,4)] = -.528
    # rand_init[(1,)] = -.243

    models = Array{Dict{Tuple, Real}, 1}()

    for current_spin = 1:(num_spins+formulation.hidden)
        index_maps[current_spin] = copy(index_maps[hidden])

        append!(models, [Dict{Tuple, Real}()])
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        nodal_stat_minus = Dict{Tuple,Array{Real,1}}()
        for p = 1:inter_order
            nodal_keys = Array{Tuple{},1}()
            neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
            if p == 1
                nodal_keys = [(current_spin,)]
            else
                perm = permutations(neighbours, p - 1)
                if length(perm) > 0
                    nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                end
            end

            for index = 1:length(nodal_keys)
                # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
                key = tuple(sort([i for i in nodal_keys[index]])...)
                nodal_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                nodal_stat_minus[key] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
                
                if !haskey(index_maps[current_spin], key)
                    index_maps[current_spin][key] = length(nodal_stat) + (current_spin != hidden ? length(index_maps[hidden]) : 0 )
                end
            end
        end

        m = Model(solver = method.solver)

        @variable(m, x[1:length(index_maps[current_spin])])
        @variable(m, z[1:length(index_maps[current_spin])])
        
        println()
        println("Optimizing node ", current_spin, " sparsity ", lambda, " entropy ", ent_reg)

        for i in union(keys(nodal_stat), keys(hidden_stat))
            # need sort tuple 
            #@variable(m, x[union([tuple(sort([i for i in k])...) for k in keys(nodal_stat)], [tuple(sort([i for i in k])...) for k in keys(hidden_stat)])])
            if !haskey(rand_init, i)
                rand_init[i] = zero_init ? 0.0 : rand()[1]*(rand()[1] > 0.5 ? 1 : -1)
            end
            println("initializing", i, " to ", rand_init[i])
            #setvalue(x[i], rand_init[i])
            setvalue(x[index_maps[current_spin][i]], rand_init[i])
        
        end
        
        # only for current_spin in inter
        # sinh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat) if hidden in inter) 
        # cosh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) 

        if current_spin == hidden
            @NLobjective(m, Min,
                sum((samples[k,1]/num_samples)/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat)) for k=1:num_conf)
                +ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter]]*stat[k] for (inter, stat) = hidden_stat)*
                tanh(sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = hidden_stat)) 
                - log(2*cosh(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = hidden_stat)))) for k=1:num_conf))
                #+lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
        else
            # @NLobjective(m, Min,
            #     sum((samples[k,1]/num_samples)*(cosh(sum(x[i]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[i]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[ii]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf))
        
            @NLobjective(m, Min,
                sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)
                    + ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
                    + sum(x[index_maps[current_spin][inter4]]*stat4[k]*
                         tanh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
                     -exp(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter1]]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)))
                     -exp(-sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter5]]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[index_maps[current_spin][inter6]]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
                      for k=1:num_conf))
                    # +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
            # @NLobjective(m, Min,
            #     sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)
            #     +ent_reg*sum((samples[k,1]/num_samples)*(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
            #     + sum(x[inter4]*stat4[k]*
            #          tanh(sum(x[i]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
            #      -exp(sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter1]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat)))
            #      -exp(-sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter5]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[inter6]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
            #       for k=1:num_conf))
                # +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
        end
        #         sum(x[inter]*stats[k]*
        #         (hidden in inter ?  
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) - exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))/
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) + exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))
        #         : 1)
        #         for (inter, stats) = nodal_stat)    
        #         -log(exp(sum(x[inter1]*stat[k] for (inter1, stat) = nodal_stat)) + exp(sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[inter3] for inter3 = keys(nodal_stat) if length(inter3)>1)
        # ) 

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[int]*nodal_stat[int][k]*
        #         (hidden in int ?  
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) - exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))/
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) + exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))
        #         : 1)
        #         for int in keys(nodal_stat))    
        #         -log(exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat))) + exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat)))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[i] for i = keys(nodal_stat) if length(i)>1)
        # ) 


        # for inter in union(keys(nodal_stat), keys(hidden_stat))
        #     @constraint(m, z[inter] >=  x[inter]) #z_plus
        #     @constraint(m, z[inter] >= -x[inter]) #z_minus
        # end

        println()
        status = solve(m)
        @assert status == :Optimal
        
        println("hRISE Objective spin ", current_spin, " : ", getobjectivevalue(m))
        

        
        xx = getvalue(x)
        ent = -sum((samples[k,1]/num_samples)*(sum(xx[index_maps[current_spin][inter]]*stat[k] for (inter, stat) = hidden_stat)*
                tanh(sum(xx[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = hidden_stat)) 
                - log(2*cosh(sum(xx[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = hidden_stat)))) for k=1:num_conf)

        println("RISE value :", getobjectivevalue(m) + ent_reg*ent)
        println("conditional entropy ", ent)

        nodal_reconstruction = Dict{Tuple, Float64}()
        
        for k in union(keys(nodal_stat), keys(hidden_stat))
            nodal_reconstruction[k] = xx[index_maps[current_spin][k]]
        end        

        if current_spin != hidden
            for inter in union(keys(hidden_stat), keys(nodal_stat))
                key= tuple(sort([i for i in inter])...)
                reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
                if !haskey(reconstruction_list, key)
                    reconstruction_list[key]= []
                end
                append!(reconstruction_list[key], deepcopy(nodal_reconstruction[inter]))
                println(inter, ": ", reconstruction[inter])
                models[current_spin][inter] = deepcopy(nodal_reconstruction[inter])
            end
        else
            for inter in union(keys(hidden_stat), keys(nodal_stat))
                hidden_model[inter] = deepcopy(nodal_reconstruction[inter])
                println(inter, ": ", hidden_model[inter])
                models[current_spin][inter] = deepcopy(nodal_reconstruction[inter])
            end
        end


        d = JuMP.NLPEvaluator(m)
        MathProgBase.initialize(d, [:Grad,:Jac,:Hess])
        #objval = MathProgBase.eval_f(d, nodal_reconstruction)
        n_params = length(union(keys(nodal_stat), keys(hidden_stat)))

        H = zeros(n_params^2) #zeros(n_params^2)-1
        #H = zeros(n_params, n_params)
        evalues = zeros(n_params)
        
        #println("Structure ", ", num params ", n_params, "size union ", size(union(keys(nodal_stat), keys(hidden_stat))), " nodal: ", length(nodal_reconstruction))
        
        keys_map = collect(keys(index_maps[current_spin]))
        keys_map = keys_map[sortperm([index_maps[current_spin][k] for k in keys_map])]
        #println(keys_map)
        #println("indices ", [index_maps[current_spin][k] for k in keys_map])
        
        my_hess = zeros(n_params, n_params)
        for i in union(keys(nodal_stat), keys(hidden_stat))
            evalues[index_maps[current_spin][i]] = nodal_reconstruction[i]#+rand()[1]
        end
        structure = MathProgBase.hesslag_structure(d)
        MathProgBase.eval_hesslag(d, H, evalues, 1.0, zeros(n_params))
        println("Hessian values ", H)
        for i = 1:length(structure[1])
            # getting correct tuples, now what?
            #println("accessing ", structure[1][i], " ", structure[2][i], " : ", H[i])
            ind1 = index_maps[current_spin][keys_map[structure[1][i]]]
            ind2 = index_maps[current_spin][keys_map[structure[2][i]]]
            my_hess[ind1, ind2] = H[i]
            my_hess[ind2, ind1] = H[i]
        end
        println("Hessian for optimization of node ", current_spin, " : ")
        pprint2d(my_hess, rounding = 10)
        #println(reshape(H, n_params, n_params))
        #hessian[current_spin] = reshape(H, n_params, n_params)
        #println(eig(reshape(H, n_params, n_params)))
        #ForwardDiff

        # f(x::Vector) =  sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))
        #     /cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))
        #     *exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)    #+lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))

        #+ sum(x[index_maps[current_spin][inter4]]*stat4[k]*(tanh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))) for (inter4,stat4) = nodal_stat if hidden in inter4)+ sum(x[index_maps[current_spin][inter4]]*stat4[k]*(tanh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))) for (inter4,stat4) = nodal_stat if hidden in inter4)

        ff(x::Vector) =  sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)

        if ent_reg > 0.0
            ff(x::Vector) =  sum((samples[k,1]/num_samples)*(cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_maps[current_spin][ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)
                    + ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
                    + sum(x[index_maps[current_spin][inter4]]*stat4[k]*
                         tanh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
                     -exp(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter1]]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)))
                     -exp(-sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_maps[current_spin][inter5]]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[index_maps[current_spin][inter6]]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
                      for k=1:num_conf)
        end
        if current_spin == hidden
            if abs(ent_reg) > 0.0
                ff(x::Vector) = sum((samples[k,1]/num_samples)/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat)) for k=1:num_conf)
                    +ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_maps[current_spin][inter]]*stat[k] for (inter, stat) = hidden_stat)*
                    tanh(sum(x[index_maps[current_spin][inter2]]*stat2[k] for (inter2, stat2) = hidden_stat)) 
                    - log(2*cosh(sum(x[index_maps[current_spin][inter3]]*stat3[k] for (inter3, stat3) = hidden_stat)))) for k=1:num_conf)
            else
                ff(x::Vector) = sum((samples[k,1]/num_samples)/cosh(sum(x[index_maps[current_spin][i]]*s[k] for (i,s) = hidden_stat)) for k=1:num_conf)
            end
            println("Hidden hessian")
        else
            println("Fwd diff hessian")
        end
        a = ForwardDiff.hessian(ff, xx)
        # random other values?
        pprint2d(a, rounding = 10)
        println()
        println("Eigenvalues (jump)")
        println(eig(my_hess)[1])
        println("Eigenvalues (fwd diff)")
        println(eig(a)[1])
        
        if current_spin == hidden
            println("***** HESSIAN ANALYTICALLY CALCULATED ******")
            pprint2d(calc_hessian(samples, nodal_reconstruction, formulation, hidden_stat, nodal_stat), rounding = 10)
        end
    end


    if formulation.symmetrization
        # reconstruction_list = Dict{Tuple,Vector{Real}}()
        # for (k,v) in reconstruction
        #     key = tuple(sort([i for i in k])...)
        #     if !haskey(reconstruction_list, key)
        #         reconstruction_list[key] = Vector{Real}()
        #     end
        #     push!(reconstruction_list[key], v)
        # end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            println("Reconstruction list : ", k, " : ", v)
            reconstruction[k] = mean(v)
        end
        #for (k,v) in hidden_model
        #end
    end


    return models
    #
    #return FactorGraph(inter_order, hidden, :spin, reconstruction), FactorGraph(inter_order, hidden, :spin, hidden_model) 
end


function learn{T <: Real}(samples::Array{T,2}, formulation::hRPLEall, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order
    hidden = formulation.hidden + num_spins 
    zero_init = false


    if formulation.hidden > 1
        error("Marginal hRPLE not implemented for more than 1 hidden node.")
    end
    reconstruction_list = Dict{Tuple,Array{Real, 1}}()
    hidden_stat = Dict{Tuple,Array{Real,1}}()
    # find hidden stats (for single hidden)
    current_spin = hidden
    for p = 1:inter_order
        nodal_keys = Array{Tuple{},1}()
        neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
        if p == 1
            nodal_keys = [(current_spin,)]
        else
            perm = permutations(neighbours, p - 1)
            if length(perm) > 0
                nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
            end
        end

        for index = 1:length(nodal_keys)
            # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
            key = tuple(sort([i for i in nodal_keys[index]])...)
            hidden_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
        end
    end
    
    rand_init = Dict{Tuple, Real}()
    rand_init[(4,)] = -.268
    rand_init[(3,)] = -.201
    rand_init[(1,4)] = -.492
    rand_init[(2,4)] = .524
    rand_init[(2,)] = -.088
    rand_init[(3,4)] = -.528
    rand_init[(1,)] = -.243
    models = Array{Dict{Tuple, Real}, 1}()
    
    reconstruction = Dict{Tuple,Real}()
    for current_spin = 1:hidden
        append!(models, [Dict{Tuple, Real}()])
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        nodal_stat_minus = Dict{Tuple,Array{Real,1}}()
        for p = 1:inter_order
            nodal_keys = Array{Tuple{},1}()
            neighbours = [i for i=1:hidden if i!=current_spin]
            if p == 1
                nodal_keys = [(current_spin,)]
            else
                perm = permutations(neighbours, p - 1)
                if length(perm) > 0
                    nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                end
            end

            for index = 1:length(nodal_keys)
                key = tuple(sort([i for i in nodal_keys[index]])...)
                # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
                
                nodal_stat[key] =  [ prod( i != hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                nodal_stat_minus[key] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
                #nodal_stat[nodal_keys[index]] =  [ prod( i != hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                #nodal_stat_minus[nodal_keys[index]] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
            end
        end

        m = Model(solver = method.solver)

        @variable(m, x[union(keys(nodal_stat), keys(hidden_stat))])
        #@variable(m, x[union([tuple(sort([i for i in k])...) for k in keys(nodal_stat)], [tuple(sort([i for i in k])...) for k in keys(hidden_stat)])])
        @variable(m, z[union(keys(nodal_stat), keys(hidden_stat))])
        for i in union(keys(nodal_stat), keys(hidden_stat))
            # need sort tuple 
            #@variable(m, x[union([tuple(sort([i for i in k])...) for k in keys(nodal_stat)], [tuple(sort([i for i in k])...) for k in keys(hidden_stat)])])
            if !haskey(rand_init, i)
                rand_init[i] = zero_init ? 0.0 : rand()[1]*(rand()[1] > 0.5 ? 1 : -1)
            end
            println("initializing", i, " to ", rand_init[i])
            setvalue(x[i], rand_init[i])    
        end
        
        # only for current_spin in inter
        # sinh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat) if hidden in inter) 
        # cosh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) 

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[inter]*stat[k]*
        #         (hidden in inter ?  
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) - exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))/
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) + exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))
        #         : 1)
        #         for (inter, stat) = nodal_stat)    
        #         -log(exp(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) + exp(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        # ) 
        if current_spin == hidden
            @NLobjective(m, Min, -sum((samples[k,1]/num_samples)*(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)*
                tanh(sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat)) 
                - log(2*cosh(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat)))) for k=1:num_conf)
                +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
        else
            @NLobjective(m, Min,
            -sum((samples[k,1]/num_samples)*(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
                + sum(x[inter4]*stat4[k]*
                    tanh(sum(x[i]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
                -exp(sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter1]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat)))
                -exp(-sum(x[i]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[i]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[inter5]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[inter6]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
                 for k=1:num_conf) +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))


            # summation of normalization constant across configs is wrong, esp without weighting 
            # @NLobjective(m, Min,
            # sum((samples[k,1]/num_samples)*(sum(x[inter3]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
            #     + sum(x[inter4]*stat4[k]*
            #         tanh(sum(x[i]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)) for k=1:num_conf)
            #     -log(exp(sum(x[inter1]*stat1[kk] for (inter1, stat1) = nodal_stat, kk = 1:num_conf)) + exp(-sum(x[inter2]*stat2[kk] for (inter2, stat2) = nodal_stat,kk = 1:num_conf))
            #     +exp(sum(x[inter5]*stat5[kk] for (inter5, stat5) = nodal_stat_minus, kk = 1:num_conf)) + exp(-sum(x[inter6]*stat6[kk] for (inter6, stat6) = nodal_stat_minus, kk = 1:num_conf)))
            #     +lambda*sum(z[inter7] for inter7 = keys(nodal_stat) if length(inter7)>1))
        end

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[int]*nodal_stat[int][k]*
        #         (hidden in int ?  
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) - exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))/
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) + exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))
        #         : 1)
        #         for int in keys(nodal_stat))    
        #         -log(exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat))) + exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat)))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[i] for i = keys(nodal_stat) if length(i)>1)
        # ) 


        for inter in union(keys(nodal_stat), keys(hidden_stat))
           @constraint(m, z[inter] >=  x[inter]) #z_plus
           @constraint(m, z[inter] >= -x[inter]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        println("hRPLE Objective spin ", current_spin, " : ", getobjectivevalue(m))

        nodal_reconstruction = getvalue(x)
        for inter = union(keys(nodal_stat), keys(hidden_stat))
            key= tuple(sort([i for i in inter])...)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
            if !haskey(reconstruction_list, key)
                reconstruction_list[key]= []
            end
            append!(reconstruction_list[key], deepcopy(nodal_reconstruction[inter]))
            println(inter, ": ", reconstruction[inter])
            models[current_spin][inter] = deepcopy(nodal_reconstruction[inter])
        end

        d = JuMP.NLPEvaluator(m)
        MathProgBase.initialize(d, [:Hess])
        n_params = length(union(keys(nodal_stat), keys(hidden_stat)))
        H = zeros(n_params^2)# n_params)
        #objval = MathProgBase.eval_f(d, nodal_reconstruction)
        hessian = MathProgBase.eval_hesslag(d, H, nodal_reconstruction, 0.0, Float64[])
        println("Hessian for optimization at node ", current_spin, " : ", typeof(hessian))
        println(hessian)

     
    end


    if formulation.symmetrization
        # reconstruction_list = Dict{Tuple,Vector{Real}}()
        # for (k,v) in reconstruction
        #     key = tuple(sort([i for i in k])...)
        #     if !haskey(reconstruction_list, key)
        #         reconstruction_list[key] = Vector{Real}()
        #     end
        #     push!(reconstruction_list[key], v)
        # end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            reconstruction[k] = mean(v)
        end
    end


    return models
    #return FactorGraph(inter_order, hidden, :spin, reconstruction) 
end


function learn{T <: Real}(samples::Array{T,2}, formulation::hRISEAll, method::NLP)
    num_conf, num_spins, num_samples = data_info(samples)
    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    ent_reg = formulation.entropy_regularization
    inter_order = formulation.interaction_order
    hidden = formulation.hidden + num_spins 
    zero_init = false
    if formulation.hidden > 1
        error("Marginal hRISE not implemented for more than 1 hidden node.")
    end
    reconstruction_list = Dict{Tuple,Array{Real, 1}}()
    hidden_model = Dict{Tuple, Real}()
    hidden_stat = Dict{Tuple,Array{Real,1}}()

    index_map= Dict{Tuple, Int64}()
    # find hidden stats (for single hidden)
    current_spin = hidden
    #index_map = Dict{Tuple, Int64}()
    for p = 1:inter_order
        nodal_keys = Array{Tuple{},1}()
        neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
        if p == 1
            nodal_keys = [(current_spin,)]
        else
            perm = permutations(neighbours, p - 1)
            if length(perm) > 0
                nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
            end
        end

        for index = 1:length(nodal_keys)
            # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
            key = tuple(sort([i for i in nodal_keys[index]])...)
            hidden_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
            index_map[key] = length(index_map)+1 #length(hidden_stat)
        end
    end
    rand_init = Dict{Tuple, Real}()
    reconstruction = Dict{Tuple,Real}()
    rand_init[(4,)] = -.268
    rand_init[(3,)] = -.201
    rand_init[(1,4)] = -.492
    rand_init[(2,4)] = .524
    rand_init[(2,)] = -.088
    rand_init[(3,4)] = -.528
    rand_init[(1,)] = -.243

    models = Array{Dict{Tuple, Real}, 1}()
    m = Model(solver = method.solver)
    prev_size = 1
    stats = Array{Dict{Tuple,Array{Real,1}}, 1}()
    stats_minus = Array{Dict{Tuple,Array{Real,1}}, 1}()
    for current_spin = 1:(num_spins+formulation.hidden)
        #index_maps[current_spin] = copy(index_maps[hidden])

        append!(models, [Dict{Tuple, Real}()])

        # different variables for different spins?
        nodal_stat = Dict{Tuple,Array{Real,1}}()
        nodal_stat_minus = Dict{Tuple,Array{Real,1}}()
        for p = 1:inter_order
            nodal_keys = Array{Tuple{},1}()
            neighbours = [i for i=1:(num_spins+formulation.hidden) if i!=current_spin]
            if p == 1
                nodal_keys = [(current_spin,)]
            else
                perm = permutations(neighbours, p - 1)
                if length(perm) > 0
                    nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                end
            end

            for index = 1:length(nodal_keys)
                # nodal stat for hiddens? note: cosh / sinh in objective takes care of +/-
                key = tuple(sort([i for i in nodal_keys[index]])...)
                nodal_stat[key] =  [ prod( i < hidden ? samples[k, 1 + i] : 1  for i=nodal_keys[index]) for k=1:num_conf ]
                nodal_stat_minus[key] =  [ prod( i != hidden ? samples[k, 1 + i] : -1  for i=nodal_keys[index]) for k=1:num_conf ]
                
                if !haskey(index_map, key)
                    index_map[key] = length(index_map)+1
                end
            end
        end
        append!(stats, [nodal_stat])
        append!(stats_minus, [nodal_stat_minus])
    end
    keys_map = collect(keys(index_map))
    keys_map = keys_map[sortperm([index_map[k] for k in keys_map])]
    new_vars = [index_map[k]for k in keys_map if index_map[k] >= prev_size]
    println("new vars ", new_vars)
    @variable(m, x[new_vars])
    @variable(m, z[new_vars])
    #@variable(m, x[prev_size:length(index_map)])
    #@variable(m, z[prev_size:length(index_map)])
    #prev_size = length(index_map)+1
        

        #@variable(m, x[1:length(index_maps[current_spin])])
        #@variable(m, z[1:length(index_maps[current_spin])])
        
        #println()
        #println("Optimizing node ", current_spin, " sparsity ", lambda, " entropy ", ent_reg)
    obj = Array{Any, 1}()
    for current_spin = 1:(num_spins+formulation.hidden)
        nodal_stat = stats[current_spin]
        nodal_stat_minus = stats_minus[current_spin]
        for i in union(keys(nodal_stat), keys(hidden_stat))
            # need sort tuple 
            #@variable(m, x[union([tuple(sort([i for i in k])...) for k in keys(nodal_stat)], [tuple(sort([i for i in k])...) for k in keys(hidden_stat)])])
            if !haskey(rand_init, i)
                rand_init[i] = zero_init ? 0.0 : rand()[1]*(rand()[1] > 0.5 ? 1 : -1)
            end
            println("initializing", i, " to ", rand_init[i])
            #setvalue(x[i], rand_init[i])
            setvalue(x[index_map[i]], rand_init[i])
        
        end
        
        # only for current_spin in inter
        # sinh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat) if hidden in inter) 
        # cosh(sum(x[inter]*stat[k] for (inter, stat) = nodal_stat)) 

        if current_spin == hidden
            append!(obj, [@NLexpression(m, current_spin,
                sum((samples[k,1]/num_samples)/cosh(sum(x[index_map[i]]*s[k] for (i,s) = hidden_stat)) for k=1:num_conf)
                +ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_map[inter]]*stat[k] for (inter, stat) = hidden_stat)*
                tanh(sum(x[index_map[inter2]]*stat2[k] for (inter2, stat2) = hidden_stat)) 
                - log(2*cosh(sum(x[index_map[inter3]]*stat3[k] for (inter3, stat3) = hidden_stat)))) for k=1:num_conf)
                +lambda*sum(z[index_map[inter7]] for inter7 = keys(nodal_stat) if length(inter7)>1))])
        else
            # @NLobjective(m, Min,
            #     sum((samples[k,1]/num_samples)*(cosh(sum(x[i]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[i]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[ii]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf))
        
            #@NLexpression(m, obj[current_spin],
            append!(obj, [@NLexpression(m, current_spin, 
                    sum((samples[k,1]/num_samples)*(cosh(sum(x[index_map[i]]*s[k] for (i,s) = hidden_stat if !(current_spin in i)))/cosh(sum(x[index_map[i]]*s[k] for (i,s) = hidden_stat))*exp(-sum(x[index_map[ii]]*ss[k] for (ii,ss) = nodal_stat if !(hidden in ii)))) for k =1:num_conf)
                    + ent_reg*sum((samples[k,1]/num_samples)*(sum(x[index_map[inter3]]*stat3[k] for (inter3, stat3) = nodal_stat if !(hidden in inter3)) 
                    + sum(x[index_map[inter4]]*stat4[k]*
                         tanh(sum(x[index_map[i]]*s[k] for (i, s)=hidden_stat)) for (inter4,stat4) = nodal_stat if hidden in inter4)
                     -exp(sum(x[index_map[i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_map[i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_map[inter1]]*stat1[k] for (inter1, stat1) = nodal_stat)) + exp(-sum(x[index_map[inter2]]*stat2[k] for (inter2, stat2) = nodal_stat)))
                     -exp(-sum(x[index_map[i]]*s[k] for (i, s)=hidden_stat))/(2*cosh(sum(x[index_map[i]]*s[k] for (i, s)=hidden_stat)))*log(exp(sum(x[index_map[inter5]]*stat5[k] for (inter5, stat5) = nodal_stat_minus)) + exp(-sum(x[index_map[inter6]]*stat6[k] for (inter6, stat6) = nodal_stat_minus))))
                      for k=1:num_conf)
                    +lambda*sum(z[index_map[inter7]] for inter7 = keys(nodal_stat) if length(inter7)>1)) ])
        end
        #         sum(x[inter]*stats[k]*
        #         (hidden in inter ?  
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) - exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))/
        #         (exp(sum(x[i]*s[k] for (i, s) = hidden_stat)) + exp(sum(x[i]*s[k] for (i, s) = hidden_stat)))
        #         : 1)
        #         for (inter, stats) = nodal_stat)    
        #         -log(exp(sum(x[inter1]*stat[k] for (inter1, stat) = nodal_stat)) + exp(sum(x[inter2]*stat2[k] for (inter2, stat2) = nodal_stat))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[inter3] for inter3 = keys(nodal_stat) if length(inter3)>1)
        # ) 

        # @NLobjective(m, Min,
        #     sum((samples[k,1]/num_samples)*(sum(x[int]*nodal_stat[int][k]*
        #         (hidden in int ?  
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) - exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))/
        #         (exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))) + exp(sum(x[inter]*hidden_stat[inter][k] for inter in keys(hidden_stat))))
        #         : 1)
        #         for int in keys(nodal_stat))    
        #         -log(exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat))) + exp(sum(x[intr]*nodal_stat[intr][k] for intr in keys(nodal_stat)))) )
        #         for k=1:num_conf) +
        #     lambda*sum(z[i] for i = keys(nodal_stat) if length(i)>1)
        # ) 


        # for inter in union(keys(nodal_stat), keys(hidden_stat))
        #     @constraint(m, z[inter] >=  x[inter]) #z_plus
        #     @constraint(m, z[inter] >= -x[inter]) #z_minus
        # end

        #println()
        #status = solve(m)
        #@assert status == :Optimal
        
        
    end
    @NLobjective(m, Min, obj[1] + obj[2] + obj[3] + obj[4])#sum(obj[i] for i = 1:(num_spins+formulation.hidden)))
    for iii in keys(index_map)
        inter = index_map[iii]
        @constraint(m, z[inter] >=  x[inter]) #z_plus
        @constraint(m, z[inter] >= -x[inter]) #z_minus
    end

    #@NLobjective(m, Min, sum(obj[i] for i = 1:(num_spins+formulation.hidden)))
    status = solve(m)
    @assert status == :Optimal 

    params_list = getvalue(x)
    nodal_reconstruction = Dict{Tuple, Float64}()
    keys_map = collect(keys(index_map))
    keys_map = keys_map[sortperm([index_map[k] for k in keys_map])]
    println("hRISE All Objective: ", getobjectivevalue(m))
    println("Params learned")
    for p = 1:length(params_list)
        nodal_reconstruction[keys_map[p]] = params_list[p] 
        println(keys_map[p], " : ", params_list[p])
    end

    # if formulation.symmetrization
    #     # reconstruction_list = Dict{Tuple,Vector{Real}}()
    #     # for (k,v) in reconstruction
    #     #     key = tuple(sort([i for i in k])...)
    #     #     if !haskey(reconstruction_list, key)
    #     #         reconstruction_list[key] = Vector{Real}()
    #     #     end
    #     #     push!(reconstruction_list[key], v)
    #     # end

    #     reconstruction = Dict{Tuple,Real}()
    #     for (k,v) in reconstruction_list
    #         println("Reconstruction list : ", k, " : ", v)
    #         reconstruction[k] = mean(v)
    #     end
    #     #for (k,v) in hidden_model
    #     #end
    # end

    return nodal_reconstruction
    #return models
    #
    #return FactorGraph(inter_order, hidden, :spin, reconstruction), FactorGraph(inter_order, hidden, :spin, hidden_model) 
end
