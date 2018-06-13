using GraphicalModelLearning
using ForwardDiff
using JuMP
using Compat 
using MathProgBase # for solver type
using Ipopt
#using Plots
#using PlotRecipes

# 3 body interaction example
params_p = Dict{Tuple, Float64}()
params_q = Dict{Tuple, Float64}()

dim = 3
order_p = 3
tree_q = true
homogenous = false
# all constant external field
h = 0.0
h_i = fill(h, dim)
# h_i = [.1 .2 .3]

step = 0.001

for i in 1:dim
    params_p[(i,)] = h_i[i]
    #params_q[(i,)] = params_p[(i,)]
end

if homogenous == true
    rn = rand(1)[1]
    params_p[(1,2)] = rn
    params_p[(1,3)] = rn
    params_p[(2,3)] = rn
else
    params_p[(1,2)] = .8  #rand(1)[1]
    params_p[(1,3)] = .2 #rand(1)[1]
    params_p[(2,3)] = 0.0 #rand(1)[1]
end
#params_p[(1,2,3)] = 0
params_p[(1,2,3)] = 0#rand(1)[1]


# init q
params_q[(4,)] =  rand(1)[1]
params_q[(1,4)]= 2 #rand(1)[1]
params_q[(2,4)]= rand(1)[1]
params_q[(3,4)] = rand(1)[1]
#params_q[(1,2)] = rand(1)[1]
#params_q[(1,3)] = rand(1)[1]
#params_q[(2,3)] = rand(1)[1]
println("*** INITIALIZATION ***")
println()
println("params p")
for i in keys(params_p)
    if length(i)==1 || i[1] < i[2]
        if params_p[i] != 0.0
            println(i, ": ", params_p[i])
        end
    end
end
println()
println("params q")
for i in keys(params_q)
    if length(i)==1 || i[1] < i[2]
        println(i, ": ", params_q[i])
    end
end


model = FactorGraph(3, 3, :spin, params_p)
num_samp = 100000
num_conf = 2^dim
samples = sample(model, num_samp)


mean = [sum(samples[k,1]/num_samp*samples[k,1+i] for k=1:num_conf) for i=1:dim]
println()
println("marginals: ", mean)
println()
println("correlations")
corr = [[sum(samples[k,1]/num_samp*samples[k,1+i]*samples[k,1+j] for k=1:num_conf) for i=1:dim] for j=1:dim]
for i = 1:size(corr)[1]
    println([j for j in corr[i,:]])
end
println()
println("*** OPTIMIZATION ***")
xents = zeros(0)
tol = .0000001
model = FactorGraph(3, 3, :spin, params_p)
num_samp = 100000
num_conf = 2^dim
samples = sample(model, num_samp)
#println(size(samples))
samples_up = hcat(samples, fill(1, num_conf))
samples_down = hcat(samples, fill(-1, num_conf))
q_samples = vcat(samples_up, samples_down)  
#println("q_samples", size(q_samples))
#println(samples_down)
# H_p is a constant in the KL divergence
# Calculate E_p log q (min KL => max this)

pstat = Dict{Tuple, Array{Float64,1}}()
qstat_up = Dict{Tuple, Array{Float64,1}}()
qstat_down = Dict{Tuple, Array{Float64,1}}()
qstat = Dict{Tuple, Array{Float64,1}}()
q_perms = keys(params_q)

    
for theta_i in keys(params_p)
    pstat[theta_i] = [prod(samples[k, 1 + theta_i[i]] for i=1:length(theta_i)) for k=1:num_conf]
end
# entropy p
p_partition = log(sum(exp(sum(pstat[theta_i][k]*params_p[theta_i] for theta_i in keys(params_p))) for k=1:num_conf))

ent_p =  sum((samples[k,1]/num_samp)*(sum(pstat[theta_i][k]*params_p[theta_i] for theta_i in keys(params_p))) for k=1:num_conf) - p_partition


while size(xents)[1] <= 1 || abs(xents[end] - xents[end-1]) > tol 
    
    # q_permutations = all parameter coupling tuples
    for theta_i in q_perms
        a = [samples_up[:,1+theta_i[jj]] for jj in 1:length(theta_i)]
        #println(a[1])
        #if length(theta_i) > 1
        #    println(a[2])
        #end
        #println(params_q[theta_i])
        qstat_up[theta_i] = [prod(samples_up[k, 1 + theta_i[i]] for i=1:length(theta_i)) for k=1:num_conf]
        qstat_down[theta_i] = [prod(samples_down[k, 1 + theta_i[i]] for i=1:length(theta_i)) for k=1:num_conf]
    end

    qpart = sum(exp(sum(qstat_up[theta_i][k]*params_q[theta_i] for theta_i in keys(params_q))) for k=1:num_conf) + sum(exp(sum(qstat_down[theta_i][k]*params_q[theta_i] for theta_i in keys(params_q))) for k=1:num_conf)
    q_partition = log(qpart)
    

    # JOINT PLUS SB PROBABILITY
    # probability of sample under q
    q_joint_plus = [exp(sum(qstat_up[theta_i][k]*params_q[theta_i] for theta_i in q_perms))/exp(q_partition) for k=1:num_conf]
    q_joint_minus = [exp(sum(qstat_down[theta_i][k]*params_q[theta_i] for theta_i in q_perms))/exp(q_partition) for k=1:num_conf]
    q_data = [q_joint_plus[k] + q_joint_minus[k] for k=1:num_conf]
    
    # cross entropy q under p
    xent = sum((samples[k,1]/num_samp)*log(q_data[k]) for k=1:num_conf)
    
    append!(xents, ent_p-xent)
    
    q_cond_plus = [q_joint_plus[k] ./ q_data[k] for k=1:num_conf]
    q_cond_minus = [q_joint_minus[k] ./ q_data[k] for k=1:num_conf]
    
    grads = Dict{Tuple, Float64}()
    for theta_i in q_perms
        if length(theta_i) == 1 
            if theta_i[1] != 4
                grads[theta_i] = 0
            else
                grads[theta_i] = sum([(samples[k,1]/num_samp - q_data[k])*(q_cond_plus[k] - q_cond_minus[k]) for k=1:num_conf])
            end
        else
            for i=1:length(theta_i)
                if theta_i[i] != 4
                    # Ep - Eq = (p_data - q_data)*sigma_i*[1*q(+1|data) + -1*q(-1|data)]
                    grads[theta_i] = sum([(samples[k,1]/num_samp - q_data[k])*samples[k, 1+theta_i[i]]*(q_cond_plus[k]-q_cond_minus[k]) for k=1:num_conf]) 
                end
            end
        end
    end     
    for key in keys(grads)
      #print(key, grads[key])
      params_q[key] = params_q[key] + step * grads[key]
    end
    
    if (size(xents)[1] <= 1 || abs(xents[end] - xents[end-1]) > tol)== false
        println()
        #println(ent_p)
        #println("correlations")
        #corr = [[sum(q_joint_plus[k]*samples_up[k,1+i]*samples_up[k,1+j]    +q_joint_minus*samples_down[k,1+i]*samples_down[k,1+j] for k=1:num_conf) for i=1:(dim)] for j=1:(dim)]
        #show(corr)
    end
end

println("final kl ", xents[end])
println()

println("iterations ", length(xents))
for k in keys(params_q)
    if length(k) == 1 || k[1] < k[2]
        println(k, ": ", params_q[k])
    end
end

@compat abstract type GMLMethod end

type NLP <: GMLMethod
    solver::MathProgBase.AbstractMathProgSolver
end
# default values
NLP() = NLP(IpoptSolver())

m = Model(solver = IpoptSolver()) #NLP().solver)

#JuMP.register(m, :calc_xent, 1, calc_xent, autodiff=true)
#JuMP.register(m, :calc_xent_array, 1, calc_xent_array, autodiff=true)

n_keys = 0 #size([k for k in keys(params_q)])
index_map = Dict{Int64, Tuple}()
#param_init = Array{Float64,1}()
for k in keys(params_q)
    add = true
    for v in values(index_map)  
        if reverse(k) == v
            add = false
        end
    end
    if add
        n_keys = n_keys + 1
        index_map[n_keys] = k
        #param_init[n_keys] = params_q[k]
    end
end
#println("index map", index_map)
    
#@variable(m, params[1:n_keys])

@variable(m, params[i=1:n_keys])
for i in 1:n_keys
    setvalue(params[i], params_q[index_map[i]])
    #println(index_map[i], " init : ", getvalue(params[i]))
end
#println(params)

#inter = key of parameter, stat = value at sample
# objective = E_p 

@NLobjective(m, Max, sum((samples[k,1]/num_samp)*
            (log(exp(sum(qstat_up[index_map[i]][k]*params[i] for i = 1:n_keys))
            + exp(sum(qstat_down[index_map[j]][k]*params[j] for j = 1:n_keys)))) 
            -log(sum(exp(sum(qstat_up[index_map[l]][kk]*params[l] 
                + qstat_down[index_map[l]][kk]*params[l] for l = 1:n_keys)) for kk=1:num_conf)) for k=1:num_conf))

status = solve(m)
optimized = deepcopy(getvalue(params))



kl = ent_p - getobjectivevalue(m)


println("optimal params")
for key in keys(optimized)
    println(index_map[key]," :", optimized[key])
end
println()
println("Ep log p: ", ent_p)
println("Ep log q: ", getobjectivevalue(m))
println("kl: ", kl)

@assert status == :Optimal