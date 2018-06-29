
% method = 'RG';
% method = 'NJ';
method = 'CLRG';
% method = 'CLNJ';
% method = 'regCLRG';
% method = 'regCLNJ';

hide = true; 
use_anima = true;
dim = 3;
hidden = 0;


a = tdfread("../example.csv", 'tab');
names = fieldnames(a);
b = getfield(a, names{1});
params = {};
couplings = getfield(a, names{2});
for i=1:size(b,1)
    param = [];
    c = strsplit(b(i,:), ",");
    for j=1:size(c,2)
        param = [param; str2num(c{j})];
    end
    params = [params, [param]];
    %params = [params, [param]];
    %append(tuples, [1])
end

n_samples = 1000000;

% construct more general
configs = dec2bin(uint8([0:2^(dim+hidden)-1]));
configs = (configs + 'a' - 145).*2-1;
conf = configs;

weights = [];
for k=1:size(configs, 1)
    evidence = 0;
    for i=1:length(params) % which coupling
        prd = 1;
        for node=1:length(params{i})
            prd = prd*configs(k,params{i}(node));
        end
        evidence = evidence + couplings(i)*prd;
    end
    weights = [weights; evidence];
end

if use_anima
    configs2=  zeros(size(configs));
    for i=1:size(configs,1)
        for j=1:size(configs,2)
            if configs(i,j) == 1
               configs2(i,j)= 2;
            else
               configs2(i,j) = 1;
            end
        end
    end
end

%configs = reshape(configs,[8,3])
%configs2 = reshape(configs2, [8,3])
weights = arrayfun(@exp, weights) ./ sum(arrayfun(@exp, weights));
y = randsample(length(weights), n_samples, true, weights);
samps = zeros(length(weights), 1+dim+hidden);
for i=1:length(weights)
    samps(i, 1) = sum(y == i);
    samps(i, 2:end) = configs(i,:);
end
%trainSamples = arrayfun(@(x) configs(x(i),:), y);
trainSamples_obs = zeros(n_samples, size(configs,2)-hidden);
trainSamples_all = zeros(n_samples, size(configs,2));
%prev = 1;
%bb = sum(n_samples);
for i = 1:n_samples
    if use_anima
        configs = configs2;
    end
    trainSamples_obs(i, :) = configs(y(i),1:dim);
    trainSamples_all(i, :) = configs(y(i),:);
end

if hide
    trainSamples = trainSamples_obs';
else
    trainSamples = trainSamples_all';
end
trainSamples_obs = trainSamples_obs';
trainSamples_all = trainSamples_all';

distance = distances(trainSamples);

switch method 
    case 'RG'
        [adjmatT,edge_distance] = RG(trainSamples, 0);
    case 'NJ'
        [adjmatT,edge_distance] = NJ(trainSamples, 0);
    case 'CLRG'
        if use_anima
            [adjmatT,edge_distance] = CLRG(trainSamples, 0); 
        else
            [adjmatT, edge_distance]= CLRG(distance, 1, n_samples);
        end
    case 'CLNJ'
        [adjmatT,edge_distance] = CLNJ(trainSamples, 0);
    case 'regCLRG'
        [adjmatT, edgeD, initNodePot, initEdgePot, ll, bic] = regCLRG_discrete(trainSamples, options);       
    case 'regCLNJ'
        [adjmatT, edgeD, initNodePot, initEdgePot, ll, bic] = regCLNJ_discrete(trainSamples, options);
        
end
drawLatentTree(adjmatT, 3)

num_nodes = size(adjmatT,1);
[~, root] = max(sum(adjmatT, 2));
msg_order = treeMsgOrder(adjmatT, root);

prob_bij = computeBnStats(trainSamples_all);
    
% previously only over obs
[node_potential,edge_potential]= marToPotBin(prob_bij, msg_order);
edge_pairs = msg_order(num_nodes:end,:);

if use_anima
    root_mar = sum(trainSamples(root, :)-1 > 0) / size(trainSamples,2);
else
    root_mar = sum(trainSamples(root, :)-1 >=0) / size(trainSamples,2);
end
root_mar = [1- root_mar, root_mar];
% need to use anima
ll_bin = logProbTreeBin(root_mar,edge_potential,edge_pairs,trainSamples);

%ll = size(trainSamples,2)*computeAvgLLBin(node_potential(root,:),edge_potential,prob_bij,edge_pairs);
%logProbTreeBin(root_mar,edgePot,edge_pairs,samples)
%edge_distance
%adjmatTree