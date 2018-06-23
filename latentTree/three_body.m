
% method = 'RG';
% method = 'NJ';
method = 'CLRG';
% method = 'CLNJ';
% method = 'regCLRG';
% method = 'regCLNJ';

hide = true; 
use_anima = true;

a = tdfread("../example.csv");
names = fieldnames(a);
b = getfield(a, names{1});
params = {};
for i=1:size(b,1)
    param = [];
    c = strsplit(b(i,:), ",");
    for j=1:size(c,2)
        param = [param; str2num(c{j})];
    end
    params = [params, [param]];
    %append(tuples, [1])
end
couplings = getfield(a, names{2});

dim = 3;
hidden = 1;
n_samples = 1000000;

% construct more general
configs = [-1 -1 -1; -1 -1 1; 1 1 1; -1 1 -1; 1 1 -1; 1 -1 1; -1 1 1; 1 -1 -1];
configs_hidden = zeros(2*size(configs,1), dim+hidden);
for i=1:size(configs,1)
    configs_hidden(2*i-1, :) = horzcat(configs(i,:), [-1]);
    configs_hidden(2*i, :) = horzcat(configs(i,:), [1]);
end

configs = configs_hidden;
"weights"
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
"replace"
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
"sample"
%configs = reshape(configs,[8,3])
%configs2 = reshape(configs2, [8,3])
weights = arrayfun(@exp, weights) ./ sum(arrayfun(@exp, weights));
y = randsample(length(weights), n_samples, true, weights);
%trainSamples = arrayfun(@(x) configs(x(i),:), y);
if hide
    trainSamples = zeros(n_samples, size(configs,2)-hidden);
else
    trainSamples = zeros(n_samples, size(configs,2));
end
%prev = 1;
%bb = sum(n_samples);
for i = 1:n_samples
    if use_anima
        configs = configs2;
    end
    if hide
        trainSamples(i, :) = configs(y(i),1:dim);
    else
        trainSamples(i, :) = configs(y(i),:);
    end
end

trainSamples = trainSamples';
distance = distances(trainSamples);
"going to clrg"
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
"hello world"
drawLatentTree(adjmatT, 3)
%edge_distance
%adjmatTree