function [ ] = run_clrg(trainSamples, use_anima_distances = false)
    if use_anima_distances
        [adjmatT,edge_distance] = CLRG(trainSamples, 0); 
    else
        [adjmatT, edge_distance]= CLRG(distance, 1, n_samples);
    end
    
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

end