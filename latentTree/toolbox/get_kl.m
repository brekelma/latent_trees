function [adjmatT, edge_distance, adjmatCL, mi] = get_kl(trainSamples, trainSamples_all)
    % DOESN"T WORK
    
    num_nodes = size(adjmatT,1);
    [~, root] = max(sum(adjmatT, 2));
    msg_order = treeMsgOrder(adjmatT, root);


    prob_bij = computeBnStats(trainSamples);

    % previously only over obs
    [node_potential,edge_potential]= marToPotBin(prob_bij, msg_order);

    edge_pairs = msg_order(num_nodes:end,:);
    "c61"

    if use_anima
        root_mar = sum(trainSamples(root, :)-1 > 0) / size(trainSamples,2);
    else
        root_mar = sum(trainSamples(root, :)-1 >=0) / size(trainSamples,2);
    end
    root_mar = [1- root_mar, root_mar];
    % need to use anima
    ll_bin_all = logProbTreeBin(root_mar,edge_potential,edge_pairs,trainSamples);

    num_nodes = size(adjmatT,1);
    [~, root] = max(sum(adjmatT, 2));
    % doesnt work if just have params of "true"
    msg_order = treeMsgOrder(adjmatT, root);

    prob_bij = computeBnStats(trainSamples);

    % try to make this over all
    [node_potential,edge_potential]= marToPotBin(prob_bij, msg_order);

    edge_pairs = msg_order(num_nodes:end,:);
    if use_anima
        root_mar = sum(trainSamples(root, :)-1 > 0) / size(trainSamples,2);
    else
        root_mar = sum(trainSamples(root, :)-1 >=0) / size(trainSamples,2);
    end
    root_mar = [1- root_mar, root_mar];
    ll_bin_obs = logProbTreeBin(root_mar,edge_potential,edge_pairs,trainSamples);

    "true " 
    sum(ll_bin_all)
    "cross"
    sum(ll_bin_obs)
    " kl: "
    sum(ll_bin_all)-sum(ll_bin_obs) 
end