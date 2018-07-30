function [adjmatT, edge_distance, adjmatCL, mi, mi_me, distance] = run_clrg(trainSamples, min_max)
    use_anima_distances = false;
    if use_anima_distances
        [adjmatT, edge_distance, usedCL] = CLRG2(trainSamples, 0, size(trainSamples, 2), min_max); 
    else
        [adjmatT, edge_distance, usedCL] = CLRG2(distances(trainSamples), 1, size(trainSamples,2), min_max);
    end
    
    prob_bij = computeBnStats(trainSamples);
    distance = distances(trainSamples);%computeDistance(prob_bij);
    nodes = size(prob_bij,2)/2;
    mi =computeMutualInformationBin(prob_bij);
    mi_me = zeros(nodes, nodes);
    ns = size(trainSamples, 2);
    if sum(sum(trainSamples(:,1:2) == 2)) > 0
       a = 1;
       b = 2;
    else
       a= -1;
       b = 1;
    end
    for i=1:nodes
        for j=1:nodes
            p00 = sum(trainSamples(i, :)==a & trainSamples(j,:)==a)/ns;
            p01 = sum(trainSamples(i, :)==a & trainSamples(j,:)==b)/ns;
            p10 = sum(trainSamples(i, :)==b & trainSamples(j,:)==a)/ns;
            p11 = sum(trainSamples(i, :)==b & trainSamples(j,:)==b)/ns;
            p1i = sum(trainSamples(i, :)==b)/ns;
            p1j = sum(trainSamples(j, :)==b)/ns;
            mi_me(i,j) = p00*log((p00+eps)/((1-p1i+eps)*(1-p1j+eps)))+p01*log((p01+eps)/((1-p1i+eps)*(p1j+eps)))+ p10*log((eps+p10)/((p1i+eps)*(1-p1j+eps))) + p11*log((p11+eps)/((p1i+eps)*(p1j+eps)));
        end 
    end
    %mi_me = mi_me + mi_me.';
    adjmatCL = ChowLiu(mi);
    %adjmatT = full(adjmatT);
    %edge_distance = full(edge_distance);
    % previously only over obs
%     [~, root] = max(sum(adjmatT, 2));
%     msg_order = treeMsgOrder(adjmatT, root);
%     [node_potential,edge_potential]= marToPotBin(prob_bij, msg_order);
%     edge_pairs = msg_order(num_nodes:end,:);
% 
%     if use_anima
%         root_mar = sum(trainSamples(root, :)-1 > 0) / size(trainSamples,2);
%     else
%         root_mar = sum(trainSamples(root, :)-1 >=0) / size(trainSamples,2);
%     end
%     root_mar = [1- root_mar, root_mar];
%     % need to use anima
%     ll_bin = logProbTreeBin(root_mar,edge_potential,edge_pairs,trainSamples);
end