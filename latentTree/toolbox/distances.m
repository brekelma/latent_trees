function distances = distances(samples)
    
% samples x configs
% matrix n x spins
d = size(samples,1);
n = size(samples,2);
dist = zeros(d,d);
for i=1:d
    for j=i:d
        p_ij = zeros(2,2);
        p(1,1) = sum((samples(i, :) == -1).*(samples(j,:) == -1))/n;
        p(2,1) = sum((samples(i, :) == 1).*(samples(j,:) == -1))/n;
        p(1,2) = sum((samples(i, :) == -1).*(samples(j,:) == 1))/n;
        p(2,2) = sum((samples(i, :) == 1).*(samples(j,:) == 1))/n;
        pi = sum((samples(i, :) == 1))/n;
        pj = sum((samples(j, :) == 1))/n;
        corr = p(2,2) - pi*pj; % p(1,1)-p(2,1)-p(1,2)+p(2,2); % E(Xi Xj)
        %corr = p(1,1) - (1-pi)*(1-pj) - p(2,1) - (pi*(1-pj)) - p(1,2) - (1-pi)*pj + p(2,2) - (1-pi)*(1-pj);
        corr = corr/sqrt((pi*(1-pi)*pj*(1-pj)));
        dist(i,j) = -log(abs(corr));
        dist(j,i) = -log(abs(corr));
    end
end

distances = dist;