clear, close all
addpath GraphCut

d = [179 174 182 162 175 165]; % heights (data)
mu = [181, 165]; % means of two classes
beta = 100; % weight of the prior term
w_s = (d(:)-mu(1)).^2; % source weight
w_t = (d(:)-mu(2)).^2; % sink weights
N = numel(d); % number of graph nodes
indices = (1:N)'; % an index for each person

% terminal and internal edge matrix
E_terminal = [indices,[w_s,w_t]]; 
E_internal = [indices(1:end-1),indices(2:end),beta*ones(N-1,2)]; 

[Scut, flow] = GraphCutMex(N,E_terminal,E_internal); % here it happens
disp(['Maximum flow: ',num2str(flow)])

% displaying results
S = 'MMMMMM';
S(Scut) = 'F';
for i=1:6
    disp(['Person ',num2str(i),' is estimated as ',S(i)])
end
