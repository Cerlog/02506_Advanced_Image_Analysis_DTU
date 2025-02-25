function [S, iter] = multilabel_MRF(U, dim, beta, max_iter)
%MULTILABEL_MRF Iterative alpha expansion.
% [S, iter] = MULTILABEL_MRF(U, dim, beta, max_iter) finds a solution for 
% the Potts model, with the 2-clique potentials in {0,beta}, and the 
% 1-clique potentials for each vertex and label given in input U.
% 
% U is a N-times-K matrix with energy of each pixel/voxel for each label,
%   where N is a total number of pixels/voxels and K is a number of labels. 
% dim is a vector indicating the dimensions (size) of problem. It must be
%   such that prod(dim)=N.
% beta is a scalar weight of the neighorhood edges (2-clique potental for
%   different labels).
% max_iter is a maximum number of alpha expansion cycles (a cycle includes
%   K alpha expansion steps), defaults to 10.
% S is a length N vector of segmentation labels form {1,...,K}. S is of
%   type uint8, so up til 255 labels are suported.
% iter is a number of actually performed cycles.
%
% Using Kolmorogov+Zabih PAMI 2004 formulation (with directed edges) of
% alpha expansion from Boykov et al. PAMI 2001 (with auxiliary vertices), 
% but with source being not-alpha and sink as alpha (opposite to Boykov 
% formulation). 
%
% vand@dtu.dk, 2014, fixed small bug 2021

if nargin<4
    max_iter = 10;
end

K = size(U,2); % number of classes
[~,S] = min(U,[],2); % initial segmentation
S = uint8(S); % assumes no more than 256 labels

% preparing for n-links, 6-neigborhood of each vertex (4-neigbhorhood for 2D)
indices = reshape(1:prod(dim),dim);
edge_x = indices(1:end-1,:,:);
edge_y = indices(:,1:end-1,:);
edge_z = indices(:,:,1:end-1); % empty for 2D segmentation
edge_n = zeros(numel(edge_x)+numel(edge_y)+numel(edge_z),4);
edge_n(:,[1,2]) = [edge_x(:),edge_x(:)+1; edge_y(:),edge_y(:)+dim(1); ...
    edge_z(:),edge_z(:)+dim(1)*dim(2)];

% preparing for t-links
edge_t = zeros(prod(dim),3);
edge_t(:,1) = indices(:);

iter = 0;
changed = true;
        
while changed && (iter<max_iter)    
    
    S_it = S;
%    for alpha = randperm(K) % expanding classes in random order
    for alpha = 1:K 
                
        % Each pair with different labels requires an adjustment in a
        % n-link (-beta) and an adjustment in a t-lint (+beta), so that
        % only configuration with 0 potential is (alpha,alpha), i.e. 
        % (relabeled,relabeled). Applies also to pairs with alpha, as alpha 
        % is always relabeled due to inf, so only other label matters.
        edge_n(:,[3,4]) = beta;
        diff_pairs = S(edge_n(:,1))~=S(edge_n(:,2)); % neighboors with different labels
        edge_n(diff_pairs,4) = 0; % adjusting n-link (minus beta)   
                
        % Finding vertices that need to have t-link adjusted, 
        % some should be adjusted multiple times, i.e. E(diff_pairs,1) is a
        % non-unique list which gets transformed to vector of occurences.
        [pos,temp,~] = unique(sort(edge_n(diff_pairs,1))); 
        from_pair = zeros(prod(dim),1);
        % So much adjusting each vertex needs:
        from_pair(pos) = [temp(2:end); numel(temp)+1] - temp;        
        
        edge_t(:,3) = U(:,alpha) ; 
        % First term in edge_t(:,2) is equivalent to U(sub2lin(indices(:),S(:)))
        % i.e. U(current label) while the second term is adjustment (+beta).
        edge_t(:,2) = U((double(S(:))-1)*prod(dim)+indices(:))+ beta*from_pair; 
        edge_t(indices(S==alpha),2) = inf; % to force re-labeling with alpha
        
        % solving
        Scut = GraphCutMex(prod(dim),edge_t,edge_n);
        S(Scut) = alpha; % source is not-alpha, source set changes to alpha
                
    end
    changed = any(S(:)~=S_it(:)); % change in labeling
    iter = iter+1;
end
if changed
    warning(['Alpha expansion not converged after ',num2str(max_iter),...
        ' iterations.'])
end
S = reshape(S,dim);