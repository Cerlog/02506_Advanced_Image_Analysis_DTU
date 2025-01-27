function [X,T,x,dim] = make_data(example_nr, n, noise)
% Generate data for training a simple neural network.
% 
% function [X,T,x,dim] = make_data(example_nr, n, noise)
% 
% Input:
%   example_nr - a number 1 - 3 for each example
%   n - number of points in each data set
%   noise - a number to increase or decrease the noise level (if changed, 
%       choose between 0.5 and 2)
% Output:
%   X - 2n x 2 array of points (there are n points in each class)
%   T - 2n x 2 target values
%   x - regular sampled points on the area covered by the points that will
%       be used for testing the neural network
%   dim - dimensionality of the area covered by the points
% 
% Authors: Vedrana Andersen Dahl and Anders Bjorholm Dahl - 25/3-2020
%   vand@dtu.dk, abda@dtu.dk
% 

if (nargin < 3)
    noise = 1;
end
if (nargin < 2)
    n = 200;
end

if ( mod(n,2) == 1 )
    n = n+1;
end

dim = [100,100];
[QX,QY] = ndgrid(1:dim(1),1:dim(2));
x = [QX(:),QY(:)];
K = [n,n];
T = [ones(K(1),1)*[1 0];ones(K(2),1)*[0 1]];

switch example_nr
    case 1
        X = [noise*10*randn(K(1),2)+[30,30];noise*10*randn(K(2),2)+[70,70]];
    case 2
        rand_ang = rand(K(1),1)*2*pi;
        X = [noise*5*randn(K(1),2) + 30*[cos(rand_ang), sin(rand_ang)];
            noise*5*randn(K(2),2)] + (dim+1)/2;
    case 3
        X = [noise*10*randn(K(1)/2,2)+[30,30];noise*10*randn(K(1)/2,2)+[70,70];
            10*randn(K(2)/2,2)+[30,70];10*randn(K(2)/2,2)+[70,30]];
    otherwise
        disp('Example number should be 1, 2, or 3.')
end




