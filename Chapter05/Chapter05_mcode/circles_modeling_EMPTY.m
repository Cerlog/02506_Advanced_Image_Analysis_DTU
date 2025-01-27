clear
close all
addpath('functions')

%% loading data
path = '../../../../Data/week5/';
I = double(imread([path, 'noisy_circles.png']));

segmentations = {}; % cell-array where I'll place diifferent segmentations
GT = double(imread([path, 'noise_free_circles.png']));

[mu, ~, S_gt] = unique(GT);
S_gt = reshape(S_gt, size(I));
segmentations{1} = S_gt;

%% finding some configurations (segmentations) using conventional methods
S_t = ones(size(I))+(I>100)+(I>160); % thresholded
segmentations{2} = S_t;

D_s = imfilter(I, fspecial('gaussian', 7 ,1), 'replicate');
S_g = ones(size(I))+(D_s>100)+(D_s>160); % gaussian fitered and thresholded
segmentations{3} = S_g;

D_m = medfilt2(I,[5 5],'symmetric');
S_m = ones(size(I))+(D_m>100)+(D_m>160); % median fitered and thresholded
segmentations{4} = S_m;

%% visualization
figure
imagesc(I, [0,255]), axis image, colormap gray

figure
N = length(segmentations);
beta = 100;

for i = 1:N
    
    subplot(3,N,i)
    imagesc(segmentations{i}), axis image
    [V1,V2] = segmentation_energy(segmentations{i}, I, mu, beta);
    title({['likelihood: ', num2str(round(V1))],['prior: ', num2str(V2)],...
        ['posterior: ', num2str(round(V1+V2))]})
    
    subplot(3, N, i+N)
    segmentation_histogram(I, segmentations{i})
    xlabel('intensity'), ylabel('count'), %legend('all','1','2','3')
    xlim([0,255]), box on
    
    subplot(3, N, i+2*N)
    err = S_gt - segmentations{i};
    imagesc(err, [-2,2]), axis image
    title(['error: ',num2str(sum(err(:)>0))])
    colormap(gca,[0,0,0.5; 0,0.25,1; 1,1,1; 1,0,0; 0.5,0,0])
end

function [U1,U2] = segmentation_energy(S, I, mu, beta)
% ﻿TODO -- add your code here

% likelihood energy
U1 = 0;

% prior energy
U2 = 0;

end

