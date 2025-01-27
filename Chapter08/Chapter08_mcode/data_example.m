% Generation of example data

example_nr = 3; % Three examples
n = 1000; % Number of points
noise = 1; % Noise level

% Make and show the data
[X,T,x,dim] = make_data(example_nr, n, noise);

cmap = interp1([-1; -0.5; 0; 0.5; 1],[0,0,0.5; 0,0.5,1; 1,1,1; 1,0,0; 0.5,0,0],linspace(-1,1,256));
X_colors = cmap([1+63,end-63],:); 

figure
scatter(X(:,2),X(:,1),20,X_colors(T(:,1)+1,:),'filled','MarkerEdgeColor',[0 0 0])
box on, axis ij image, axis([0.5,dim(2)+0.5,0.5,dim(1)+0.5])
title('training')

%% Before training, you should make data have zero mean

c = mean(X,1);
xc = x-c;
Xc = X-c;

figure
scatter(Xc(:,2),Xc(:,1),20,X_colors(T(:,1)+1,:),'filled','MarkerEdgeColor',[0 0 0])
axis equal
box on, axis ij image, axis([0.5-c(2),dim(2)+0.5-c(2),0.5-c(1),dim(1)+0.5-c(1)])
title('Zero mean training')












