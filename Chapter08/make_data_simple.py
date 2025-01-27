#%%

import numpy as np


def make_data(example_nr, n = 200, noise = 1):
    '''
    Generate data for training a simple neural network.
    
        Arguments:
            example_nr: a number 1 to 3 for each example.
            n: number of points in each class set.
            noise: noise level, best between 0.5 and 2.
        Returns:
            X: 2 x 2n array of points (there are n points in each class).
            T: 2 x 2n target values.
            x: grid points for testing the neural network.
            dim: size of the area covered by the grid points.

        Authors: Vedrana Andersen Dahl and Anders Bjorholm Dahl - 25/3-2020
        vand@dtu.dk, abda@dtu.dk
    '''

    rg = np.random.default_rng()
    
    dim = (100, 100)
    
    QX, QY = np.meshgrid(range(0, dim[0]), range(0, dim[1]))
    x_grid = np.c_[np.ravel(QX), np.ravel(QY)]
    
    #  Targets: first half class 0, second half class 1
    T = np.vstack((np.tile([True, False], (n, 1)), 
                   np.tile([False, True], (n, 1))))
    
    if example_nr == 1 :  # two separated clusters

        X = np.vstack((np.tile([30., 30.], (n, 1)), 
                       np.tile([70., 70.], (n, 1))))
        X += rg.normal(size=X.shape, scale=10*noise)  # add noise

    elif example_nr == 2 :  # concentric clusters

        rand_ang = 2 * np.pi * rg.uniform(size=n)
        X = np.vstack((30 * np.array([np.cos(rand_ang), np.sin(rand_ang)]).T, 
                       np.tile([0., 0.], (n, 1))))
        X += [50, 50]  # center
        X += rg.normal(size=X.shape, scale=5*noise)# add noise
    
    elif example_nr == 3 :  # 2x2 checkerboard 
        n1 = n//2
        n2 = n//2 + n%2  # if n is odd n2 will have 1 element more

        X = np.vstack((np.tile([30., 30.], (n1, 1)), 
                       np.tile([70., 70.], (n2, 1)),
                       np.tile([30. ,70.], (n1, 1)),
                       np.tile([70., 30.], (n2, 1))))
        X += rg.normal(size=X.shape, scale=10*noise)  # add noise

    else:
        print('No data returned - example_nr must be 1, 2, or 3')
    
    o = rg.permutation(range(2*n))
    
    return X[o].T, T[o].T, x_grid.T, dim


#%% Test of the data generation
if __name__ == "__main__":
    #%%

    import matplotlib.pyplot as plt
    
    n = 1000
    noise = 1
    
    fig, ax = plt.subplots(1, 3)
    for i, a in enumerate(ax):
        example_nr = i + 1
        X, T, x_grid, dim = make_data(example_nr, n, noise)
        a.scatter(X[0][T[0]], X[1][T[0]], c='r', alpha=0.3, s=15)
        a.scatter(X[0][T[1]], X[1][T[1]], c='g', alpha=0.3, s=15)
        a.set_aspect('equal', 'box')
        a.set_title(f'Example {i} data')
    
    plt.show()
    
    
    #%% Before training, you should make data zero mean
    
    c = np.mean(X, axis=1, keepdims=True)
    X_c = X - c
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_c[0][T[0]], X_c[1][T[0]], c='r', alpha=0.3, s=15)
    ax.scatter(X_c[0][T[1]], X_c[1][T[1]], c='g', alpha=0.3, s=15)
    ax.set_aspect('equal', 'box')
    plt.title('Zero-mean data')
    plt.show()








# %%
