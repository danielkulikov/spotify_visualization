"""
Created on Mon Jul 20 11:18:13 2020

@author: Daniel Kulikov

Matrix factorization using numpy.
"""

import numpy as np

class matrix_factorizer():
    def __init__(self, M, params, dim, data_type, I=None):
        """
        Initializes a matrix factorizer that trains by stochastic gradient descent.
        
        Arguments
        M: user-item rating matrix
        I (optional): gives the matrix P (renamed here), which is a matrix of 
            the binarized values of M, which is used for the implicit case
        data_type: implicit or explicit data
        params: hyperparameter dictionary with keys:
            eps: learning rate
            lmbda: regularization parameter
            num_iter: number of iterations to run SGD
        dim: number of latent dimensions to select
        
        Implicit model based on http://yifanhu.net/PUB/cf.pdf (ALS optimization)
        Explicit model based on (Stochastic Gradient Descent optimization)
        """
        self.params = params
        self.M = M
        self.total_users, self.total_items = M.shape
        self.dim = dim
        self.data_type = data_type
        self.P, self.Q = None, None
        if(I is not None):
            self.I = I
          
    def train(self):
        """
        Trains the matrix factorization model.
        """
        # get num_iterations and hyperparameters
        num_iter = self.params["num_iter"]
  
        # initialize our P,Q matrices (effectively our "weights")
        self.P = np.random.normal(scale=1, size=(self.total_users, self.dim)).astype('float64')
        self.Q = np.random.normal(scale=1, size=(self.total_items, self.dim)).astype('float64')
        # initialize our biases
        self.b_u = np.zeros(self.total_users)
        self.b_i = np.zeros(self.total_items)
        #self.b = np.mean(self.M[np.where(self.M != 0)])
        #self.b = np.zeros(self.M.shape)
        
        # for the explicit case we use SGD
        if(self.data_type == "explicit"):
            for k in range(num_iter):
                # compute loss
                loss = self.compute_loss()
                #print(loss)
                # run one iteration of SGD and update parameters
                for u in range(self.total_users):
                    for i in range(self.total_items):
                        if (self.M[u, i] > 0):
                            self.update_parameters_explicit(u, i)
                        # print update
                #if(k % 10 == 0):
                    #print("Epoch: ", k, " Loss: ", loss)
                            
        # for the implicit case we use ALS (alternating least squares)
        if(self.data_type == "implicit"):
            for k in range(num_iter):
                # does one sweep of ALS updates to P, Q
                self.update_parameters_implicit()
                loss = self.mean_squared_error_implicit()
                #loss = 1
                # print update
                #print("Epoch: ", k, " Loss: ", loss)
                  
    def compute_loss(self):
        """
        Compute the loss of the current state of the model.
        """
        if(self.params["loss"] == "mse"):
            if(self.data_type == "explicit"):
                return self.mean_squared_error()
            #if(self.data_type == "implicit"):
                #return self.mean_squared_error_implicit()
    
    def update_parameters_explicit(self, u, i):
        """
        Adjusts parameters based on one datapoint (user u and item i).
        Updates taken from lecture slides at https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf
        """
        # get error for this particular prediction
        e_ui = self.M[u, i] - self.compute_rating(u, i)
        
        # update interaction matrices
        # P <= P + eps(e_ui * h_i - lambda*w_u)
        # Q <= Q + eps(e_ui * w_u - lambda*h_i)
        # where e_ui is the error, h_i are the inferred parameters for item i 
        # w_u are the inferred parameters for user u
        # and eps, lambda are the learning rate and reg constant respectively.
        eps = self.params["eps"]
        lmbda = self.params["lmbda"]
        self.P[u, :] += eps*(e_ui*self.Q[i, :] - lmbda*self.P[u, :])
        self.Q[i, :] += eps*(e_ui*self.P[u, :] - lmbda*self.Q[i, :])
        # update biases
        self.b_u[u] += eps * (e_ui - lmbda * self.b_u[u])
        self.b_i[i] += eps * (e_ui - lmbda * self.b_i[i])
        
    def update_parameters_implicit(self):
        """
        Does one sweep of ALS parameter adjustment to P, Q.        
        """
        # first we update each of P_u (one row corresponding to the preferences
        # of one user)
        #C = 1. + self.params["eps"]*self.M
        C = self.M
        h = self.params["lmbda"]*np.eye(self.dim)
        
        for u in range(self.P.shape[0]):
            # compute Cu
            Cu = np.diag(C[u, :])
            # compute p_u
            p_u = self.I[u, :]
            # compute terms
            t1 = np.dot(self.Q.T, self.Q)
            t2 = t2 = np.dot(np.dot(self.Q.T, Cu - np.eye(Cu.shape[0])), self.Q)
            t3 = np.add(np.add(t1, t2), h)
            t4 = np.linalg.inv(t3)
            t5 = np.dot(t4, np.dot(self.Q.T, np.dot(Cu, p_u)))   
            # update user features
            self.P[u, :] = t5
            
        for i in range(self.Q.shape[0]):
            # compute Ci
            Ci = np.diag(C[:, i])
            # compute q_i
            p_i =  self.I[:, i]
            # compute terms 
            t1 = np.dot(self.P.T, self.P)
            t2 = t2 = np.dot(np.dot(self.P.T, Ci - np.eye(Ci.shape[0])), self.P)
            t3 = np.add(np.add(t1, t2), h)
            t4 = np.linalg.inv(t3)
            t5 = np.dot(t4, np.dot(self.P.T, np.dot(Ci, p_i)))   
            # update item features
            self.Q[i, :] = t5
        
        # now we do a similar update to each of Q_i (one row corresponding
        #to the preferences of one item)
        
    def compute_rating(self, u, i):
        """
        Computes the predicted rating for a particular user/item pair.
        """
        return self.b_u[u] + self.b_i[i] + self.P[u, :].dot(self.Q[i, :].T)
        #return self.P[u, :].dot(self.Q[i, :].T)
        
    def compute_large_matrix(self):
        return self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        #return self.P.dot(self.Q.T)
    
    def mean_squared_error(self, regularize=False):
        """
        Computes the total mean squared error for the matrix for the explicit case.
        """
        # would like to vectorize this but the 0 condition is quite tricky
        loss = 0
        x_inds, y_inds = self.M.nonzero()
        inds = zip(x_inds, y_inds)
        pred = self.compute_large_matrix()
        
        for x_ind, y_ind in inds:
            loss += np.power((self.M[x_ind,y_ind] - pred[x_ind,y_ind]), 2)
        print(np.sqrt(loss))
        return np.sqrt(loss)
    
    def mean_squared_error_implicit(self, regularize=False):
        """
        Computes the total mean squared error for the matrix for the implicit case.
        """
        #C = 1. + self.params["eps"]*self.M
        C = 1.
        loss = np.sum(np.multiply(C, np.power(self.M - self.compute_large_matrix(), 2)))
        reg = self.params["lmbda"]*(np.sum(np.linalg.norm(self.M, axis=0)) + np.sum(np.linalg.norm(self.M, axis=1)))
        reg_loss = loss + reg
        return reg_loss/(self.M.shape[0] * self.M.shape[1])
    
    def update_predictors(self):
        pass
    
    def get_model(self):
        return self.P, self.Q
    
    def recommend(self, u):
        """
        Get some recommendations (indices) for the user at row u of the rating matrix.
        """
        zero_inds = np.where(self.M[u, :]==0)[0]
        mat = self.compute_large_matrix()
        pred_ratings = mat[u, :][zero_inds]
        top_five = np.argsort(pred_ratings)[-5:]
        
        return top_five

if __name__ == "__main__":
    # test the matrix factorization on the small movielens dataset - 100k ratings
    # load data
    movie_data = np.genfromtxt ('C:/Users/Daniel/Documents/GitHub/matrix_factorization/ml-latest-small/ratings.csv', delimiter=",")[1:, 0:3]
    movie_data = movie_data[np.logical_and(movie_data[:,0] <=100, movie_data[:,1] <= 2000)]
    
    # lets take a subset of 100 users and 2000 movies
    max_users = movie_data[:,0].max()
    max_movies = movie_data[:,1].max()
    movie_data = movie_data[np.logical_and(movie_data[:,0] <= max_users ,movie_data[:, 1] <= max_movies)]
    
    # set up the rating matrix 
    rating_matrix = np.zeros((int(max_users), int(max_movies)))
    
    # Aside: we're not worrying about model evaluation here (splitting into train/test sets)
    # we just want to get some predictions and see if they are reasonable
    for i in range(movie_data.shape[0]):
            row = movie_data[i, :]
            rating_matrix[int(row[0])-1, int(row[1])-1] = int(row[2])
                
    # convert to implicit data if we're doing implicit data
    i_matrix = np.copy(rating_matrix)
    i_matrix[i_matrix > 0] = 1
    
    # cross-validate train the model
    epsilons = [0.001, 0.1, 0.5]
    lmbdas = [0.01, 0.1, 1]
    dims = [20, 30, 50, ]
    cross_validation_results = np.zeros((len(epsilons), len(lmbdas), len(dims)))
    for s in range(3):
        for t in range(3):
            for u in range(3):
                params = {}
                # set up parameters
                num_iter = 15
                eps = epsilons[s]
                lmbda = lmbdas[t]
                dim = dims[u]
                params["eps"], params["lmbda"], params["num_iter"] = eps, lmbda, num_iter
                params["loss"] = "mse"
    
                # train
                mf = matrix_factorizer(rating_matrix, params, dim, "implicit", i_matrix)
                mf.train()
                model = mf.get_model()
                
    
    