# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:18:12 2020

@author: Daniel Kulikov

Trains a matrix factorization network to recommend Spotify songs based
on the songs in a particular playlist. Evaluates the model based on
"""
import numpy as np    
import pandas as pd
import matrix_factorizer as mf
import random

class song_recommender:
    
    def __init__(self, row_limit):
        self.row_limit = row_limit
        self.p_map = None
        self.p_ind = None
        self.t_map = None
        self.t_map_rev = None
        self.t_ind = None
        self.sp_mf = None

    def recommend(self, playlist_tracks):
        """
        Returns a list of recommended spotify songs based on the inputted playlist.
        """
        rating_matrix, df = self.get_rating_matrix()
        user_row = np.zeros(len(self.t_map_rev))
        
        s_names = playlist_tracks[:, 1]
        a_names = playlist_tracks[:, 3]
        # get list of track names in the correct format: "ArtistSong"
        
        # get appended track names
        track_names = []
        for i in range(playlist_tracks.shape[0]):
            track_names.append(a_names[i] + s_names[i])

            
        # np.save('track_names', np.asarray(track_names))
            
        # convert this to row-form using the mappings
        for j in range(len(track_names)):
            if (track_names[j] in self.t_map_rev):
               user_row[self.t_map_rev[track_names[j]]] = 5
                
        # don't need to update tracks since we're assuming the user has tracks
        # within the selected set (big assumption and non-ideal, however 
        # is just something we have to live with since this is a toy dataset 
        # with little practical applicability)   
        
        sp_mf = self.train(recommend=True, to_append=user_row, rating_matrix=rating_matrix)   
        
        top_five = sp_mf.recommend(rating_matrix.shape[0]-1)
        top_five_str = []
        for l in range(5):
            # convert to song titles
            print(top_five[l])
            print(self.t_map[top_five[l]])   
            top_five_str.append(self.t_map[top_five[l]])
        np.save('top', top_five_str)

    def train(self, recommend=False, to_append=None, rating_matrix=None):
        """
        Trains a matrix factorization model on spotify playlist/track data for 
        recommendation purposes. Uses cross-validation to lean the hyper-parameters
        and evaluates the model based on withheld training samples (user-item pairs).
        """

        # cross-validation
        
        if(rating_matrix is None):
            rating_matrix = self.get_rating_matrix()
            
        params = {}
        epsilons = [0.001, 0.01]
        lmbdas = [1]
        dims = [3]
        
        # add row if necessary
        if(recommend):
            rating_matrix  = np.vstack([rating_matrix, to_append.T])
        
        eval_results = np.zeros((len(epsilons), len(lmbdas), len(dims)))
        #for index, (eps,lmbda,dim) in enumerate(zip(epsilons, lmbdas, dims)):
        #for i in range(1):
            # pick params
        num_iter = 100
        params["eps"], params["lmbda"], params["num_iter"] = 0.01, 0.01, 50
        params["loss"] = "mse"
        # withhold 1% (?) of the playlist-song pairs and set them to 0 in the rating matrix
        # keep their indices within the mappings so we can evaluate
        # get 500 random indices (later replace with 1%)
        random_inds_pl = np.asarray(random.sample(range(0, rating_matrix.shape[0] - 1), 20))
        random_inds_song = np.asarray(random.sample(range(0, rating_matrix.shape[1] - 1), 20))
        
        for i in range(len(random_inds_pl)):
            for j in range(len(random_inds_song)):
                # set each of these random user/item pairs to 0
                rating_matrix[random_inds_pl[i], random_inds_song[j]] = 0

        # train
        sp_mf = mf.matrix_factorizer(rating_matrix, params, 4, "explicit")
        sp_mf.train()
        # evaluate
        evl = self.evaluate_model(sp_mf, random_inds_pl, random_inds_song)
        #eval_results[i] = evl
        
        self.sp_mf = sp_mf
    
        np.save('eval_results.npy', eval_results)
        
        return sp_mf
        
    def evaluate_model(self, model, pl_inds, song_inds):
        
        """
        Evaluates a MF model by withholding user-item pairs and comparing them to the actual values. 
        """
        error = 0
        
        for i in range(len(pl_inds)):
            for j in range(len(song_inds)):
                # get difference between withheld actual rating value and predicted rating value
                error += np.abs(5 - model.compute_rating(pl_inds[i], song_inds[j]))
                
        return error
        
    def get_rating_matrix(self):
        """
        Computes the User-Item rating matrix to use for training the model. 
        """
        song_data = pd.read_csv('C:/Users/Daniel/Documents/GitHub/matrix_factorization/spotify_dataset.csv', delimiter=',', error_bad_lines=False, warn_bad_lines=False)
        
        with open('C:/Users/Daniel/Documents/GitHub/matrix_factorization/spotify_dataset.csv') as myfile:
            head = [next(myfile) for x in range(5)]
            print(head)
            
        # get row_limit number of rows (full dataset is too computationally expensive
        # without parallelization)
        x = song_data.to_numpy()[0:self.row_limit]
        x = x[:, 1:4]
        
        # get rid of some bad rows
        inds = []
        for i in range(x.shape[0]):
            if(not isinstance(x[i,0], str) or not isinstance(x[i,1], str) or not isinstance(x[i,2], str)):
                inds.append(i)
               
        print("removed bad rows")
                
        x = np.delete(x, inds, 0)
        
        z = pd.DataFrame(x)
        z.columns = ["aid", "tid", "pid"]
        z["song_id"] = z["aid"] + z["tid"]
        del z["aid"]
        del z["tid"]
        x = z.to_numpy()
        
        # index playlist_ids
        self.p_map, self.p_ind = np.unique(x[:,0], return_inverse=True)
        
        print("mapped playlist")
        
        # index artist-track names as ids, and reverse for predictions
        self.t_map, self.t_ind = np.unique(x[:,1], return_inverse=True)
        t_map_rev = dict(enumerate(self.t_map))
        t_map_rev = dict((v,k) for k,v in t_map_rev.items())
        self.t_map_rev = t_map_rev
        
        print("mapped tracks")
        # create rating matrix
        rating_matrix = np.zeros((self.p_map.shape[0], self.t_map.shape[0]))
        
        for i in range(self.p_ind.shape[0]):
            rating_matrix[self.p_ind[i], self.t_ind[i]] = 5
        print("created rating matrix")
        
        return rating_matrix, z
        
# 
if __name__ == "__main__":
    """
    Example usage: load in the sptoify data set, set # of rows to take from
    it for training, then get the model. Use sr.recommend() to convert the playlist
    into row form for predictions, and then use cosine similarity to get recommendations
    from the nearest playlist. :)
    """
    data = 'C:/Users/Daniel/Documents/GitHub/matrix_factorization/spotify_dataset.csv'
    row_limit = 10000
    sr = song_recommender(row_limit)
    rm, df = sr.get_rating_matrix()

    #mf, p_map, p_ind, t_map, t_ind = sr.train(data)
    
