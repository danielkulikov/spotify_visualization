# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:06:22 2020

@author: Daniel Kulikov

Doing some data visualization using spotipy (a library for the Spotify API).
"""

import spotipy 
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from pyvis.network import Network
from spotipy.oauth2 import SpotifyClientCredentials

class sp_visualizer():
    """
    Class that visualizes various graphs using the Spotify API.
    """
    def __init__(self, spotify_client):
        self.artist_network = None
        self.cmap = cm.get_cmap('Blues', 12)
        self.sp = spotify_client

    def related_artists_graph(self, artist_id, artist_name="", depth=1, cmap=None):
        """
        Constructs an interactive similarity graph in pyvis representing
        the related artists feature in the Spotify API (to a depth of n).
        """
        nodes = []
        edges = []
        weights = []
        popularities = []
        
        # create network
        related_artists_net = Network(height="800px", width="800px", bgcolor="#222222", font_color="white")
        
        # get a list of names and ids
        related = self.sp.artist_related_artists(artist)['artists']
        
        # get dictionary key/value pairs we want (name, api, genre)
        required_feats = []
       
        for i in range(len(related)):
            # 0th index is the name, 1st index is the URI, 
            # 3rd is space for related artists, 4th is the popularity value
             required_feats.append([related[i]['name'], related[i]['uri'], [], related[i]['popularity']])
    
        # get a 1-d list of nodes for the graph
        for i in range(len(required_feats)):
            related_artists = self.sp.artist_related_artists(required_feats[i][1])['artists']
            # add to nodes and pops
            nodes.append(required_feats[i][0])
            popularities.append(required_feats[i][3])
            for j in range(len(related_artists)):
                required_feats[i][2].append([related_artists[j]['name'], related_artists[j]['uri']])
                nodes.append(related_artists[j]['name'])
                # add to nodes and pops
                popularities.append(related_artists[j]['popularity'])
       
          
        # get unique nodes
        nodes, popularities = np.asarray(nodes), np.asarray(popularities)
        popularities = popularities.reshape((popularities.shape[0], 1))
        unique_nodes, ind = np.unique(nodes, axis=0, return_index=True)
        popularities = popularities[ind]
        nodes = unique_nodes
        
        # gets a unique list of edges for the graph, along with corresponding similarity weights
        for i in range(len(required_feats)):
            nd = required_feats[i][0]
            for j in range(len(required_feats[i][2])):
                adj = required_feats[i][2][j]
                edges.append([nd, adj[0]])
                weights.append(self.weight(j))

        
        edges, weights = np.asarray(edges), np.asarray(weights)
        weights = weights.reshape((weights.shape[0], 1))
        unique_edges, ind = np.unique(edges, axis=0, return_index=True)
        weights = weights[ind]
        edges = np.append(edges, weights, 1)

        # convert list to pandas dataframe
        edge_data = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])
        edge_data = edge_data.sort_values(by = ["Source", "Weight"])
        
        # add nodes and edges to the graph
        sources = edge_data['Source']
        targets = edge_data['Target']
        weights = edge_data['Weight']

        # zip the data
        edge_data = zip(sources, targets, weights)
        
        # use color-map
        for i in range(len(nodes)):
            pop = popularities[i]
            value = self.get_node_size(pop.item())
            
            x = None
            y = None
            
            color=self.get_node_colour(pop/100)
            
            if (nodes[i] == artist_name):
                x, y = 400, 400
                color = "#FFFFFF"
                
            related_artists_net.add_node(nodes[i], value=value, \
                color=color, \
                image = "https://i.ytimg.com/vi/cSblrT8hBDc/maxresdefault.jpg", x=x, y=y)
        
        # add edges to the graph
        for e in edge_data:
            source = e[0]
            dst = e[1]
            #w = e[2]
            related_artists_net.add_edge(source, dst, width=1, arrowStrikethrough=False)
        
        #neighbor_map = related_artists_net.get_adj_list()
        
        # populate hover data with artist names and other data we might want
        # for node in related_artists_net.nodes:
        #   node["image"] = "https://i.ytimg.com/vi/cSblrT8hBDc/maxresdefault.jpg"
            
        # show graph
        related_artists_net.show("artist_graph.html")
    
    def visualize_playlist(self, user_id, playlist_id):
        """
        Visualizes a playlist using spotipy. Genre breakdown (clustering),
        etc.
        """
        tracks_data = self.get_tracks_data(user_id, playlist_id)
        
        # we now have a numpy array of all the track info we need, in the order:
        # 0: track id
        # 1: track name
        # 2: track album
        # 3: track artist
        # 4: track duration (ms)
        # 5: track artist id
        # 6: popularity 
        # 7: danceability [0,1]
        # 8: energy [0,1]
        # 9: valence [0,1]
        # 10: key (pitch class notation)
        # 11: loudness [-60, 0]?
        # 12: mode (major/minor)
        # 13: instrumentalness [0,1]
        # 14: tempo (int)
        
        # get the top n genres in the playlist, along with their counts
        top_n_genres, counts = self.get_genre_breakdown(tracks_data, 10)
        
        # plot valence vs. energy with plotly - DASH to make it interactive
        
        # recommend some artist or songs using recommendation algorithm :~)
            # TO-DO
        
        
    def get_genre_breakdown(self, tracks_data, n):
        """
        Returns the top n genres of the playlist, along with their count
        """
        # NB: why can't we get the genre of a particular song? Weird move, spotify...
        # so, to get the *approximate* genre breakdown of the playlist, we have to 
        # get a list of all the unique artists and query the artist genres.
        
        # lets do this first - get unique artists
        artist_ids = np.unique(tracks_data[:, 6], axis=0)
        
        artist_info = self.batch_get_artist_features(artist_ids)
        
        # get their genres - this is a list of lists
        genres = [artist_info[x]["genres"] for x in range(len(artist_info))] 

        # merge the sublists
        genres = np.asarray([j for i in genres for j in i])
        
        # get unique genres with corresponding counts
        unique_genres, counts = np.unique(genres, return_counts=True)
        
        # get sorted indices
        sorted_indices = np.argsort(counts)
        
        # apply
        sorted_genres = unique_genres[sorted_indices][-n:]
        counts = counts[sorted_indices][-n:]
        
        return sorted_genres, counts
        
    def get_tracks_data(self, user_id, playlist_id):
        """
        Gets track info and audio features of all the tracks in a given playlist.
        Returns a numpy array.
        """
        # get playlist tracks
        tracks_data = get_playlist_tracks(user_id, playlist_id)
        
        # get what we want out of tracks , "artists", "album", "duration_ms", "popularity"
        required_keys = ["id", "name", "album", "artists", "duration_ms", "popularity"]
        
        # hacking it a little to get the values in a nice format u.u
        pl_track_vals = []
        for i in range(len(tracks_data)):
            vals = [tracks_data[i]["track"][x] for x in required_keys]
            for j in range(len(vals)):
                if(isinstance(vals[j], list)):
                    vals[j] = vals[j][0]
                if(isinstance(vals[j], dict)):
                    if("name" in vals[j]):
                        vals[j] = vals[j]["name"]
                # get artist id
            vals.append(str(tracks_data[i]["track"]["artists"][0]["id"]))
            pl_track_vals.append(vals)

        audio_feats = []
        # can only request audio features for sets of size 100, so we partition our tracks
        subsets = [pl_track_vals[x:x+100] for x in range(0, len(pl_track_vals), 100)]

        # request the audio features of the playlist
        for i in range(len(subsets)):
            subset = np.asarray(subsets[i])
            a_f = self.sp.audio_features(tracks=subset[:, 0])
            for j in range(len(subset)):
                audio_feats.append(a_f[j])
                
        required_audio_feats = ["danceability", "energy", "valence", 
                                "key", "loudness", "mode", "instrumentalness", "tempo"]
        
        # extract  the required keys from the audio features
        filtered_audio_feats = []
        for i in range(len(audio_feats)):
             filtered_audio_feats.append([audio_feats[i][x] for x in required_audio_feats])
        
        # combine track data and audio features 
        track_data = np.concatenate((np.asarray(pl_track_vals), np.asarray(filtered_audio_feats)), axis=1)
        
        return track_data
            
        
    def batch_get_audio_features(self, playlist_id):
        """
        Gets the audio features of a playlist with batch size 100 (the maximum
        allowed for a request).
        """
        pass
    
    def batch_get_artist_features(self, artist_ids):
        """
        Gets the artist info of a list of artist ids with batch size 50
        (the maximum allowed for a request).
        """
        subsets = [artist_ids[x:x+50] for x in range(0, len(artist_ids), 100)]
        artists = []

        # get batch artist info
        for subset in subsets:
            info = self.sp.artists(subset)["artists"]
            for artist in info:
                artists.append(artist)
        return artists
    
    def weight(self, x):
        """
        Computes a function for similarity based on its position in the 
        related artists list (given some artist).
        """
        return (x+1)**(0.3)    
    
    def get_node_colour(self, x):
        """
        Colour map for artist popularity values.
        """
        c_m = self.cmap
        return colors.rgb2hex(c_m(x)[:, :3][0])
      
    def get_node_size(self, x):
        """
        Defines the scaling of nodes based on the popularity value.
        """
        return x**2
    
def get_playlist_tracks(username,playlist_id, sp):
    """
    Gets the tracks of a playlist using spotipy. 
    """
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks   

def read_csv_data(csv):
    """
    Reads in a csv file into pandas dataframe format.
    """
    return pd.read_csv(csv)

def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks
 
if __name__ == "__main__":
    # set up spotify client and authenticate
    
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # artist URI - e.g. I use 
    artist = 'spotify:artist:2YZyLoL8N0Wb9xBt1NhZWg'
    artist_name = sp.artist(artist)['name']
    
    # create visualizer object
    sp_vis = sp_visualizer(sp)
    #sp_vis.related_artists_graph(artist, artist_name=artist_name)
    sp_vis.visualize_playlist('dkulikov', "6Mj2GTsBFWtSygUOl7ijws")