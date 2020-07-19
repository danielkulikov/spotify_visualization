# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:06:22 2020

@author: Daniel Kulikov

Doing some data visualization using spotipy (a library for the Spotify API).
"""

import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from pyvis.network import Network
import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib import colors

class sp_visualizer():
    """
    Class that visualizes various graphs using the Spotify API.
    """
    def __init__(self):
        self.artist_network = None

    def related_artists_graph(self, artist_id, depth=1, cmap=None):
        """
        Constructs an interactive similarity graph in pyvis representing
        the related artists feature in the Spotify API (to a depth of n).
        """
        nodes = []
        edges = []
        weights = []
        popularities = []
        
        # create network
        related_artists_net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white")
        
        # get a list of names and ids
        related = sp.artist_related_artists(artist)['artists']
        
        # get dictionary key/value pairs we want (name, api, genre)
        required_feats = []
       
        for i in range(len(related)):
             required_feats.append([related[i]['name'], related[i]['uri'], [], related[i]['popularity']])
    
        # get a 1-d list of nodes for the graph
        for i in range(len(required_feats)):
            related_artists = sp.artist_related_artists(required_feats[i][1])['artists']
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
            related_artists_net.add_node(nodes[i], value=value, \
                color=self.get_node_colour(pop/100), \
                image = "https://i.ytimg.com/vi/cSblrT8hBDc/maxresdefault.jpg")
        
        # add edges to the graph
        for e in edge_data:
            source = e[0]
            dst = e[1]
            #w = e[2]

            related_artists_net.add_edge(source, dst, width=1, arrowStrikethrough=False)
        
        neighbor_map = related_artists_net.get_adj_list()
        
        # populate hover data with artist names and other data we might want
       # for node in related_artists_net.nodes:
         #   node["image"] = "https://i.ytimg.com/vi/cSblrT8hBDc/maxresdefault.jpg"
            
        # show graph
        related_artists_net.show("aaa.html")
    
    def visualize_playlist(self, playlist_id):
        """
        Visualizes a playlist using spotipy. Genre breakdown (clustering),
        etc.
        """
        pass
    
    def weight(self, x):
        """
        Computes a function for similarity based on its position in the 
        related artists list (given some artist).
        """
        return (x+1)**1.5
    
    def get_node_colour(self, x):
        """
        Colour map for artist popularity values.
        """
        c_m = cm.get_cmap('Blues', 12)
        c_m = colors.rgb2hex(c_m(x)[:, :3][0])
        return c_m
    
    def get_node_size(self, x):
        """
        Defines the scaling of nodes based on the popularity value.
        """
        return x**2
    
def get_playlist_tracks(username,playlist_id):
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
 
if __name__ == "__main__":
    # client ids
    cid ="607e8d3ba9664c3f8da9dc813e020779" 
    secret = "11912db189014b7c948928cc3c85cb0d"
    
    # authenticate  requests
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # artist URI - e.g. I use 
    artist = 'spotify:artist:2YZyLoL8N0Wb9xBt1NhZWg'
    sp_vis = sp_visualizer()
    sp_vis.related_artists_graph(artist)