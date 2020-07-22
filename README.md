# spotify_visualization
For the purposes of some data visualization and practice with recommender systems in python, using Spotipy (a python library for the Spotify API). 

Create a visualizer: \
\
sp = sp_visualizer(spotify_client) \
\
With it you can call: \
  sp.related_artist_graph(artist_id, artist_name) \
  for a (default 2-depth) network visualization of the related artists feature provided by the Spotify API \
  \
Or, you can call: \
  sp.visualize_playlist(user_id, playlist_id)  \ 
  which: a) retrieves a top-n genre breakdown of the playlist \
  b) generates a valence vs. energy plot of all the tracks in the playlist using bokeh (generates a webpage file) \
  c) recommends some tracks based on your playlist (WIP: homebrew matrix factorization is complete in the matrix_factorization repo). \
