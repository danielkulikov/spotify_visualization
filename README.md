# spotify_visualization
For the purposes of some data visualization and practice with recommender systems in python, using Spotipy (a python library for the Spotify API). 

Create a visualizer: \
\
sp = sp_visualizer(spotify_client) \
\
Then, call: \
\
  sp.related_artist_graph(artist_id, artist_name) \
  \
  for a (default 2-depth) network visualization of the related artists feature provided by the Spotify API \
 \
Or call: \
\
  sp.visualize_playlist(user_id, playlist_id)  \
  \
  a) retrieve a top-n genre breakdown of the playlist \
  b) generate a valence vs. energy plot of all the tracks in the playlist using bokeh (generates a webpage file) \
  c) get recommended some tracks based on your playlist (WIP: homebrew matrix factorization is complete in the matrix_factorization repo). \
