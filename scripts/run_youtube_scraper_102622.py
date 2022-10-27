#!/usr/bin/python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from tqdm import tqdm
import ast
import spacy
from googleapiclient.discovery import build
from youtube_helpers import get_channel_playlists, get_video_ids, get_video_details
nlp = spacy.load("en_core_web_lg")

path = '/Users/jetcalz07/Desktop/MIDS/W266_NLP/nlp_podcast_segmentation/'
load_scripts = False

api_key = 'AIzaSyDTr6GYriQpML3rwVB_M4aVhixseOxVO4U'
channel_ids = ['UCSHZKyawb77ixDdsGog4iWA', # Lex Fridman
               'UCESLZhusAkFfsNsApnjF_Cg', # All-In
               'UCNNTZgxNQuBrhbO0VrG8woA', # No Jumper
               'UC0-swBG9Ne0Vh4OuoJ2bjbA', # rSlash
               'UC2D2CMWXMOVWx7giW1n3LIg', # Andrew Huberman
               'UCxcTeAKWJca6XyJ37_ZoKIQ' , # Pat McAfee
               'UC5PstSsGrRwj2o6asQpC4Rg' , # Flagrant
               'UC5sqmi33b7l9kIYa0yASOmQ' , # Fresh and Fit
               'UCbk_QsfaFZG6PdQeCvaYXJQ' , # Jay Shetty
               'UCGeBogGDZ9W3dsGx-mWQGJA', # Logan Paul
               'UCL_f53ZEJxp8TtlOkHwMV9Q', # JP
               'UCAVojJ1k03GZzjSbdXXunkw', # Shane2
              ]

channel_ids = ['UCSHZKyawb77ixDdsGog4iWA', # Lex Fridman
              ]
youtube = build('youtube', 'v3', developerKey=api_key)

## Execute functions to collect data from Youtube channels

if not load_scripts:
    # Get playlist ids for channel
    channel_df = pd.DataFrame(get_channel_playlists(youtube, channel_ids))

    # Loop through channels, get video ids
    v_ids = np.array([])
    for idx, row in channel_df.iterrows():
        print(f"Begin channel {idx+1}")
        playlist_id = row['playlist_id']
        channel_vids = get_video_ids(youtube, playlist_id)
        v_ids = np.append(v_ids, channel_vids)

    # Loop through videos, get timestamp, script information
    vid_details = pd.DataFrame(get_video_details(youtube, v_ids, splitter='spacy', nlp = nlp))
    print(f"Total Channels: {len(channel_df)}, Total Videos: {len(v_ids)}")
    print(f"Total videos with timestamps & scripts: {len(vid_details)}")
    vid_details.to_csv(path+f'data/yt_scripts_segments_v102722_{len(vid_details)}.csv', index=False)

else:
    vid_details = pd.read_csv(path+'data/yt_scripts_segments_v102722.csv')
    vid_details['Transcript'] = vid_details['Transcript'].apply(lambda x: ast.literal_eval(x))

print("Done!")