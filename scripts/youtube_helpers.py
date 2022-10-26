import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
#from googleapiclient.discovery import build

path = '/Users/jetcalz07/Desktop/MIDS/W266_NLP/nlp_podcast_segmentation/'
load_scripts = True


def get_channel_playlists(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
                part='snippet, contentDetails',
                id=','.join(channel_ids))
    response = request.execute() 
    
    for i in range(len(response['items'])):
        data = dict(channel = response['items'][i]['snippet']['title'],
                    playlist_id = response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
        all_data.append(data)
    
    return all_data


## Get video ids
def get_video_ids(youtube, playlist_id):
    
    # Request first 50 videos in playlist
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId = playlist_id,
                maxResults = 50)
    response = request.execute()
    
    # unpack the video ids
    video_ids = []
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
    
    # check if there are more results
    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    # collect info from next results pages
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token)
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
        
    return video_ids


def get_transcript(vid_id): ## plugged into get_video_details
    flag = None
    try: # check if any transcripts available
        transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
        try: # search for manual transcript first, if available
            script_type = 'manual'
            transcript = transcript_list.find_manually_created_transcript(['en-US'])
            print("Manual available")
        except: # else, use auto-generated one
            script_type = 'auto'
            transcript = transcript_list.find_generated_transcript(['en'])
        script = transcript.fetch()
    except: # there may be no subtitles allow for video
        flag = True
        
    if not flag: # transcript is available
        return script, script_type
    else: # none available
        return None, None


def timestamp_to_secs(time_str):
    t = np.array(time_str.split(':'))
    # Add 0 hours if none
    if len(t) < 3:
        t = np.insert(t, 0, '0')
    t.astype(int)
    h, m, s = t
    return round(int(h) * 3600 + int(m) * 60 + int(s), 0)


def extract_segments(desc, first_n=12): ## plugged into get_video_details
    # break description into chunks and find which chunk contains timestamps
    line_matches = np.array([])
    times = np.array([])
    times_secs = np.array([])
    desc_chunks = desc.split("\n\n")
    for chunk in desc_chunks: # loop through chunks
        for line in chunk.split("\n"): # loop through lines in a chunk
            # the timestamp will most always be within first 12 characters of line: "(00:00:00)" is 10
            first_chars = line[:first_n]
            match_hrs = re.search(r'[0-9][:][0-9][0-9][:][0-9][0-9]', first_chars) # >=1 hour
            match_mins = re.search(r'[0-9][:][0-9][0-9]', first_chars) # <1 hour
            if match_hrs != None:
                span = match_hrs.span()
                time = line[span[0]:span[1]]
                times = np.append(times, time)
                times_secs = np.append(times_secs, timestamp_to_secs(time))
                line_matches = np.append(line_matches, line) # also append full line (time + summary) to list
            elif match_mins != None: 
                span = match_mins.span()
                time = line[span[0]:span[1]]
                times = np.append(times, time) # string timestamp
                times_secs = np.append(times_secs, timestamp_to_secs(time)) # time converted to # of seconds
                line_matches = np.append(line_matches, line) # also append full line (time + summary) to list
            else:
                continue

    return line_matches, times, times_secs


# unpack words from script, map word to time
def unpack_words(script):
    utter_words = []
    utter_times = []
    all_text = ''

    for line in script:
        for word in line['text'].split(" "):
            if (word != ""):
                word = word.replace("'","")
                utter_words.append(word)
                utter_times.append(line['start'])
                all_text = all_text + ' ' + word
            
    return utter_words, utter_times, all_text




#### ---- Below functions needed for Spacy sentence splitter ---- ####
# unpack script into string of words, and word-time mapping
def unpack_words(script, replace_list = ["'", "_", "[", "]"]):
    utter_words = []
    utter_times = []
    all_text = ''

    for line in script:
        for word in line['text'].split(" "):
            if (word != ""):
                for str in replace_list:
                    word = word.replace(str,"")
                utter_words.append(word)
                utter_times.append(line['start'])
                all_text = all_text + ' ' + word
            
    return utter_words, utter_times, all_text

#sentence splitting with Spacy
def spacy_split(nlp, all_text):
    about_doc = nlp(all_text)
    sentences = list(about_doc.sents)

    return sentences

# after sentence split, map each word to a sentence ind
def word_sentence_map(sentences, replace_str = [" ", "\n", "_", "[", "]"]):
    z = 0 # sentence index
    sent_inds = []
    sent_word_lists = [] # store the words for each sentence
    sent_words = [] # all words after splitting

    for sentence in sentences:
        word_list = []
        for word in sentence.text.split(" "):
            for str in replace_str:
                word = word.replace(str, "")
            sent_inds.append(z)
            sent_words.append(word)
            word_list.append(word)
        z += 1
        sent_word_lists.append(word_list)

    return sent_words, sent_inds, sent_word_lists

## Account for differences in word lists pre/post splitter
# Basically check the text and if a word does't equal the neighbor, check if the next
def map_sentence_time(utter_words, sent_words, utter_times, sent_inds, sent_word_lists, word_retention_thresh = 0.97):
    # check if the utter words and sent words map
    if np.all(utter_words == sent_words):
        print("Match")
        final_words = utter_words
        final_sentence_inds = sent_inds
        final_times = utter_times
        final_sent_word_lists = sent_word_lists

    else:
        rows = np.min([len(utter_words), len(sent_words)])
        w_t = utter_words
        w_s = sent_words
        final_words = []
        final_sentence_inds = []
        final_times = []
        final_sent_word_lists = []

        w_t_i = 0 # utter word ind
        w_s_i = 0 # sentence word ind
        check_n = 5 # if misaligned, check words in +=n direction to re-align
        sent_ind = -1 # trigger new sentence list at beginning
        sent_word_list = []

        for i in range(rows): # loop through all words
            if sent_inds[w_s_i] != sent_ind: # check if new sentence start
                if i != 0: # don't append if first sentence
                    final_sent_word_lists.append(sent_word_list)
                elif i == (rows-1): # make sure last sentence gets added before quiting
                    final_sent_word_lists.append(sent_word_list)
                    break
                sent_word_list = [] # start new sentence list
                sent_ind = sent_inds[w_s_i] # align indexes

            # check if the two lists are equal at indices
            if w_t[w_t_i] == w_s[w_s_i]:
                final_words.append(w_t[w_t_i])
                final_sentence_inds.append(sent_ind)
                final_times.append(utter_times[w_t_i])
                sent_word_list.append(w_t[w_t_i])
                w_t_i += 1
                w_s_i += 1

            else:
                # check if utter word list gets ahead, if so, look at next n sentence map words to adjust inds and re-align
                next_w_s = w_s[w_s_i+1] # something is wrong at current index, so lets skip it and go to next word
                check_w_t_next = w_t[w_t_i+1:w_t_i+check_n]
                if next_w_s in check_w_t_next: # if left list gets ahead, find out by how much, adjust indices, return to top of loop and should get match
                    add_ind = np.where(check_w_t_next == next_w_s)[0][0] # take first ind where match is found
                    w_t_i += 1 + add_ind # skip ahead of sentence word list
                    w_s_i += 1

                # check if sentence map gets ahead
                next_w_t = w_t[w_t_i+1]
                check_w_s_next = np.array(w_s[w_s_i+1:w_s_i+check_n])
                if next_w_t in check_w_s_next: # if right list gets ahead, find out by how much, adjust indices, return to top of loop and should get match
                    add_ind = np.where(check_w_s_next == next_w_t)[0][0]
                    w_s_i += 1 + add_ind # skip ahead of sentence word list
                    w_t_i += 1


    # Track how many words retained from initial script, ensure > 98%
    retention = len(final_words)/len(utter_words)
    if retention < word_retention_thresh:
        print("Retention problem")
        f_s_t = [None]
    else:
        s_t_df = pd.DataFrame({'word': final_words, 'sentence_ind': final_sentence_inds, 'time': final_times})
        f_s_t = pd.DataFrame((s_t_df.groupby(['sentence_ind'])['time'].first())).reset_index()
        f_s_t = np.array(f_s_t['time'])
        print(f"final_sentence_inds length: {len(np.unique(final_sentence_inds))}")
        print(f"final_sent_word_lists length: {len(final_sent_word_lists)}")

    return f_s_t, final_sent_word_lists

#### ---- End Spacy specific functions ---- ####


# after mapping words-times-sentences, return model input and output labels
def get_transition_labels(script, segment_times, splitter='yt_simple', nlp = None):
    if splitter == 'yt_simple':
        # parse transcript, get sentences and the corresponding times
        f_s_t = [round(line['start']) for line in script] #f_s_t = final sentence times
        f_s_w_l = [[line['text']] for line in script] # f_s_w_l = final list of sentence word lists

    if splitter == 'spacy':
        utter_words, utter_times, all_text = unpack_words(script)
        sentences = spacy_split(nlp, all_text)
        sent_words, sent_inds, sent_word_lists = word_sentence_map(sentences)
        f_s_t, f_s_w_l = map_sentence_time(utter_words, sent_words, utter_times, sent_inds, sent_word_lists)

    # for each timestamp, get the index of the closest utterance, assign transition label
    if (None in f_s_t) | (len(f_s_t) != len(f_s_w_l)):
        if None in f_s_t: print("f_s_t issue")
        else: print("Mismatch of sentence inds issue")
        transitions = None
    else:
        transitions = np.zeros(len(f_s_t))
        for t in segment_times:
            closest_idx = (np.abs(f_s_t - t)).argmin()
            transitions[closest_idx] = 1
    
    return f_s_w_l, transitions


def get_video_details(youtube, video_ids, splitter='yt_simple', nlp = None):
    all_video_stats = []
    
    # Step 1: Loop through pages of 50 videos each
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                    part='snippet',
                    id=','.join(video_ids[i:i+50]))
        response = request.execute()
        
        # Step 2: Loop through videos in page
        for video in tqdm(response['items']):
            video_stats = dict(Channel = video['snippet']['channelTitle'],
                               Title = video['snippet']['title'],
                               Description = video['snippet']['description']
                               )
            # Step 3: use separate API to get transcript with words and times
            script, script_type = get_transcript(video['id'])
            if script == None: # no transcript available, continue to next video in page
                continue
            else:
                sents = [line['text'] for line in script]
                word_list, word_times, all_words = unpack_words(script)
                video_stats.update({
                    'Transcript': script,
                    'Transcript-Source': script_type,
                    'Word_List': word_list,
                    'Word_Time_List': word_times,
                    'All_Words': all_words,
                    'Sentences': sents})
            
                # Step 4: Extract the timestamp/segment info from description
                segment_all, segment_times, segment_secs = extract_segments(video['snippet']['description'])
                if len(segment_all) == 0: # could not find a clear timestamp/segment chunk in description
                    continue # skip this episode, continue to next
                else: 
                    inputs, outputs = get_transition_labels(script, segment_secs, splitter, nlp)
                    video_stats.update({
                        'Sentence_Word_Lists': inputs,
                        'Transition_Labels': outputs,
                        'Segment_Times': segment_times,
                        'Segment_Times_Secs': segment_secs,
                        'Segments_All': segment_all})
                    all_video_stats.append(video_stats) # add to dict
    
    return all_video_stats