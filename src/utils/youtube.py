"""
This module contains functions to interact with the YouTube API.
"""
# Standard imports
import sys
import os
import re
import json
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Third-party imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer

# Custom imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download the punkt tokenizer for sentence splitting (only needed once)
nltk.download('punkt')

def create_youtube_client():
    """
    Create and return a YouTube API client.

    Returns:
        googleapiclient.discovery.Resource: YouTube API client.
    """
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY environment variable is not set")
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key)

def get_playlist_ids(youtube_client, channel_id):
    """
    Get the playlist IDs for a given YouTube channel ID.

    Args:
        youtube_client (googleapiclient.discovery.Resource): YouTube API client.
        channel_id (str): YouTube channel ID.

    Returns:
        List[str]: List of playlist IDs associated with the channel.
    """
    playlist_ids = []
    next_page_token = None

    while True:
        # Prepare the request to fetch playlists
        request = youtube_client.playlists().list(
            part='id',
            channelId=channel_id,
            maxResults=50,  # Maximum allowed by the API
            pageToken=next_page_token
        )
        
        # Execute the request
        response = request.execute()

        # Extract playlist IDs from the response and add them to the list
        playlist_ids.extend(item['id'] for item in response['items'])

        # Get the next page token for pagination
        next_page_token = response.get('nextPageToken')
        
        # If there's no next page, exit the loop
        if not next_page_token:
            break

    return playlist_ids


def get_podcast_ids(youtube_client: build, playlist_id: str) -> List[str]:
    """
    Get the podcast ids for a given playlist id.

    Args:
        youtube (googleapiclient.discovery.Resource): YouTube API client.
        playlist_id (str): YouTube playlist ID.

    Returns:
        List[str]: List of podcast IDs.
    """
    podcast_ids = []
    next_page_token = None

    while True:
        # Prepare the request to fetch playlist items
        request = youtube_client.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,  # Maximum allowed by the API
            pageToken=next_page_token
        )
        
        # Execute the request
        response = request.execute()

        # Extract podcast IDs from the response and add them to the list
        podcast_ids.extend(item['contentDetails']['videoId'] for item in response['items'])

        # Get the next page token for pagination
        next_page_token = response.get('nextPageToken')
        
        # If there's no next page, exit the loop
        if not next_page_token:
            break

    return podcast_ids

def get_transcript(podcast_id: str) -> Tuple[Optional[List[Dict]], str]:
    """
    Fetch the transcript for a YouTube podcast.

    Args:
        podcast_id (str): YouTube podcast ID.

    Returns:
        Tuple[Optional[List[Dict]], str]: A tuple containing the transcript (if available) and its type.
    """
    try:
        # Get list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(podcast_id)
        
        try:
            # Try to find a manually created English transcript
            transcript = transcript_list.find_manually_created_transcript(['en-US', 'en'])
            script_type = 'manual'
        except:
            # If not available, use auto-generated English transcript
            transcript = transcript_list.find_generated_transcript(['en-US', 'en'])
            script_type = 'auto'
        
        # Fetch and return the transcript
        return transcript.fetch(), script_type
    except Exception as e:
        print(f"Error fetching transcript for podcast {podcast_id}: {str(e)}")
        return None, 'unavailable'

def prepare_transcript(transcript: List[Dict], n: int = 4) -> Tuple[List[str], List[float]]:
    """
    Prepare the transcript by mapping words to times, splitting into sentences using NLTK,
    and assigning the time of the first word to each sentence. Validates the alignment with podcast duration.

    Args:
        transcript (List[Dict]): The raw transcript from YouTube.
        n (int): Number of sentences to combine into a single sentence. Default is 4.
    Returns:
        Tuple[List[str], List[float]]: A tuple containing a list of cleaned sentences and
        a list of corresponding start times in seconds.

    Raises:
        ValueError: If the last word's time is not within 30 seconds of the podcast end.
    """
    # Step 1: Map each word to its time
    word_time_map = []
    for item in transcript:
        words = word_tokenize(item['text'])
        for word in words:
            word_time_map.append((word, item['start']))

    # Step 2: Combine all words into a single text
    full_text = " ".join(word for word, _ in word_time_map)

    # Step 3: Use NLTK to split the text into sentences
    sentences = sent_tokenize(full_text)

    cleaned_sentences = []
    sentence_start_times = []

    word_index = 0
    for i in range(0, len(sentences), n):
        # Combine N sentences to reduce sequence length
        combined_sentence = " ".join(sentences[i:i+n])
        # Clean up the sentence
        cleaned_sentence = combined_sentence.strip()
        cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
        
        # Find the start time of the first word in this sentence
        sentence_words = word_tokenize(cleaned_sentence)
        while word_index < len(word_time_map) and word_time_map[word_index][0] != sentence_words[0]:
            word_index += 1
        
        if word_index < len(word_time_map):
            sentence_start_times.append(word_time_map[word_index][1])
        else:
            # If we can't find the start time, use the last known time
            sentence_start_times.append(word_time_map[-1][1])

        cleaned_sentences.append(cleaned_sentence)
        
        # Move the word_index to the end of the current sentence
        word_index += len(sentence_words)

    # Validate alignment with podcast duration
    last_word_time = word_time_map[-1][1]
    last_sentence_time = sentence_start_times[-1]

    if last_sentence_time - last_word_time > 30:
        raise ValueError(f"Last word time ({last_word_time:.2f}s) is more than 30 seconds before the podcast end ({duration_seconds:.2f}s)")

    return cleaned_sentences, sentence_start_times

def create_segment_indicators(sentences: List[str], sentence_start_times: List[float], segments: List[Dict]) -> List[int]:
    """
    Create an array of segment indicators for each sentence.

    Args:
        sentences (List[str]): List of sentences from the transcript.
        sentence_start_times (List[float]): List of start times for each sentence.
        segments (List[Dict]): List of segment dictionaries containing start times.

    Returns:
        List[int]: Array of segment indicators (1 for start of a new segment, 0 otherwise).
    """
    # Initialize the segment indicators array with zeros
    segment_indicators = [0] * len(sentences)
    segment_index = 0

    # Iterate through each sentence and its start time
    for i, sentence_time in enumerate(sentence_start_times):
        # Check if there are more segments to process
        if segment_index < len(segments):
            # Get the start time of the current segment
            segment_start_time = segments[segment_index]['start_secs']
            # If the sentence start time is greater than or equal to the segment start time
            if sentence_time >= segment_start_time:
                segment_indicators[i] = 1
                segment_index += 1

    return segment_indicators

def extract_segments(description: str) -> List[Dict]:
    """Extracts segments from a description using timestamps.

    Args:
        desc (str): The description to extract segments from.

    Returns:
        list: A list of dictionaries containing the start time, start time in seconds, and description for each segment.
    """
    # Initialize the result list
    result = []

    # Define the timestamp pattern
    timestamp_pattern = r'(\d{1,2}:?\d{1,2}:?\d{2}|\d{1,2}:\d{2})\s*-?\s*(.*)'
    
    # Check each line in the description for a timestamp pattern
    for line in description.split('\n'):
        match = re.match(timestamp_pattern, line.strip())
        if match:
            # Extract the time string and description
            time_str, segment_description = match.groups()
            
            # Normalize time format
            parts = time_str.split(':')
            if len(parts) == 2:
                time_str = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
            else:
                time_str = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"

            # Convert to seconds
            h, m, s = map(int, time_str.split(':'))
            start_secs = h * 3600 + m * 60 + s
            
            # Create a dictionary for the segment
            segment = {
                "start_time": time_str,
                "start_secs": start_secs,
                "segment_description": segment_description.strip()
            }
            
            # Append the segment to the result list
            result.append(segment)
    
    return result

def embed_sentences(sentences: List[str], model_name: str = 'all-MiniLM-L6-v2') -> List[List[float]]:
    """
    Embed a list of sentences using a pre-trained sentence transformer model.

    Args:
        sentences (List[str]): List of sentences to embed.
        model_name (str): Name of the pre-trained model to use. Default is 'all-MiniLM-L6-v2'.

    Returns:
        List[List[float]]: List of sentence embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings.tolist()

def get_podcast_details(youtube_client: build, podcast_id: str, mode: str = 'train', n: int = 4) -> Dict:
    """
    Fetch podcast details from YouTube API and get the transcript.

    Args:
        youtube_client (googleapiclient.discovery.Resource): YouTube API client.
        podcast_id (str): YouTube podcast ID.
        mode (str): 'train' or 'inference'. Default is 'train'.
        n (int): Number of sentences to combine into a single sentence. Default is 4.
    Returns:
        Dict: Dictionary containing podcast details and transcript information.
    """
    # Request podcast details from the API
    request = youtube_client.videos().list(
        part="snippet,contentDetails,statistics",
        id=podcast_id
    )
    response = request.execute()
    
    # Return empty dict if no podcast found
    if not response['items']:
        return None
    
    # Extract relevant information from the API response
    podcast_data = response['items'][0]
    
    # Get podcast duration
    duration = podcast_data['contentDetails']['duration']

    # Get transcript and prepare sentences with start times
    transcript_raw, script_type = get_transcript(podcast_id)
    try:
        sentences, sentence_start_times = prepare_transcript(transcript_raw, n)
        sentence_embeddings = embed_sentences(sentences)
    except ValueError as e:
        logging.error(f"Error in transcript preparation for podcast {podcast_id}: {str(e)}")
        return None

    # Get segments and target segment transition indicators for training
    if mode == 'train':
        segments = extract_segments(podcast_data['snippet']['description'])
        segment_indicators = create_segment_indicators(sentences, sentence_start_times, segments)
    else:
        segments = None
        segment_indicators = None

    # Need to ensure segments available for training
    result = {
        'title': podcast_data['snippet']['title'],
        'segments': segments,
        'duration': duration,
        'sentences': sentences,
        'sentence_start_times': sentence_start_times,
        'sentence_embeddings': sentence_embeddings,
        'segment_indicators': segment_indicators,
        'transcript_type': script_type,
    }
    if mode == 'train' and not segments:
        return None
    else:
        return result

def get_podcast_id_from_url(url: str) -> str:
    """
    Extract the podcast ID from a YouTube URL.

    Args:
        url (str): The YouTube podcast URL.

    Returns:
        str: The extracted podcast ID.

    Raises:
        ValueError: If the podcast ID cannot be extracted from the URL.
    """
    # Common YouTube URL patterns
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Unable to extract podcast ID from the provided URL.")

def process_channels(channels: Dict[str, str], mode: str = 'train', n: int = 4) -> Dict[str, Dict]:
    """
    Process all channels, retrieve podcast IDs and details, and save them to individual JSON files.

    Args:
        channels (Dict[str, str]): A dictionary with channel names as keys and channel IDs as values.
        mode (str): 'train' or 'inference'. Default is 'train'.
        n (int): Number of sentences to combine into a single sentence. Default is 4.
    Returns:
        Dict[str, Dict]: A dictionary with podcast IDs as keys and channel ID, playlist ID, and details filepath as values.
    """
    # Create YouTube client
    youtube_client = create_youtube_client()

    # Iterate through channels, playlists, and podcasts to get details
    result = {}
    for channel_name, channel_id in tqdm(channels.items(), total=len(channels)):
        logging.info(f"Processing channel: {channel_name}")
        try:
            # Get playlist IDs for the channel
            playlist_ids = get_playlist_ids(youtube_client, channel_id)
            for playlist_id in playlist_ids:
                # Get podcast IDs for the playlist
                podcast_ids = get_podcast_ids(youtube_client, playlist_id)
                for podcast_id in podcast_ids:
                    try:
                        # Get podcast details
                        podcast_details = get_podcast_details(youtube_client, podcast_id, mode, n)

                        # Save podcast details to a JSON file
                        if podcast_details:
                            # Create directory if it doesn't exist
                            os.makedirs('podcasts', exist_ok=True)
                            
                            # Save podcast details to a JSON file
                            filepath = f"podcasts/{podcast_id}.json"
                            with open(filepath, 'w') as f:
                                json.dump(podcast_details, f, indent=2)
                            
                            # Add to result dictionary
                            result[podcast_id] = {
                                'channel_name': channel_name,
                                'channel_id': channel_id,
                                'playlist_id': playlist_id,
                                'details_filepath': filepath
                            }
                            logging.info(f"Processed podcast: {podcast_id}")
                        else:
                            logging.warning(f"Skipped podcast (no details): {podcast_id}")
                    except Exception as e:
                        logging.error(f"Error processing podcast {podcast_id}: {str(e)}")
                break
        except Exception as e:
            logging.error(f"Error processing channel {channel_name}: {str(e)}")
            break
        break
    
    # Save the result to a JSON file
    with open(PODCAST_METADATA_FILENAME, 'w') as f:
        json.dump(result, f, indent=2)

    return result
