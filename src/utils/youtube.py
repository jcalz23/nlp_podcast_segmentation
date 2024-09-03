"""
This module contains functions to interact with the YouTube API.
"""
# Standard imports
import sys
import os
import re
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Third-party imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer

# Custom imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from utils.aws import save_json_to_s3

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

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

def prepare_transcript(transcript: List[Dict], n_chunks: int = 20) -> List[Dict]:
    """
    Prepare the transcript by combining every n consecutive phrases into a single sentence,
    cleaning the sentences, and validating the timing.

    Args:
        transcript (List[Dict]): The raw transcript from YouTube.
        n_chunks (int): Number of phrases to combine into a single sentence. Default is 20.
    Returns:
        List[Dict]: A list of dictionaries, each containing a cleaned, combined sentence and its start time.

    Raises:
        ValueError: If the last sentence's start time is not within 30 seconds of the last phrase's start time.
    """
    sentence_texts = []
    sentence_start_times = []
    for i in range(0, len(transcript), n_chunks):
        # Get the next n phrases (or fewer if we're at the end)
        phrase_group = transcript[i:i+n_chunks]
        
        # Combine the text of all phrases in the group
        combined_text = " ".join(item['text'] for item in phrase_group)
        
        # Clean up the combined text
        combined_text = combined_text.strip()
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = re.sub(r'[^\w\s.,!?-]', '', combined_text)
        
        # Use sentence_tokenize to split into sentences, then rejoin
        sentences = sent_tokenize(combined_text)
        cleaned_sentences = [s.capitalize() for s in sentences]
        final_text = " ".join(cleaned_sentences)
        
        # Use the start time of the first phrase in the group
        start_time = phrase_group[0]['start']
        
        # Add the combined sentence and start time to the result
        sentence_texts.append(final_text)
        sentence_start_times.append(start_time)
    
    # Validation step
    last_phrase_time = transcript[-1]['start']
    last_sentence_time = sentence_start_times[-1]
    
    if last_phrase_time - last_sentence_time > 30:
        raise ValueError(f"Last sentence start time ({last_sentence_time:.2f}s) is more than 30 seconds before the last phrase time ({last_phrase_time:.2f}s)")
    
    return sentence_texts, sentence_start_times

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

def get_podcast_details(youtube_client: build, podcast_id: str, mode: str = 'train', n_chunks: int = 20) -> Dict:
    """
    Fetch podcast details from YouTube API and get the transcript.

    Args:
        youtube_client (googleapiclient.discovery.Resource): YouTube API client.
        podcast_id (str): YouTube podcast ID.
        mode (str): 'train' or 'inference'. Default is 'train'.
        n_chunks (int): Number of sentences to combine into a single sentence. Default is 20.
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
        sentences, sentence_start_times = prepare_transcript(transcript_raw, n_chunks)
        sentence_embeddings = embed_sentences(sentences)
    except ValueError as e:
        logging.error(f"Error in transcript preparation for podcast {podcast_id}: {str(e)}")
        return None

    # Get segments and target segment transition indicators for training
    segments = extract_segments(podcast_data['snippet']['description'])
    segment_indicators = create_segment_indicators(sentences, sentence_start_times, segments)

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
        return None, None
    else:
        # Save podcast details to S3
        s3_file_key = f"{S3_DATA_DIR}/podcasts/{podcast_id}.json"
        save_json_to_s3(result, S3_BUCKET_NAME, s3_file_key)
        return result, s3_file_key

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


def process_channels(channels: Dict[str, str], mode: str = 'train', n_chunks: int = 4) -> Dict[str, Dict]:
    """
    Process all channels, retrieve podcast IDs and details, and save them to individual JSON files.

    Args:
        channels (Dict[str, str]): A dictionary with channel names as keys and channel IDs as values.
        mode (str): 'train' or 'inference'. Default is 'train'.
        n (int): Number of sentences to combine into a single sentence. Default is 20.
    Returns:
        Dict[str, Dict]: A dictionary with podcast IDs as keys and channel ID, playlist ID, and details filepath as values.
    """
    # Create YouTube client
    youtube_client = create_youtube_client()

    # Iterate through channels, playlists, and podcasts to get details
    result = {}
    for channel_name, channel_id in tqdm(channels.items(), desc="Processing channels", unit="channel"):
        try:
            # Get playlist IDs for the channel
            playlist_ids = get_playlist_ids(youtube_client, channel_id)
            for playlist_id in tqdm(playlist_ids, desc=f"Playlists for {channel_name}", leave=False):
                podcast_ids = get_podcast_ids(youtube_client, playlist_id)
                for podcast_id in tqdm(podcast_ids, desc=f"Podcasts in playlist {playlist_id}", leave=False):
                    try:
                        # Get podcast details
                        podcast_details, s3_file_key = get_podcast_details(youtube_client, podcast_id, mode, n_chunks)

                        # Save podcast details to AWS S3
                        if podcast_details:                            
                            # Add to result dictionary
                            result[podcast_id] = {
                                'channel_name': channel_name,
                                'channel_id': channel_id,
                                'playlist_id': playlist_id,
                                'details_filepath': s3_file_key
                            }
                        else:
                            logging.info(f"Skipped podcast (no details): {podcast_id}")
                    except Exception as e:
                        logging.error(f"Error processing podcast {podcast_id}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing channel {channel_name}: {str(e)}")
        break
    
    # Save the result to a JSON file
    s3_file_key = f"{S3_DATA_DIR}/{PODCAST_METADATA_FILENAME}"
    save_json_to_s3(result, S3_BUCKET_NAME, s3_file_key)

    return result


def process_playlists(playlists: Dict[str, str], mode: str = 'train', n_chunks: int = 4, run_name: str = None) -> Dict[str, Dict]:
    """
    Process all playlists, retrieve podcast IDs and details, and save them to individual JSON files.

    Args:
        playlists (Dict[str, str]): A dictionary with channel names as keys and playlists IDs as values.
        mode (str): 'train' or 'inference'. Default is 'train'.
        n (int): Number of sentences to combine into a single sentence. Default is 20.
    Returns:
        Dict[str, Dict]: A dictionary with podcast IDs as keys and channel ID, playlist ID, and details filepath as values.
    """
    # Create YouTube client
    youtube_client = create_youtube_client()

    # Iterate through channels, playlists, and podcasts to get details
    result = {}
    # Get playlist IDs for the channel
    for channel_name, playlist_id in tqdm(playlists.items(), desc="Processing playlists"):
        podcast_ids = get_podcast_ids(youtube_client, playlist_id)
        for podcast_id in tqdm(podcast_ids, desc=f"Podcasts in playlist {playlist_id}", leave=False, total=len(podcast_ids)):
            try:
                # Get podcast details
                podcast_details, s3_file_key = get_podcast_details(youtube_client, podcast_id, mode, n_chunks)

                # Save podcast details
                if podcast_details:
                    result[podcast_id] = {
                        'channel_name': channel_name,
                        'playlist_id': playlist_id,
                        'details_filepath': s3_file_key
                    }
                else:
                    logging.info(f"Skipped podcast (no details): {podcast_id}")
            except Exception as e:
                logging.error(f"Error processing podcast {podcast_id}: {str(e)}")
    
    # Save the aggregate results to a JSON file
    s3_file_key = f"{S3_DATA_DIR}/{str(run_name)}/{PODCAST_METADATA_FILENAME}"
    save_json_to_s3(result, S3_BUCKET_NAME, s3_file_key)

    return result
