import json
import os
import logging
import sys
import tiktoken
import argparse
import boto3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from utils.aws import read_json
from utils.openai import call_openai
from modeling.prompts import PODCAST_SEGMENTATION_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_transcript_prompt(video_id):
    """
    Fetch transcript from S3 and prepare input for LLM.
    Returns the prompt and its token count.
    """
    # Construct the file key for the transcript in S3
    file_key = f'{S3_DATA_DIR}/podcasts/{video_id}.json'
    transcript_data = read_json(S3_BUCKET_NAME, file_key)

    # Extract sentences and their start times from the transcript data
    sentences = transcript_data['sentences']
    sentence_start_times = [int(x) for x in transcript_data['sentence_start_times']]

    # Construct the prompt by combining timestamps and sentences
    transcript_prompt = ""
    for sentence, timestamp in zip(sentences, sentence_start_times):
        transcript_prompt += f"\n[{timestamp}] {sentence} "

    # Remove any trailing whitespace from the prompt
    transcript_prompt = transcript_prompt.strip()

    return transcript_prompt, transcript_data

def create_segment_dict(segments):
    """
    Create a dictionary of {start_secs: segment_description} from the segments list.
    
    Args:
    segments (list): List of dictionaries containing segment information.
    
    Returns:
    dict: Dictionary with start_secs as keys and segment_description as values.
    """
    return {segment['start_secs']: segment['segment_description'] for segment in segments}

def find_different_video_id(video_id):
    """
    Find a different video_id in the S3 bucket.
    """
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f'{S3_DATA_DIR}/podcasts/')
    different_video_id = None

    for obj in response.get('Contents', []):
        file_name = obj['Key'].split('/')[-1]
        if file_name.endswith('.json'):
            potential_video_id = file_name[:-5]  # Remove '.json' extension
            if potential_video_id != video_id:
                different_video_id = potential_video_id
                break
    logger.info(f"Found different video_id: {different_video_id}")

    return different_video_id

def prepare_inputs(video_id, few_shot=True):
    """
    Fetch transcript from S3 and prepare input for LLM.
    Returns the prompt and its token count.
    """
    # Get main video transcript and labels
    transcript_prompt, transcript_data = get_transcript_prompt(video_id)

    # Get one-shot example transcript and labels
    if few_shot:
        # Get transcript, targets, fill in prompt
        different_video_id = find_different_video_id(video_id)

        # Get transcript and targets
        ex_transcript_prompt, ex_transcript_data  = get_transcript_prompt(different_video_id)
        ex_target_output = create_segment_dict(ex_transcript_data['segments'])

        # Create few-shot prompt
        few_shot_template = """Here is an example from the same podcast:\nUser: {ex_transcript}\nAgent: {ex_target}"""
        few_shot_prompt = few_shot_template.format(ex_transcript=ex_transcript_prompt, ex_target=ex_target_output)
    else:
        few_shot_prompt = ""

    # Add to final prompt
    prompt = PODCAST_SEGMENTATION_PROMPT.format(
        transcript=transcript_prompt, few_shot_examples=few_shot_prompt
    )

    # Count tokens in the prompt using the GPT-4 tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt))
    logger.info(f"Token count: {token_count}")

    return prompt, transcript_data


def process_llm_output(response, sentence_start_times):
    """
    Process LLM output and create binary array of topic transitions.
    """
    try:
        content = response.choices[0].message.content
        topic_dict = json.loads(content)
    except (AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing LLM response: {e}")
        raise

    # Get final output as binary array
    binary_transitions = [0] * len(sentence_start_times)
    transition_times = [int(time) for time in topic_dict.keys()]

    for pred_transition_time in transition_times:
        if pred_transition_time in sentence_start_times:
            index = sentence_start_times.index(pred_transition_time)
            binary_transitions[index] = 1

    return binary_transitions, topic_dict

def llm_inference(video_id, model='gpt-4o', temperature=0.1, few_shot=True):
    # Prepare input
    prompt, transcript_data = prepare_inputs(video_id, few_shot)
    
    # Call OpenAI
    response = call_openai(prompt, model=model, temperature=temperature)
    
    # Process outputs
    sentence_start_times = [int(x) for x in transcript_data['sentence_start_times']]
    actual_topic_transitions = transcript_data['segment_indicators']
    pred_topic_transitions, topic_dict = process_llm_output(response, sentence_start_times)
    
    return pred_topic_transitions, actual_topic_transitions, topic_dict

