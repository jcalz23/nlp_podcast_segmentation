import os
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_openai(prompt, model, **kwargs):
    """
    Call OpenAI API with the given prompt and model.
    """
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = input("Please enter your OpenAI API key: ")

    client = openai.OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies topic transitions in podcast transcripts."},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return response
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise