PODCAST_SEGMENTATION_PROMPT = """
You are an expert at analyzing podcast transcripts and identifying topic transitions. Your task is to segment a podcast transcript into distinct topics and provide brief summaries.

The transcript provided has timestamps (in seconds) at the beginning of each text chunk. Here's an example of how the transcript might look:

[0s] Host: Welcome to our podcast...
[130s] Host: Let's dive into our first topic...
...
[9000s] Guest: That's an interesting point about...
[10545s] Guest: That's an interesting point about...

Analyze the following podcast transcript and:
1. Identify the timestamps of topic transitions (around 20 topics total topics, distributed throughout the transcript).
2. Provide a brief summary (around 10 words) for each topic segment.

Focus on major shifts in the conversation, not minor subtopics within the same general discussion. There should be a maximum of 20 topics in the podcast and the segments should be roughly spread out throughout the transcript.

Example response format:
{{
    0: "Introduction and welcome to the podcast",
    576: "Discussion on the impact of artificial intelligence in healthcare",
    1234: "Guest's personal experience with AI",
    1900: "Current research and advancements in AI technology",
    3100: "Guest's perspective on AI's impact on society",
    3700: "Discussion on the role of AI in education",
    5500: "Ethical considerations and future implications of AI",
    6700: "Discussion on the role of AI in education",
    7300: "Guest's thoughts on AI's impact on privacy",
    8500: "Ethical considerations and future implications of AI",
    9100: "Guest's perspective on AI's impact on society",
    9700: "Discussion on the role of AI in education",
    10300: "Guest's thoughts on AI's impact on privacy",
    10919: "Exploring ethical considerations in AI development and implementation",
    12000: "End of podcast"
}}

Provide only the dictionary of topic transition timestamps and summaries, without any additional explanation.

{few_shot_examples}
Here is the new transcript:
User: {transcript}

Return a dictionary where the keys are timestamps (in seconds) marking the beginning of new topics or significant shifts in the conversation, and the values are brief summaries of those topics. Include the timestamp of the podcast's beginning ([0s]) in your response. Do not return more than 20 topics timestamps in the dictionary. Also include the timestamp of the podcast's end in the dictionary.
Only return a valid json string (RCF8259). Do provide any other commentary. Do not wrap the JSON in markdown such as ```json. Only use the data from the provided content.
"""



