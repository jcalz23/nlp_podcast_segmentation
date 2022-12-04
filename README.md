# nlp_podcast_segmentation

### Abstract
This project aims to use natural language processing to divide podcast episodes into topic segments with timestamps. We primarily reference two previous works: one unsupervised topic segmentation approach on a corpus of meeting transcripts, and one supervised approach on a corpus of Wikipedia articles. This paper discusses the implementation of both unsupervised and supervised topic segmentation methods on a corpus of podcast transcripts from Youtube videos. All model variants are evaluated using the WindowDiff and Pk error metrics. Ultimately, the supervised learning method using transformer encoders produced the best results and in many cases, produced practically useful podcast topic segments.

### Motivation
With a vast and expanding library of podcasts, topic timestamps allow users to quickly identify which sections of each episode are worth listening to and which are worth skipping, saving the user time and increasing the quality of average content consumed. Despite the value provided to consumers, the majority of the podcast library does not provide them - likely due to the time required to manually parse an episode. This research aims to provide a tool for podcasters to quickly input their podcast transcript and receive a set of reliable topic segments for users to navigate across.

### Methods
All models built rely on pre-trained transformer models to extract sentence embeddings from podcast transcripts. The unsupervised approach uses the sequence of embedded sentences and for each timestep, it calculates the semantic similarity with the previous sentence. In development of the unsupervised approach, we explored different pretrained embedding models, similarity formulas, and thresholding logic to optimally segment podcast topics. The supervised approach attempts to use a recurrent neural network to automatically derive features across the sequence and classify topic transitions.

## File structure
nlp_podcast_segmentation

    .
    ├── scripts                 # All model scripts
    └── README.md

### Unsupervised Approach
In the unsupervised approach we usde two sets of files to determine the predictions and the performance metrics associated with them. The process is as follows:
1. Create Embeddings: To create embeddings for each Dataset, process the transcriptions with the create_embeddings.ipynb noteboook which will take each podcast and pass it through a pre-trained model that will generate a tensor per podcast and will save it in the ./data/embeddings/pre-trained model name/ folder
2. Predict and Measure: Create the predictions of the processed dataset with the process_embedings.ipynb notebook. This pipeline will take the embeddings generated in the previous step and calculate the predictions for each podcast and the dataset as a whole.
3. Calibrate Threshold for Unsupervised Model: Run the scripts/calibrate_threshold_z.ipynb determines optimal Z value for each train episode and averages to get the Z param for deriving thresholds on test set


### Supervised approach
For the supervised approach to generate and traind the models, we need to:
- Generate synthetic training episodes:
    - Run scripts/gen_seg_df_splitn5.ipynb to separate the topic segments of each train episode into rows
    - Run scripts/gen_synthetic_splitn5.ipynb to create new episodes from shuffling train segments and concatenating together up to max sequence length
- Train Supervised Models
	- scripts/train_transformer.ipynb has the final architecture configuration
	- scripts/train_bilstm_dl.ipynb and scripts/train_lstm_dl.ipynb contain the unsuccessful RNN model training code