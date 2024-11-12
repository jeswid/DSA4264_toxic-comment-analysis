# Technical Report

**Project: Problem 2 - Online Safety in Social Media**  
**Members: Eu Shae-Anne, Tiffany Irene Prasetio, Wang Tingyu Kelly, Duangporn Sirikulwattananon, Jessica Widyawati**  
Last updated on {last updated date}

## Section 1: Context

This project was initiated in response to rising concerns about the increasing toxicity and hatefulness observed in online discussions within Singaporean Reddit communities. Evidence indicates a growing trend of polarising and extreme content on social media, with recent survey results showing a notable rise in harmful content perceived by users (66% in the latest poll, up from 57% the previous year). The project’s main objective is to assess and quantify this trend in toxicity, specifically within Singapore subreddits, to inform policy decisions and potential partnerships aimed at mitigating online harm.

This project continues efforts by the Ministry of Digital Development and Innovation’s (MDDI) Online Trust and Safety department, which oversees online safety initiatives addressing issues like misinformation and toxic behavior on social media. Stakeholders such as the Ministry of Community, Culture, and Youth and social media companies (e.g., Meta, Google, TikTok) are also invested in this initiative. Previous groundwork includes discussions with these stakeholders, engagement with social media platforms, and exploratory studies on public perception and online safety, all of which support the current project scope.


## Section 2: Scope

### 2.1 Problem

The primary problem being addressed is the noticeable increase in hateful and toxic content on Singapore-focused Reddit threads, affecting users who interact with these platforms, particularly young individuals. The Online Trust and Safety department at MDDI, tasked with managing online hate speech and toxic content, is the main stakeholder facing this issue. This problem has grown over recent years, as indicated by survey data, posing a risk to the social fabric of Singapore—a diverse and multicultural society. Without intervention, there is a risk of increased societal polarisation, potentially influencing vulnerable individuals and compromising societal cohesion.

The importance of addressing this problem is underscored by the potential impact on public safety and well-being, particularly among youth. Metrics from the Online Safety Poll highlight an increase in perceived harmful content, illustrating the urgency of a targeted solution. Traditional methods of manually reviewing online content are insufficient given the scale, so data science and natural language processing (NLP) are essential to accurately assess and quantify trends in hateful speech. Machine learning techniques such as large language models (LLMs) offer scalability and precision, making it feasible to evaluate a vast amount of content (i.e., social media comments) in a meaningful and systematic way.

### 2.2 Success Criteria

Success for this project will be measured by its ability to deliver actionable insights that MDDI can use to influence policy or recommend interventions to social media platforms. Specifically, success will be defined by:

1. **Business Goal** : Providing a comprehensive analysis that highlights key trends and drivers behind the increase in hatefulness and toxicity on Singapore subreddits. This analysis should directly inform policy considerations and strategies for social media regulation or engagement with platforms.
2. **Operational Goal**: Developing a large-language model capable of accurately detecting and measuring toxicity levels in large datasets. This model should demonstrate robustness in categorising content by toxicity (inclusive of hatefulness), ideally achieving a high accuracy rate. 


### 2.3 Assumptions

The key assumptions underlying this project are:
1. **Data Sufficiency**: The dataset provided contains all Reddit comments from selected Singapore subreddits, which is assumed to be representative of the overall toxicity (inclusive of hatefulness) trends. The analysis assumes that this dataset is both complete and sufficiently broad to capture necessary patterns over time.
2. **Computational Resources**: Given that our team only had access to the free-tier Google Colab environment, we were limited by 16GB of RAM and intermittent GPU availability. This constraint impacted our ability to process the entire dataset at once, necessitating creative solutions such as splitting the data by year and pre-generating embeddings to avoid session timeouts. We also could not conduct extensive tuning on the entire dataset and all hyperparameters due to computational constraints.
3. **Stakeholder Support and Compliance**: Social media platforms and other agencies will be receptive to the project’s findings and recommendations, enabling any insights to be translated into tangible actions.
4. **Toxicity and hatefulness definition**: We included the definition of hatefulness inside the definition of toxicity, and we adopted a framework discussed under section 4 (LLM methodology) to classify a comment as toxic.

### 2.4 Final Workflow: NLP First, Then LLM
**Key Hypotheses**: We hypothesised that (1) toxicity on Reddit increased over time from 2020 to 2023, (2) distinct drivers of toxicity could be identified through topic modeling, and (3) sentiment analysis and topic modelling could complement each other in revealing these trends.

Initially, our workflow was to have the LLM team train their model and flag toxic comments, which we would then use for topic modelling. However, we realised that this would give us a skewed representation of popular topics on Reddit. For example, Topic A has a high proportion of toxic comments (e.g. 90% of comments are toxic) but is not very popular overall (e.g. 100 comments total). Very few Reddit users would interact with Topic A and thus it does not have a large potential impact on public safety. In comparison, a very popular topic B (e.g. 5000 comments) has a smaller proportion of toxic comments (e.g 10%). Running topic modelling on the toxic comments would make it easier to pick up Topic A (90 comments) as compared to Topic B (50 comments), but Topic B would have a larger potential impact on public safety as it is a much more popular topic. 

As such, we shifted to an **NLP First workflow**, where we ran topic modelling on the full dataset first and then passed the most prominent topics to the LLM team for toxicity labelling. This transition also allowed us to avoid bottlenecks in the model development process and ensured that we could identify the key discussion themes without waiting for the LLM to flag comments as toxic.

In the final overview of our workflow:
1. **Topic Modeling First**: We experimented with various different topic modelling strategies, and applied the best performing model BERTopic to the entire dataset, which allowed us to extract broad discussion themes across the comments. We identified the top 15 most prominent topics per year and passed these topics to the LLM team. 
2. **Toxicity Tagging by LLM**: We experimented with various different LLM models to identify which model aligned most closely with our definition of toxicity. We applied the best LLM model, Toxic BERT, to flag individual comments from the top 15 topics as toxic or non-toxic. This strategy allowed us to identify the drivers of toxicity in the dataset while avoiding the bias that early filtering might introduce.

## Section 3: NLP Methodology

### 3.1 Technical Assumptions

In this project, several technical assumptions influenced our the NLP model development process and methodology, including:

1. **Data Quality**: The Reddit dataset was unstructured, containing slang, informal language, and a mix of English and Chinese characters. It also contained inconsistent formatting (such as non-standard abbreviations and special characters), which affected our preprocessing steps. We also observed some imbalance in the volume of comments across different topics, which later influenced our choice of clustering methods.
2. **Sampling Methodology and NLP Parameter Tuning**: Given the massive size of our dataset, it would take multiple hours to run topic modelling on all the comments, which would be infeasible for the tuning step of our model. As such, we randomly sampled 2.5% of the dataset, stratified by year and by subreddit, to obtain a representative sample to fine tune the models’ HDBScan component. Our tuning was aimed towards identifying generalised, rather than specific topics. We assumed that all other parameters we did not tune (including other HDBScan default parameters and other modular comments of BERTopic) give optimal results for our topic modelling results using BERTopic. For more information on the parameters tuned, see section 3.4. 
3. **Final number of topics chosen**: For each year, we identified the top 15 topics, out of a list of around 200-250. These top 15 topics represented about 7-10% of the dataset. The number of topics was chosen based on a balance of 1) identifying the key drivers which had the most potential impact on public safety and 2) ensuring the topics represented a substantive proportion of all comments. 

### 3.2 Data


**Collection**

We collected data from Reddit spanning four years, from 2020 to 2023. This dataset was derived from public subreddits discussing relevant social and political topics. Each comment included a timestamp and was tagged with basic metadata (e.g., subreddit, comment length).

**Cleaning** 

*Jupyter Notebook:* `data_cleaning_updated.ipynb`

*Dataset:* `Reddit-Threads_2022-2023.csv` and `Reddit-Threads_2020-2021.csv`

The cleaning process involved multiple steps to ensure the data was ready for analysis:
- **Stopword Removal**: We removed common stopwords, focusing on those contextualized to Singaporean slang and local expressions. This helped eliminate non-informative words specific to our dataset's region (Casas et al., 2020).
- **Chinese Character Removal**: Comments containing significant portions of Chinese characters were removed as the focus was on English language discussions.
- **Punctuation and Special Characters**: We stripped non-alphanumeric characters to retain meaningful textual content. Emojis and special symbols, common in social media, were handled with caution as they sometimes contributed to sentiment expression (Bird et al., 2009).
- **Short Comments**: Comments shorter than 8 words were excluded to eliminate non-substantive entries.


**Feature Engineering**

For our analysis, we chose not to apply traditional feature engineering steps like tokenization or lemmatization, as BERTopic is designed to work directly with raw text. Unlike models such as LDA, which require manual preprocessing like tokenization and vectorization, BERTopic leverages pre-trained transformer models to generate contextual embeddings from the raw input (Grootendorst, 2020). These embeddings capture deeper semantic meaning and are more effective at handling informal language, slang, and misspellings common in Reddit comments. By bypassing tokenization and lemmatization, BERTopic can preserve the nuances of language that are often lost in traditional preprocessing methods, making it particularly suitable for social media data.

**Data Splitting**

To address our computational constraints, we split the data into separate datasets by year (2020, 2021, 2022, and 2023). This allowed us to analyse each year individually while ensuring that each dataset could be processed efficiently in Google Colab's free-tier environment. We did not perform train-test splits, as topic modelling aims to understand latent structures rather than predict outcomes (Blei et al., 2003).

### 3.3 Experimental Design

*Jupyter Notebook:* `Build and Test NLP.ipynb`

*Dataset:* `cleaned_data_2223.csv`

**Initial Workflow: SpaCy Sentiment Filtering and LDA**

Initially, our approach centred on flagging toxic comments using **SpaCy**'s sentiment analysis model. Since the LLM team was still training a model to identify toxicity, we opted to use SpaCy’s polarity scores to flag comments. Specifically, comments with a polarity score below -0.2 were classified as negative, with the assumption that highly negative sentiment would likely correlate with toxicity (Honnibal et al., 2020). About 7.8% of the data had a polarity score of less than -0.2. 

<div style="text-align: center;">
    <img src="report-images/space_histogram.png" alt="Histogram of Polarity Scores" />
    <p><em>Figure 1: Histogram of Spacy Polarity Scores</em></p>
</div>

 After filtering these negative comments, we applied **Latent Dirichlet Allocation (LDA)** for topic modelling. LDA was chosen for its simplicity and interpretability in extracting topics from a text corpus. Given that LDA is widely used for traditional topic modelling, and its computational efficiency, it was a natural first choice for our project (Blei et al., 2003).

 Optimal number of topics was tuned using grid search, and the optimal model was chosen with the highest coherence score.

 ```python
 #hyperparameter tuning using grid search
n_topics_list = [3, 5, 10, 15, 20, 25]
coherence_scores = []

texts_tokenized = [text.split() for text in texts_preprocessed]

# It should take around 15-30 seconds for each iteration
for n_topics in tqdm(n_topics_list):

    lda = LatentDirichletAllocation(n_components = n_topics, random_state = 2024)
    lda.fit(X)
    lda_topics = lda.components_
    lda_topics_words = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]] for topic in lda_topics]
    coherence_model_lda = CoherenceModel(topics = lda_topics_words,
                                         texts = texts_tokenized,
                                         dictionary = dictionary,
                                         coherence = 'c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Number of topics: {n_topics} | Coherence Score: {coherence_lda}")
    coherence_scores.append(coherence_lda)
``` 

However, after experimenting with different numbers of topics and coherence scores, we identified two key limitations:
- **Low Coherence Scores**: LDA struggled to produce meaningful topics in this dataset, likely due to the informal and unstructured nature of Reddit comments (Röder, Both, & Hinneburg, 2015). The coherence scores remained low, indicating that the generated topics lacked semantic clarity.

- **Bag-of-Words Limitations**: LDA’s bag-of-words approach could not effectively capture the context behind slang, informal phrasing, and non-standard grammar, which are common in Reddit discussions. This led to less meaningful topic separation (Blei et al., 2003).

```python
#fit the model with the optimal number of topics (highest coherence score)
lda = LatentDirichletAllocation(n_components = 10, random_state = 2024)
lda.fit(X)
no_top_words = 10
tf_feature_names = vectorizer.get_feature_names_out()
print_topics(lda_model, vectorizer)
```

<div style="text-align: center;">
    <img src="report-images/lda_topics.png" alt="topics" />
    <p><em>Figure 2: LDA Topics</em></p>
</div>

**Transition to BERTopic and Vader Sentiment Analysis**

Recognizing the limitations of LDA and SpaCy, we decided to transition to **Vader** for sentiment analysis and **BERTopic** for topic modelling.

- **Vader**: Unlike SpaCy’s general sentiment analysis model, Vader is specifically designed for social media text. It accounts for the informal language, slang, and even emoticons often found in Reddit comments, making it more accurate in identifying sentiment in this context (Hutto & Gilbert, 2014). We found that SpaCy’s sentiment analysis model was too rigid and generalised, leading to a less accurate classification of comments. Vader, on the other hand, provided more nuanced polarity scores and better handled informal language, aligning more effectively with our data. We classified comments that had a negative polarity score greater than 0.2 as being a toxic comment. This represented about 2.7% of the data. 

<div style="text-align: center;">
    <img src="report-images/vader_histogram.png" alt="Histogram of Polarity Scores" />
    <p><em>Figure 3: Histogram of Vader Negative Polarity Scores</em></p>
</div>

- **BERTopic**: BERTopic outperformed LDA by using contextual embeddings generated by transformers like BERT. These embeddings captured the more nuanced meanings in social media text and allowed for more meaningful clustering of topics. Moreover, BERTopic uses HDBScan, a density-based clustering method that performed better with the noisy and unstructured nature of Reddit comments (Grootendorst, 2020). The topics generated by BERTopic were much more coherent and picked up on local current affairs. 

<div style="text-align: center;">
    <img src="report-images/bert_topics.png" alt="topics" />
    <p><em>Figure 4: BERTopic Topics</em></p>
</div>

### 3.4 Results and Evaluation

*Jupyter Notebook:* `HDBSCAN Fine Tuning.ipynb`

*Dataset:* `data_2022_long.csv` and `data_2020_long.csv`

**BERTopic Tuning and HDBScan**

For this project, we aimed to find general, rather than specific topics in our data. For example, rather than finding topics such as “Football” or “Olympics”, we aimed to identify the broader topic of “Sports”. This is because broader topics are able to capture a larger proportion of Reddit comments. For clustering in BERTopic, we used HDBScan, which we tuned for the following parameters:

- **Min Cluster Size**: We experimented with values from 250 to 450, choosing 300 as the optimal size for capturing broad topic themes while maintaining cluster cohesion.
- **Min Samples**: We set this parameter to 5, balancing the need for reliable clusters without excluding too many comments.

The code below shows our setting up of the range of min cluster size and min sample values

```python
# Set up parameter distribution for tuning using grid search
min_samples_values = [5, 10, 15, 20]
min_cluster_sizes = [250, 300, 350, 400]
```
- **Other parameters**: We decided to keep default parameter options for computing distances (euclidean) and determining clusters (extent of mass). Euclidean distance is widely used for text data clustering due to its compatibility with vectorized representations (bertopic: embeddings). For Reddit’s text data, Euclidean effectively captures feature differences, making it a reliable choice without the need for alternative metrics. Reddit's content is dense and varied, with overlapping topics. The EOM method supports nuanced clustering and noise filtering, which aligns well with Reddit’s structured discussions and subtopics, generating clearer, major topic-oriented clusters.

The code below shows our tuning function, with the HDBSCAN model initiated keeping the metric = 'euclidean' and cluster_selection_method = 'eom' constant, while tuning for min_cluster_size and min_samples. We chose to randomly choose different combinations of parameter values to tune on so as to improve computational efficiency. 
```python
# Tuning HDBSCAN parameters using the true DBCV score from validity_index
def randomized_hdbscan_search(embeddings, min_samples_values, min_cluster_sizes, n_iter=10):
    random.seed(42)
    
    best_dbvc_score = -np.inf
    best_params = None
    best_model = None

    # Convert embeddings to dtype float64 for HDBSCAN compatibility
    embeddings = embeddings.astype(np.float64)

    # Sample parameter combinations randomly
    param_combinations = list(product(min_cluster_sizes, min_samples_values))
    sampled_combinations = random.sample(param_combinations, min(n_iter, len(param_combinations)))

    for min_cluster_size, min_samples in sampled_combinations:
        # Initialize and fit HDBSCAN with sampled parameters
        # Explicitly set gen_min_span_tree to True
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            gen_min_span_tree=True # This line is added to force the MST generation
        )
        clusterer.fit(embeddings)

        # Only consider models with more than one cluster
        if len(set(clusterer.labels_)) > 1:  # Exclude noise-only cases
            dbvc_score = clusterer.relative_validity_
            if dbvc_score > best_dbvc_score:
                best_dbvc_score = dbvc_score
                best_params = (min_cluster_size, min_samples)
                best_model = clusterer

    return best_model, best_params, best_dbvc_score
```
We tuned HDBScan on sampled dataset (2.5%) stratified for comments by year, and source of the subreddit, to save computational cost, due to computational constraints in terms of access to GPU and RAM space. Additionally, exploring additional distance metrics and clustering methods for HDBSCAN significantly increases computational demands, especially with high-dimensional Reddit data. Given resource limitations, we focused on key parameters (min_cluster_size, min_samples) that impact clustering performance without overextending processing capacity. 

**Evaluation Metric: HDBSCAN Relative Validity Index**

We used the HDBSCAN Relative Validity Index to evaluate cluster quality in BERTopic. The relative validity index, as implemented in HDBSCAN, measures the clustering's consistency by considering both intra-cluster density and inter-cluster separation in a comparative manner. This provides a quantitative metric that helps assess the relative validity of different clustering solutions without requiring prior knowledge of the number of clusters (Campello et al., 2013). This metric is particularly suitable for density-based clustering methods like HDBSCAN, as it can account for density variations while balancing computational efficiency.

Below is the specific code portion containing the relative validity index in the randomized_hdbscan_search() function:

```python
dbvc_score = clusterer.relative_validity_
```
The validity index was preferred over traditional metrics like the silhouette score and coherence score, which are generally more suited for centroid-based clustering models such as k-means (Maas et al., 2021). These traditional metrics are less effective for density-based clustering algorithms, which can have clusters of varying shapes and densities. The validity index better aligns with the nature of HDBSCAN and offers a more reliable measure of clustering quality for such models. The chosen values for min_cluster_size and min_samples are the values that correspond to the highest validity index score (dbcv_score) relative to other combinations of parameter values.

As we ran our tuning based on a randomly sampled dataset and implemented random search for parameter-tuning, running the tuning notebook may yield slightly different best parameters and best DBCV score.

**Training and Embedding Strategy**

*Jupyter Notebook:* `Bertopic Modelling.ipynb`

*Dataset:* `data_2022_long.csv` and `data_2020_long.csv`

To optimise the runtime and manage Google Colab’s limitations, we pre-generated the embeddings for each year’s dataset. This allowed us to bypass the computationally expensive embedding process during each model run. By saving the embeddings, we could rerun BERTopic without recomputing the text embeddings, reducing the risk of interrupted sessions due to Colab timeouts (Devlin et al., 2019). 

We chose the all-MiniLM-L6-v2 model from SentenceTransformer is an efficient, compact model well-suited for generating high-quality sentence embeddings for semantic similarity tasks. Its architecture, designed to balance speed and accuracy, enables effective clustering and topic modeling with minimal resource use, which is ideal for real-time applications and large datasets (Reimers & Gurevych, 2020).

```python
# Embedding model chosen
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# After embeddings were generated, we saved them as .npy files
np.save("flattened_embeddings_2020.npy", flattened_embeddings_2020)
np.save("flattened_embeddings_2022.npy", flattened_embeddings_2022)
```

**Running the optimal model and generating top 15 topics**

We ran our optimal BERTopic model on the rest of the dataset. Given limited GPU and RAM space, we ran the BERTopic model separately each year, and identified the top 15 topics of each year. This represented about 7-10% of the dataset.  

```python
#Best parameters
best_hdbscan = HDBSCAN(cluster_selection_method='eom', metric='euclidean',
        min_cluster_size=300, min_samples=5)

#Best bertopic model
topic_model = BERTopic(hdbscan_model=best_hdbscan)

#fit topic modelling to the preprocessed text data
topics, probabilities = topic_model.fit_transform(filtered_df["text"], np.array(filtered_embeddings))
```
For example, the topics identified for 2020 can be seen below. 
<div style="text-align: center;">
    <img src="report-images/bert_2020_topics.png" alt="2020 Top 15 Topics" />
    <p><em>Figure 5: 2020 Top 15 Topics</em></p>
</div>


As BERTopic uses UMAP which is stochastic in nature, running the model may result in slightly different results as compared to the ones our team derived. To our model results, see `2020top15topics.csv`, `2021top15topics.csv`, `2022top15topics.csv` and `2023top15topics.csv` along with the corresponding datasets labelled with topics. 

## Section 4: LLM Methodology

### 4.1 Toxicity Definition

**What is toxic?**

Before examining and labelling the toxicity and hatefulness of comments, it is important to define these terms to ensure consistency and accuracy in labelling. Toxicity refers to content that is rude, disrespectful, or profane, including the use of slurs. Hatefulness is defined as content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. In this project, we use toxicity models that combine both toxic and hateful elements in their output. Therefore, when we label a comment as toxic, it implies the presence of either toxicity or hatefulness.

### 4.2 Technical Assumptions

In this project, several technical assumptions influenced our LLM model selection and development process, including:

- **Data Preprocessing:** To ensure consistency with topic modelling, we worked on LLM models after processing the raw data in the same manner, including data cleaning and filtering of comments with more than 8 words. We assumed that the toxicity trend of comments with less than 8 words would not differ significantly with our current data selection. 

- **Sampling Methodology:** For sampling data to be manually labelled, we only sampled and stratified by subReddit thread from comments in 2022 and 2023. It was assumed that this sample would be representative and contain a substantial amount of toxic comments for us to differentiate LLM models’ performance and select the best one based on the metric we chose.

- **Comparison of Selected Topics:** To continue on toxicity analysis, we ran LLM on the data with topics labelled by NLP. As the top 15 topics were chosen for each year, there were some variations in the sub-topics selected across time. For example, under the transport category, buses and cars were chosen in some years while flights and cycling could appear in others. We assumed that these variations would not introduce bias or alter the overarching trends in our subsequent analysis.

### 4.3 Labelling Mechanism

**Rule-based Labelling**

Given that our dataset is sourced from Singapore’s main threads, where comments often include Singlish words and acronyms, we face challenges in adopting pretrained models that are not fine-tuned to detect toxic Singlish expressions. This language-specific variation means that toxic comments in Singlish may go undetected, as these models may not recognise such terms accurately. To bridge this gap, we developed a rule-based approach called the Singlish Toxic Dictionary, which includes a curated list of common toxic words in Singlish.
Before using the LLM model, each comment is checked against this dictionary. If a comment contains any toxic Singlish words, it will immediately be labelled as 'toxic.' Comments that remain unlabelled after this initial check are then processed by the pretrained model. This two-step approach improves the detection of toxicity in Singlish content, ensuring a more thorough and culturally relevant labelling process.


```python
singlish_toxic_dict = ['ahbeng', 'ahlian', 'baka', 'bloody hell', 'bloody idiot', 'bodoh', 'bo liao','buay pai seh', 'buay tahan', 'cb', 'cb kia', 'cb knn', 'cb', 'cb lao jia', 'cb lao knn', 'cb lor', 'cb sia', 'cb sia kia', 'ccb', 'chbye kia', 'chao chbye', 'chao chee bye', 'chow chibai', 'chow kar', 'chow tu lan', 'cibai', 'dumb ass', 'dumb', 'fuck', 'fuck you', 'fking', 'fucker', 'fucker sia', 'gila babi', 'gundu', 'hao lian kia', 'hopeless', 'idiot', 'idiot', 'ji bai', 'jiat lat', 'jialat kia', 'jibai', 'joker', 'kan', 'kan ni na', 'kena sai', 'kia si lang', 'knn', 'knn cb kia', 'knnccb', 'knnbccb', 'kns', 'kns cb', 'lampa', 'lan pa', 'lanjiao', 'lanjiao kia', 'lj', 'loser', 'nabei', 'no use kia', 'noob', 'pok gai', 'pui', 'sabo kia', 'sibei jialat', 'sibei sian', 'si gina', 'siol', 'slut', 'siao lang', 'stupid', 'suck', 'sua gu', 'tmd', 'tiok knn', 'tiok tiam', 'useless', 'what knn', 'what the fuck', 'wtf', 'wu liao kia', 'you die ah', 'you die']
```

**Manual Labelling Mechanism**

To help us identify the best-performing model for our dataset, we first aimed to test and compare all shortlisted LLM models on a labelled sample. For this, we manually labelled 200 comments, categorising them as toxic (marked as 1) or non-toxic (marked as 0) based on our toxicity criteria. This approach allowed us to evaluate model performance before applying the best model to the remaining dataset.

**Sampling for Manual Labelling**

*Jupyter Notebook:* `Get_Manual_Labelling_Data.ipynb`

*Dataset:* `Reddit-Threads_2022-2023.csv` and `cleaned_data_2223.csv`

Random sampling stratified by subreddit threads is performed on the cleaned `Reddit-Threads_2022-2023.csv`. Proportion of comments for each subreddit thread is calculated and used for determining the number of comments to be included in the manual sample, with rSingaporeRaw accounting for approximately 18% of total comments, 2% for rSingaporeHappening, and 80% for rSingapore. This ensures that the sampled data reflects the proportion of each subReddit thread in the whole dataset adequately.


### 4.4 Experimental Models

Given the variety of pre-trained toxic detection models available, the question arose as to which model would be most effective for our purposes. Therefore, we explored and experimented with several models, including Google Jigsaw's Perspective, the VADER sentiment analysis tool, and various BERT-based models.

**Perspective**

*Jupyter Notebook:* `Perspective_Labelling.ipynb`

*Dataset:* `manual_label_sample.xlsx`

Perspective provides a tool for assessing the potential impact of comments on a conversation, with attributes like *TOXICITY* and *IDENTITY_ATTACK*. This model outputs a probability score between 0 and 1, where a higher score indicates a greater likelihood of the comment exhibiting the specified attribute.

<u>Implementation</u>

After obtaining an API key, we configured a client to interact with Perspective’s comment analysis model. We structured each request to include the comment text, specify the language, and request scores for the attributes *TOXICITY* and *IDENTITY_ATTACK*—in line with our definition of toxicity, which covers both toxicity and hatefulness elements. Upon receiving the response, we parsed it to extract scores for each attribute and experimented with different threshold values to classify comments accurately. For example, if the *TOXICITY* or *IDENTITY_ATTACK* score exceeds 0.3, the comment is labelled as *toxic*. Else, the comment is classified as *non-toxic*.

```python
# API call for TOXICITY and IDENTITY_SCORE scores
def get_toxicity(text):
    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"

    # Data to send to the API
    data = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {
            'TOXICITY': {},
            'IDENTITY_ATTACK': {}
        }
    }

    # Send request to API
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    # Check for success and return result
    if response.status_code == 200:
        result = response.json()
        toxicity_score = result['attributeScores']['TOXICITY']['summaryScore']['value']
        identity_attack_score = result['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value']
        return toxicity_score, identity_attack_score
    
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
# To avoid hitting rate limits
def toxicity_delay(text):
    time.sleep(1)  
    return get_toxicity(text)

# Label toxicity
def label_toxicity(df, t):
    # Create a copy of the DataFrame within the function
    df_copy = df.copy()

    # Iterate over rows and label based on toxicity thresholds
    for index, row in df_copy.iterrows():
        if pd.isnull(row['result']):  # If result is None, check toxicity
            scores = toxicity_delay(row['text'])

            if scores is not None:
                toxic_score, identity_attack_score = scores

                # Set threshold for toxicity
                if toxic_score > t or identity_attack_score > t:
                    df_copy.at[index, 'result'] = 'toxic'
                else:
                    df_copy.at[index, 'result'] = 'non-toxic'
    
    return df_copy  # Return the modified copy
```

**Vader**

*Jupyter Notebook:* `Vadar_Labelling.ipynb`

*Dataset:* `manual_label_sample.xlsx`

Vader is a lexicon and rule-based text sentiment analysis tool that is specifically attuned to sentiments expressed in social media, making it a well-suited LLM model in analysing Reddit comments. It is the simplest and fastest model among the three. We were curious to see how the sentiment-based model would perform in this case, even though the model itself was not designed to identify toxic comments.

<u>Implementation</u>

- **Vader Model Selection:** We used the C.J. Hutto’s version of Vader model and retrieved polarity score from the model’s output. This version included customised negation words, idioms, and degree adverbs to modify or scale the original Vader score (Hutto, C.J. & Gilbert, E.E, 2014). It assigns a score between 0 and 1 to “negative”, “neutral”, and “positive” respectively for each target text. A compound score is then computed based on these 3 subscores, with a range of -1 to 1. A more negative score indicates a more intense negative sentiment.

```python
analyzer = SentimentIntensityAnalyzer() # C.J. Hutto's Implementation can be found in Vadar_Labelling.ipynb

def get_sentiment_scores(text):
    return analyzer.polarity_scores(text)

# Apply the sentiment analysis function to the 'text' column
sentiment_results = labelled['text'].apply(get_sentiment_scores)
```
- **Classification Thresholds for Toxicity:** The final compound score was compared against a threshold when labelling toxic comments. We explored thresholds from -0.4 to -0.8 and classified comments as toxic if the score was lower than the threshold.


**Huggingface Models**

*Jupyter Notebook:* `huggingface.ipynb`

*Dataset:* `manual_label_sample.xlsx`

To identify the most effective Hugging Face model for toxicity classification, we initially tested a diverse set of 13 models on our manually labelled sample. This selection included both models specifically trained for toxic comment detection and general sentiment analysis models, enabling us to compare performance across different types of language models. The list of tested models is as follows:

1. **Toxic Comment Classifiers:**

    - unitary/toxic-bert
    - unitary/unbiased-toxic-roberta
    - pykeio/lite-toxic-comment-classification
    - martin-ha/toxic-comment-model
    - JungleLee/bert-toxic-comment-classification
    - ZiruiXiong/bert-base-finetuned-toxic-comment-classification
    - longluu/distilbert-toxic-comment-classifier
    - prabhaskenche/toxic-comment-classification-using-RoBERTa

2. **General Sentiment Analysis Models:**

    - cardiffnlp/twitter-roberta-base-sentiment
    - roberta-base
    - bert-base-uncased
    - nlptown/bert-base-multilingual-uncased-sentiment
    - siebert/sentiment-roberta-large-english

Through preliminary testing, we observed that only three model general sentiment models: `unitary/toxic-bert`, `pykeio/lite-toxic-comment-classification` and `unitary/unbiased-toxic-roberta` showed consistent results suitable for our toxic classification task, with the first two performing best. The general sentiment analysis models often struggled to accurately detect nuanced toxic language, highlighting the advantage of using specialised toxic classifiers. Based on these findings, we narrowed our focus to the top 3 toxic comment classifiers, picking the one with the strongest metrics.

<u>Implementation</u>

- **Dynamic Text Chunking for Model Compatibility:** To accommodate each model’s maximum token limit of 512 tokens, we divided longer text entries into smaller segments, or “chunks.” This chunking approach allowed each model to process lengthy entries effectively without losing context. For each entry, the model calculated a toxicity score for each chunk, and these scores were averaged to generate a final toxicity score.

```python
def classify_toxicity_by_dynamic_chunks(text, tokenizer, classifier, max_length=512):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids'][0]
        total_tokens = len(input_ids)

        if total_tokens <= max_length:
            num_chunks = 1
            chunk_size = total_tokens
        else:
            num_chunks = math.ceil(total_tokens / max_length)
            chunk_size = math.ceil(total_tokens / num_chunks)

        toxicity_scores = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_tokens)
            chunk = tokenizer.decode(input_ids[start:end], skip_special_tokens=True)
            result = classifier(chunk)
            toxicity_scores.append(result[0]['score'])

        return sum(toxicity_scores) / len(toxicity_scores)
    except Exception as e:
        print(f"Warning: Error processing text chunk: {str(e)}")
        return 0.0
```

- **Classification Threshold for Toxicity:** To classify entries as *toxic* or *non-toxic*, we experimented with thresholds ranging from 0.1 to 0.9 in increments of 0.1. For each threshold, we generated predicted labels based on the model's toxicity scores in order to compare these predictions against our manually labelled data. Entries with scores above the chosen threshold were ultimately labelled as "toxic," while those below were classified as "non-toxic."

**Confusion Matrics and Evaluation Metrics**

*Jupyter Notebook:* `Perspective_Labelling.ipynb`, `Vadar_Labelling.ipynb` and `huggingface.ipynb`

*Dataset:* `manual_label_sample.xlsx`

To determine which model performed best for toxicity detection, we generated confusion matrix values, including True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN), for each Hugging Face model, along with two additional models, VADER and PerspectiveAI. 

<u>Metric Selection: Emphasis on Recall</u>

Instead of focusing solely on accuracy, we prioritised recall as our primary metric. This decision was based on the importance of minimising false negatives in toxicity detection. Given the context of this task, it is preferable to classify a non-toxic comment as toxic (false positive) than to miss labelling a toxic comment (false negative). High recall indicates that a model effectively identifies toxic content, even if some non-toxic content is incorrectly flagged as toxic.

In this context:

- **Recall** = TP / (TP + FN), measuring the model’s ability to correctly identify toxic comments.

- **Precision** and **accuracy** were also evaluated, but they held secondary importance compared to recall.

**Choosing the Model**

To select the final model for toxicity labelling, we compared the recall values across different experimental models. For Perspective, the threshold tested ranged from 0.2 to 0.5, as we found that values below 0.2 or above 0.5 did not improve the results. For Vader, thresholds between -0.4 and -0.8 were tested for classifying toxic comments. For the Hugging Face models—`unitary/toxic-bert` and `pykeio/lite-toxic-comment-classification—thresholds` from 0.1 to 0.9 were applied. The best performing thresholds for each model were 0.2, -0.4, 0.1, and 0.1 for Perspective, Vader, `toxic-bert`, and `lite-toxic-comment-classification`, respectively.

Upon comparing the performance across different models, we observed notable differences in both recall scores and computational efficiency.

While Perspective produced the highest recall score, it was highly time-consuming. Labelling 200 comments took approximately 5 minutes, making it difficult to scale, especially with a large volume of text data. This limitation was primarily due to the Queries Per Second (QPS) quota imposed by the API, which was capped at 1. Although we attempted to request an increase in the QPS, the tradeoff between improved speed and the project's timeline made it clear that this approach would not be feasible for our needs.

Vader, despite having the lowest recall score, was also deemed unsuitable for toxicity labelling. This is because it tends to score text based mainly on sentiment. For example, texts expressing negative emotions like sadness are likely to receive highly negative scores, even though they are not necessarily toxic in nature. This made Vader less reliable for our purpose.

In contrast, `unitary/toxic-bert` with a threshold of 0.1 emerged as the second-best performing model, with a recall score in the range of 0.65 to 0.75. While it did not outperform Perspective in recall, it offered a significantly faster processing time, making it more suitable for our project timeline. This recall range is adequate given our use case, as it reduces over-flagging of non-toxic content without requiring an extremely high threshold, which would be essential for more high-stakes applications like banking fraud detection. Although this threshold may introduce some false positives, its effectiveness in consistently detecting toxic content outweighs these drawbacks. Given its balance of performance and efficiency, we selected unitary/toxic-bert at a threshold of 0.1 as our final toxicity labelling model.

|          | Perspective (0.2) | Vader (-0.4)     | unitary/toxic-bert<br>(0.1)  | pykeio/lite-toxic-comment-<br>classification (0.1)
| -------- | ----------------- | ---------------- | ---------------------------- | ------------------- |
| Recall   | 0.73              | 0.48             | 0.65                         | 0.63                |


### 4.5 Labelling Toxicity

To uncover insights on overall toxicity trends, and identify specific topics contributing to rising toxicity levels, we apply our chosen model on the entire dataset and the sample curated by the NLP team.

**Whole Dataset**

After deciding on the final model, we run unitary/toxic-bert with a threshold at 0.1 on the whole Reddit dataset from Jan 2020 to Oct 2023 to investigate the general toxicity trend. These are cleaned data with comments more than 8 words to align with the NLP team’s data for a fairer comparison. 

**NLP Output**

Similarly, after receiving the top 15 common topics from each year from the NLP team, we ran the `unitary/toxic-bert` to label toxicity. We then decided to narrow down the topics for deeper analysis. Although some topics overlapped across the years, we were unable to automatically merge them due to the specificity of the content in the output. For example, a topic might be labelled as "1_football_games_sports_players" with associated terms given as ['football', 'games', 'sports', 'players', 'team', 'badminton', 'athletes', 'gt', 'world', 'gold']. As a result, we manually classified such topics into broader categories; in this case, labelling it under *"Sports"*.

We then conducted a final classification on a Miro board to better visualise the groupings. From this board (Figure X), we observed that certain topics only contained data from a single year. Such topics were deemed less meaningful for analysis, as they lacked relevance across multiple years. Instead, we prioritised groupings with data spanning 3 to 4 years, indicating sustained topic relevance. This approach aligns with our objective of deriving useful and impactful insights for recommendations. Ultimately, we focused on the following nine categories: Religion, SG Politics, Covid, Sports, Housing, Music, Gaming, Transport, and Media.

<div style="text-align: center;">
    <img src="miro_9_topics.jpg" alt="2020 Top 15 Topics" />
    <p><em>Figure 5: 2020 Top 15 Topics</em></p>
</div>


*Jupyter Notebook:* `HDBSCAN Fine Tuning.ipynb`

*Dataset:* `data_2022_long.csv` and `data_2020_long.csv`

**BERTopic Tuning and HDBScan**

## Section 5: Frontend 

We developed a frontend dashboard interface for our toxic comment analysis to enhance usability and interpretability for non-technical end users, such as policy makers. Key functions of this dashboard include adjustable date and topic filters, which allow users to tailor their search to specific periods or topics of interest. The frontend generates interactive graphs that visually represent trends in toxic comment frequency, providing an overview of levels of toxicity and hate over time. If policy makers wish to take a closer look at the actual toxic comments, they can use the ‘Toxic Comment’ feature to retrieve toxic comments from a specified topic and chosen month, allowing for further analysis. 

This dashboard allows end users to easily obtain an understanding of the levels of toxicity and hate in the most popular Reddit topics and take a deep dive into specific time periods and topics, making the tracker a valuable tool for maintaining online community standards. 

To run the frontend dashboard, run `streamlit run app.py` in your terminal. It requires the `topic_relabelled_full.csv` dataset.

### Section 6: Findings

*In this subsection, you should report the results from your experiments in a summary table, keeping only the most relevant results for your experiment (ie your best model, and two or three other options which you explored). You should also briefly explain the summary table and highlight key results.*

*Interpretability methods like LIME or SHAP should also be reported here, using the appropriate tables or charts.*

### 6.2 Discussion

*In this subsection, you should discuss what the results mean for the business user – specifically how the technical metrics translate into business value and costs, and whether this has sufficiently addressed the business problem.*

*You should also discuss or highlight other important issues like interpretability, fairness, and deployability.*

### 6.3 Recommendations

*In this subsection, you should highlight your recommendations for what to do next. For most projects, what to do next is either to deploy the model into production or to close off this project and move on to something else. Reasoning about this involves understanding the business value, and the potential IT costs of deploying and integrating the model.*

*Other things you can recommend would typically relate to data quality and availability, or other areas of experimentation that you did not have time or resources to do this time round.*