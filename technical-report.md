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
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
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

*Jupyter Notebook:* `HDBSCAN Fine Tuning.ipynb`

*Dataset:* `data_2022_long.csv` and `data_2020_long.csv`

We used the HDBSCAN Relative Validity Index to evaluate cluster quality in BERTopic. The relative validity index, as implemented in HDBSCAN, measures the clustering's consistency by considering both intra-cluster density and inter-cluster separation in a comparative manner. This provides a quantitative metric that helps assess the relative validity of different clustering solutions without requiring prior knowledge of the number of clusters (Campello et al., 2013). This metric is particularly suitable for density-based clustering methods like HDBSCAN, as it can account for density variations while balancing computational efficiency.

Below is the specific code portion containing the relative validity index in the randomized_hdbscan_search() function:

```python
dbvc_score = clusterer.relative_validity_
```
The validity index was preferred over traditional metrics like the silhouette score and coherence score, which are generally more suited for centroid-based clustering models such as k-means (Maas et al., 2021). These traditional metrics are less effective for density-based clustering algorithms, which can have clusters of varying shapes and densities. The validity index better aligns with the nature of HDBSCAN and offers a more reliable measure of clustering quality for such models. The chosen values for min_cluster_size and min_samples are the values that correspond to the highest validity index score (dbcv_score in the code).

**Training and Embedding Strategy**

*Jupyter Notebook:* `Bertopic Modelling.ipynb`

*Dataset:* `data_2022_long.csv` and `data_2020_long.csv`

To optimise the runtime and manage Google Colab’s limitations, we pre-generated the embeddings for each year’s dataset. This allowed us to bypass the computationally expensive embedding process during each model run. By saving the embeddings, we could rerun BERTopic without recomputing the text embeddings, reducing the risk of interrupted sessions due to Colab timeouts (Devlin et al., 2019).

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