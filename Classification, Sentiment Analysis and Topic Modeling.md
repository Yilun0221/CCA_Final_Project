# Classification, Sentiment Analysis and Topic Modeling

***Yilun Xu, MACSS Student, Division of the Social Sciences, University of Chicago***

## Import Packages


```python
import lucem_illud_2020 

import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import scipy #For hierarchical clustering and some visuals
#import scipy.cluster.hierarchy
import gensim#For topic modeling
import requests #For downloading our datasets
import numpy as np #for arrays
import pandas as pd #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn as sns #Makes the graphics look nicer

%matplotlib inline

import itertools
import json
import warnings
warnings.filterwarnings('ignore')
import random
from collections import Counter
import re
import string
from wordcloud import STOPWORDS,WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

nltk.download('vader_lexicon')
nltk.download('punkt')
from gensim.models import ldaseqmodel
import glob
random.seed(1234)
import pyLDAvis
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\mac\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\mac\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

## Data Preparation

In this part, we will read all the twitter data and do basic processing on them. The tweet data we used were the tweets related to Covid19 in March. We will extract some of the columns in these datasets and calculate their sentiment scores using polarity scores based on the tweet content. After processing all the tweets, we divided them into three data sets according to the time they were generated: early, middle and late.

The early tweets included all tweets before March 12. The reason we chose this date as the time node is that this day is the day when Covid19 was determined to be a global pandemic. Mid-term tweets include all data from March 12 to March 20, while late tweets include all data from March 21 to March 31. In different analysis sections, we will use Early, Middle and Late to refer to the three datasets.


```python
def new_sentence(text):
    '''
    This function is to clean a text with some regular expression methods. We
    want to return a new string with all marks and numbers deleted and all
    letters in the lower case format.

    Input:
    text (string): a text which needs to be processed.

    Ouput:
    new (string): a processed text.
    '''

    text = re.sub(r'[{}]+'.format(string.punctuation), '', text)
    text = re.sub(r'[{}]+'.format(string.digits), '', text)
    word_list = text.strip().lower().split()
    picked_words = [word for word in word_list if word not in STOPWORDS]
    new = ' '.join(picked_words)
    return new
```


```python
def add_sentiment(path):
    '''
    This function is to get a DataFrame from a csv, extract necessary columns 
    and add a column describing the sentiment category of the text.
    
    Input:
    path (string): the path of a csv to be read
    
    Output:
    df (DataFrame): a processed DataFrame
    '''
    df = pd.read_csv(path)
    df = df[['status_id','user_id','created_at','screen_name','text','followers_count','friends_count']]
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(new_sentence)
    SID = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['clean_text'].apply(SID.polarity_scores)
    return df
```


```python
t_early = add_sentiment('slim_tweets_0125\\2020-03-00 Coronavirus Tweets (pre 2020-03-12).CSV')
t_12 = add_sentiment('slim_tweets_0125\\2020-03-12 Coronavirus Tweets.CSV')
t_13 = add_sentiment('slim_tweets_0125\\2020-03-13 Coronavirus Tweets.CSV')
t_14 = add_sentiment('slim_tweets_0125\\2020-03-14 Coronavirus Tweets.CSV')
t_15 = add_sentiment('slim_tweets_0125\\2020-03-15 Coronavirus Tweets.CSV')
t_16 = add_sentiment('slim_tweets_0125\\2020-03-16 Coronavirus Tweets.CSV')
t_17 = add_sentiment('slim_tweets_0125\\2020-03-17 Coronavirus Tweets.CSV')
t_18 = add_sentiment('slim_tweets_0125\\2020-03-18 Coronavirus Tweets.CSV')
t_19 = add_sentiment('slim_tweets_0125\\2020-03-19 Coronavirus Tweets.CSV')
t_20 = add_sentiment('slim_tweets_0125\\2020-03-20 Coronavirus Tweets.CSV')
t_middle = pd.concat([t_12,t_13,t_14,t_15,t_16,t_17,t_18,t_19,t_20],axis=0)
t_21 = add_sentiment('slim_tweets_0125\\2020-03-21 Coronavirus Tweets.CSV')
t_22 = add_sentiment('slim_tweets_0125\\2020-03-22 Coronavirus Tweets.CSV')
t_23 = add_sentiment('slim_tweets_0125\\2020-03-23 Coronavirus Tweets.CSV')
t_24 = add_sentiment('slim_tweets_0125\\2020-03-24 Coronavirus Tweets.CSV')
t_25 = add_sentiment('slim_tweets_0125\\2020-03-25 Coronavirus Tweets.CSV')
t_26 = add_sentiment('slim_tweets_0125\\2020-03-26 Coronavirus Tweets.CSV')
t_27 = add_sentiment('slim_tweets_0125\\2020-03-27 Coronavirus Tweets.CSV')
t_28 = add_sentiment('slim_tweets_0125\\2020-03-28 Coronavirus Tweets.CSV')
t_29 = add_sentiment('slim_tweets_0125\\2020-03-29 Coronavirus Tweets.CSV')
t_30 = add_sentiment('slim_tweets_0125\\2020-03-30 Coronavirus Tweets.CSV')
t_31 = add_sentiment('slim_tweets_0125\\2020-03-31 Coronavirus Tweets.CSV')
t_late = pd.concat([t_21,t_22,t_23,t_24,t_25,t_26,t_27,t_28,t_29,t_30,t_31],axis=0)
tweets = [t_early,t_middle,t_late]
```

## WordCloud

In this part, we want to make a word cloud for the tweets in each stage. This section serves more like an EDA process to help us get a basic understanding of the tweets.


```python
plt.figure(figsize=(8,6))
wc = WordCloud(background_color="white", max_words=500, width= 1200, height = 1200, mode ='RGBA', scale=.5).generate(t_early['text'].sum())
plt.imshow(wc)
plt.axis("off")
plt.title('WordCloud for Tweets in Early March',size = 16,y = 1.02)
```




    Text(0.5, 1.02, 'WordCloud for Tweets in Early March')




![png](output_11_1.png)



```python
plt.figure(figsize=(8,6))
wc = WordCloud(background_color="white", max_words=500, width= 1200, height = 1200, mode ='RGBA', scale=.5).generate(t_middle['text'].sum())
plt.imshow(wc)
plt.axis("off")
plt.title('WordCloud for Tweets in Middle March',size = 16,y = 1.02)
```




    Text(0.5, 1.02, 'WordCloud for Tweets in Middle March')




![png](output_12_1.png)



```python
plt.figure(figsize=(8,6))
wc = WordCloud(background_color="white", max_words=500, width= 1200, height = 1200, mode ='RGBA', scale=.5).generate(t_late['text'].sum())
plt.imshow(wc)
plt.axis("off")
plt.title('WordCloud for Tweets in Late March',size = 16,y = 1.02)
```




    Text(0.5, 1.02, 'WordCloud for Tweets in Late March')




![png](output_13_1.png)


## Clustering

### Cluster Number Selection

In this part, we want to pick the optimal cluster number. We will use the average silhouette score as the standard. We will pick the best one from the list: 3, 4, 5 and 6. We will calculate the Silhouette score for each possible cluster number for each tweet dataset in the three stages, and finally pick the best number.


```python
def plotSilhouette(n_clusters, X, reduced_data, pca):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,5))
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = cmap(float(i) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    projected_centers = pca.transform(centers)
    # Draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()
    print("For n_clusters = {}, The average silhouette_score is : {:.3f}".format(n_clusters, silhouette_avg))
```

#### Early


```python
possible_numbers = [3,4,5,6]
exampleTFVectorizer_early = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
exampleTFVects_early = exampleTFVectorizer_early.fit_transform(t_early['text'])
X = exampleTFVects_early.toarray()
PCA = sklearn.decomposition.PCA
pca_early = PCA(n_components = 2).fit(exampleTFVects_early.toarray())
reduced_data_early = pca_early.transform(exampleTFVects_early.toarray())
for i in possible_numbers:
    plotSilhouette(i, X, reduced_data_early,pca_early)
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_19_1.png)


    For n_clusters = 3, The average silhouette_score is : 0.008
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_19_4.png)


    For n_clusters = 4, The average silhouette_score is : 0.008
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_19_7.png)


    For n_clusters = 5, The average silhouette_score is : 0.010
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_19_10.png)


    For n_clusters = 6, The average silhouette_score is : 0.011
    

#### Middle


```python
exampleTFVectorizer_middle = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
exampleTFVects_middle = exampleTFVectorizer_middle.fit_transform(t_middle['text'])
X = exampleTFVects_middle.toarray()
PCA = sklearn.decomposition.PCA
pca_middle = PCA(n_components = 2).fit(exampleTFVects_middle.toarray())
reduced_data_middle = pca_middle.transform(exampleTFVects_middle.toarray())
for i in possible_numbers:
    plotSilhouette(i, X, reduced_data_middle, pca_middle)
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_21_1.png)


    For n_clusters = 3, The average silhouette_score is : 0.009
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_21_4.png)


    For n_clusters = 4, The average silhouette_score is : 0.011
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_21_7.png)


    For n_clusters = 5, The average silhouette_score is : 0.013
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_21_10.png)


    For n_clusters = 6, The average silhouette_score is : 0.013
    

#### Late


```python
exampleTFVectorizer_late = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
exampleTFVects_late = exampleTFVectorizer_late.fit_transform(t_late['text'])
X = exampleTFVects_late.toarray()
PCA = sklearn.decomposition.PCA
pca_late = PCA(n_components = 2).fit(exampleTFVects_late.toarray())
reduced_data_late = pca_late.transform(exampleTFVects_late.toarray())
for i in possible_numbers:
    plotSilhouette(i, X, reduced_data_late, pca_late)
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_23_1.png)


    For n_clusters = 3, The average silhouette_score is : 0.010
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_23_4.png)


    For n_clusters = 4, The average silhouette_score is : 0.012
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_23_7.png)


    For n_clusters = 5, The average silhouette_score is : 0.014
    

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](output_23_10.png)


    For n_clusters = 6, The average silhouette_score is : 0.016
    

According to the above results, we found the for all the three tweets datasets, the best cluster number is 6.

### K-means and Hierarchical Clustering with Wald's Method

According to the previous analysis, when we are using K-means and Hierarchical Clustering with Wald's Method to cluster the datasets, we will assign a total of 6 different sentiment categories to each observation, including negative, pretty negative, slightly negative, slightly positive, pretty positive and positive. We will see how these two methods work in the three datasets respectively.


```python
colors = {'negative':'purple','pretty negative':'blue','slightly negative':'green','slightly positive':'yellow',
         'pretty positive':'orange','positive':'red'}
colors = {0:'purple',1:'blue',2:'green',3:'yellow',4:'orange',5:'red'}
```


```python
def convert_cluster(score):
    '''
    This function is to convert a sentiment analysis score into a sentiment
    analysis label. The ranges for different categories are as follows:
    negative: [-1, -0.2)
    pretty negative: [-0.2, -0.05)
    slightly negative: [-0.05,0)
    slightly positive: [0, 0.05)
    pretty positive;[0.05,0.2)
    positive:[0.2, 1]

    Input:
    score (number): the analysis score

    Output:
    the analysis category (string)
    '''
    if score < -0.2:
        return 'negative'
    if score >= -0.2 and score <-0.05:
        return 'pretty negative'
    if score >=-0.05 and score <0:
        return 'slightly negative'
    if score >= 0 and score <0.05:
        return 'slightly positive'
    if score >= 0.05 and score <0.2:
        return 'pretty positive'
    return 'positive'
for df in tweets:
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: convert_cluster(x['compound']))
```

#### early


```python
numClusters = 6
km_early = sklearn.cluster.KMeans(n_clusters=numClusters, init='k-means++')
km_early.fit(exampleTFVects_early)
exampleTFVectorizer_early.vocabulary_.get('trump')
```




    904



We trained the model, and we can consult the word 'trump' in the model.


```python
exampleTransformer_early = sklearn.feature_extraction.text.TfidfTransformer().fit(exampleTFVects_early)
exampleTF_early = exampleTransformer_early.transform(exampleTFVects_early)
# Let's see the shape of the transformed vectors.
print(exampleTF_early.shape)
```

    (4144, 1000)
    


```python
try:
    print(exampleTFVectorizer_early.vocabulary_['covid'])
except KeyError:
    print('This word is missing')
    print('The available words are: {} ...'.format(list(ngTFVectorizer.vocabulary_.keys())[:10]))
```

    211
    

We can also consult other words in the model, and we can also get the list for available words in the model.


```python
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("For our clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(t_early['sentiment'], km_early.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(t_early['sentiment'], km_early.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(t_early['sentiment'], km_early.labels_)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(t_early['sentiment'], km_early.labels_)))
```

    The available metrics are: ['ConfusionMatrixDisplay', 'PrecisionRecallDisplay', 'RocCurveDisplay', 'SCORERS', 'accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'balanced_accuracy_score', 'brier_score_loss', 'calinski_harabasz_score', 'calinski_harabaz_score', 'check_scoring', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'davies_bouldin_score', 'dcg_score', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'max_error', 'mean_absolute_error', 'mean_gamma_deviance', 'mean_poisson_deviance', 'mean_squared_error', 'mean_squared_log_error', 'mean_tweedie_deviance', 'median_absolute_error', 'multilabel_confusion_matrix', 'mutual_info_score', 'nan_euclidean_distances', 'ndcg_score', 'normalized_mutual_info_score', 'pairwise', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_chunked', 'pairwise_kernels', 'plot_confusion_matrix', 'plot_precision_recall_curve', 'plot_roc_curve', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss']
    For our clusters:
    Homogeneity: 0.018
    Completeness: 0.018
    V-measure: 0.018
    Adjusted Rand Score: 0.016
    


```python
terms = exampleTFVectorizer_early.get_feature_names()
print("Top terms per cluster:")
order_centroids = km_early.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
```

    Top terms per cluster:
    Cluster 0:
     covid
     19
     coronavirus
     coronavirusoutbreak
     pandemic
     outbreak
     covid_19
     covid2019
     cases
     health
    
    
    Cluster 1:
     people
     coronavirus
     coronavirusoutbreak
     don
     italy
     just
     need
     virus
     think
     infected
    
    
    Cluster 2:
     coronavirus
     covid_19
     covid2019
     covidー19
     corona
     coronavirusupdate
     coronavirusoutbreak
     china
     covid
     pandemic
    
    
    Cluster 3:
     amp
     coronavirus
     just
     pandemic
     like
     time
     don
     health
     virus
     trump
    
    
    Cluster 4:
     cases
     confirmed
     new
     total
     coronavirus
     deaths
     coronavirusoutbreak
     reported
     china
     number
    
    
    Cluster 5:
     coronavirusoutbreak
     coronavirus
     covid2019
     coronavirusupdate
     corona
     covidー19
     virus
     realdonaldtrump
     just
     trump
    
    
    


```python
components = pca_early.components_
keyword_ids_early = list(set(order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words(features) from the kmeans model.
words_early = [terms[i] for i in keyword_ids_early]#Turn the ids into words.
x = components[:,keyword_ids_early][0,:] #Find the coordinates of those words in the biplot.
y = components[:,keyword_ids_early][1,:]
```


```python
colordict = {'negative': 'blue','pretty negative': 'green','slightly negative': 'yellow',
             'slightly positive':'pink','pretty positive':'orange','positive':'red'}
colors = [colordict[c] for c in t_early['sentiment']]
print("The categories' colors are:\n{}".format(colordict.items()))
```

    The categories' colors are:
    dict_items([('negative', 'blue'), ('pretty negative', 'green'), ('slightly negative', 'yellow'), ('slightly positive', 'pink'), ('pretty positive', 'orange'), ('positive', 'red')])
    


```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data_early[:, 0], reduced_data_early[:, 1], color = colors, alpha = 0.5, label = colors)
plt.xticks(())
plt.yticks(())
plt.title('True Classes (Early)')
plt.show()
```


![png](output_39_0.png)



```python
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data_early[:, 0], reduced_data_early[:, 1], color = colors, alpha = 0.3, label = colors)
for i, word in enumerate(words_early):
    ax.annotate(word, (x[i],y[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes (Early)')
plt.show()
```


![png](output_40_0.png)


In the above two figures, we get the clustered true labels of the observations, one without annotated vocabulary and one with annotated vocabulary.


```python
sentimentCategories = list(colordict.keys())
colors_p = [colordict[sentimentCategories[l]] for l in km_early.labels_]
```


```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reduced_data_early[:, 0], reduced_data_early[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters (Early)\n k = 6')
plt.show()
```


![png](output_43_0.png)


In the above figure, we get the clustered predicted labels of the observations.


```python
hier_early = t_early.sample(n=200)
hierTFVectorizer_early = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
hierTFVects_early = hierTFVectorizer_early.fit_transform(hier_early['text'])
CoocMat_early = hierTFVects_early * hierTFVects_early.T
CoocMat_early.setdiag(0)
examplelinkage_matrix_early = scipy.cluster.hierarchy.ward(CoocMat_early.toarray())
ax = scipy.cluster.hierarchy.dendrogram(examplelinkage_matrix_early, p=6, truncate_mode='level')
plt.title('Hierarchical Clustering (Early)',y=1.02,size=16)
plt.show()
```


![png](output_45_0.png)



```python
hierarchicalClusters_early = scipy.cluster.hierarchy.fcluster(examplelinkage_matrix_early, 6, 'maxclust')
print("For our complete clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(hier_early['sentiment'], hierarchicalClusters_early)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(hier_early['sentiment'], hierarchicalClusters_early)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(hier_early['sentiment'], hierarchicalClusters_early)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(hier_early['sentiment'], hierarchicalClusters_early)))
```

    For our complete clusters:
    Homogeneity: 0.041
    Completeness: 0.043
    V-measure: 0.042
    Adjusted Rand Score: 0.004
    

We applied Hierarchical Clustering with Wald's Method on the Early tweets, with the original sampled with n = 200.

#### Middle


```python
numClusters = 6
km_middle = sklearn.cluster.KMeans(n_clusters=numClusters, init='k-means++')
km_middle.fit(exampleTFVects_middle)
exampleTFVectorizer_middle.vocabulary_.get('trump')
```




    903




```python
exampleTransformer_middle = sklearn.feature_extraction.text.TfidfTransformer().fit(exampleTFVects_middle)
#train
exampleTF_middle = exampleTransformer_middle.transform(exampleTFVects_middle)
print(exampleTF_middle.shape)
```

    (38555, 1000)
    


```python
try:
    print(exampleTFVectorizer_middle.vocabulary_['covid'])
except KeyError:
    print('This word is missing')
    print('The available words are: {} ...'.format(list(ngTFVectorizer.vocabulary_.keys())[:10]))
```

    198
    


```python
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("For our clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(t_middle['sentiment'], km_middle.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(t_middle['sentiment'], km_middle.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(t_middle['sentiment'], km_middle.labels_)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(t_middle['sentiment'], km_middle.labels_)))
```

    The available metrics are: ['ConfusionMatrixDisplay', 'PrecisionRecallDisplay', 'RocCurveDisplay', 'SCORERS', 'accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'balanced_accuracy_score', 'brier_score_loss', 'calinski_harabasz_score', 'calinski_harabaz_score', 'check_scoring', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'davies_bouldin_score', 'dcg_score', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'max_error', 'mean_absolute_error', 'mean_gamma_deviance', 'mean_poisson_deviance', 'mean_squared_error', 'mean_squared_log_error', 'mean_tweedie_deviance', 'median_absolute_error', 'multilabel_confusion_matrix', 'mutual_info_score', 'nan_euclidean_distances', 'ndcg_score', 'normalized_mutual_info_score', 'pairwise', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_chunked', 'pairwise_kernels', 'plot_confusion_matrix', 'plot_precision_recall_curve', 'plot_roc_curve', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss']
    For our clusters:
    Homogeneity: 0.010
    Completeness: 0.009
    V-measure: 0.009
    Adjusted Rand Score: 0.001
    


```python
terms = exampleTFVectorizer_middle.get_feature_names()
print("Top terms per cluster:")
order_centroids = km_middle.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
```

    Top terms per cluster:
    Cluster 0:
     coronavirusoutbreak
     people
     just
     like
     time
     cases
     trump
     virus
     new
     don
    
    
    Cluster 1:
     covid_19
     coronaviruspandemic
     coronaoutbreak
     coronavirusupdate
     corona
     just
     people
     virus
     like
     covid19
    
    
    Cluster 2:
     amp
     covid19
     people
     health
     covid_19
     time
     like
     trump
     need
     help
    
    
    Cluster 3:
     covid
     19
     covid19
     coronavirusoutbreak
     covid_19
     cases
     test
     pandemic
     health
     frontline
    
    
    Cluster 4:
     covid19
     pandemic
     people
     time
     help
     just
     health
     need
     like
     friends
    
    
    Cluster 5:
     home
     stay
     safe
     covid19
     work
     working
     covid_19
     people
     time
     don
    
    
    


```python
components = pca_middle.components_
keyword_ids_middle = list(set(order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words from the kmeans model.
words_middle = [terms[i] for i in keyword_ids_middle]#Turn the ids into words.
x = components[:,keyword_ids_middle][0,:] #Find the coordinates of those words in the biplot.
y = components[:,keyword_ids_middle][1,:]
```


```python
colordict = {'negative': 'blue','pretty negative': 'green','slightly negative': 'yellow',
             'slightly positive':'pink','pretty positive':'orange','positive':'red'}
colors = [colordict[c] for c in t_middle['sentiment']]

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data_middle[:, 0], reduced_data_middle[:, 1], color = colors, alpha = 0.5, label = colors)
for i, word in enumerate(words_middle):
    ax.annotate(word, (x[i],y[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes (Middle)')
plt.show()
```


![png](output_55_0.png)



```python
colors_p = [colordict[sentimentCategories[l]] for l in km_middle.labels_]
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reduced_data_middle[:, 0], reduced_data_middle[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters (Middle)\n k = 6')
plt.show()
```


![png](output_56_0.png)


In the above two figures, we get the clustered true and predicted labels of the observations of Middle tweets.


```python
hier_middle = t_middle.sample(n=200)
hierTFVectorizer_middle = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
hierTFVects_early = hierTFVectorizer_middle.fit_transform(hier_middle['text'])
CoocMat_middle = hierTFVects_early * hierTFVects_early.T
CoocMat_middle.setdiag(0)
examplelinkage_matrix_middle = scipy.cluster.hierarchy.ward(CoocMat_middle.toarray())
ax = scipy.cluster.hierarchy.dendrogram(examplelinkage_matrix_middle, p=6, truncate_mode='level')
plt.title('Hierarchical Clustering (Middle)',y=1.02,size=16)
plt.show()
```


![png](output_58_0.png)



```python
hierarchicalClusters_middle = scipy.cluster.hierarchy.fcluster(examplelinkage_matrix_middle, 6, 'maxclust')
print("For our complete clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(hier_middle['sentiment'], hierarchicalClusters_middle)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(hier_middle['sentiment'], hierarchicalClusters_middle)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(hier_middle['sentiment'], hierarchicalClusters_middle)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(hier_middle['sentiment'], hierarchicalClusters_middle)))
```

    For our complete clusters:
    Homogeneity: 0.047
    Completeness: 0.043
    V-measure: 0.045
    Adjusted Rand Score: 0.013
    

We applied Hierarchical Clustering with Wald's Method on the Middle tweets, with the original sampled with n = 200.

#### Late


```python
numClusters = 6
km_late = sklearn.cluster.KMeans(n_clusters=numClusters, init='k-means++')
km_late.fit(exampleTFVects_late)
exampleTFVectorizer_late.vocabulary_.get('trump')
```




    901




```python
exampleTransformer_late = sklearn.feature_extraction.text.TfidfTransformer().fit(exampleTFVects_late)
#train
exampleTF_late = exampleTransformer_late.transform(exampleTFVects_late)
print(exampleTF_late.shape)
```

    (52327, 1000)
    


```python
try:
    print(exampleTFVectorizer_late.vocabulary_['president'])
except KeyError:
    print('This word is missing')
    print('The available words are: {} ...'.format(list(ngTFVectorizer.vocabulary_.keys())[:10]))
```

    667
    


```python
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("For our clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(t_late['sentiment'], km_late.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(t_late['sentiment'], km_late.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(t_late['sentiment'], km_late.labels_)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(t_late['sentiment'], km_late.labels_)))
```

    The available metrics are: ['ConfusionMatrixDisplay', 'PrecisionRecallDisplay', 'RocCurveDisplay', 'SCORERS', 'accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'balanced_accuracy_score', 'brier_score_loss', 'calinski_harabasz_score', 'calinski_harabaz_score', 'check_scoring', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'davies_bouldin_score', 'dcg_score', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'max_error', 'mean_absolute_error', 'mean_gamma_deviance', 'mean_poisson_deviance', 'mean_squared_error', 'mean_squared_log_error', 'mean_tweedie_deviance', 'median_absolute_error', 'multilabel_confusion_matrix', 'mutual_info_score', 'nan_euclidean_distances', 'ndcg_score', 'normalized_mutual_info_score', 'pairwise', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_chunked', 'pairwise_kernels', 'plot_confusion_matrix', 'plot_precision_recall_curve', 'plot_roc_curve', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss']
    For our clusters:
    Homogeneity: 0.011
    Completeness: 0.010
    V-measure: 0.010
    Adjusted Rand Score: 0.001
    


```python
terms = exampleTFVectorizer_late.get_feature_names()
print("Top terms per cluster:")
order_centroids = km_late.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
```

    Top terms per cluster:
    Cluster 0:
     coronaviruspandemic
     people
     coronavirusoutbreak
     covid19
     just
     trump
     lockdown
     covidー19
     like
     pandemic
    
    
    Cluster 1:
     covid_19
     quarantine
     people
     lockdown
     stayathome
     coronaviruspandemic
     time
     like
     stayhome
     corona
    
    
    Cluster 2:
     amp
     stay
     home
     covid19
     safe
     friends
     family
     people
     support
     share
    
    
    Cluster 3:
     covid19
     pandemic
     time
     help
     support
     need
     ll
     stayhome
     like
     health
    
    
    Cluster 4:
     covid
     19
     covid19
     pandemic
     coronavirusoutbreak
     covid_19
     health
     new
     coronaviruspandemic
     positive
    
    
    Cluster 5:
     cases
     new
     deaths
     total
     confirmed
     covid19
     number
     000
     reported
     positive
    
    
    


```python
components = pca_late.components_
keyword_ids_late = list(set(order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words from the kmeans model.
words_late = [terms[i] for i in keyword_ids_late]#Turn the ids into words.
x = components[:,keyword_ids_late][0,:] #Find the coordinates of those words in your biplot.
y = components[:,keyword_ids_late][1,:]
```


```python
colordict = {'negative': 'blue','pretty negative': 'green','slightly negative': 'yellow',
             'slightly positive':'pink','pretty positive':'orange','positive':'red'}
colors = [colordict[c] for c in t_late['sentiment']]
```


```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data_late[:, 0], reduced_data_late[:, 1], color = colors, alpha = 0.5, label = colors)
for i, word in enumerate(words_late):
    ax.annotate(word, (x[i],y[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes (Late)')
plt.show()
```


![png](output_69_0.png)



```python
colors_p = [colordict[sentimentCategories[l]] for l in km_late.labels_]
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reduced_data_late[:, 0], reduced_data_late[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters (Late)\n k = 6')
plt.show()
```


![png](output_70_0.png)


In the above two figures, we get the clustered true and predicted labels of the observations of Late tweets.


```python
hier_late = t_late.sample(n=200)
hierTFVectorizer_late = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, 
                                                                      stop_words='english', norm='l2')
hierTFVects_early = hierTFVectorizer_late.fit_transform(hier_late['text'])
CoocMat_late = hierTFVects_early * hierTFVects_early.T
CoocMat_late.setdiag(0)
examplelinkage_matrix_late = scipy.cluster.hierarchy.ward(CoocMat_late.toarray())
ax = scipy.cluster.hierarchy.dendrogram(examplelinkage_matrix_late, p=6, truncate_mode='level')
plt.title('Hierarchical Clustering (Late)',y=1.02,size=16)
```




    Text(0.5, 1.02, 'Hierarchical Clustering (Late)')




![png](output_72_1.png)



```python
hierarchicalClusters_late = scipy.cluster.hierarchy.fcluster(examplelinkage_matrix_late, 6, 'maxclust')
print("For our complete clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(hier_late['sentiment'], hierarchicalClusters_late)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(hier_late['sentiment'], hierarchicalClusters_late)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(hier_late['sentiment'], hierarchicalClusters_late)))
print("Adjusted Rand Score: {:0.3f}".format(sklearn.metrics.adjusted_rand_score(hier_late['sentiment'], hierarchicalClusters_late)))
```

    For our complete clusters:
    Homogeneity: 0.052
    Completeness: 0.054
    V-measure: 0.053
    Adjusted Rand Score: 0.007
    

We applied Hierarchical Clustering with Wald's Method on the Late tweets, with the original sampled with n = 200.

## Sentiment Analysis

This part is to do some sentiment analysis on the tweets. In the first step, we want to label all the reviews with their emotion(1 if the review is emotionally positive and -1 if negative) and train models to make predictions.This step is an unsupervised machine learning case.We will use three different perceptron models and also tune the hyperparameters to pick the optimal choice. In the second process, we want to use the unigram model to analyze which words most contribute to the reviews' emotion genre. This case aims to find words that strongly contribute to positive or negative emotion, and the results can be used for further movie analysis.


```python
def convert(score):
    '''
    This function is to convert a sentiment analysis score into a sentiment
    analysis label. The label will be -1 if the analysis score is not bigger
    than 0.2 and 1 if the analysis score is bigger than 0.2.

    Input:
    score (number): the analysis score

    Output:
    the analysis label (number)
    '''
    if score <= 0.2:
        return -1
    return 1
```


```python
for df in tweets:
    df['sentiment'] = df['sentiment_scores'].apply(lambda x: convert(x['compound']))
```


```python
def bag_of_words(texts):
    '''
    This function is to extract unique words from all the texts for further
    unigram analysis. However, Since we are unable to create a too large numpy
    array, we just pick about 10% of the total word number with higher frequency
    to do the following analysis.

    Input:
    texts (list): a list of strings.

    Output:
    dictionary (dictionary): a dictionary where the keys are the unique words
      and the values are their own index.
    '''

    dictionary = {}
    all_words = []
    for text in texts:
        word_list = text.split()
        all_words.extend(word_list)

    words_count = dict(Counter(all_words))
    words_tuple = sorted(words_count.items(), key=lambda x: (-x[1], x[0]))
    for i, content in enumerate(words_tuple):
        dictionary[content[0]] = i

    return dictionary
```


```python
def extract_bow_feature_vectors(reviews, dictionary):
    '''
    This function is to get the bag-of-words feature matrix representation of
    the data. The shape of the output will be (n, m), where n is the number of
    reviews and m the total number of words included in the dictionary that we
    want to analyze.

    Inputs:
    reviews (list): texts which are to be analyzed.
    dictionary (dictionary): a dictionary where the keys are the words and the
      values are their index.

    Ouput:
    feature_matrix (numpy matrix): the feature matrix representation of the
      texts.
    '''
    feature_matrix = np.zeros([len(reviews), len(dictionary)])
    for i, text in enumerate(reviews):
        word_list = text.split()
        for word in word_list:
            if word in dictionary:
                count = word_list.count(word)
                feature_matrix[i, dictionary[word]] = count
    return feature_matrix
```


```python
def get_order(n_samples):
    '''
    This function is to get a fixed order, which will be used in querrying the
    elements in feature matrix representation of texts.

    Input:
    n_samples (number): total number of orders.

    Ouput:
    a list of ordered numbers which represent an index.
    '''

    try:
        with open(str(n_samples) + '.txt') as temp_file:
            line = temp_file.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
```


```python
def perceptron(feature_matrix, labels, times):
    '''
    This function is to run the perceptron algorithm on data. We know that
    sklearn package contains algorithm for perceptron, however we want to
    emphasize the differences among the three algorithms. Therefore, we show the
    entire computing process of this algorithm.

    Inputs:
    feature_matrix (numpy matrix): the feature matrix representation of the
      reviews.
    labels (list): labels of each review.
    times (number): the number of iteration times.

    Ouput:
    (theta, theta_0) (tuple): the trained theta and theta_0 for this algorithm.
    '''

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    time = 1
    while time <= times:
        time += 1
        for i in get_order(feature_matrix.shape[0]):
            ips = 1e-8
            label = labels[i]
            feature_vector = feature_matrix[i]
            check = float(label*(theta.dot(feature_vector) + theta_0))
            if abs(check) < ips or check < 0:
                theta = theta + label*feature_vector
                theta_0 = theta_0 + label
    return (theta, theta_0)
```


```python
def average_perceptron(feature_matrix, labels, times):
    '''
    This function is to run the average perceptron algorithm on data.

    Inputs:
    feature_matrix (numpy matrix): the feature matrix representation of the
      texts.
    labels (list): labels of each text.
    times (number): the number of iteration times.

    Ouput:
    (theta_mean, theta_0_mean) (tuple): the trained theta and theta_0 for this
      algorithm.
    '''

    theta, theta_0 = np.zeros(feature_matrix.shape[1]), 0

    theta_sum, theta_0_sum = np.zeros(feature_matrix.shape[1]), 0
    amount = feature_matrix.shape[0]
    time = 1
    while time <= times:
        time += 1
        for i in get_order(feature_matrix.shape[0]):
            ips = 1e-8
            label = labels[i]
            feature_vector = feature_matrix[i]
            check = float(label * (theta.dot(feature_vector) + theta_0))
            if abs(check) < ips or check < 0:
                theta = theta + label * feature_vector
                theta_0 = theta_0 + label
            theta_sum += theta
            theta_0_sum += theta_0
    theta_mean = (1 / (amount*times)) * theta_sum
    theta_0_mean = (1 / (amount*times)) * theta_0_sum

    return (theta_mean, theta_0_mean)
```


```python
def pegasos_once(feature_vector, label, lambda_, eta, current_theta,\
    current_theta_0):
    '''
    This function is to run the pegasos algorithm on data for one step.

    Inputs:
    feature_matrix (numpy matrix): the feature matrix representation of the
      reviews.
    label (list): labels of each text.
    lambda_ (number): a parameter used to update the algorithm.
    eta (number): a parameter used to update the algorithm.
    current_theta (numpy array): the initialized theta
    current_theta_0 (number): the initialized theta_0

    Ouputs:
    (current_theta, current_theta_0) (tuple): the trained theta and theta_0
      for this algorithm.
    '''

    check = label * (current_theta.dot(feature_vector) + current_theta_0)

    if check <= 1:
        current_theta = (1 - eta*lambda_) * current_theta +\
        eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
    else:
        current_theta = (1 - eta*lambda_) * current_theta

    return (current_theta, current_theta_0)
```


```python
def pegasos(feature_matrix, labels, times, lambda_):
    '''
    This function is to run the entire process of pegasos algorithm on data.

    Inputs:
    feature_matrix (numpy matrix): the feature matrix representation of the
      texts.
    label (list): labels of each text.
    times (number): the number of iteration times.
    lambda_ (number): a parameter used to update the algorithm.

    Ouput:
    (current_theta, current_theta_0) (tuple): the trained theta and theta_0
      for this algorithm.
    '''

    theta, theta_0 = np.zeros(feature_matrix.shape[1]), 0
    time = 1
    while time <= times:
        time += 1
        for i in get_order(feature_matrix.shape[0]):
            eta = 1/np.sqrt(time)
            time += 1
            theta, theta_0 = pegasos_once(feature_matrix[i, :],\
            labels[i], lambda_, eta, theta, theta_0)
    return (theta, theta_0)
```


```python
def classify(feature_matrix, theta, theta_0):
    '''
    This function is to classify observations.

    Inputs:
    feature_matrix (numpy matrix): the feature matrix representation of the
      texts.
    theta (numpy array): coefficients of different words.
    theta_0 (number): coefficient of the constant part.

    Ouput:
    check (numpy array): the classfied results of the observations.
    '''

    check = theta.dot(feature_matrix.T) + theta_0
    check[check > 0] = 1
    check[check < 0] = -1
    check[abs(check) < 1e-9] = -1

    return check
```


```python
def accuracy(preds, targets):
    '''
    This function is to get the accuracy of the predictions over their true
    values.

    Inputs:
    preds (list or numpy series): the prediction values.
    targets (list or numpy series): the true values.

    Ouput:
    A number shows the accuracy of the prediction.
    '''

    return (preds == targets).mean()
```


```python
def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix,\
    train_labels, val_labels, **kwargs):
    '''
    This function is to get the accuracy of a certain algorithm on the training
    and validation datasets.

    Inputs:
    classifier (function): an algorithm.
    train_feature_matrix (numpy matrix): the feature matrix representation of
      the texts in the training dataset.
    val_feature_matrix (numpy matrix): the feature matrix representation of the
      texts in the validation dataset.
    train_labels (list): the sentiment labels of the texts in the training
      dataset.
    val_labels (list): the sentiment labels of the texts in the validation
      dataset.
    **kwargs: other parameters that may be needed in the algorithm function.

    Ouput:
    (train_ac, val_ac) (tuple): the accuracies on the training and validation
      datasets.
    '''

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    pred_train = classify(train_feature_matrix, theta, theta_0)
    pred_val = classify(val_feature_matrix, theta, theta_0)

    train_ac = accuracy(pred_train, train_labels)
    val_ac = accuracy(pred_val, val_labels)

    return (train_ac, val_ac)
```


```python
def tune_one_param(train_fn, param_vals, train_feats, train_labels, val_feats,\
    val_labels):
    '''
    This function is to tune one parameter of a certain algorithm on the
    training and validation datasets.

    Inputs:
    train_fn (function): an algorithm.
    param_vals (list): a list of possible chosen values for the parameter.
    train_feats (numpy matrix): the feature matrix representation of the texts
      in the training dataset.
    train_labels (list): the sentiment labels of the texts in the training
      dataset.
    val_feats (numpy matrix): the feature matrix representation of the texts
      in the validation dataset.
    val_labels (list): the sentiment labels of the texts in the validation
      dataset.

    Ouput:
    chosen_param (number): the optimal value for the parameter.
    '''

    chosen_param = 'undecided'
    chosen_cv_score = 0

    for val in param_vals:
        theta, theta_0 = train_fn(train_feats, train_labels, val)

        val_preds = classify(val_feats, theta, theta_0)
        val_acc = accuracy(val_preds, val_labels)
        if val_acc > chosen_cv_score:
            chosen_cv_score = val_acc
            chosen_param = val

    return chosen_param

TS = [1, 5, 10, 15, 25, 50]
LS = [0.001, 0.01, 0.1, 1, 10]
```


```python
def tune_two_params(train_fn, param_vals_1, param_vals_2, train_feats,\
    train_labels, val_feats, val_labels):
    '''
    This function is to tune two parameters of a certain algorithm on the
    training and validation datasets.

    Inputs:
    train_fn (function): an algorithm.
    param_vals_1 (list): a list of possible chosen values for the first
      parameter.
    param_vals_2 (list): a list of possible chosen values for the second
      parameter.
    train_feats (numpy matrix): the feature matrix representation of the texts
      in the training dataset.
    train_labels (list): the sentiment labels of the texts in the training
      dataset.
    val_feats (numpy matrix): the feature matrix representation of the texts
      in the validation dataset.
    val_labels (list): the sentiment labels of the texts in the validation
      dataset.

    Ouput:
    chosen_params (list): contains the optimal values for the two parameters.
    '''

    chosen_params = 'undecided'
    chosen_cv_score = 0

    for val_1 in param_vals_1:
        for val_2 in param_vals_2:
            theta, theta_0 = train_fn(train_feats, train_labels, val_1, val_2)

            val_preds = classify(val_feats, theta, theta_0)
            val_acc = accuracy(val_preds, val_labels)
            if val_acc > chosen_cv_score:
                chosen_params = [val_1, val_2]

    return chosen_params
```


```python
def most_explanatory_word(theta, wordlist, emotion):
    """
    This cuntion is to get the word associated with the bag-of-words feature
    having largest weight in the diretion of postive or negative emotion.

    Inputs:
    theta (numpy array): coefficients of different words.
    wordlist (list): a list of words corresponding to the theta.
    emotion (str): 'positive' or 'negative'

    Ouput:
    a list of words arranged according to their coeffients ascendingly if we
    want to get the top words indicating the negative emotion and descendingly
    if postive signals are wanted.
    """
    if emotion == 'positive':
        return [word for (theta_i, word) in sorted(zip(theta, wordlist))[::-1]]
    return [word for (theta_i, word) in sorted(zip(theta, wordlist))]
```

### Early


```python
X_early, X_TEST_early, Y_early, Y_TEST_early = train_test_split(t_early['text'],\
    t_early['sentiment'], test_size=0.2, train_size=0.8)
X_TRAIN_early, X_CV_early, Y_TRAIN_early, Y_CV_early = train_test_split(X_early, Y_early, test_size=0.25,\
    train_size=0.75)
X_TRAIN_LIST_early = X_TRAIN_early.tolist()
X_CV_LIST_early = X_CV_early.tolist()
X_TEST_LIST_early = X_TEST_early.tolist()
Y_TRAIN_LIST_early = Y_TRAIN_early.tolist()
Y_CV_LIST_early = Y_CV_early.tolist()
Y_TEST_LIST_early = Y_TEST_early.tolist()
```


```python
DICTIONARY_early = bag_of_words(X_TRAIN_LIST_early)
```


```python
TRAIN_BOW_FEATURES_early = extract_bow_feature_vectors(X_TRAIN_LIST_early, DICTIONARY_early)
CV_BOW_FEATURES_early = extract_bow_feature_vectors(X_CV_LIST_early, DICTIONARY_early)
TEST_BOW_FEATURES_early = extract_bow_feature_vectors(X_TEST_LIST_early, DICTIONARY_early)
```


```python
PERCEPTRON_SCORES_early = classifier_accuracy(perceptron, TRAIN_BOW_FEATURES_early,\
    CV_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_CV_LIST_early, times=10)

AVG_PERCEPTRON_SCORES_early = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_early, CV_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_CV_LIST_early, times=10)

PEGASOS_SCORES_early = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_early,\
    CV_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_CV_LIST_early, times=10, lambda_=0.01)

ALGO_SCORES_early = {'algorithm': ['perceptron', 'average_perceptron', 'pegasos'],\
'train accuracy': [PERCEPTRON_SCORES_early[0], AVG_PERCEPTRON_SCORES_early[0],\
PEGASOS_SCORES_early[0]], 'cv accuracy': [PERCEPTRON_SCORES_early[1],\
AVG_PERCEPTRON_SCORES_early[1], PEGASOS_SCORES_early[1]]}

ALGO_SCORES_DF_early = pd.DataFrame(ALGO_SCORES_early)
ALGO_SCORES_DF_early
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algorithm</th>
      <th>train accuracy</th>
      <th>cv accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>perceptron</td>
      <td>0.972647</td>
      <td>0.728589</td>
    </tr>
    <tr>
      <th>1</th>
      <td>average_perceptron</td>
      <td>0.996782</td>
      <td>0.740651</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pegasos</td>
      <td>0.809332</td>
      <td>0.718938</td>
    </tr>
  </tbody>
</table>
</div>



We applied the three algorithms to the Early tweets, and we saw that the accuracies are relatively high. Therefore, we continue to tune the hyperparameters.


```python
DATA_early = (TRAIN_BOW_FEATURES_early, Y_TRAIN_LIST_early, CV_BOW_FEATURES_early, Y_CV_LIST_early)
```


```python
PERCEPTRON_CHOSEN_T_early = tune_one_param(perceptron, TS, *DATA_early)
AVG_PERCEPTRON_CHOSEN_T_early = tune_one_param(average_perceptron, TS, *DATA_early)
PEGASOS_CHOSEN_TL_early = tune_two_params(pegasos, TS, LS, *DATA_early)

TUNED_PERCEPTRON_TEST_SCORE_early = classifier_accuracy(perceptron,\
    TRAIN_BOW_FEATURES_early, TEST_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_TEST_LIST_early,\
    times=PERCEPTRON_CHOSEN_T_early)[1]

TUNED_AVG_PERCEPTRON_TEST_SCORE_early = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_early, TEST_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_TEST_LIST_early,\
    times=AVG_PERCEPTRON_CHOSEN_T_early)[1]

TUNED_PEGASOS_TEST_SCORE_early = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_early,\
    TEST_BOW_FEATURES_early, Y_TRAIN_LIST_early, Y_TEST_LIST_early, times=PEGASOS_CHOSEN_TL_early[0],\
    lambda_=PEGASOS_CHOSEN_TL_early[1])[1]

print('Perceptron (early):', TUNED_PERCEPTRON_TEST_SCORE_early)
print('Average perceptron (early):', TUNED_AVG_PERCEPTRON_TEST_SCORE_early)
print('Pegasos (early):', TUNED_PEGASOS_TEST_SCORE_early)
```

    Perceptron (early): 0.7322074788902292
    Average perceptron (early): 0.7201447527141134
    Pegasos (early): 0.6815440289505428
    

We tuned the hyperparameters and trained the model, and we got the highest accuracy scores for each model as shown above. Since the score of the average perceptron is the highest, we choose this model.


```python
BEST_THETAS_early = perceptron(TRAIN_BOW_FEATURES_early, Y_TRAIN_LIST_early, PERCEPTRON_CHOSEN_T_early)
WORDLIST_early = list(DICTIONARY_early.keys())
POSITIVE_WORDS_early = most_explanatory_word(BEST_THETAS_early[0], WORDLIST_early, 'positive')
TOP_10_POSITIVE_WORDS_early = POSITIVE_WORDS_early[:10]
NEGATIVE_WORDS_early = most_explanatory_word(BEST_THETAS_early[0], WORDLIST_early, 'negative')
TOP_10_NEGATIVE_WORDS_early = NEGATIVE_WORDS_early[:10]
print("Top 10 words that strongly indicate positive emotion include (early): ", '\n',\
    TOP_10_POSITIVE_WORDS_early)
print("Top 10 words that strongly indicate negative emotion include (early): ", '\n',\
    TOP_10_NEGATIVE_WORDS_early)
```

    Top 10 words that strongly indicate positive emotion include (early):  
     ['positive', 'best', 'safe', 'sure', 'good', 'help', 'hand', 'chance', 'Please', 'protect']
    Top 10 words that strongly indicate negative emotion include (early):  
     ['sick', 'stop', 'die', 'emergency', 'fear', 'crisis', 'deaths', 'dying', 'infected', 'isolation']
    

With the optimal model, we get the top words that strongly indicate positive/negative emotion.

### Middle


```python
sample_middle = round(len(t_middle) * 0.2) + 1
t_sentiment_middle = t_late.sample(n=sample_middle)
X_middle, X_TEST_middle, Y_middle, Y_TEST_middle = train_test_split(t_sentiment_middle['text'],\
    t_sentiment_middle['sentiment'], test_size=0.2, train_size=0.8)
X_TRAIN_middle, X_CV_middle, Y_TRAIN_middle, Y_CV_middle = train_test_split(X_middle, Y_middle, test_size=0.25,\
    train_size=0.75)
X_TRAIN_LIST_middle = X_TRAIN_middle.tolist()
X_CV_LIST_middle = X_CV_middle.tolist()
X_TEST_LIST_middle = X_TEST_middle.tolist()
Y_TRAIN_LIST_middle = Y_TRAIN_middle.tolist()
Y_CV_LIST_middle = Y_CV_middle.tolist()
Y_TEST_LIST_middle = Y_TEST_middle.tolist()

DICTIONARY_middle = bag_of_words(X_TRAIN_LIST_middle)

TRAIN_BOW_FEATURES_middle = extract_bow_feature_vectors(X_TRAIN_LIST_middle, DICTIONARY_middle)
CV_BOW_FEATURES_middle = extract_bow_feature_vectors(X_CV_LIST_middle, DICTIONARY_middle)
TEST_BOW_FEATURES_middle = extract_bow_feature_vectors(X_TEST_LIST_middle, DICTIONARY_middle)
```


```python
PERCEPTRON_SCORES_middle = classifier_accuracy(perceptron, TRAIN_BOW_FEATURES_middle,
                                               CV_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_CV_LIST_middle, times=10)

AVG_PERCEPTRON_SCORES_middle = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_middle, CV_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_CV_LIST_middle, times=10)

PEGASOS_SCORES_middle = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_middle,\
    CV_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_CV_LIST_middle, times=10, lambda_=0.01)

ALGO_SCORES_middle = {'algorithm': ['perceptron', 'average_perceptron', 'pegasos'],\
'train accuracy': [PERCEPTRON_SCORES_middle[0], AVG_PERCEPTRON_SCORES_middle[0],\
PEGASOS_SCORES_middle[0]], 'cv accuracy': [PERCEPTRON_SCORES_middle[1],\
AVG_PERCEPTRON_SCORES_middle[1], PEGASOS_SCORES_middle[1]]}

ALGO_SCORES_DF_middle = pd.DataFrame(ALGO_SCORES_middle)
ALGO_SCORES_DF_middle
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algorithm</th>
      <th>train accuracy</th>
      <th>cv accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>perceptron</td>
      <td>0.980761</td>
      <td>0.699935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>average_perceptron</td>
      <td>0.996974</td>
      <td>0.749190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pegasos</td>
      <td>0.784911</td>
      <td>0.691510</td>
    </tr>
  </tbody>
</table>
</div>



We applied the three algorithms to the Middle tweets, and we saw that the accuracies are relatively high. Therefore, we continue to tune the hyperparameters.


```python
DATA_middle = (TRAIN_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, CV_BOW_FEATURES_middle, Y_CV_LIST_middle)

PERCEPTRON_CHOSEN_T_middle = tune_one_param(perceptron, TS, *DATA_middle)
AVG_PERCEPTRON_CHOSEN_T_middle = tune_one_param(average_perceptron, TS, *DATA_middle)
PEGASOS_CHOSEN_TL_middle = tune_two_params(pegasos, TS, LS, *DATA_middle)

TUNED_PERCEPTRON_TEST_SCORE_middle = classifier_accuracy(perceptron,\
    TRAIN_BOW_FEATURES_middle, TEST_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_TEST_LIST_middle,\
    times=PERCEPTRON_CHOSEN_T_middle)[1]

TUNED_AVG_PERCEPTRON_TEST_SCORE_middle = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_middle, TEST_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_TEST_LIST_middle,\
    times=AVG_PERCEPTRON_CHOSEN_T_middle)[1]

TUNED_PEGASOS_TEST_SCORE_middle = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_middle,\
    TEST_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, Y_TEST_LIST_middle, times=PEGASOS_CHOSEN_TL_middle[0],\
    lambda_=PEGASOS_CHOSEN_TL_middle[1])[1]

print('Perceptron (middle):', TUNED_PERCEPTRON_TEST_SCORE_middle)
print('Average perceptron (middle):', TUNED_AVG_PERCEPTRON_TEST_SCORE_middle)
print('Pegasos (middle):', TUNED_PEGASOS_TEST_SCORE_middle)
```

    Perceptron (middle): 0.7258587167854829
    Average perceptron (middle): 0.729747245625405
    Pegasos (middle): 0.5910563836681789
    

We tuned the hyperparameters and trained the model, and we got the highest accuracy scores for each model as shown above. Since the accuracy score of the average perceptron is the highest, we choose this model.


```python
BEST_THETAS_middle = average_perceptron(TRAIN_BOW_FEATURES_middle, Y_TRAIN_LIST_middle, AVG_PERCEPTRON_CHOSEN_T_middle)
WORDLIST_middle = list(DICTIONARY_middle.keys())
POSITIVE_WORDS_middle = most_explanatory_word(BEST_THETAS_middle[0], WORDLIST_middle, 'positive')
TOP_10_POSITIVE_WORDS_middle = POSITIVE_WORDS_middle[:10]
NEGATIVE_WORDS_middle = most_explanatory_word(BEST_THETAS_middle[0], WORDLIST_middle, 'negative')
TOP_10_NEGATIVE_WORDS_middle = NEGATIVE_WORDS_middle[:10]
print("Top 10 words that strongly indicate positive emotion include (middle): ", '\n',\
    TOP_10_POSITIVE_WORDS_middle)
print("Top 10 words that strongly indicate negative emotion include (middle): ", '\n',\
    TOP_10_NEGATIVE_WORDS_middle)
```

    Top 10 words that strongly indicate positive emotion include (middle):  
     ['best', 'safe', 'positive', 'support', 'hope', 'important', 'good', 'Please', 'save', 'care']
    Top 10 words that strongly indicate negative emotion include (middle):  
     ['crisis', 'death', 'avoid', 'fighting', 'alone', 'kill', 'crisis.', 'worst', 'stop', 'infected']
    

With the optimal model, we get the top words that strongly indicate positive/negative emotion.

### Late


```python
sample_late = round(len(t_late) * 0.2) + 1
t_sentiment_late = t_late.sample(n=sample_late)
X_late, X_TEST_late, Y_late, Y_TEST_late = train_test_split(t_sentiment_late['text'],\
    t_sentiment_late['sentiment'], test_size=0.2, train_size=0.8)
X_TRAIN_late, X_CV_late, Y_TRAIN_late, Y_CV_late = train_test_split(X_late, Y_late, test_size=0.25,\
    train_size=0.75)
X_TRAIN_LIST_late = X_TRAIN_late.tolist()
X_CV_LIST_late = X_CV_late.tolist()
X_TEST_LIST_late = X_TEST_late.tolist()
Y_TRAIN_LIST_late = Y_TRAIN_late.tolist()
Y_CV_LIST_late = Y_CV_late.tolist()
Y_TEST_LIST_late = Y_TEST_late.tolist()

DICTIONARY_late = bag_of_words(X_TRAIN_LIST_late)

TRAIN_BOW_FEATURES_late = extract_bow_feature_vectors(X_TRAIN_LIST_late, DICTIONARY_late)
CV_BOW_FEATURES_late = extract_bow_feature_vectors(X_CV_LIST_late, DICTIONARY_late)
TEST_BOW_FEATURES_late = extract_bow_feature_vectors(X_TEST_LIST_late, DICTIONARY_late)
```


```python
PERCEPTRON_SCORES_late = classifier_accuracy(perceptron, TRAIN_BOW_FEATURES_late,\
    CV_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_CV_LIST_late, times=10)

AVG_PERCEPTRON_SCORES_late = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_late, CV_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_CV_LIST_late, times=10)

PEGASOS_SCORES_late = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_late,\
    CV_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_CV_LIST_late, times=10, lambda_=0.01)

ALGO_SCORES_late = {'algorithm': ['perceptron', 'average_perceptron', 'pegasos'],\
'train accuracy': [PERCEPTRON_SCORES_late[0], AVG_PERCEPTRON_SCORES_late[0],\
PEGASOS_SCORES_late[0]], 'cv accuracy': [PERCEPTRON_SCORES_late[1],\
AVG_PERCEPTRON_SCORES_late[1], PEGASOS_SCORES_late[1]]}

ALGO_SCORES_DF_late = pd.DataFrame(ALGO_SCORES_late)
ALGO_SCORES_DF_late
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algorithm</th>
      <th>train accuracy</th>
      <th>cv accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>perceptron</td>
      <td>0.992037</td>
      <td>0.752986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>average_perceptron</td>
      <td>0.997133</td>
      <td>0.757764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pegasos</td>
      <td>0.733397</td>
      <td>0.693741</td>
    </tr>
  </tbody>
</table>
</div>



We applied the three algorithms to the Late tweets, and we saw that the accuracies are relatively high. Therefore, we continue to tune the hyperparameters.


```python
DATA_late = (TRAIN_BOW_FEATURES_late, Y_TRAIN_LIST_late, CV_BOW_FEATURES_late, Y_CV_LIST_late)

PERCEPTRON_CHOSEN_T_late = tune_one_param(perceptron, TS, *DATA_late)
AVG_PERCEPTRON_CHOSEN_T_late = tune_one_param(average_perceptron, TS, *DATA_late)
PEGASOS_CHOSEN_TL_late = tune_two_params(pegasos, TS, LS, *DATA_late)

TUNED_PERCEPTRON_TEST_SCORE_late = classifier_accuracy(perceptron,\
    TRAIN_BOW_FEATURES_late, TEST_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_TEST_LIST_late,\
    times=PERCEPTRON_CHOSEN_T_late)[1]

TUNED_AVG_PERCEPTRON_TEST_SCORE_late = classifier_accuracy(average_perceptron,\
    TRAIN_BOW_FEATURES_late, TEST_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_TEST_LIST_late,\
    times=AVG_PERCEPTRON_CHOSEN_T_late)[1]

TUNED_PEGASOS_TEST_SCORE_late = classifier_accuracy(pegasos, TRAIN_BOW_FEATURES_late,\
    TEST_BOW_FEATURES_late, Y_TRAIN_LIST_late, Y_TEST_LIST_late, times=PEGASOS_CHOSEN_TL_late[0],\
    lambda_=PEGASOS_CHOSEN_TL_late[1])[1]

print('Perceptron (late):', TUNED_PERCEPTRON_TEST_SCORE_late)
print('Average perceptron (late):', TUNED_AVG_PERCEPTRON_TEST_SCORE_late)
print('Pegasos (late):', TUNED_PEGASOS_TEST_SCORE_late)
```

    Perceptron (late): 0.7511938872970392
    Average perceptron (late): 0.7459407831900668
    Pegasos (late): 0.5840496657115568
    

We tuned the hyperparameters and trained the model, and we got the highest accuracy scores for each model as shown above. Since the accuracy score of the perceptron is the highest, we choose this model.


```python
BEST_THETAS_late = perceptron(TRAIN_BOW_FEATURES_late, Y_TRAIN_LIST_late, PERCEPTRON_CHOSEN_T_late)
WORDLIST_late = list(DICTIONARY_late.keys())
POSITIVE_WORDS_late = most_explanatory_word(BEST_THETAS_late[0], WORDLIST_late, 'positive')
TOP_10_POSITIVE_WORDS_late = POSITIVE_WORDS_late[:10]
NEGATIVE_WORDS_late = most_explanatory_word(BEST_THETAS_late[0], WORDLIST_late, 'negative')
TOP_10_NEGATIVE_WORDS_late = NEGATIVE_WORDS_late[:10]
print("Top 10 words that strongly indicate positive emotion include (late): ", '\n',\
    TOP_10_POSITIVE_WORDS_late)
print("Top 10 words that strongly indicate negative emotion include (late): ", '\n',\
    TOP_10_NEGATIVE_WORDS_late)
```

    Top 10 words that strongly indicate positive emotion include (late):  
     ['great', 'positive', 'best', 'love', 'hand', 'Thank', 'hope', 'please', 'free', 'safe.']
    Top 10 words that strongly indicate negative emotion include (late):  
     ['crisis', 'crisis.', 'death', 'infected', 'critical', 'stop', 'wrong', 'ass', 'fuck', 'isolation']
    

With the optimal model, we get the top words that strongly indicate positive/negative emotion.

## Gensim

In this part, we will use the Gensim model to analyze the topics of tweets in each dataset and visualize the results.


```python
def dropMissing(wordLst, vocab):
    '''
    This function is to drop missing information.
    
    Input:
    wordLst (list/array): all words
    vocab (list/array): non-missing words
    
    Output:
    a list/array of non-missing words
    '''
    return [w for w in wordLst if w in vocab]
```


```python
def add_tokens(df, TFVectorizer):
    '''
    This function is to extract tokens from tweets.
    
    Input:
    df (DataFrame): a DataFrame of tweets
    TFVectorizer (TfidfVectorizer): the TfidfVectorizer of the texts
    
    Output:
    df (DataFrame): a DataFrame of tweets with tokens
    '''
    df['tokenized_text'] = df['text'].apply(lambda x: lucem_illud_2020.word_tokenize(x))
    df['normalized_tokens'] = df['tokenized_text'].apply(lambda x: lucem_illud_2020.normalizeTokens(x))
    df['reduced_tokens'] = df['normalized_tokens'].apply(lambda x: dropMissing(x, TFVectorizer.vocabulary_.keys()))
    return df
```


```python
vectorizers = [exampleTFVectorizer_early, exampleTFVectorizer_middle, exampleTFVectorizer_late]
for i,df in enumerate(tweets):
    df = add_tokens(df, vectorizers[i])
```

#### Early


```python
dictionary_early = gensim.corpora.Dictionary(t_early['reduced_tokens'])
corpus_early = [dictionary_early.doc2bow(text) for text in t_early['reduced_tokens']]
senlda_early = gensim.models.ldamodel.LdaModel(corpus=corpus_early, id2word=dictionary_early, num_topics=10, alpha='auto', eta='auto')
sen1Bow_early = dictionary_early.doc2bow(t_early['reduced_tokens'][0])
sen1lda_early = senlda_early[sen1Bow_early]
print("The topics of the text: {}".format(t_early['status_id'][0]))
print("are: {}".format(sen1lda_early))
```

    The topics of the text: 1236600394521939968
    are: [(0, 0.020813175), (1, 0.021905515), (2, 0.021785706), (3, 0.021964626), (4, 0.02160765), (5, 0.8067039), (6, 0.020254398), (7, 0.021406451), (8, 0.021155838), (9, 0.022402665)]
    

In the above analysis, we saw the topics related to the first observation in Early tweets and the weight of each topic.


```python
ldaDF_early = pd.DataFrame({'status_id' : t_early['status_id'],'time':t_early['created_at'],
                          'topics' : [senlda_early[dictionary_early.doc2bow(l)] for l in t_early['reduced_tokens']]})
```


```python
topicsProbDict_early = {i : [0] * len(ldaDF_early) for i in range(senlda_early.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(ldaDF_early['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict_early[topicNum][index] = prob

#Update the DataFrame
for topicNum in range(senlda_early.num_topics):
    ldaDF_early['topic_{}'.format(topicNum)] = topicsProbDict_early[topicNum]
```


```python
ldaDFV_early = ldaDF_early[:10][['topic_%d' %x for x in range(10)]]
ldaDFVisN_early = ldaDF_early[:10][['status_id']]
ldaDFVis_early = np.asmatrix(ldaDFV_early)
ldaDFVisNames_early = np.asmatrix(ldaDFVisN_early)
```


```python
N = 10
ind = np.arange(N)
K = senlda_early.num_topics  # N documents, K topics
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, np.asarray(ldaDFVis_early)[:, k], width, color=color)
    else:
        p = plt.bar(ind, np.asarray(ldaDFVis_early)[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += np.asarray(ldaDFVis_early)[:, k]
    plots.append(p)
    

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')

plt.title('Topics in Tweets (Early)')
plt.xticks(ind+width/2, ldaDFVisNames_early, rotation='vertical')

plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', frameon=True,  bbox_to_anchor = (1, .5))

plt.show()
```


![png](output_130_0.png)



```python
plt.pcolor(np.asarray(ldaDFVis_early), norm=None, cmap='Purples')
plt.yticks(np.arange(ldaDFVis_early.shape[0])+0.5, ldaDFVisNames_early);
plt.xticks(np.arange(ldaDFVis_early.shape[1])+0.5, topic_labels);

# flip the y-axis so the texts are in the order we anticipate 
plt.gca().invert_yaxis()

# rotate the ticks on the x-axis
plt.xticks(rotation=90)

# add a legend
plt.colorbar(cmap='Blues')
plt.tight_layout()  # fixes margins
plt.show()
```


![png](output_131_0.png)


We visualized the topics of the first ten observations in the Early tweets.


```python
for i in range(10):
    print('Topic {}:'.format(i))
    print(senlda_early.show_topic(i))
```

    Topic 0:
    [('think', 0.04591542), ('coronavirus', 0.038818363), ('health', 0.023974415), ('time', 0.021463739), ('people', 0.021448337), ('amp', 0.01870268), ('public', 0.01807296), ('coronavirusoutbreak', 0.016315486), ('organization', 0.014536712), ('fast', 0.013932205)]
    Topic 1:
    [('case', 0.102400854), ('coronavirusoutbreak', 0.05808223), ('coronavirus', 0.043263737), ('new', 0.039640557), ('total', 0.03931524), ('like', 0.029159054), ('bring', 0.019257136), ('report', 0.019170959), ('virus', 0.018340314), ('question', 0.017302591)]
    Topic 2:
    [('coronavirus', 0.03381089), ('hand', 0.027338825), ('wash', 0.02526485), ('people', 0.02381832), ('amp', 0.022545978), ('coronavirusoutbreak', 0.022343675), ('come', 0.018127855), ('break', 0.018122712), ('sick', 0.016674949), ('home', 0.015745623)]
    Topic 3:
    [('coronavirusoutbreak', 0.1070956), ('coronavirus', 0.07383032), ('covid2019', 0.025352921), ('case', 0.023177305), ('italy', 0.02151454), ('covidー19', 0.021355106), ('india', 0.017629316), ('coronavirusinindia', 0.0151377935), ('care', 0.015128553), ('govt', 0.013101128)]
    Topic 4:
    [('coronavirus', 0.0678875), ('coronavirusoutbreak', 0.045459576), ('close', 0.023463342), ('kill', 0.021446884), ('test', 0.019177957), ('wake', 0.017941289), ('virus', 0.017274546), ('country', 0.016470771), ('school', 0.01576392), ('emergency', 0.013606327)]
    Topic 5:
    [('trump', 0.035059325), ('lot', 0.02758604), ('coronavirusoutbreak', 0.023239274), ('like', 0.023161292), ('thing', 0.020941816), ('response', 0.018711917), ('coronavirus', 0.017592434), ('way', 0.01635971), ('say', 0.01578801), ('look', 0.014896209)]
    Topic 6:
    [('people', 0.05644505), ('case', 0.036167003), ('say', 0.025384063), ('tell', 0.02459114), ('need', 0.022348123), ('coronavirusoutbreak', 0.022335237), ('work', 0.019567125), ('quarantine', 0.017574431), ('coronavirus', 0.017218754), ('infection', 0.016174197)]
    Topic 7:
    [('coronavirus', 0.089660436), ('coronavirusoutbreak', 0.037495222), ('corona', 0.035862766), ('covid_19', 0.02297679), ('coronavirusupdate', 0.021527173), ('covid', 0.020563245), ('covid2019', 0.019558506), ('people', 0.019029085), ('virus', 0.016352678), ('case', 0.016017504)]
    Topic 8:
    [('travel', 0.034117885), ('new', 0.02684537), ('coronavirus', 0.025478676), ('case', 0.023704711), ('state', 0.022928745), ('cancel', 0.019393349), ('city', 0.018603386), ('test', 0.017896714), ('spread', 0.015289535), ('people', 0.0141945835)]
    Topic 9:
    [('coronavirus', 0.11546659), ('coronavirusoutbreak', 0.03330844), ('covid_19', 0.023592634), ('pandemic', 0.02292067), ('amp', 0.01951913), ('coverage', 0.017401602), ('outbreak', 0.01631846), ('news', 0.016148306), ('virus', 0.013447266), ('people', 0.013252167)]
    


```python
topicsDict_early = {}
for topicNum in range(senlda_early.num_topics):
    topicWords = [w for w, p in senlda_early.show_topic(topicNum)]
    topicsDict_early['Topic_{}'.format(topicNum)] = topicWords

wordRanksDF_early = pd.DataFrame(topicsDict_early)
wordRanksDF_early
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>think</td>
      <td>case</td>
      <td>coronavirus</td>
      <td>coronavirusoutbreak</td>
      <td>coronavirus</td>
      <td>trump</td>
      <td>people</td>
      <td>coronavirus</td>
      <td>travel</td>
      <td>coronavirus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>coronavirus</td>
      <td>coronavirusoutbreak</td>
      <td>hand</td>
      <td>coronavirus</td>
      <td>coronavirusoutbreak</td>
      <td>lot</td>
      <td>case</td>
      <td>coronavirusoutbreak</td>
      <td>new</td>
      <td>coronavirusoutbreak</td>
    </tr>
    <tr>
      <th>2</th>
      <td>health</td>
      <td>coronavirus</td>
      <td>wash</td>
      <td>covid2019</td>
      <td>close</td>
      <td>coronavirusoutbreak</td>
      <td>say</td>
      <td>corona</td>
      <td>coronavirus</td>
      <td>covid_19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time</td>
      <td>new</td>
      <td>people</td>
      <td>case</td>
      <td>kill</td>
      <td>like</td>
      <td>tell</td>
      <td>covid_19</td>
      <td>case</td>
      <td>pandemic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>people</td>
      <td>total</td>
      <td>amp</td>
      <td>italy</td>
      <td>test</td>
      <td>thing</td>
      <td>need</td>
      <td>coronavirusupdate</td>
      <td>state</td>
      <td>amp</td>
    </tr>
    <tr>
      <th>5</th>
      <td>amp</td>
      <td>like</td>
      <td>coronavirusoutbreak</td>
      <td>covidー19</td>
      <td>wake</td>
      <td>response</td>
      <td>coronavirusoutbreak</td>
      <td>covid</td>
      <td>cancel</td>
      <td>coverage</td>
    </tr>
    <tr>
      <th>6</th>
      <td>public</td>
      <td>bring</td>
      <td>come</td>
      <td>india</td>
      <td>virus</td>
      <td>coronavirus</td>
      <td>work</td>
      <td>covid2019</td>
      <td>city</td>
      <td>outbreak</td>
    </tr>
    <tr>
      <th>7</th>
      <td>coronavirusoutbreak</td>
      <td>report</td>
      <td>break</td>
      <td>coronavirusinindia</td>
      <td>country</td>
      <td>way</td>
      <td>quarantine</td>
      <td>people</td>
      <td>test</td>
      <td>news</td>
    </tr>
    <tr>
      <th>8</th>
      <td>organization</td>
      <td>virus</td>
      <td>sick</td>
      <td>care</td>
      <td>school</td>
      <td>say</td>
      <td>coronavirus</td>
      <td>virus</td>
      <td>spread</td>
      <td>virus</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fast</td>
      <td>question</td>
      <td>home</td>
      <td>govt</td>
      <td>emergency</td>
      <td>look</td>
      <td>infection</td>
      <td>case</td>
      <td>people</td>
      <td>people</td>
    </tr>
  </tbody>
</table>
</div>



We can see the detailed information of the specific words and their weight in each topic.


```python
topic1_df_early = pd.DataFrame(senlda_early.show_topic(1, topn=50))
plt.figure()
topic1_df_early.plot.bar(legend = False)
plt.title('Probability Distribution of Words, Topic 1 (Early)')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_136_1.png)



```python
senlda1_early = gensim.models.ldamodel.LdaModel(corpus=corpus_early, id2word=dictionary_early, num_topics=10, eta = 0.00001)
senlda2_early = gensim.models.ldamodel.LdaModel(corpus=corpus_early, id2word=dictionary_early, num_topics=10, eta = 0.9)
topic11_df_early = pd.DataFrame(senlda1_early.show_topic(1, topn=50))
topic21_df_early = pd.DataFrame(senlda2_early.show_topic(1, topn=50))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
topic11_df_early.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.0001')
topic21_df_early.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
plt.show()
```

    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <ipython-input-90-fd2c792b6dd4>:8: DeprecationWarning: invalid escape sequence \e
      topic11_df_early.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.0001')
    <ipython-input-90-fd2c792b6dd4>:9: DeprecationWarning: invalid escape sequence \e
      topic21_df_early.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
    


![png](output_137_1.png)


We can make the topics more unique by changing the parameters of the model, where $\alpha$ controls the sparsity of document-topic loadings, and $\eta$ controls the sparsity of topic-word loadings. We visualized the distribution of words over any single topic. The above example shows how different $\eta$ values can change the shape of the distribution.

#### Middle


```python
dictionary_middle = gensim.corpora.Dictionary(t_middle['reduced_tokens'])
corpus_middle = [dictionary_middle.doc2bow(text) for text in t_middle['reduced_tokens']]
senlda_middle = gensim.models.ldamodel.LdaModel(corpus=corpus_middle, id2word=dictionary_middle, num_topics=10, alpha='auto', eta='auto')
sen1Bow_middle = dictionary_middle.doc2bow(t_middle['reduced_tokens'][0].astype(str))
sen1lda_middle = senlda_middle[sen1Bow_middle]
print("The topics of the text: {}".format(t_middle.iloc[0,0]))
print("are: {}".format(sen1lda_middle))
```

    The topics of the text: 1.2381593661943767e+18
    are: [(0, 0.10399658), (1, 0.09794796), (2, 0.09202929), (3, 0.09449395), (4, 0.1037459), (5, 0.10408751), (6, 0.09946841), (7, 0.103212886), (8, 0.097014695), (9, 0.10400281)]
    

In the above analysis, we saw the topics related to the first observation in Middle tweets and the weight of each topic.


```python
ldaDF_middle = pd.DataFrame({'status_id' : t_middle['status_id'],'time':t_middle['created_at'],
                          'topics' : [senlda_middle[dictionary_middle.doc2bow(l)] for l in t_middle['reduced_tokens']]})
```


```python
topicsProbDict_middle = {i : [0] * len(ldaDF_middle) for i in range(senlda_middle.num_topics)}
```


```python
for index, topicTuples in enumerate(ldaDF_middle['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict_middle[topicNum][index] = prob
for topicNum in range(senlda_middle.num_topics):
    ldaDF_middle['topic_{}'.format(topicNum)] = topicsProbDict_middle[topicNum]
```


```python
ldaDFV_middle = ldaDF_middle[:10][['topic_%d' %x for x in range(10)]]
ldaDFVisN_middle = ldaDF_middle[:10][['status_id']]
ldaDFVis_middle = np.asmatrix(ldaDFV_middle)
ldaDFVisNames_middle = np.asmatrix(ldaDFVisN_middle)
```


```python
N = 10
ind = np.arange(N)
K = senlda_middle.num_topics  # N documents, K topics
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)
for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, np.asarray(ldaDFVis_middle)[:, k], width, color=color)
    else:
        p = plt.bar(ind, np.asarray(ldaDFVis_middle)[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += np.asarray(ldaDFVis_middle)[:, k]
    plots.append(p)
plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')
plt.title('Topics in Tweets (Middle)')
plt.xticks(ind+width/2, ldaDFVisNames_middle, rotation='vertical')
plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', frameon=True,  bbox_to_anchor = (1, .5))
plt.show()
```


![png](output_146_0.png)



```python
plt.pcolor(np.asarray(ldaDFVis_middle), norm=None, cmap='Purples')
plt.yticks(np.arange(ldaDFVis_middle.shape[0])+0.5, ldaDFVisNames_middle);
plt.xticks(np.arange(ldaDFVis_middle.shape[1])+0.5, topic_labels);
plt.gca().invert_yaxis()
plt.xticks(rotation=90)
plt.colorbar(cmap='Blues')
plt.tight_layout()  # fixes margins
plt.show()
```


![png](output_147_0.png)


We visualized the topics of the first ten observations in the Middle tweets.


```python
for i in range(10):
    print('Topic {}:'.format(i))
    print(senlda_middle.show_topic(i))
```

    Topic 0:
    [('home', 0.07641599), ('stay', 0.06887814), ('covid19', 0.060535856), ('work', 0.040641744), ('safe', 0.037533693), ('time', 0.03553239), ('people', 0.021671344), ('house', 0.015721085), ('feel', 0.013739011), ('love', 0.013548531)]
    Topic 1:
    [('test', 0.08219249), ('help', 0.073763855), ('covid19', 0.06463081), ('friend', 0.030900152), ('family', 0.024662724), ('positive', 0.023636324), ('community', 0.022492716), ('people', 0.02192543), ('need', 0.020780692), ('try', 0.019398713)]
    Topic 2:
    [('amp', 0.07409614), ('covid19', 0.048194837), ('know', 0.033816732), ('people', 0.032821245), ('hand', 0.032354333), ('question', 0.02131847), ('time', 0.017597305), ('wash', 0.016834345), ('mask', 0.015884884), ('tell', 0.015763277)]
    Topic 3:
    [('case', 0.08764967), ('covid19', 0.0659801), ('new', 0.044063196), ('death', 0.03760174), ('italy', 0.031733233), ('virus', 0.026515119), ('patient', 0.022712007), ('china', 0.021875456), ('government', 0.020272674), ('spread', 0.018401127)]
    Topic 4:
    [('covid19', 0.06186864), ('update', 0.034711577), ('close', 0.032034613), ('business', 0.023308678), ('school', 0.02056599), ('march', 0.018025106), ('learn', 0.0176354), ('state', 0.017305627), ('amp', 0.014860011), ('city', 0.014728377)]
    Topic 5:
    [('covid19', 0.058476985), ('pandemic', 0.030761328), ('quarantine', 0.023648528), ('crisis', 0.022812668), ('health', 0.022129273), ('self', 0.021010019), ('isolation', 0.01522387), ('public', 0.013847146), ('work', 0.013318218), ('company', 0.013161165)]
    Topic 6:
    [('covid19', 0.049927987), ('social', 0.03466602), ('let', 0.029720146), ('people', 0.029582072), ('distance', 0.027891088), ('right', 0.023806531), ('need', 0.01755226), ('come', 0.016758865), ('stop', 0.016332073), ('know', 0.015289237)]
    Topic 7:
    [('covid19', 0.09922011), ('support', 0.02807544), ('trump', 0.026144074), ('share', 0.024691334), ('time', 0.024062388), ('care', 0.020991426), ('need', 0.02001129), ('video', 0.018597191), ('news', 0.018496756), ('thank', 0.01795503)]
    Topic 8:
    [('covid19', 0.065414324), ('like', 0.037821565), ('fight', 0.03007433), ('day', 0.027697206), ('china', 0.024823576), ('look', 0.024630165), ('world', 0.024512146), ('watch', 0.015959654), ('play', 0.015710486), ('amp', 0.01567805)]
    Topic 9:
    [('coronavirusupdate', 0.072137736), ('covid_19', 0.062250074), ('covid19', 0.04994507), ('coronavirusoutbreak', 0.04166461), ('coronaviruspandemic', 0.03797972), ('corona', 0.02589288), ('lockdown', 0.023708403), ('socialdistancing', 0.023701005), ('coronacrisis', 0.019779168), ('covid2019', 0.019146958)]
    


```python
topicsDict_middle = {}
for topicNum in range(senlda_middle.num_topics):
    topicWords = [w for w, p in senlda_middle.show_topic(topicNum)]
    topicsDict_middle['Topic_{}'.format(topicNum)] = topicWords
```


```python
wordRanksDF_middle = pd.DataFrame(topicsDict_middle)
wordRanksDF_middle
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>home</td>
      <td>test</td>
      <td>amp</td>
      <td>case</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>coronavirusupdate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stay</td>
      <td>help</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>update</td>
      <td>pandemic</td>
      <td>social</td>
      <td>support</td>
      <td>like</td>
      <td>covid_19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>covid19</td>
      <td>covid19</td>
      <td>know</td>
      <td>new</td>
      <td>close</td>
      <td>quarantine</td>
      <td>let</td>
      <td>trump</td>
      <td>fight</td>
      <td>covid19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>work</td>
      <td>friend</td>
      <td>people</td>
      <td>death</td>
      <td>business</td>
      <td>crisis</td>
      <td>people</td>
      <td>share</td>
      <td>day</td>
      <td>coronavirusoutbreak</td>
    </tr>
    <tr>
      <th>4</th>
      <td>safe</td>
      <td>family</td>
      <td>hand</td>
      <td>italy</td>
      <td>school</td>
      <td>health</td>
      <td>distance</td>
      <td>time</td>
      <td>china</td>
      <td>coronaviruspandemic</td>
    </tr>
    <tr>
      <th>5</th>
      <td>time</td>
      <td>positive</td>
      <td>question</td>
      <td>virus</td>
      <td>march</td>
      <td>self</td>
      <td>right</td>
      <td>care</td>
      <td>look</td>
      <td>corona</td>
    </tr>
    <tr>
      <th>6</th>
      <td>people</td>
      <td>community</td>
      <td>time</td>
      <td>patient</td>
      <td>learn</td>
      <td>isolation</td>
      <td>need</td>
      <td>need</td>
      <td>world</td>
      <td>lockdown</td>
    </tr>
    <tr>
      <th>7</th>
      <td>house</td>
      <td>people</td>
      <td>wash</td>
      <td>china</td>
      <td>state</td>
      <td>public</td>
      <td>come</td>
      <td>video</td>
      <td>watch</td>
      <td>socialdistancing</td>
    </tr>
    <tr>
      <th>8</th>
      <td>feel</td>
      <td>need</td>
      <td>mask</td>
      <td>government</td>
      <td>amp</td>
      <td>work</td>
      <td>stop</td>
      <td>news</td>
      <td>play</td>
      <td>coronacrisis</td>
    </tr>
    <tr>
      <th>9</th>
      <td>love</td>
      <td>try</td>
      <td>tell</td>
      <td>spread</td>
      <td>city</td>
      <td>company</td>
      <td>know</td>
      <td>thank</td>
      <td>amp</td>
      <td>covid2019</td>
    </tr>
  </tbody>
</table>
</div>



We can see the detailed information of the specific words and their weight in each topic.


```python
topic1_df_middle = pd.DataFrame(senlda_middle.show_topic(1, topn=50))
plt.figure()
topic1_df_middle.plot.bar(legend = False)
plt.title('Probability Distribution of Words, Topic 1 (Middle)')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_153_1.png)



```python
senlda1_middle = gensim.models.ldamodel.LdaModel(corpus=corpus_middle, id2word=dictionary_middle, num_topics=10, eta = 0.00001)
senlda2_middle = gensim.models.ldamodel.LdaModel(corpus=corpus_middle, id2word=dictionary_middle, num_topics=10, eta = 0.9)
topic11_df_middle = pd.DataFrame(senlda1_middle.show_topic(1, topn=50))
topic21_df_middle = pd.DataFrame(senlda2_middle.show_topic(1, topn=50))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
topic11_df_middle.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.0001')
topic21_df_middle.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
plt.show()
```

    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <ipython-input-102-f934895b1ff8>:8: DeprecationWarning: invalid escape sequence \e
      topic11_df_middle.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.0001')
    <ipython-input-102-f934895b1ff8>:9: DeprecationWarning: invalid escape sequence \e
      topic21_df_middle.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
    


![png](output_154_1.png)


We can make the topics more unique by changing the parameters of the model, where $\alpha$ controls the sparsity of document-topic loadings, and $\eta$ controls the sparsity of topic-word loadings. We visualized the distribution of words over any single topic. The above example shows how different $\eta$ values can change the shape of the distribution.

#### Late


```python
dictionary_late = gensim.corpora.Dictionary(t_late['reduced_tokens'])
corpus_late = [dictionary_late.doc2bow(text) for text in t_late['reduced_tokens']]
senlda_late = gensim.models.ldamodel.LdaModel(corpus=corpus_late, id2word=dictionary_late, num_topics=10, alpha='auto', eta='auto')
sen1Bow_late = dictionary_late.doc2bow(t_late['reduced_tokens'][0].astype(str))
sen1lda_late = senlda_late[sen1Bow_late]
```


```python
print("The topics of the text: {}".format(t_late.iloc[0,0]))
print("are: {}".format(sen1lda_late))
```

    The topics of the text: 1241225643234406406
    are: [(0, 0.11585152), (1, 0.063173644), (2, 0.052012824), (3, 0.09811876), (4, 0.1086971), (5, 0.107367024), (6, 0.095483504), (7, 0.107394665), (8, 0.11803488), (9, 0.1338661)]
    

In the above analysis, we saw the topics related to the first observation in Late tweets and the weight of each topic.


```python
ldaDF_late = pd.DataFrame({'status_id' : t_late['status_id'],'time':t_late['created_at'],
                          'topics' : [senlda_late[dictionary_late.doc2bow(l)] for l in t_late['reduced_tokens']]})
topicsProbDict_late = {i : [0] * len(ldaDF_late) for i in range(senlda_late.num_topics)}
```


```python
for index, topicTuples in enumerate(ldaDF_late['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict_late[topicNum][index] = prob
for topicNum in range(senlda_late.num_topics):
    ldaDF_late['topic_{}'.format(topicNum)] = topicsProbDict_late[topicNum]
```


```python
ldaDFV_late = ldaDF_late[:10][['topic_%d' %x for x in range(10)]]
ldaDFVisN_late = ldaDF_late[:10][['status_id']]
ldaDFVis_late = np.asmatrix(ldaDFV_late)
ldaDFVisNames_late = np.asmatrix(ldaDFVisN_late)
```


```python
N = 10
ind = np.arange(N)
K = senlda_late.num_topics  # N documents, K topics
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)
for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, np.asarray(ldaDFVis_late)[:, k], width, color=color)
    else:
        p = plt.bar(ind, np.asarray(ldaDFVis_late)[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += np.asarray(ldaDFVis_late)[:, k]
    plots.append(p)
plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')
plt.title('Topics in Tweets (Late)')
plt.xticks(ind+width/2, ldaDFVisNames_late, rotation='vertical')
plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', frameon=True,  bbox_to_anchor = (1, .5))
plt.show()
```


![png](output_163_0.png)



```python
plt.pcolor(np.asarray(ldaDFVis_late), norm=None, cmap='Purples')
plt.yticks(np.arange(ldaDFVis_late.shape[0])+0.5, ldaDFVisNames_late);
plt.xticks(np.arange(ldaDFVis_late.shape[1])+0.5, topic_labels);
plt.gca().invert_yaxis()
plt.xticks(rotation=90)
plt.colorbar(cmap='Blues')
plt.tight_layout()  # fixes margins
plt.show()
```


![png](output_164_0.png)


We visualized the topics of the first ten observations in the Middle tweets.


```python
for i in range(10):
    print('Topic {}:'.format(i))
    print(senlda_late.show_topic(i))
```

    Topic 0:
    [('covid19', 0.066672936), ('stay', 0.057962157), ('covid_19', 0.056892365), ('home', 0.054604493), ('stayhome', 0.031354822), ('corona', 0.031260733), ('stayathome', 0.02789843), ('safe', 0.026973065), ('quarantine', 0.025258195), ('coronavirusoutbreak', 0.02384225)]
    Topic 1:
    [('amp', 0.102784514), ('fight', 0.0960389), ('people', 0.060596302), ('come', 0.057696007), ('covid19', 0.05599865), ('pm', 0.03598253), ('month', 0.02844215), ('govt', 0.025155386), ('protect', 0.023669776), ('join', 0.023293259)]
    Topic 2:
    [('mask', 0.12159391), ('family', 0.11716833), ('share', 0.09970049), ('try', 0.091367), ('time', 0.08832946), ('friend', 0.084712505), ('covid19', 0.06314208), ('wear', 0.062003314), ('stand', 0.040351037), ('face', 0.032329205)]
    Topic 3:
    [('covid19', 0.059163857), ('lockdown', 0.03335915), ('people', 0.026694383), ('thank', 0.022851406), ('covid_19', 0.022695845), ('world', 0.022619877), ('country', 0.022040343), ('day', 0.020627635), ('woman', 0.017515626), ('know', 0.017491883)]
    Topic 4:
    [('case', 0.089493304), ('covid19', 0.0676281), ('death', 0.058061488), ('test', 0.05347156), ('new', 0.05064008), ('report', 0.032410096), ('positive', 0.030220447), ('total', 0.025319505), ('china', 0.022873435), ('update', 0.018928606)]
    Topic 5:
    [('covid19', 0.07561164), ('people', 0.021091294), ('covid_19', 0.020096814), ('time', 0.01899627), ('food', 0.017799366), ('pay', 0.017319687), ('watch', 0.01728033), ('amp', 0.016084097), ('happen', 0.013862759), ('make', 0.013076377)]
    Topic 6:
    [('covid19', 0.08749873), ('pandemic', 0.02796408), ('virus', 0.025060961), ('people', 0.019384572), ('doctor', 0.018695682), ('world', 0.017523397), ('god', 0.016871974), ('cause', 0.015247), ('listen', 0.014396088), ('story', 0.014028104)]
    Topic 7:
    [('covid19', 0.097672574), ('deliver', 0.042538956), ('support', 0.03133358), ('hospital', 0.02910455), ('official', 0.02701613), ('worker', 0.026785228), ('act', 0.023936668), ('work', 0.023421925), ('sign', 0.022289023), ('patient', 0.022131525)]
    Topic 8:
    [('covid19', 0.07391739), ('trump', 0.037415236), ('like', 0.026668973), ('coronavirustruth', 0.021182599), ('say', 0.020524198), ('covid_19', 0.018428221), ('people', 0.015613652), ('president', 0.01525129), ('stop', 0.013791927), ('know', 0.013313101)]
    Topic 9:
    [('covid19', 0.10200227), ('amp', 0.027953854), ('help', 0.022347318), ('pandemic', 0.01873707), ('business', 0.018544083), ('read', 0.016254703), ('service', 0.015611273), ('response', 0.013460681), ('need', 0.0128309), ('crisis', 0.012103763)]
    


```python
topicsDict_late = {}
for topicNum in range(senlda_late.num_topics):
    topicWords = [w for w, p in senlda_late.show_topic(topicNum)]
    topicsDict_late['Topic_{}'.format(topicNum)] = topicWords
```


```python
wordRanksDF_late = pd.DataFrame(topicsDict_late)
wordRanksDF_late
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>covid19</td>
      <td>amp</td>
      <td>mask</td>
      <td>covid19</td>
      <td>case</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
      <td>covid19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stay</td>
      <td>fight</td>
      <td>family</td>
      <td>lockdown</td>
      <td>covid19</td>
      <td>people</td>
      <td>pandemic</td>
      <td>deliver</td>
      <td>trump</td>
      <td>amp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>covid_19</td>
      <td>people</td>
      <td>share</td>
      <td>people</td>
      <td>death</td>
      <td>covid_19</td>
      <td>virus</td>
      <td>support</td>
      <td>like</td>
      <td>help</td>
    </tr>
    <tr>
      <th>3</th>
      <td>home</td>
      <td>come</td>
      <td>try</td>
      <td>thank</td>
      <td>test</td>
      <td>time</td>
      <td>people</td>
      <td>hospital</td>
      <td>coronavirustruth</td>
      <td>pandemic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>stayhome</td>
      <td>covid19</td>
      <td>time</td>
      <td>covid_19</td>
      <td>new</td>
      <td>food</td>
      <td>doctor</td>
      <td>official</td>
      <td>say</td>
      <td>business</td>
    </tr>
    <tr>
      <th>5</th>
      <td>corona</td>
      <td>pm</td>
      <td>friend</td>
      <td>world</td>
      <td>report</td>
      <td>pay</td>
      <td>world</td>
      <td>worker</td>
      <td>covid_19</td>
      <td>read</td>
    </tr>
    <tr>
      <th>6</th>
      <td>stayathome</td>
      <td>month</td>
      <td>covid19</td>
      <td>country</td>
      <td>positive</td>
      <td>watch</td>
      <td>god</td>
      <td>act</td>
      <td>people</td>
      <td>service</td>
    </tr>
    <tr>
      <th>7</th>
      <td>safe</td>
      <td>govt</td>
      <td>wear</td>
      <td>day</td>
      <td>total</td>
      <td>amp</td>
      <td>cause</td>
      <td>work</td>
      <td>president</td>
      <td>response</td>
    </tr>
    <tr>
      <th>8</th>
      <td>quarantine</td>
      <td>protect</td>
      <td>stand</td>
      <td>woman</td>
      <td>china</td>
      <td>happen</td>
      <td>listen</td>
      <td>sign</td>
      <td>stop</td>
      <td>need</td>
    </tr>
    <tr>
      <th>9</th>
      <td>coronavirusoutbreak</td>
      <td>join</td>
      <td>face</td>
      <td>know</td>
      <td>update</td>
      <td>make</td>
      <td>story</td>
      <td>patient</td>
      <td>know</td>
      <td>crisis</td>
    </tr>
  </tbody>
</table>
</div>



We can see the detailed information of the specific words and their weight in each topic.


```python
topic1_df_late = pd.DataFrame(senlda_late.show_topic(1, topn=50))
plt.figure()
topic1_df_late.plot.bar(legend = False)
plt.title('Probability Distribution of Words, Topic 1 (Late)')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_170_1.png)



```python
senlda1_late = gensim.models.ldamodel.LdaModel(corpus=corpus_late, id2word=dictionary_late, num_topics=10, eta = 0.00001)
senlda2_late = gensim.models.ldamodel.LdaModel(corpus=corpus_late, id2word=dictionary_late, num_topics=10, eta = 0.9)
topic11_df_late = pd.DataFrame(senlda1_late.show_topic(1, topn=50))
topic21_df_late = pd.DataFrame(senlda2_late.show_topic(1, topn=50))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
topic11_df_late.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.00001')
topic21_df_late.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
plt.show()
```

    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <>:8: DeprecationWarning: invalid escape sequence \e
    <>:9: DeprecationWarning: invalid escape sequence \e
    <ipython-input-114-3bb9b6c0c0ba>:8: DeprecationWarning: invalid escape sequence \e
      topic11_df_late.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.00001')
    <ipython-input-114-3bb9b6c0c0ba>:9: DeprecationWarning: invalid escape sequence \e
      topic21_df_late.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
    


![png](output_171_1.png)


We can make the topics more unique by changing the parameters of the model, where $\alpha$ controls the sparsity of document-topic loadings, and $\eta$ controls the sparsity of topic-word loadings. We visualized the distribution of words over any single topic. The above example shows how different $\eta$ values can change the shape of the distribution.

## Dynamic Topic Modelling

In this part, we want to create a dynamic model for the tweets in March. We will see the change of numbers per day and per stage, and explore the dynamic model of each stage.

### Change per Day


```python
all_tweets = [t_early,t_12,t_13,t_14,t_15,t_16,t_17,t_18,t_19,t_20,t_21,t_22,t_23,t_24,t_25,t_26,t_27,t_28,t_29,t_30,t_31]
lens_day = [len(x) for x in all_tweets]
```


```python
days = []
for i in range(12,32):
    sub_str = '03-'+ str(i)
    days.append(sub_str)
days.insert(0,'before 03-12')
```


```python
fig = plt.figure(figsize = (19,6))
plt.plot(days,lens_day)
plt.title('Number of Tweets in March per Day', size = 18, y = 1.02)
```




    Text(0.5, 1.02, 'Number of Tweets in March per Day')




![png](output_178_1.png)


### Change per Stage


```python
middle_tweets = sum(lens_day[1:10])
late_tweets = sum(lens_day[10:])
lens_stage = [lens_day[0], middle_tweets, late_tweets]
stages = ['early', 'middle', 'late']
plt.plot(stages,lens_stage)
plt.title('Number of Tweets in March per Stage', size = 18, y = 1.02)
```




    Text(0.5, 1.02, 'Number of Tweets in March per Stage')




![png](output_180_1.png)



```python
t_all = pd.concat([t_early, t_middle, t_late],axis = 0)
```


```python
dictionary_all = gensim.corpora.Dictionary(t_all['reduced_tokens'])
corpus_all = [dictionary_early.doc2bow(text) for text in t_all['reduced_tokens']]
```


```python
#ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus_all, id2word=dictionary_all, time_slice=lens_stage, num_topics=10)
#ldaseq.save("ldaseqmodel")
ldaseq = ldaseqmodel.LdaSeqModel.load("ldaseqmodel")
```

Since getting the model is very time-consuming, we suggest you direcely load the generated model instead of getting the model again.


```python
# Topics for the early stage
early_lda = ldaseq.print_topics(time=0) 
```


```python
#Topics for the middle stage
middle_lda = ldaseq.print_topics(time=1) 
```


```python
#Topics for the late stage
late_lda = ldaseq.print_topics(time=2) 
```


```python
def get_df(lda):
    '''
    This function is get a DataFrame of top 10 words in each of the topics of an Idaseq Model.
    
    Input:
    lda (an Idaseq Model): a model
    
    Output:
    a processed DataFrame
    '''
    topics = {}
    for i in range(len(lda)):
        topic = [m[0] for m in lda[i][:10]]
        string = "Topic_"+str(i)
        topics[string] = topic
    return pd.DataFrame(topics)
```


```python
get_df(early_lda)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>help</td>
      <td>like</td>
      <td>coronavirusoutbreak</td>
      <td>hand</td>
      <td>test</td>
      <td>amp</td>
      <td>stay</td>
      <td>day</td>
      <td>pandemic</td>
      <td>case</td>
    </tr>
    <tr>
      <th>1</th>
      <td>health</td>
      <td>coronaviruspandemic</td>
      <td>virus</td>
      <td>spread</td>
      <td>close</td>
      <td>work</td>
      <td>home</td>
      <td>quarantine</td>
      <td>trump</td>
      <td>new</td>
    </tr>
    <tr>
      <th>2</th>
      <td>need</td>
      <td>know</td>
      <td>corona</td>
      <td>covid_19</td>
      <td>school</td>
      <td>business</td>
      <td>people</td>
      <td>week</td>
      <td>world</td>
      <td>death</td>
    </tr>
    <tr>
      <th>3</th>
      <td>support</td>
      <td>think</td>
      <td>covidー19</td>
      <td>watch</td>
      <td>positive</td>
      <td>pay</td>
      <td>safe</td>
      <td>year</td>
      <td>country</td>
      <td>report</td>
    </tr>
    <tr>
      <th>4</th>
      <td>care</td>
      <td>people</td>
      <td>coronavirusupdate</td>
      <td>ask</td>
      <td>state</td>
      <td>food</td>
      <td>time</td>
      <td>feel</td>
      <td>crisis</td>
      <td>lockdown</td>
    </tr>
    <tr>
      <th>5</th>
      <td>community</td>
      <td>look</td>
      <td>coronaviruspandemic</td>
      <td>wash</td>
      <td>say</td>
      <td>home</td>
      <td>social</td>
      <td>self</td>
      <td>live</td>
      <td>italy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>thank</td>
      <td>good</td>
      <td>china</td>
      <td>question</td>
      <td>order</td>
      <td>company</td>
      <td>try</td>
      <td>covid_19</td>
      <td>president</td>
      <td>total</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hospital</td>
      <td>thing</td>
      <td>covid2019</td>
      <td>stop</td>
      <td>uk</td>
      <td>help</td>
      <td>family</td>
      <td>time</td>
      <td>response</td>
      <td>update</td>
    </tr>
    <tr>
      <th>8</th>
      <td>medical</td>
      <td>right</td>
      <td>covid_19</td>
      <td>video</td>
      <td>march</td>
      <td>service</td>
      <td>let</td>
      <td>month</td>
      <td>say</td>
      <td>country</td>
    </tr>
    <tr>
      <th>9</th>
      <td>information</td>
      <td>covid_19</td>
      <td>covid</td>
      <td>soon</td>
      <td>people</td>
      <td>essential</td>
      <td>share</td>
      <td>house</td>
      <td>global</td>
      <td>india</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_df(middle_lda)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>help</td>
      <td>like</td>
      <td>coronavirusoutbreak</td>
      <td>hand</td>
      <td>test</td>
      <td>amp</td>
      <td>stay</td>
      <td>day</td>
      <td>pandemic</td>
      <td>case</td>
    </tr>
    <tr>
      <th>1</th>
      <td>health</td>
      <td>coronaviruspandemic</td>
      <td>virus</td>
      <td>spread</td>
      <td>close</td>
      <td>work</td>
      <td>home</td>
      <td>quarantine</td>
      <td>trump</td>
      <td>new</td>
    </tr>
    <tr>
      <th>2</th>
      <td>need</td>
      <td>know</td>
      <td>corona</td>
      <td>covid_19</td>
      <td>school</td>
      <td>business</td>
      <td>people</td>
      <td>week</td>
      <td>world</td>
      <td>death</td>
    </tr>
    <tr>
      <th>3</th>
      <td>support</td>
      <td>people</td>
      <td>covidー19</td>
      <td>watch</td>
      <td>positive</td>
      <td>pay</td>
      <td>time</td>
      <td>year</td>
      <td>country</td>
      <td>report</td>
    </tr>
    <tr>
      <th>4</th>
      <td>care</td>
      <td>think</td>
      <td>coronaviruspandemic</td>
      <td>ask</td>
      <td>state</td>
      <td>food</td>
      <td>safe</td>
      <td>feel</td>
      <td>crisis</td>
      <td>lockdown</td>
    </tr>
    <tr>
      <th>5</th>
      <td>community</td>
      <td>look</td>
      <td>china</td>
      <td>wash</td>
      <td>say</td>
      <td>home</td>
      <td>social</td>
      <td>self</td>
      <td>live</td>
      <td>italy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>thank</td>
      <td>good</td>
      <td>coronavirusupdate</td>
      <td>question</td>
      <td>order</td>
      <td>help</td>
      <td>try</td>
      <td>covid_19</td>
      <td>president</td>
      <td>total</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hospital</td>
      <td>right</td>
      <td>covid2019</td>
      <td>video</td>
      <td>uk</td>
      <td>company</td>
      <td>family</td>
      <td>time</td>
      <td>response</td>
      <td>update</td>
    </tr>
    <tr>
      <th>8</th>
      <td>medical</td>
      <td>thing</td>
      <td>covid_19</td>
      <td>stop</td>
      <td>march</td>
      <td>job</td>
      <td>let</td>
      <td>month</td>
      <td>say</td>
      <td>india</td>
    </tr>
    <tr>
      <th>9</th>
      <td>doctor</td>
      <td>covid_19</td>
      <td>covid</td>
      <td>soon</td>
      <td>people</td>
      <td>service</td>
      <td>share</td>
      <td>house</td>
      <td>global</td>
      <td>country</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_df(late_lda)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic_0</th>
      <th>Topic_1</th>
      <th>Topic_2</th>
      <th>Topic_3</th>
      <th>Topic_4</th>
      <th>Topic_5</th>
      <th>Topic_6</th>
      <th>Topic_7</th>
      <th>Topic_8</th>
      <th>Topic_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>help</td>
      <td>like</td>
      <td>coronavirusoutbreak</td>
      <td>hand</td>
      <td>test</td>
      <td>amp</td>
      <td>stay</td>
      <td>day</td>
      <td>pandemic</td>
      <td>case</td>
    </tr>
    <tr>
      <th>1</th>
      <td>health</td>
      <td>coronaviruspandemic</td>
      <td>virus</td>
      <td>spread</td>
      <td>close</td>
      <td>work</td>
      <td>home</td>
      <td>quarantine</td>
      <td>trump</td>
      <td>new</td>
    </tr>
    <tr>
      <th>2</th>
      <td>support</td>
      <td>know</td>
      <td>corona</td>
      <td>covid_19</td>
      <td>positive</td>
      <td>business</td>
      <td>people</td>
      <td>week</td>
      <td>world</td>
      <td>death</td>
    </tr>
    <tr>
      <th>3</th>
      <td>need</td>
      <td>people</td>
      <td>covidー19</td>
      <td>watch</td>
      <td>state</td>
      <td>pay</td>
      <td>safe</td>
      <td>year</td>
      <td>country</td>
      <td>lockdown</td>
    </tr>
    <tr>
      <th>4</th>
      <td>care</td>
      <td>think</td>
      <td>coronaviruspandemic</td>
      <td>video</td>
      <td>say</td>
      <td>food</td>
      <td>time</td>
      <td>feel</td>
      <td>crisis</td>
      <td>report</td>
    </tr>
    <tr>
      <th>5</th>
      <td>thank</td>
      <td>right</td>
      <td>china</td>
      <td>ask</td>
      <td>school</td>
      <td>home</td>
      <td>family</td>
      <td>self</td>
      <td>live</td>
      <td>italy</td>
    </tr>
    <tr>
      <th>6</th>
      <td>deliver</td>
      <td>look</td>
      <td>covid2019</td>
      <td>wash</td>
      <td>order</td>
      <td>job</td>
      <td>try</td>
      <td>covid_19</td>
      <td>president</td>
      <td>total</td>
    </tr>
    <tr>
      <th>7</th>
      <td>hospital</td>
      <td>good</td>
      <td>covid_19</td>
      <td>question</td>
      <td>uk</td>
      <td>company</td>
      <td>social</td>
      <td>time</td>
      <td>response</td>
      <td>update</td>
    </tr>
    <tr>
      <th>8</th>
      <td>medical</td>
      <td>thing</td>
      <td>covid</td>
      <td>stop</td>
      <td>march</td>
      <td>help</td>
      <td>share</td>
      <td>month</td>
      <td>say</td>
      <td>india</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mask</td>
      <td>covid_19</td>
      <td>coronavirusupdate</td>
      <td>soon</td>
      <td>people</td>
      <td>essential</td>
      <td>let</td>
      <td>house</td>
      <td>global</td>
      <td>rate</td>
    </tr>
  </tbody>
</table>
</div>




```python
doc_topic, topic_term, doc_lengths, term_frequency, vocab = ldaseq.dtm_vis(time=0, corpus=corpus_all)
vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, 
                               vocab=vocab, term_frequency=term_frequency)
pyLDAvis.display(vis_wrapper)
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el878030360836667602109931664"></div>
<script type="text/javascript">

var ldavis_el878030360836667602109931664_data = {"mdsDat": {"x": [-0.1377326313456319, -0.007021822882079555, -0.04192771992800253, -0.13781762052291738, 0.16087605886422485, -0.09511335477837107, -0.16887924269779028, 0.16828516716115186, 0.3040171691301882, -0.044686003000771635], "y": [-0.1654916479121922, 0.03106507577043725, 0.26040569242416606, 0.14489531917407766, 0.07119635519883653, -0.14040188150908445, -0.024813569200553186, -0.22548236927584056, 0.055725860175650385, -0.007098834845497392], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [13.735901177090582, 11.885135125484444, 11.765683418857694, 10.071806233637467, 9.446077565065968, 9.375805341189949, 9.184986404894916, 8.796615975662627, 8.733511625190234, 7.004477132926123]}, "tinfo": {"Term": ["amp", "case", "coronavirusoutbreak", "test", "work", "stay", "virus", "home", "new", "day", "pandemic", "death", "trump", "coronaviruspandemic", "like", "know", "corona", "help", "close", "covid\u30fc19", "safe", "world", "hand", "quarantine", "coronavirusupdate", "social", "try", "china", "think", "spread", "thing", "look", "paper", "listen", "toilet", "happen", "bad", "actually", "panic", "roll", "know", "right", "good", "thread", "stupid", "funny", "think", "like", "oh", "shit", "talk", "buy", "fuck", "idea", "probably", "maybe", "believe", "mind", "worth", "music", "coronaviruspandemic", "great", "want", "lot", "read", "people", "news", "need", "covid_19", "time", "let", "way", "come", "say", "trump", "president", "americans", "leader", "war", "american", "political", "gop", "leadership", "potus", "administration", "america", "maga", "economy", "pandemic", "lead", "threat", "economic", "global", "foxnews", "world", "hoax", "blame", "msnbc", "save", "failure", "million", "biden", "joebiden", "crisis", "response", "amid", "trumpvirus", "action", "country", "live", "fear", "market", "say", "wrong", "impact", "outbreak", "time", "need", "people", "covid_19", "government", "state", "sign", "community", "medical", "information", "doctor", "deliver", "supply", "wear", "copy", "nhs", "website", "healthcare", "mask", "care", "hospital", "patient", "provide", "guidance", "page", "medicine", "act", "staff", "priority", "official", "access", "visit", "click", "research", "protection", "help", "health", "support", "safety", "free", "thank", "offer", "need", "fight", "public", "advice", "local", "emergency", "update", "risk", "covid_19", "protect", "pandemic", "amp", "work", "business", "pay", "essential", "company", "remote", "small", "food", "tax", "money", "job", "price", "non", "tech", "insurance", "industry", "attention", "usual", "email", "cost", "team", "service", "plan", "technology", "online", "meeting", "office", "govt", "security", "poor", "ppl", "impact", "hard", "home", "help", "continue", "time", "need", "people", "change", "leave", "government", "today", "test", "close", "school", "order", "april", "minister", "cancel", "florida", "negative", "event", "till", "california", "positive", "university", "city", "gov", "league", "monday", "football", "uk", "remain", "result", "march", "department", "open", "governor", "county", "tuesday", "town", "borisjohnson", "state", "shut", "ny", "pm", "say", "break", "place", "government", "people", "public", "health", "coronaviruspandemic", "travel", "covid_19", "week", "spread", "update", "quarantine", "self", "feel", "house", "ago", "year", "isolation", "isolate", "white", "sense", "fun", "day", "wait", "week", "month", "old", "enjoy", "game", "sorry", "common", "long", "end", "black", "past", "st", "happy", "room", "hear", "crazy", "nature", "man", "today", "play", "come", "tell", "start", "covid_19", "make", "time", "leave", "like", "people", "live", "think", "stay", "safe", "try", "stand", "healthy", "god", "italian", "social", "family", "home", "love", "share", "practice", "responsible", "let", "die", "forward", "strong", "respond", "away", "elderly", "protect", "remember", "time", "dear", "support", "italy", "people", "flattenthecurve", "catch", "hope", "pm", "fight", "help", "covid_19", "live", "coronaviruspandemic", "come", "need", "care", "coronavirusoutbreak", "virus", "corona", "covid\u30fc19", "covid", "coronavirusupdate", "chinese", "wuhan", "coronavirusindia", "covid2019", "coronavirususa", "coronavirustruth", "wuhanvirus", "cure", "coronavirusupdates", "coronavirussa", "coronaoutbreak", "covid19india", "vaccine", "coronaviruschallenge", "pakistan", "coronaalert", "scare", "coronavirus", "deadly", "coronavid19", "china", "trumpdemic", "coverage", "covid2019uk", "viral", "coronavirusinindia", "sarscov2", "coronavirusuk", "flu", "kill", "coronaviruspandemic", "covid_19", "world", "spread", "india", "disease", "people", "stop", "case", "death", "report", "total", "rate", "rise", "spain", "york", "toll", "france", "number", "south", "korea", "germany", "new", "mar", "infection", "lockdown", "africa", "india", "italy", "increase", "japan", "iran", "population", "update", "usa", "dead", "nigeria", "worldwide", "high", "country", "china", "late", "bring", "day", "today", "march", "news", "covid_19", "break", "hand", "wash", "question", "video", "soon", "answer", "post", "clean", "story", "conference", "press", "touch", "message", "air", "washyourhands", "water", "sanitizer", "soap", "slow", "second", "avoid", "ask", "facebook", "hit", "watch", "hygiene", "cough", "simple", "prevent", "spread", "face", "check", "stop", "travel", "use", "covid_19", "real"], "Freq": [8342.0, 5785.0, 5534.0, 5072.0, 4640.0, 4418.0, 4037.0, 4849.0, 4000.0, 4491.0, 4519.0, 3160.0, 3350.0, 6737.0, 4318.0, 3582.0, 2574.0, 4730.0, 2469.0, 2365.0, 2389.0, 3419.0, 2105.0, 2349.0, 2248.0, 2374.0, 2134.0, 2715.0, 3108.0, 3476.0, 1990.323107437475, 2041.1523713761128, 810.691058621194, 952.9460847972007, 902.7777284978184, 1167.416676008726, 640.1869671689537, 512.9161090214354, 1067.863820190355, 210.49502597045708, 3400.078487061123, 1911.5610794353083, 2031.5189980049724, 389.6163857435508, 242.42641931961128, 109.04724395772695, 2634.097805868235, 3578.5150155750507, 269.87487850451805, 459.52636420158996, 804.1249197252947, 727.1709521759931, 651.9748302460007, 419.6791028877345, 241.95060360197476, 372.2817721880061, 450.915963719771, 284.65816712327427, 187.70923042987155, 240.14589066099367, 3434.458991309503, 903.0247830353503, 1123.368297454217, 559.9740648652071, 810.1643646188529, 2584.1686696026845, 826.6238366902654, 1357.3439762817998, 1802.8833119645085, 1184.928509733586, 709.0781821881135, 614.2346234676838, 565.9031802742786, 546.8909969520341, 3338.38693666992, 1229.2071303666485, 744.550517894007, 739.2546744811428, 523.4903187579248, 581.8719031316859, 348.23121719910716, 183.1350439611113, 495.97462350261276, 189.01904292248784, 287.8154546301019, 761.0091798152939, 262.77940598298176, 795.784673430207, 4006.2058837351206, 604.2563456652745, 352.45866226616823, 508.5661178786475, 897.4139093439595, 106.74694847370893, 2792.673567311342, 244.7762059235161, 340.10372718192286, 49.047467300334304, 824.3055274102641, 111.2278186431353, 256.1996178244425, 103.98775830297326, 24.546805541773374, 1488.8638706075933, 1095.936592004247, 478.9625507993659, 227.30753744273736, 534.7479871959388, 1529.7712806531092, 1423.5300115481243, 457.56906105689757, 488.2226792217192, 927.3256616295905, 519.7097653082054, 472.52276348402233, 602.0255602299833, 800.674187927729, 765.055129189856, 729.9741883665987, 568.2796759144733, 494.2962797534442, 493.29995574865706, 1187.4118726446632, 1563.7531708736328, 1358.8193290451682, 1288.7984036661417, 1269.0259621979144, 635.3930758074702, 741.668043445682, 279.80346349560375, 386.99892942214086, 677.2721380395417, 404.7985938843566, 907.0810312234825, 1126.4452036195842, 1833.5269413006636, 1400.2016694384104, 1231.5638612307962, 712.7607765845978, 262.5548976271366, 247.46231211005494, 214.8376297544114, 825.3845610149657, 801.3352196407129, 125.53571652119093, 816.1544159518083, 326.2923547339888, 427.2886735026779, 121.04164593190715, 281.54689568135706, 120.51738761920011, 3243.566907744785, 2395.357177149226, 1956.61064179219, 482.2945164638453, 830.1173482903458, 1459.9613908706062, 447.99841809354325, 2338.012694668554, 1087.0788136915096, 821.6308871470219, 473.82430214484384, 485.8828817360607, 495.5004963015237, 543.5515886349141, 502.2933239677953, 525.1076588305078, 489.8091457366094, 494.0857017639937, 8336.453518520259, 4631.830663935226, 1739.965651067443, 1188.6189481789584, 774.0167760479401, 852.0551358850831, 264.97293482640987, 565.8227883592098, 988.3005054260183, 222.5430913223167, 735.9652570649263, 762.2212086873596, 158.24825942813845, 355.79361768818495, 102.49216586528104, 82.60970850690057, 317.9710350549705, 107.97267952781235, 86.29612291474555, 236.4104409078104, 269.5876828991609, 577.8521078520822, 781.0406603604283, 754.8473041377417, 171.82772976153888, 563.7334999800029, 87.3891450208052, 339.37042375022804, 442.5971517619555, 146.8850252782109, 203.0710760608646, 213.00446502756543, 378.29905787961707, 323.2792288077197, 924.6274007341392, 847.214259773098, 343.6200841320155, 729.0925104080499, 694.2318438146856, 760.7142729915513, 306.88648193644593, 307.9270873885069, 299.4662572006274, 281.17623486607846, 5062.821706152942, 2458.722937870667, 1880.938332717123, 1155.0874473423748, 521.4461209832536, 616.8348671939868, 817.0087708883146, 368.71392251063065, 271.96074242096347, 493.1696678741136, 364.54608003869686, 259.64706863095853, 1601.7556758652265, 171.52938958113958, 837.4959723390186, 303.7074144933289, 108.82829310007374, 260.8553274956414, 102.41215027528149, 1079.816055502618, 362.4061762970198, 380.39701242160294, 1058.792797788687, 92.01573454196688, 759.6975066519092, 321.0451526542858, 471.84837476954726, 72.89660861619173, 84.25060255315344, 97.80460545637261, 1490.2125652949067, 499.6471243938293, 110.28467077558467, 629.0631974082085, 1487.4333969486745, 703.5173020513623, 512.1743882534145, 791.0793128451551, 868.8022457867276, 449.22651560655896, 573.1484802159705, 660.56790586074, 404.2627708992773, 632.3221240959435, 429.1016814921151, 414.34628084458717, 390.4129835264174, 2340.174149699135, 1384.8128607877688, 1520.2920465804252, 937.7432701536292, 604.2071137092794, 1640.3244099621857, 894.7518510799604, 707.648054651384, 349.3097498503535, 341.1051892136851, 231.8691940853989, 3769.9611958932933, 670.3054076265462, 1987.5964863750628, 983.2004279599989, 782.7358678912668, 252.13834178567302, 423.35599846740547, 199.29129375723448, 221.43971624683053, 883.4509465413205, 713.4685038577143, 80.08349109926347, 251.26145406121657, 75.77656010900395, 312.86205068169016, 234.21573426401682, 586.8495706348165, 179.1661615461065, 138.5074049106803, 406.35723115203473, 856.3978044216028, 356.94650475113343, 861.4683397831559, 577.9714842623836, 566.4417242692042, 1339.812846848092, 447.14613008163656, 993.4073537578151, 426.1735027923953, 730.5272090479291, 901.4768460083478, 523.4055123387353, 462.34099201275154, 4409.548004882711, 2383.9287593527843, 2123.6694151328866, 946.2612033534773, 643.0513972021691, 949.8037798384099, 627.3487987708536, 2225.316653264505, 1864.8047636920865, 3916.059852604988, 1027.1930761548617, 1476.2886638860114, 413.62213638703525, 288.7914869533644, 1662.57706936039, 829.6231053204671, 199.83624784984588, 210.1771108820151, 140.5493411411075, 364.4782254689147, 153.9241999449014, 563.011313910377, 284.66203614623095, 2299.675948247104, 138.4349477545048, 971.3277529956331, 734.3997486763991, 2685.858268172242, 150.92245464427597, 64.4729786566602, 345.177339937684, 259.1567298248394, 452.9290582158074, 620.3484886950391, 935.7855535700592, 422.0784299170102, 534.3841637872875, 369.92059706695625, 389.91721156849206, 294.3039081721314, 5526.998324953657, 4029.6333614401483, 2565.804594465219, 2355.7322737835366, 1389.179956310208, 2233.8676768896703, 766.3751329458637, 495.4160804587254, 562.5381398261618, 1860.624535548492, 411.50904880835606, 247.90008266906668, 267.26637472302247, 509.93586278668033, 572.1326404041317, 96.94157009757535, 963.936268728451, 158.91026557139824, 525.7831282816483, 158.33333072875476, 296.86503583369154, 53.32132732583732, 22.616513100513032, 1337.3988885799297, 229.17829799993052, 48.47866523690003, 1969.433470467603, 20.26003023697103, 12.944713107568656, 16.677140806134663, 145.23356833371415, 35.198437256614426, 94.62838603750583, 227.37115525433254, 377.94711146396327, 419.6966749622321, 2054.8970883209604, 1420.8867312554833, 616.6420194822325, 593.1925588572701, 297.34430804428854, 251.90862427573887, 394.3089750243508, 264.60804559601894, 5776.368063894028, 3151.9828914423406, 1808.228587691743, 1297.2204221822594, 794.015897279844, 520.2567927905793, 474.0447540587673, 357.4031252237235, 441.18618607823987, 332.3043460757723, 614.899816550791, 444.3743196424995, 279.85782471173366, 335.3703300973051, 3818.080566264684, 112.15413938785754, 581.1632817780653, 1752.768154109871, 247.35142442179472, 830.6233596126015, 1478.124610613689, 448.81967680205184, 83.78850926284578, 357.0946223790586, 310.52905952136416, 1270.005466783376, 509.6784229141788, 213.33130397508555, 189.74620853186954, 163.4876170435702, 465.718041373175, 858.6351366525913, 738.7012095878551, 484.2897542421001, 331.6818413294353, 713.7462342847723, 459.29123891526797, 381.57985617570483, 399.96802024870857, 438.9892323357738, 358.00508350629, 2096.6013254553154, 1062.4607153680493, 995.8825237708637, 845.6086802367391, 808.7646150512694, 523.0445703972204, 678.6233156719306, 567.2290153197484, 633.9758927585609, 509.6113686597325, 474.64884409720855, 501.90362361529975, 667.7801968947037, 424.79963065271045, 304.6214691956877, 465.939367873498, 236.78823748498505, 143.13993709121112, 508.42464399008827, 115.31023419188163, 582.9841058680407, 1085.6061664221654, 196.44051009518788, 518.7325403758197, 1181.3497374845333, 118.03509496191982, 369.07096752171987, 222.48274336680717, 492.24059283611956, 1930.7033292453614, 655.1409364337588, 586.1137813022884, 865.8533076724998, 515.7361437777143, 523.9313129377348, 1683.9609404441796, 401.39234270312636], "Total": [8342.0, 5785.0, 5534.0, 5072.0, 4640.0, 4418.0, 4037.0, 4849.0, 4000.0, 4491.0, 4519.0, 3160.0, 3350.0, 6737.0, 4318.0, 3582.0, 2574.0, 4730.0, 2469.0, 2365.0, 2389.0, 3419.0, 2105.0, 2349.0, 2248.0, 2374.0, 2134.0, 2715.0, 3108.0, 3476.0, 2000.607796045099, 2055.2729924895702, 819.458676996352, 963.4865182046128, 914.6025353815427, 1182.7790889574096, 653.5682447871644, 527.334138214147, 1109.1039143031835, 221.23025125099298, 3582.1616721228656, 2079.572003736248, 2224.5118989061666, 451.557283637132, 281.49233002305505, 127.74902440395587, 3108.805646188741, 4318.257075523915, 342.9600561240355, 597.9991053391221, 1056.7725378902765, 960.7323271270683, 873.6484208588226, 568.9523972888855, 334.43721894000487, 534.8191256186292, 675.6132919781644, 428.976561774119, 283.88377458731713, 368.94029257579126, 6737.167592278851, 1609.9756060035086, 2128.41275372442, 956.4496807402867, 1673.1625585492984, 9060.568985900063, 1989.753714550745, 5712.177496307746, 9623.053247841237, 6257.8671572461, 2581.9768813772753, 2049.8023269934056, 2829.1634003768813, 3251.0810059960686, 3350.5754468356163, 1239.084863516912, 756.4463094431278, 753.0329600849227, 533.666907138771, 594.8498220719295, 363.68179527189625, 192.34550557097364, 522.1693283108818, 202.3193584186382, 312.78409401615954, 839.1911379916801, 293.641364386453, 895.9144350266718, 4519.766073693988, 683.7339866246808, 405.2880374530369, 592.5949394804178, 1052.069454105277, 125.18089455148608, 3419.3591065438677, 300.210069272849, 425.3193078330464, 62.630142673072335, 1068.4087644283493, 145.3530417659339, 341.12301534740004, 142.42755771214348, 33.81091366500104, 2079.875159737344, 1536.1281643787843, 674.4669634753853, 318.1684901193093, 826.738885078069, 2876.525335419494, 3068.9563761854065, 807.7673654484626, 913.9109651915763, 3251.0810059960686, 1190.9995041359643, 1023.2725336909446, 1865.2482102674762, 6257.8671572461, 5712.177496307746, 9060.568985900063, 9623.053247841237, 2283.5150950080997, 2309.6150067703866, 1197.2852133384793, 1577.4888144140507, 1370.7877785028197, 1302.5592917227723, 1283.0570772112567, 643.9139257742956, 757.6543004863856, 288.09046924689227, 401.10807451668796, 713.9872845783215, 429.8223320964169, 978.4829835784795, 1246.7957976916364, 2138.637581893725, 1720.451379851141, 1577.1474283840505, 913.9898906606938, 339.2123879743533, 320.93988796603736, 278.75127396406737, 1117.956564997112, 1093.4739946124241, 171.73657923447226, 1121.2742846902722, 448.5351784348826, 592.3068113511885, 172.85048379991795, 404.4989137700739, 174.784259127388, 4730.9044952071745, 3608.693881736888, 2936.423311646993, 714.2039060959665, 1274.1275829092476, 2355.616599444339, 677.141690684107, 5712.177496307746, 2173.0609676983827, 1618.613948911684, 765.0110823360714, 910.625743917561, 1132.8020447072836, 2212.6783750871136, 1403.8799022440596, 9623.053247841237, 1439.1308549719124, 4519.766073693988, 8342.56343042624, 4640.798076569315, 1752.1584989019714, 1199.2102596172235, 782.1502957314632, 866.2877725245191, 272.98542283047186, 591.9898498417093, 1122.7866630272663, 257.2537304553409, 916.9198923644173, 968.6686542507306, 203.11101485827527, 457.85292313606146, 136.3154283026926, 111.7534187339862, 435.3018157108971, 148.73505951395504, 121.31861390946305, 347.88981524566873, 397.46816767225346, 863.6993643030628, 1174.4402313379112, 1136.6746330452827, 266.7215278856328, 879.2926392867039, 138.66890946563316, 539.0288302994824, 705.7889541405963, 277.6018058234316, 389.41673077925384, 418.65646384865096, 1023.2725336909446, 842.1190046377511, 4849.52289982238, 4730.9044952071745, 1159.0890981717212, 6257.8671572461, 5712.177496307746, 9060.568985900063, 1008.8475221855063, 1165.333463766624, 2283.5150950080997, 2683.557847331756, 5072.277784697287, 2469.6138888405835, 1892.5321877539086, 1166.3249523173524, 529.8626383928984, 627.3742375467207, 833.8513125712541, 378.5642831277343, 280.3326633850415, 509.62232177238803, 382.11312029360295, 280.2044628577591, 1810.5189230289322, 195.15266503934419, 954.8892905244804, 353.8814974343095, 130.63553504859019, 347.7266607443008, 136.5991850123972, 1441.302764558187, 489.4757585066676, 514.1489752581411, 1449.3963154230335, 127.13373264374619, 1083.565503403426, 467.24499632321636, 706.9792699764893, 109.51808216313138, 127.83531079904994, 148.48680237611052, 2309.6150067703866, 798.9278692726222, 168.07712480788518, 1073.3616365407763, 3251.0810059960686, 1484.552461988188, 1127.2979382977564, 2283.5150950080997, 9060.568985900063, 1618.613948911684, 3608.693881736888, 6737.167592278851, 1164.0734417274316, 9623.053247841237, 2425.5601536109752, 3476.6014182542476, 2212.6783750871136, 2349.2432895503052, 1393.2985769858826, 1529.7305540877421, 949.2378298376997, 612.7299398556014, 1664.5830833132488, 907.9963546632, 719.3718372124822, 358.7910099843081, 352.34270896803184, 274.9424345126852, 4491.90726145771, 810.1910351961656, 2425.5601536109752, 1207.7343189584055, 994.3436606494412, 320.9521975530268, 549.4728770156094, 270.67655493465395, 302.3856706868717, 1228.7493847538567, 996.0754942020062, 112.58254839274137, 354.5328236210905, 107.72750024532827, 445.5902449023911, 342.07156966017635, 968.4089548871955, 310.78751606631903, 263.5567577208087, 910.0004892835115, 2683.557847331756, 837.8120553802056, 2829.1634003768813, 1702.418535145991, 1661.9451957947088, 9623.053247841237, 1227.140973122207, 6257.8671572461, 1165.333463766624, 4318.257075523915, 9060.568985900063, 3068.9563761854065, 3108.805646188741, 4418.472943904951, 2389.589926002966, 2134.7190884331953, 957.4706394819414, 654.2657139104326, 966.7931785586509, 638.5715268740881, 2374.530425009883, 2128.1281935703996, 4849.52289982238, 1345.4849696279919, 1956.847189871223, 583.1289728463391, 433.8616887179096, 2581.9768813772753, 1590.2251777471765, 385.17909723273846, 413.42814045559015, 309.51687379990534, 841.9296682083933, 368.8331058948232, 1439.1308549719124, 739.599979112898, 6257.8671572461, 385.84391210620316, 2936.423311646993, 2221.8685511793515, 9060.568985900063, 569.7597829342692, 244.360774477272, 1439.131758222476, 1073.3616365407763, 2173.0609676983827, 4730.9044952071745, 9623.053247841237, 3068.9563761854065, 6737.167592278851, 2829.1634003768813, 5712.177496307746, 2138.637581893725, 5534.795301293627, 4037.81340489366, 2574.6068854417035, 2365.632223899339, 1396.9359015946932, 2248.551018947691, 775.2590774143237, 504.3557861321149, 573.0235723185605, 1897.145925596418, 420.0789159762269, 253.4503208828014, 276.5666179763012, 538.4702382819498, 633.8126759555076, 108.70133941302127, 1105.3964034247144, 184.84239300332402, 619.5653330756476, 198.9488454269973, 373.6920410096208, 68.31799060999404, 29.419364746572974, 1751.09383859306, 305.1069738123876, 66.32900298783109, 2715.6164266554642, 28.564005194190973, 18.884060595419406, 25.080957355859557, 229.5841512469962, 53.881290984152535, 160.43080709635157, 426.0279916579016, 754.1257185829928, 911.4083946317512, 6737.167592278851, 9623.053247841237, 3419.3591065438677, 3476.6014182542476, 1137.267606185847, 901.1299872121115, 9060.568985900063, 2171.9376749299054, 5785.017483677269, 3160.697461367903, 1816.502713839283, 1305.4576646926437, 805.7037492064981, 529.3986661193121, 482.39463173213977, 364.2899988969507, 450.7655999008004, 339.93134699397825, 629.8066981693273, 455.98689088272283, 289.19496508763626, 347.5385074008919, 4000.38791463547, 125.87791111275916, 772.8382459968082, 2333.465540352996, 332.15268997193505, 1137.267606185847, 2221.8685511793515, 723.9715663733683, 137.3354029170588, 601.5431996824409, 538.1463893433503, 2212.6783750871136, 888.8034879095138, 380.3637642730906, 342.26627964929776, 323.68260067210423, 1060.018897014002, 2876.525335419494, 2715.6164266554642, 1539.087949227871, 904.9321197029502, 4491.90726145771, 2683.557847331756, 1449.3963154230335, 1989.753714550745, 9623.053247841237, 1484.552461988188, 2105.3866056049915, 1070.1066565890371, 1005.1329563642395, 853.5118326066722, 817.7269766536407, 531.3813091400382, 690.2626782702374, 576.9694038819528, 645.0357356881105, 518.7047406808321, 483.4386327409466, 511.52634704350316, 680.6918103584824, 436.7179918026367, 314.0630749729838, 480.68550532860354, 244.9921782563478, 149.7325632370291, 559.0351367726215, 135.49027632352306, 692.0496660943621, 1302.9859174779253, 254.56823472880734, 672.3522780848064, 1551.3977389533634, 157.95750791379325, 497.1406163042734, 307.9011580118844, 712.024535471428, 3476.6014182542476, 1312.7784598794121, 1175.5380333432006, 2171.9376749299054, 1164.0734417274316, 1263.165315571497, 9623.053247841237, 937.6054495509816], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.5793, -3.554, -4.4774, -4.3158, -4.3698, -4.1128, -4.7136, -4.9352, -4.2019, -5.8259, -3.0438, -3.6196, -3.5588, -5.2102, -5.6846, -6.4835, -3.299, -2.9926, -5.5774, -5.0451, -4.4856, -4.5862, -4.6953, -5.1358, -5.6866, -5.2557, -5.064, -5.524, -5.9404, -5.6941, -3.0337, -4.3696, -4.1512, -4.8474, -4.4781, -3.3182, -4.458, -3.962, -3.6782, -4.0979, -4.6114, -4.7549, -4.8369, -4.8711, -2.9174, -3.9165, -4.4178, -4.425, -4.7701, -4.6643, -5.1777, -5.8204, -4.8241, -5.7887, -5.3683, -4.3959, -5.4593, -4.3513, -2.735, -4.6266, -5.1657, -4.799, -4.2311, -6.3601, -3.0958, -5.5302, -5.2013, -7.1378, -4.3161, -6.319, -5.4846, -6.3863, -7.83, -3.7248, -4.0312, -4.859, -5.6043, -4.7488, -3.6977, -3.7697, -4.9047, -4.8398, -4.1983, -4.7773, -4.8725, -4.6303, -4.3451, -4.3906, -4.4376, -4.688, -4.8275, -4.8295, -3.941, -3.6656, -3.8061, -3.859, -3.8745, -4.5662, -4.4116, -5.3864, -5.0621, -4.5024, -5.0171, -4.2103, -3.9937, -3.5065, -3.7761, -3.9045, -4.4513, -5.45, -5.5092, -5.6506, -4.3046, -4.3342, -6.1879, -4.3159, -5.2327, -4.963, -6.2244, -5.3802, -6.2287, -2.9361, -3.2392, -3.4415, -4.8419, -4.2989, -3.7343, -4.9157, -3.2634, -4.0292, -4.3092, -4.8597, -4.8345, -4.8149, -4.7224, -4.8013, -4.7569, -4.8265, -4.8178, -1.8367, -2.4243, -3.4034, -3.7845, -4.2135, -4.1174, -5.2854, -4.5268, -3.9691, -5.4599, -4.2639, -4.2288, -5.8009, -4.9907, -6.2353, -6.4509, -5.1031, -6.1832, -6.4073, -5.3995, -5.2682, -4.5057, -4.2044, -4.2385, -5.7186, -4.5305, -6.3947, -5.038, -4.7724, -5.8754, -5.5515, -5.5037, -4.9294, -5.0865, -4.0357, -4.1231, -5.0255, -4.2732, -4.3222, -4.2308, -5.1386, -5.1352, -5.163, -5.2261, -2.2712, -2.9935, -3.2614, -3.749, -4.5443, -4.3763, -4.0953, -4.8909, -5.1952, -4.6001, -4.9023, -5.2416, -3.422, -5.6562, -4.0705, -5.0848, -6.1111, -5.2369, -6.1719, -3.8164, -4.9081, -4.8597, -3.836, -6.2789, -4.168, -5.0293, -4.6442, -6.5119, -6.3671, -6.2179, -3.4942, -4.587, -6.0978, -4.3567, -3.4961, -4.2448, -4.5622, -4.1275, -4.0338, -4.6934, -4.4498, -4.3078, -4.7988, -4.3515, -4.7392, -4.7742, -4.8337, -3.0355, -3.5601, -3.4668, -3.95, -4.3895, -3.3908, -3.9969, -4.2315, -4.9375, -4.9612, -5.3473, -2.5586, -4.2857, -3.1988, -3.9026, -4.1306, -5.2635, -4.7452, -5.4987, -5.3933, -4.0096, -4.2233, -6.4104, -5.2669, -6.4656, -5.0477, -5.3372, -4.4187, -5.6051, -5.8625, -4.7862, -4.0407, -4.9159, -4.0348, -4.4339, -4.4541, -3.5932, -4.6906, -3.8923, -4.7386, -4.1997, -3.9894, -4.5331, -4.6571, -2.3813, -2.9964, -3.112, -3.9204, -4.3067, -3.9166, -4.3314, -3.0652, -3.242, -2.5, -3.8383, -3.4756, -4.7479, -5.1072, -3.3568, -4.0519, -5.4754, -5.4249, -5.8273, -4.8744, -5.7364, -4.4396, -5.1216, -3.0324, -5.8425, -3.8942, -4.1738, -2.8771, -5.7561, -6.6066, -4.9288, -5.2154, -4.6571, -4.3426, -3.9315, -4.7277, -4.4918, -4.8596, -4.8069, -5.0883, -2.1123, -2.4282, -2.8796, -2.9651, -3.4932, -3.0182, -4.088, -4.5243, -4.3972, -3.201, -4.7098, -5.2166, -5.1414, -4.4954, -4.3803, -6.1556, -3.8586, -5.6613, -4.4648, -5.665, -5.0364, -6.7533, -7.611, -3.5312, -5.2952, -6.8485, -3.1442, -7.721, -8.169, -7.9156, -5.7513, -7.1687, -6.1797, -5.3031, -4.7949, -4.6901, -3.1017, -3.4706, -4.3054, -4.3442, -5.0348, -5.2006, -4.7525, -5.1514, -2.0609, -2.6667, -3.2224, -3.5545, -4.0454, -4.4682, -4.5612, -4.8436, -4.633, -4.9164, -4.301, -4.6258, -5.0882, -4.9072, -2.475, -6.0026, -4.3574, -3.2535, -5.2117, -4.0003, -3.4239, -4.6159, -6.2942, -4.8445, -4.9842, -3.5757, -4.4887, -5.3596, -5.4768, -5.6257, -4.5789, -3.9671, -4.1176, -4.5398, -4.9183, -4.1519, -4.5928, -4.7782, -4.7311, -4.638, -4.8419, -2.8538, -3.5335, -3.5982, -3.7618, -3.8063, -4.2422, -3.9818, -4.1611, -4.0498, -4.2682, -4.3393, -4.2834, -3.9979, -4.4502, -4.7828, -4.3578, -5.0347, -5.538, -4.2705, -5.7542, -4.1337, -3.512, -5.2215, -4.2505, -3.4274, -5.7309, -4.5909, -5.097, -4.3029, -2.9362, -4.017, -4.1283, -3.7381, -4.2563, -4.2405, -3.073, -4.5069], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.98, 1.9783, 1.9744, 1.9742, 1.9721, 1.9721, 1.9645, 1.9574, 1.9473, 1.9354, 1.933, 1.9009, 1.8944, 1.8376, 1.8358, 1.8269, 1.8195, 1.7973, 1.7455, 1.7218, 1.7119, 1.7066, 1.6925, 1.6809, 1.6614, 1.6229, 1.5808, 1.575, 1.5715, 1.5558, 1.3114, 1.4069, 1.3461, 1.4498, 1.2599, 0.7306, 1.1067, 0.5481, 0.3104, 0.321, 0.6928, 0.78, 0.3758, 0.2027, 2.1262, 2.1219, 2.114, 2.1114, 2.1106, 2.1078, 2.0865, 2.0808, 2.0784, 2.0619, 2.0467, 2.0321, 2.0188, 2.0114, 2.0093, 2.0063, 1.9902, 1.977, 1.9709, 1.9706, 1.9274, 1.9257, 1.9063, 1.8854, 1.8705, 1.8623, 1.8436, 1.8153, 1.8097, 1.7956, 1.7922, 1.7876, 1.7936, 1.6942, 1.4984, 1.3617, 1.5615, 1.5029, 0.8754, 1.3006, 1.3572, 0.999, 0.0737, 0.1195, -0.3888, -0.6994, 0.5995, 0.5862, 2.1317, 2.1312, 2.1312, 2.1294, 2.129, 2.1267, 2.1187, 2.1108, 2.1042, 2.0872, 2.08, 2.0642, 2.0385, 1.9861, 1.934, 1.8927, 1.8913, 1.8838, 1.88, 1.8795, 1.8366, 1.8291, 1.8266, 1.8224, 1.8218, 1.8134, 1.7837, 1.7776, 1.7682, 1.7625, 1.7302, 1.734, 1.7474, 1.7115, 1.6616, 1.7269, 1.2467, 1.4473, 1.4619, 1.6609, 1.5118, 1.3131, 0.7361, 1.1122, -0.7683, 1.0622, -0.0735, 2.2947, 2.2935, 2.2884, 2.2866, 2.285, 2.2789, 2.2656, 2.2502, 2.1678, 2.1505, 2.0756, 2.0557, 2.0458, 2.0432, 2.0102, 1.9933, 1.9814, 1.9751, 1.9548, 1.9091, 1.9072, 1.8935, 1.8875, 1.8861, 1.8557, 1.8509, 1.8337, 1.8328, 1.8288, 1.6589, 1.6443, 1.6197, 1.3004, 1.338, 0.6382, 0.5755, 1.0796, 0.1456, 0.1879, -0.182, 1.1053, 0.9645, 0.264, 0.0395, 2.3577, 2.3552, 2.3534, 2.3499, 2.3436, 2.3426, 2.3392, 2.3332, 2.3293, 2.3268, 2.3125, 2.2834, 2.2371, 2.2305, 2.2284, 2.2067, 2.1769, 2.0721, 2.0715, 2.0708, 2.059, 2.0583, 2.0456, 2.0363, 2.0045, 1.9843, 1.9552, 1.9525, 1.9426, 1.942, 1.9214, 1.8902, 1.9382, 1.8253, 1.5776, 1.6128, 1.5707, 1.2995, 0.015, 1.0778, 0.5196, 0.0373, 1.302, -0.3629, 0.6274, 0.2325, 0.6248, 2.3632, 2.3609, 2.3608, 2.3549, 2.353, 2.3524, 2.3523, 2.3506, 2.3403, 2.3346, 2.1966, 2.1918, 2.1775, 2.1679, 2.1613, 2.1278, 2.1257, 2.1063, 2.0609, 2.0555, 2.0371, 2.0334, 2.0264, 2.0227, 2.0152, 2.0134, 1.9883, 1.8662, 1.8162, 1.7237, 1.5608, 1.2249, 1.5138, 1.1779, 1.2868, 1.2907, 0.3954, 1.3575, 0.5266, 1.3611, 0.5902, 0.0594, 0.5983, 0.4613, 2.3856, 2.3852, 2.3824, 2.3758, 2.3703, 2.3699, 2.3699, 2.3227, 2.2555, 2.1738, 2.1177, 2.1058, 2.0441, 1.9806, 1.9474, 1.7369, 1.7314, 1.7111, 1.5981, 1.5504, 1.5137, 1.4491, 1.4328, 1.3865, 1.3626, 1.2813, 1.2805, 1.1717, 1.0592, 1.0552, 0.9599, 0.9665, 0.8194, 0.356, 0.0571, 0.4037, -0.1467, 0.3532, -0.2968, 0.4043, 2.4294, 2.4288, 2.4274, 2.4266, 2.4252, 2.4243, 2.4193, 2.4129, 2.4123, 2.4114, 2.4102, 2.4087, 2.3966, 2.3764, 2.3284, 2.3163, 2.2939, 2.2796, 2.2667, 2.2025, 2.2006, 2.183, 2.1678, 2.1613, 2.1446, 2.1173, 2.1095, 2.0873, 2.0532, 2.0227, 1.9729, 2.005, 1.9029, 1.8029, 1.74, 1.6553, 1.2434, 0.5179, 0.7179, 0.6625, 1.0893, 1.1562, -0.7037, 0.3257, 2.4365, 2.4352, 2.4334, 2.4317, 2.4234, 2.4206, 2.4205, 2.4189, 2.4165, 2.4153, 2.414, 2.4122, 2.4052, 2.4024, 2.3914, 2.3226, 2.153, 2.1518, 2.1432, 2.1238, 2.0304, 1.9599, 1.9439, 1.9165, 1.8881, 1.8828, 1.8819, 1.8597, 1.8481, 1.755, 1.6155, 1.229, 1.1361, 1.2817, 1.4343, 0.5985, 0.6728, 1.1034, 0.8336, -0.6494, 1.0157, 2.6544, 2.6514, 2.6494, 2.6493, 2.6476, 2.6428, 2.6416, 2.6416, 2.6413, 2.6409, 2.6403, 2.6396, 2.6395, 2.631, 2.6281, 2.6275, 2.6246, 2.6136, 2.5637, 2.4973, 2.4871, 2.4761, 2.3994, 2.3992, 2.3861, 2.3673, 2.3607, 2.3337, 2.2895, 2.0704, 1.9636, 1.9627, 1.739, 1.8445, 1.7786, 0.9156, 1.8102]}, "token.table": {"Topic": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 5, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 9, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 6, 7, 8, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 8, 9, 10], "Freq": [0.0022294795326631843, 0.0022294795326631843, 0.7268103276481981, 0.22294795326631842, 0.033442192989947765, 0.0022294795326631843, 0.0022294795326631843, 0.0022294795326631843, 0.0022294795326631843, 0.0022294795326631843, 0.03130711970020983, 0.22362228357292738, 0.7379535357906603, 0.0008944891342917094, 0.0008944891342917094, 0.0008944891342917094, 0.0008944891342917094, 0.0008944891342917094, 0.0008944891342917094, 0.0008944891342917094, 0.0024191434999590467, 0.6471208862390451, 0.17901661899696947, 0.1136997444980752, 0.050802013499139986, 0.0024191434999590467, 0.0024191434999590467, 0.0024191434999590467, 0.0012095717499795234, 0.0012095717499795234, 0.9728177313483051, 0.0018963308603280802, 0.0018963308603280802, 0.0037926617206561603, 0.00568899258098424, 0.0037926617206561603, 0.0018963308603280802, 0.0037926617206561603, 0.0018963308603280802, 0.0037926617206561603, 0.0031970935195583714, 0.9207629336328109, 0.006394187039116743, 0.006394187039116743, 0.05435058983249231, 0.0031970935195583714, 0.0031970935195583714, 0.0031970935195583714, 0.0031970935195583714, 0.0031970935195583714, 0.11372384271131471, 0.0013071706058771805, 0.6195988671857835, 0.07843023635263083, 0.04052228878219259, 0.003921511817631541, 0.07058721271736774, 0.0013071706058771805, 0.0013071706058771805, 0.06928004211149057, 0.003010663559835972, 0.13849052375245471, 0.006021327119671944, 0.003010663559835972, 0.003010663559835972, 0.003010663559835972, 0.003010663559835972, 0.009031990679507917, 0.7436338992794851, 0.09031990679507916, 0.0016320403736688048, 0.0016320403736688048, 0.0016320403736688048, 0.0016320403736688048, 0.0016320403736688048, 0.9857523856959581, 0.0016320403736688048, 0.0016320403736688048, 0.0016320403736688048, 0.0016320403736688048, 0.004579614390844351, 0.004579614390844351, 0.0022898071954221754, 0.0022898071954221754, 0.0022898071954221754, 0.004579614390844351, 0.0022898071954221754, 0.0022898071954221754, 0.0022898071954221754, 0.9731680580544245, 0.002383247283552497, 0.906825591391725, 0.0011916236417762484, 0.002383247283552497, 0.0011916236417762484, 0.0011916236417762484, 0.0011916236417762484, 0.0011916236417762484, 0.0810304076407849, 0.0011916236417762484, 0.003362193154960983, 0.9783982080936461, 0.003362193154960983, 0.003362193154960983, 0.0016810965774804916, 0.0016810965774804916, 0.0016810965774804916, 0.003362193154960983, 0.0016810965774804916, 0.0016810965774804916, 0.002643941777536515, 0.9848683121323518, 0.0013219708887682576, 0.002643941777536515, 0.0013219708887682576, 0.0013219708887682576, 0.0013219708887682576, 0.0013219708887682576, 0.0013219708887682576, 0.0013219708887682576, 0.0014826523079013564, 0.7101904554847498, 0.0014826523079013564, 0.08599383385827868, 0.15864379694544514, 0.0014826523079013564, 0.0014826523079013564, 0.035583655389632554, 0.0014826523079013564, 0.0014826523079013564, 0.00011986723365541229, 0.9992132597515168, 0.00011986723365541229, 0.00011986723365541229, 0.00011986723365541229, 0.00011986723365541229, 0.00011986723365541229, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.0018818877946955863, 0.9842273166257915, 0.0018872815849652154, 0.0018872815849652154, 0.0018872815849652154, 0.0018872815849652154, 0.9832737057668772, 0.0018872815849652154, 0.0018872815849652154, 0.0018872815849652154, 0.0018872815849652154, 0.0007674680029816528, 0.0007674680029816528, 0.0007674680029816528, 0.16040081262316544, 0.0007674680029816528, 0.0007674680029816528, 0.0007674680029816528, 0.0007674680029816528, 0.0007674680029816528, 0.8334702512380749, 0.08068037246372367, 0.1479140161834934, 0.0067233643719769726, 0.7261233521735131, 0.0067233643719769726, 0.0067233643719769726, 0.0067233643719769726, 0.0067233643719769726, 0.013446728743953945, 0.0067233643719769726, 0.002889965992306825, 0.0014449829961534124, 0.002889965992306825, 0.002889965992306825, 0.004334948988460237, 0.002889965992306825, 0.13293843564611393, 0.004334948988460237, 0.0014449829961534124, 0.8424250867574394, 0.17103566418609364, 0.11283602845610345, 0.0011877476679589835, 0.002375495335917967, 0.002375495335917967, 0.25061475793934557, 0.43234015113707003, 0.009501981343671868, 0.005938738339794919, 0.010689729011630853, 0.9792397429107301, 0.0030601241965960316, 0.0015300620982980158, 0.0015300620982980158, 0.0015300620982980158, 0.006120248393192063, 0.0015300620982980158, 0.0015300620982980158, 0.0015300620982980158, 0.0015300620982980158, 0.6675416326986299, 0.14653352912896753, 0.002960273315736718, 0.017761639894420308, 0.02812259649949882, 0.007400683289341794, 0.03404314313097225, 0.08732806281423318, 0.005920546631473436, 0.002960273315736718, 0.1544644895509871, 0.7301957687864845, 0.0070211131614085045, 0.03510556580704252, 0.0070211131614085045, 0.0070211131614085045, 0.0070211131614085045, 0.0070211131614085045, 0.028084452645634018, 0.028084452645634018, 0.026647113987281368, 0.008882371329093789, 0.008882371329093789, 0.008882371329093789, 0.017764742658187577, 0.7105897063275031, 0.008882371329093789, 0.17764742658187577, 0.035529485316375155, 0.008882371329093789, 0.007053524128224179, 0.7993994011987403, 0.00235117470940806, 0.00470234941881612, 0.00235117470940806, 0.007053524128224179, 0.00235117470940806, 0.17163575378678836, 0.00235117470940806, 0.00235117470940806, 0.25591499979740745, 0.006734605257826511, 0.006734605257826511, 0.006734605257826511, 0.6599913152669981, 0.03367302628913255, 0.020203815773479534, 0.006734605257826511, 0.006734605257826511, 0.006734605257826511, 0.07881163043797568, 0.048499464884908104, 0.000673603678957057, 0.004041622073742342, 0.4742169899857681, 0.06129793478509219, 0.00673603678957057, 0.08420045986963212, 0.24115011706662642, 0.001347207357914114, 0.14144707344680926, 0.10498024982380376, 0.015470773658244764, 0.08950947616555899, 0.011050552613031974, 0.16023301288896363, 0.018785939442154356, 0.0430971551908247, 0.36687834675266157, 0.04751737623603749, 0.0005707246237293441, 0.0005707246237293441, 0.0017121738711880323, 0.9930608452890587, 0.0005707246237293441, 0.0005707246237293441, 0.0005707246237293441, 0.0005707246237293441, 0.0005707246237293441, 0.0005707246237293441, 0.7567144140699302, 0.0020817452931772497, 0.0020817452931772497, 0.15404915169511646, 0.0010408726465886249, 0.07806544849414686, 0.0010408726465886249, 0.0010408726465886249, 0.0010408726465886249, 0.0020817452931772497, 0.007137645059619429, 0.0035688225298097145, 0.007137645059619429, 0.0035688225298097145, 0.9278938577505258, 0.0035688225298097145, 0.007137645059619429, 0.0035688225298097145, 0.03211940276828743, 0.0035688225298097145, 0.003597763719708296, 0.0011992545732360986, 0.0011992545732360986, 0.003597763719708296, 0.9797909863338926, 0.003597763719708296, 0.0011992545732360986, 0.0023985091464721973, 0.0023985091464721973, 0.0023985091464721973, 0.0009351748126622914, 0.0004675874063311457, 0.8575553032113212, 0.0009351748126622914, 0.0004675874063311457, 0.0009351748126622914, 0.13747069746135682, 0.0004675874063311457, 0.0004675874063311457, 0.0004675874063311457, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.00017286032459219917, 0.9984412348445424, 0.00017286032459219917, 0.22507704077159713, 0.016369239328843425, 0.020461549161054284, 0.02455385899326514, 0.016369239328843425, 0.14732315395959084, 0.2619078292614948, 0.14732315395959084, 0.004092309832210856, 0.13913853429516912, 0.18238633287356798, 0.25672858812094623, 0.07632471538730834, 0.3043076314792683, 0.01784214125937078, 0.08722824615692382, 0.0019824601399300867, 0.0009912300699650434, 0.01784214125937078, 0.05550888391804243, 0.12164642567396275, 0.000850674305412327, 0.25520229162369806, 0.11824372845231344, 0.000850674305412327, 0.001701348610824654, 0.000850674305412327, 0.000850674305412327, 0.000850674305412327, 0.4984951429716236, 0.00036824051813222884, 0.00036824051813222884, 0.00036824051813222884, 0.00036824051813222884, 0.00036824051813222884, 0.00036824051813222884, 0.00036824051813222884, 0.7250655802023586, 0.27212974289971714, 0.00036824051813222884, 0.0012898913784218323, 0.0012898913784218323, 0.0012898913784218323, 0.0012898913784218323, 0.0012898913784218323, 0.0012898913784218323, 0.0012898913784218323, 0.9880567958711236, 0.0012898913784218323, 0.0012898913784218323, 0.001047241821563149, 0.001047241821563149, 0.002094483643126298, 0.002094483643126298, 0.8765414046483558, 0.001047241821563149, 0.001047241821563149, 0.001047241821563149, 0.11205487490725696, 0.001047241821563149, 0.00173319415773492, 0.00173319415773492, 0.00346638831546984, 0.00173319415773492, 0.00173319415773492, 0.00173319415773492, 0.00173319415773492, 0.00173319415773492, 0.00173319415773492, 0.9827210874356996, 0.2429845672206324, 0.005785346838586486, 0.7000269674689649, 0.023141387354345944, 0.005785346838586486, 0.005785346838586486, 0.005785346838586486, 0.005785346838586486, 0.005785346838586486, 0.005785346838586486, 0.0004049215970636903, 0.0004049215970636903, 0.0004049215970636903, 0.0008098431941273806, 0.9957022071796144, 0.0004049215970636903, 0.0004049215970636903, 0.0004049215970636903, 0.0004049215970636903, 0.0004049215970636903, 0.20005914113147422, 0.09366726572410011, 0.002120768280545663, 0.06256266427609707, 0.07458035119918915, 0.3043302482583026, 0.1307807106336492, 0.07281304429873443, 0.004241536561091326, 0.054786513914096294, 0.00661407002341408, 0.00330703501170704, 0.00330703501170704, 0.00661407002341408, 0.00330703501170704, 0.7308547375872558, 0.00661407002341408, 0.12236029543316049, 0.00330703501170704, 0.11243919039803936, 0.0006339189164846435, 0.001267837832969287, 0.9914491853819826, 0.001267837832969287, 0.001267837832969287, 0.0006339189164846435, 0.001267837832969287, 0.0006339189164846435, 0.001267837832969287, 0.0006339189164846435, 0.003463052458027265, 0.002308701638684843, 0.004617403277369686, 0.9835068980797432, 0.0011543508193424216, 0.0011543508193424216, 0.0011543508193424216, 0.0011543508193424216, 0.0011543508193424216, 0.0011543508193424216, 0.0019278790448057949, 0.0038557580896115898, 0.0019278790448057949, 0.0038557580896115898, 0.0019278790448057949, 0.0019278790448057949, 0.0019278790448057949, 0.0019278790448057949, 0.0019278790448057949, 0.9832183128509553, 0.004313732229805893, 0.1975689361251099, 0.23035330107163468, 0.2967847774106454, 0.06901971567689429, 0.010352957351534143, 0.04745105452786482, 0.003450985783844714, 0.07937267302842843, 0.061254997663243674, 0.00249309366610219, 0.00747928099830657, 0.9648272487815476, 0.00498618733220438, 0.00498618733220438, 0.00498618733220438, 0.00249309366610219, 0.00498618733220438, 0.00249309366610219, 0.00038840881132361243, 0.00038840881132361243, 0.00038840881132361243, 0.00038840881132361243, 0.00038840881132361243, 0.00038840881132361243, 0.00038840881132361243, 0.9966570098563894, 0.00038840881132361243, 0.00038840881132361243, 0.05854973139995794, 0.014637432849989486, 0.043912298549968455, 0.043912298549968455, 0.02927486569997897, 0.7757839410494427, 0.014637432849989486, 0.014637432849989486, 0.11941417539539712, 0.0009046528439045237, 0.0009046528439045237, 0.0009046528439045237, 0.0009046528439045237, 0.0009046528439045237, 0.0009046528439045237, 0.8720853415239608, 0.0009046528439045237, 0.0009046528439045237, 0.060305444373012086, 0.04522908327975906, 0.015076361093253022, 0.04522908327975906, 0.030152722186506043, 0.030152722186506043, 0.030152722186506043, 0.723665332476145, 0.015076361093253022, 0.001142143245508149, 0.001142143245508149, 0.0005710716227540745, 0.0005710716227540745, 0.16389755573041936, 0.0005710716227540745, 0.0005710716227540745, 0.7635227596221975, 0.06681537986222671, 0.0005710716227540745, 0.010052835419614858, 0.005026417709807429, 0.005026417709807429, 0.005026417709807429, 0.005026417709807429, 0.16084536671383773, 0.010052835419614858, 0.7941739981495738, 0.005026417709807429, 0.005026417709807429, 0.0017451289062225016, 0.0017451289062225016, 0.0017451289062225016, 0.0017451289062225016, 0.0017451289062225016, 0.0017451289062225016, 0.0017451289062225016, 0.9825075742032684, 0.005235386718667505, 0.0017451289062225016, 0.07423727098848601, 0.05567795324136451, 0.018559317747121502, 0.05567795324136451, 0.037118635494243005, 0.037118635494243005, 0.037118635494243005, 0.6495761211492526, 0.018559317747121502, 0.018559317747121502, 0.00018067515518889628, 0.00018067515518889628, 0.00018067515518889628, 0.00018067515518889628, 0.9985915827290298, 0.00036135031037779256, 0.00018067515518889628, 0.5097097486391083, 0.0007421516433300938, 0.00029686065733203746, 0.00029686065733203746, 0.0981124472482384, 0.00029686065733203746, 0.07926179550765401, 0.30502432540866853, 0.00593721314664075, 0.00044529098599805625, 0.018399037314533965, 0.02759855597180095, 0.009199518657266982, 0.009199518657266982, 0.009199518657266982, 0.009199518657266982, 0.009199518657266982, 0.8923533097548974, 0.018399037314533965, 0.009199518657266982, 0.00394554639551004, 0.00394554639551004, 0.00394554639551004, 0.00394554639551004, 0.9784955060864898, 0.00394554639551004, 0.00394554639551004, 0.1643084524272519, 0.0023472636061035988, 0.0023472636061035988, 0.0023472636061035988, 0.1666557160333555, 0.11501591669907633, 0.007041790818310796, 0.5328288385855169, 0.0023472636061035988, 0.0046945272122071975, 0.0008894616947299638, 0.0008894616947299638, 0.0004447308473649819, 0.0004447308473649819, 0.0008894616947299638, 0.0004447308473649819, 0.0004447308473649819, 0.9935287130133695, 0.0013341925420949456, 0.0004447308473649819, 0.001577753235200676, 0.001577753235200676, 0.001577753235200676, 0.001577753235200676, 0.003155506470401352, 0.001577753235200676, 0.001577753235200676, 0.9024748505347868, 0.08362092146563584, 0.001577753235200676, 0.0047610101910308305, 0.0023805050955154152, 0.0023805050955154152, 0.0023805050955154152, 0.0023805050955154152, 0.0023805050955154152, 0.0023805050955154152, 0.980768099352351, 0.0023805050955154152, 0.0023805050955154152, 0.005031849498068915, 0.26417209864861807, 0.027675172239379035, 0.6792996822393036, 0.007547774247103373, 0.0025159247490344574, 0.0025159247490344574, 0.005031849498068915, 0.0025159247490344574, 0.005031849498068915, 0.002011503319591882, 0.002011503319591882, 0.002011503319591882, 0.002011503319591882, 0.002011503319591882, 0.23936889503143394, 0.002011503319591882, 0.002011503319591882, 0.002011503319591882, 0.7422447249294044, 0.00139056658071001, 0.5318917171215788, 0.00139056658071001, 0.007648116193905055, 0.06848540409996799, 0.0003476416451775025, 0.011819815936035085, 0.07787172851976056, 0.29862417320747464, 0.000695283290355005, 0.00141446863078921, 0.00141446863078921, 0.00282893726157842, 0.00141446863078921, 0.6676291937325071, 0.00141446863078921, 0.00141446863078921, 0.00141446863078921, 0.31966991055836147, 0.00141446863078921, 0.052954712517845025, 0.052954712517845025, 0.052954712517845025, 0.052954712517845025, 0.052954712517845025, 0.6884112627319853, 0.052954712517845025, 0.0007158524588411215, 0.0007158524588411215, 0.0007158524588411215, 0.0007158524588411215, 0.0007158524588411215, 0.0007158524588411215, 0.0007158524588411215, 0.9943190653303177, 0.0007158524588411215, 0.0007158524588411215, 0.005410014357377514, 0.005410014357377514, 0.010820028714755028, 0.010820028714755028, 0.005410014357377514, 0.005410014357377514, 0.07033018664590769, 0.8601922828230248, 0.027050071786887574, 0.005410014357377514, 0.0005271075811870527, 0.0005271075811870527, 0.0005271075811870527, 0.0005271075811870527, 0.0005271075811870527, 0.0005271075811870527, 0.0005271075811870527, 0.9809472085891049, 0.015286119854424527, 0.0005271075811870527, 0.03987088633864984, 0.03987088633864984, 0.03987088633864984, 0.03987088633864984, 0.6778050677570473, 0.03987088633864984, 0.15948354535459935, 0.18736257127169814, 0.05902492539230424, 0.0545564891390136, 0.028577208596626172, 0.06567562121115542, 0.13924894370719662, 0.09726642635069853, 0.14766623060293016, 0.045619616632432324, 0.17499643373352172, 0.0004227199773055473, 0.0004227199773055473, 0.0004227199773055473, 0.0004227199773055473, 0.0004227199773055473, 0.0004227199773055473, 0.0004227199773055473, 0.9959282665318695, 0.0008454399546110946, 0.0004227199773055473, 0.3828982627944636, 0.0032176324604576774, 0.0032176324604576774, 0.01287052984183071, 0.0032176324604576774, 0.5759562104219242, 0.009652897381373031, 0.0032176324604576774, 0.0032176324604576774, 0.0032176324604576774, 0.0004807980879613393, 0.7159083529744341, 0.1543361862355899, 0.12644989713383223, 0.0004807980879613393, 0.0004807980879613393, 0.0004807980879613393, 0.0004807980879613393, 0.0004807980879613393, 0.0004807980879613393, 0.0074284514084983645, 0.009285564260622955, 0.0037142257042491822, 0.0018571128521245911, 0.0018571128521245911, 0.0074284514084983645, 0.0074284514084983645, 0.9471275545835415, 0.009285564260622955, 0.0037142257042491822, 0.0002226225836362171, 0.0002226225836362171, 0.0002226225836362171, 0.0002226225836362171, 0.0004452451672724342, 0.8392871403085386, 0.0002226225836362171, 0.0002226225836362171, 0.158952524716259, 0.0002226225836362171, 0.07887186640223919, 0.17614716829833418, 0.002629062213407973, 0.002629062213407973, 0.002629062213407973, 0.06835561754860729, 0.04469405762793554, 0.05258124426815945, 0.5599902514558982, 0.007887186640223918, 0.006555077961704716, 0.21304003375540329, 0.003277538980852358, 0.003277538980852358, 0.003277538980852358, 0.003277538980852358, 0.003277538980852358, 0.7505564266151901, 0.013110155923409432, 0.003277538980852358, 0.14254468782407873, 0.21252117093771739, 0.04405926714562433, 0.062201318323234354, 0.046650988742425764, 0.04405926714562433, 0.3576575803585975, 0.07775164790404294, 0.0025917215968014316, 0.005183443193602863, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.0003163858648993298, 0.9972482461626875, 0.0003163858648993298, 0.00155300259859657, 0.00155300259859657, 0.9861566501088219, 0.00155300259859657, 0.00155300259859657, 0.00155300259859657, 0.00155300259859657, 0.00310600519719314, 0.00155300259859657, 0.007865733029346329, 0.007865733029346329, 0.19664332573365823, 0.007865733029346329, 0.7236474386998623, 0.007865733029346329, 0.007865733029346329, 0.031462932117385316, 0.007865733029346329, 0.0006288417602698691, 0.19053905336177035, 0.0006288417602698691, 0.0006288417602698691, 0.06225533426671704, 0.0006288417602698691, 0.5219386610239913, 0.0012576835205397382, 0.22260998313553368, 0.0006288417602698691, 0.003329153443535165, 0.28075860707146555, 0.19198118191052785, 0.002219435629023443, 0.056595608540097805, 0.023304074104746154, 0.02108463847572271, 0.27964888925695386, 0.04771786602400403, 0.09321629641898461, 0.0015587771078329805, 0.0007793885539164903, 0.9890440749200262, 0.0007793885539164903, 0.0015587771078329805, 0.0007793885539164903, 0.0015587771078329805, 0.0015587771078329805, 0.0007793885539164903, 0.0007793885539164903, 0.001687493316896684, 0.8589340983004122, 0.003374986633793368, 0.12993698540104467, 0.001687493316896684, 0.001687493316896684, 0.001687493316896684, 0.001687493316896684, 0.001687493316896684, 0.001687493316896684, 0.002232356039603781, 0.8884777037623048, 0.002232356039603781, 0.10157219980197203, 0.0011161780198018906, 0.0011161780198018906, 0.0011161780198018906, 0.0011161780198018906, 0.0011161780198018906, 0.0011161780198018906, 0.0623588274273804, 0.0027112533664078436, 0.22232277604544318, 0.05422506732815687, 0.013556266832039218, 0.173520215450102, 0.4175330184268079, 0.005422506732815687, 0.0433800538625255, 0.0027112533664078436, 0.005748946684707236, 0.005748946684707236, 0.15809603382944898, 0.6783757087954538, 0.005748946684707236, 0.008623420027060853, 0.005748946684707236, 0.002874473342353618, 0.002874473342353618, 0.12360235372120557, 0.0008827667681853455, 0.26924386429653036, 0.4378523170199314, 0.09004221035490524, 0.1977397560735174, 0.0008827667681853455, 0.0008827667681853455, 0.0008827667681853455, 0.0008827667681853455, 0.0008827667681853455, 0.0732876176805083, 0.18874071402651454, 0.001003939968226141, 0.002007879936452282, 0.010039399682261411, 0.7158091973452386, 0.001003939968226141, 0.0030118199046784235, 0.0030118199046784235, 0.001003939968226141, 0.009347186350092988, 0.003115728783364329, 0.003115728783364329, 0.006231457566728658, 0.003115728783364329, 0.7851636534078109, 0.17136508308503812, 0.003115728783364329, 0.003115728783364329, 0.012462915133457316, 0.0012785266533266535, 0.0012785266533266535, 0.002557053306653307, 0.9895796296748298, 0.0012785266533266535, 0.0012785266533266535, 0.0012785266533266535, 0.0012785266533266535, 0.0012785266533266535, 0.0012785266533266535, 0.005886712319755661, 0.005886712319755661, 0.0019622374399185533, 0.003924474879837107, 0.9673830578798469, 0.003924474879837107, 0.003924474879837107, 0.0019622374399185533, 0.0019622374399185533, 0.003924474879837107, 0.0007617431505479279, 0.18434184243259855, 0.31002946227300665, 0.0015234863010958558, 0.0007617431505479279, 0.0007617431505479279, 0.0007617431505479279, 0.0007617431505479279, 0.0007617431505479279, 0.4989417636088928, 0.01178465963436449, 0.01178465963436449, 0.16498523488110284, 0.007856439756242992, 0.003928219878121496, 0.003928219878121496, 0.01178465963436449, 0.007856439756242992, 0.003928219878121496, 0.7699310961118132, 0.006879800985591537, 0.7636579094006607, 0.1100768157694646, 0.013759601971183075, 0.041278805913549224, 0.02751920394236615, 0.006879800985591537, 0.006879800985591537, 0.02751920394236615, 0.006879800985591537, 0.00046989650483521, 0.00046989650483521, 0.00093979300967042, 0.11935371222814334, 0.00046989650483521, 0.00046989650483521, 0.8763569815176666, 0.00046989650483521, 0.00046989650483521, 0.00046989650483521, 0.1956008706941018, 0.5669949289740419, 0.0024759603885329343, 0.0024759603885329343, 0.0012379801942664671, 0.027235564273862277, 0.10275235612411678, 0.09903841554131737, 0.0012379801942664671, 0.0024759603885329343, 0.0006537098950712611, 0.0006537098950712611, 0.0006537098950712611, 0.0006537098950712611, 0.0006537098950712611, 0.9936390405083169, 0.0006537098950712611, 0.0006537098950712611, 0.0006537098950712611, 0.0006537098950712611, 0.0018407214797275737, 0.20293954313996498, 0.5002160621159681, 0.0004601803699318934, 0.0004601803699318934, 0.0004601803699318934, 0.2084617075791477, 0.0842130076975365, 0.0004601803699318934, 0.0004601803699318934, 0.07547040224311644, 0.04914351773970372, 0.2422073374313969, 0.02632688450341271, 0.08249090477735982, 0.0017551256335608472, 0.2650239706676879, 0.07196015097599473, 0.042123015205460336, 0.14392030195198946, 0.002641559292751826, 0.002641559292751826, 0.002641559292751826, 0.002641559292751826, 0.974735379025424, 0.002641559292751826, 0.002641559292751826, 0.002641559292751826, 0.002641559292751826, 0.002641559292751826, 0.310293090706, 0.0066301942458547, 0.00132603884917094, 0.00132603884917094, 0.00265207769834188, 0.11403934102870085, 0.00265207769834188, 0.5012426849866154, 0.05569363166517948, 0.00397811654751282, 0.11043950207394718, 0.0008906411457576384, 0.001781282291515277, 0.8799534520085468, 0.0008906411457576384, 0.0026719234372729155, 0.0008906411457576384, 0.0008906411457576384, 0.0008906411457576384, 0.0008906411457576384, 0.16105513365986313, 0.014641375787260285, 0.007320687893630142, 0.007320687893630142, 0.7467101651502744, 0.021962063680890424, 0.007320687893630142, 0.014641375787260285, 0.007320687893630142, 0.014641375787260285, 0.3219281650818007, 0.0051923897593838825, 0.0051923897593838825, 0.1194249644658293, 0.01817336415784359, 0.0051923897593838825, 0.5192389759383883, 0.0025961948796919413, 0.0025961948796919413, 0.023965318435762734, 0.8547630242088708, 0.007988439478587578, 0.007988439478587578, 0.007988439478587578, 0.007988439478587578, 0.0718959553072882, 0.007988439478587578, 0.015976878957175155, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.0029417704746650345, 0.9766677975887915, 0.0029417704746650345, 0.0015697015172007733, 0.0015697015172007733, 0.6514261296383209, 0.1993520926844982, 0.0015697015172007733, 0.1404882857894692, 0.0007848507586003866, 0.0015697015172007733, 0.0007848507586003866, 0.0015697015172007733, 0.746295631552867, 0.02403712923713222, 0.001144625201768201, 0.001144625201768201, 0.002289250403536402, 0.028615630044205025, 0.1121732697732837, 0.08241301452731047, 0.001144625201768201, 0.001144625201768201, 0.014548499969056064, 0.003637124992264016, 0.003637124992264016, 0.010911374976792048, 0.003637124992264016, 0.8438129982052517, 0.08729099981433638, 0.014548499969056064, 0.01818562496132008, 0.8532354787722725, 0.007827848429103416, 0.007827848429103416, 0.007827848429103416, 0.007827848429103416, 0.007827848429103416, 0.007827848429103416, 0.015655696858206832, 0.007827848429103416, 0.08610633272013758, 0.10919556271072417, 0.08735645016857933, 0.0018199260451787361, 0.0018199260451787361, 0.016379334406608624, 0.7698287171106054, 0.0018199260451787361, 0.0018199260451787361, 0.0018199260451787361, 0.0072797041807149444, 0.002877378991693954, 0.002877378991693954, 0.002877378991693954, 0.002877378991693954, 0.01438689495846977, 0.002877378991693954, 0.002877378991693954, 0.002877378991693954, 0.9639219622174745, 0.002877378991693954, 0.0009505075887318115, 0.8526053070924349, 0.001901015177463623, 0.0009505075887318115, 0.0009505075887318115, 0.0009505075887318115, 0.0009505075887318115, 0.0009505075887318115, 0.1387741079548445, 0.0009505075887318115, 0.002068694778113465, 0.007240431723397128, 0.0010343473890567325, 0.0010343473890567325, 0.0010343473890567325, 0.0010343473890567325, 0.982630019603896, 0.0010343473890567325, 0.0010343473890567325, 0.0010343473890567325, 0.9134588135937469, 0.00044953681771345814, 0.00044953681771345814, 0.08226523764156284, 0.00044953681771345814, 0.00044953681771345814, 0.00044953681771345814, 0.00044953681771345814, 0.00044953681771345814, 0.0008990736354269163, 0.01559693319110609, 0.9514129246574714, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.0051989777303686965, 0.002825804703693582, 0.10172896933296895, 0.005651609407387164, 0.01412902351846791, 0.8590446299228489, 0.002825804703693582, 0.002825804703693582, 0.002825804703693582, 0.008477414111080746, 0.002825804703693582, 0.0021896067212037696, 0.21633314405493245, 0.16509634677876425, 0.13093848192798543, 0.3463957832944364, 0.0013137640327222618, 0.06919157239003912, 0.0656882016361131, 0.0013137640327222618, 0.00043792134424075393, 0.0042804096688849325, 0.29320806231861785, 0.0042804096688849325, 0.0021402048344424662, 0.6870057518560316, 0.0021402048344424662, 0.0021402048344424662, 0.0021402048344424662, 0.0021402048344424662, 0.0021402048344424662, 0.0014168541376758294, 0.002833708275351659, 0.002833708275351659, 0.6276663829903923, 0.24794947409327014, 0.0014168541376758294, 0.0014168541376758294, 0.10768091446336303, 0.005667416550703318, 0.0014168541376758294, 0.5608780633897581, 0.14472268966756768, 0.15403960101955702, 0.13416352346864643, 0.0006211274234659557, 0.0006211274234659557, 0.0012422548469319115, 0.0006211274234659557, 0.0006211274234659557, 0.002484509693863823, 0.0029480055429921583, 0.0029480055429921583, 0.7753254578069375, 0.15624429377858437, 0.05011609423086669, 0.0029480055429921583, 0.0058960110859843165, 0.0029480055429921583, 0.0029480055429921583, 0.0029480055429921583, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.0004749721487435064, 0.9960165959151329, 0.9866593101748878, 0.0025363992549482977, 0.0008454664183160992, 0.0016909328366321984, 0.0016909328366321984, 0.0016909328366321984, 0.0008454664183160992, 0.0008454664183160992, 0.0008454664183160992, 0.0008454664183160992, 0.10772228644842675, 0.0022442143010088906, 0.006732642903026672, 0.006732642903026672, 0.0022442143010088906, 0.7024390762157828, 0.1391412866625512, 0.0022442143010088906, 0.0022442143010088906, 0.02693057161210669, 0.2137465120828492, 0.002374961245364991, 0.002374961245364991, 0.38355624112644604, 0.0011874806226824955, 0.0035624418680474863, 0.004749922490729982, 0.0011874806226824955, 0.0011874806226824955, 0.38711868299449353, 0.00027710856968524394, 0.1102892107347271, 0.6636750243961593, 0.0008313257090557318, 0.1587832104296448, 0.00027710856968524394, 0.00027710856968524394, 0.0005542171393704879, 0.06456629673666184, 0.00027710856968524394, 0.001021990179474383, 0.06234140094793736, 0.9269450927832653, 0.002043980358948766, 0.001021990179474383, 0.001021990179474383, 0.001021990179474383, 0.001021990179474383, 0.001021990179474383, 0.001021990179474383, 0.001528431000950934, 0.001528431000950934, 0.003056862001901868, 0.003056862001901868, 0.001528431000950934, 0.001528431000950934, 0.9827811336114506, 0.001528431000950934, 0.001528431000950934, 0.001528431000950934, 0.37897212551359544, 0.0020652431908097842, 0.0020652431908097842, 0.0020652431908097842, 0.0020652431908097842, 0.6061488765026717, 0.0010326215954048921, 0.0010326215954048921, 0.0010326215954048921, 0.005163107977024461, 0.00042275214010897437, 0.00042275214010897437, 0.6857039712567564, 0.17903553133615063, 0.00021137607005448719, 0.00021137607005448719, 0.13105316343378204, 0.00021137607005448719, 0.00021137607005448719, 0.002536512840653846, 0.0037735176337589035, 0.0028301382253191777, 0.1018849761114904, 0.1603744994347534, 0.1707516729275904, 0.09245118202709314, 0.0037735176337589035, 0.0037735176337589035, 0.4396148043329123, 0.02075434698567397, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.0014873155525679147, 0.21417343956977972, 0.7719167717827478, 0.1332400345427641, 0.81609521157443, 0.003331000863569102, 0.003331000863569102, 0.003331000863569102, 0.026648006908552815, 0.003331000863569102, 0.006662001727138204, 0.003331000863569102, 0.003331000863569102, 0.0002062058517213366, 0.0002062058517213366, 0.0002062058517213366, 0.19074041284223636, 0.0002062058517213366, 0.0002062058517213366, 0.8075021153407541, 0.0002062058517213366, 0.0002062058517213366, 0.0002062058517213366, 0.20915388620967967, 0.004864043865341388, 0.002084590228003452, 0.09241683344148637, 0.0006948634093344839, 0.24876110054174527, 0.23972787622039698, 0.0013897268186689679, 0.0006948634093344839, 0.20081552529766586, 0.0005812428132008725, 0.0005812428132008725, 0.8137399384812215, 0.0005812428132008725, 0.0720741088369082, 0.0005812428132008725, 0.0005812428132008725, 0.0005812428132008725, 0.10927364888176402, 0.0005812428132008725, 0.0010534767669035901, 0.0021069535338071803, 0.0021069535338071803, 0.0021069535338071803, 0.0010534767669035901, 0.9881612073555675, 0.0010534767669035901, 0.0010534767669035901, 0.0010534767669035901, 0.0010534767669035901, 0.01899244954938943, 0.0063308165164631435, 0.05064653213170515, 0.0063308165164631435, 0.0063308165164631435, 0.0063308165164631435, 0.1456087798786523, 0.012661633032926287, 0.0063308165164631435, 0.7470363489426509, 0.7381988405380513, 0.00351523257399072, 0.04745563974887472, 0.12479075637667057, 0.00351523257399072, 0.00351523257399072, 0.07381988405380513, 0.00175761628699536, 0.00175761628699536, 0.00351523257399072, 0.0009772567591480243, 0.4622424470770155, 0.1446340003539076, 0.36940305495795317, 0.0009772567591480243, 0.0009772567591480243, 0.0009772567591480243, 0.00684079731403617, 0.012704337868924316, 0.0009772567591480243, 0.0013812697161704245, 0.004143809148511273, 0.14365205048172414, 0.21824061515492704, 0.005525078864681698, 0.0013812697161704245, 0.002762539432340849, 0.0013812697161704245, 0.6201901025605205, 0.0013812697161704245, 0.0008793005221996842, 0.0008793005221996842, 0.0008793005221996842, 0.0008793005221996842, 0.0008793005221996842, 0.0008793005221996842, 0.0017586010443993683, 0.2611522550933062, 0.7306987339479376, 0.0008793005221996842, 0.0045945133418149745, 0.22283389707802628, 0.029864336721797335, 0.730527621348581, 0.0022972566709074872, 0.0022972566709074872, 0.0022972566709074872, 0.0022972566709074872, 0.0022972566709074872, 0.0022972566709074872, 0.0012939318223183923, 0.0012939318223183923, 0.04658154560346213, 0.0025878636446367846, 0.006469659111591962, 0.0012939318223183923, 0.0025878636446367846, 0.07892984116142193, 0.751774388766986, 0.10739634125242657, 0.002303158112696876, 0.0015354387417979174, 0.9895902690887578, 0.0007677193708989587, 0.0007677193708989587, 0.0007677193708989587, 0.0007677193708989587, 0.0015354387417979174, 0.0015354387417979174, 0.0007677193708989587, 0.008948272109512496, 0.008948272109512496, 0.11632753742366246, 0.7427065850895372, 0.02684481632853749, 0.008948272109512496, 0.008948272109512496, 0.08053444898561248, 0.0016623909978999137, 0.2776192966492856, 0.0016623909978999137, 0.0016623909978999137, 0.0016623909978999137, 0.0016623909978999137, 0.0016623909978999137, 0.11636736985299397, 0.5934735862502692, 0.0016623909978999137, 0.0013901016807593319, 0.0013901016807593319, 0.0013901016807593319, 0.0013901016807593319, 0.0027802033615186637, 0.9841919899776069, 0.0013901016807593319, 0.0013901016807593319, 0.0013901016807593319, 0.0013901016807593319, 0.002202652014767012, 0.001101326007383506, 0.001101326007383506, 0.002202652014767012, 0.0033039780221505183, 0.9856867766082379, 0.002202652014767012, 0.001101326007383506, 0.001101326007383506, 0.001101326007383506, 0.001565995284655367, 0.001565995284655367, 0.004697985853966101, 0.001565995284655367, 0.001565995284655367, 0.001565995284655367, 0.9818790434789151, 0.001565995284655367, 0.001565995284655367, 0.001565995284655367, 0.0009001432595724057, 0.00045007162978620286, 0.00045007162978620286, 0.0009001432595724057, 0.00045007162978620286, 0.00045007162978620286, 0.3303525762630729, 0.00045007162978620286, 0.6652058688240078, 0.00045007162978620286, 0.0072814436682719875, 0.08009588035099187, 0.0072814436682719875, 0.0072814436682719875, 0.10194021135580783, 0.0072814436682719875, 0.0072814436682719875, 0.18203609170679969, 0.611641268134847, 0.0072814436682719875, 0.00206468950060845, 0.2033719158099323, 0.001032344750304225, 0.7866466997318193, 0.001032344750304225, 0.00206468950060845, 0.001032344750304225, 0.001032344750304225, 0.001032344750304225, 0.001032344750304225, 0.02957624895641723, 0.7394062239104308, 0.02957624895641723, 0.02957624895641723, 0.02957624895641723, 0.05915249791283446, 0.02957624895641723, 0.02957624895641723, 0.02957624895641723, 0.19859373807186834, 0.21834339158177787, 0.0010972029727727532, 0.0010972029727727532, 0.0010972029727727532, 0.0021944059455455064, 0.11410910916836633, 0.4608252485645563, 0.0010972029727727532, 0.0010972029727727532, 0.9491475570350478, 0.0002791610461867788, 0.0002791610461867788, 0.0002791610461867788, 0.0005583220923735576, 0.04801569994412595, 0.0002791610461867788, 0.0002791610461867788, 0.0002791610461867788, 0.0002791610461867788, 0.0034578748620224587, 0.0069157497240449175, 0.0034578748620224587, 0.0034578748620224587, 0.0034578748620224587, 0.0034578748620224587, 0.0034578748620224587, 0.0034578748620224587, 0.9682049613662884, 0.0034578748620224587, 0.056526984077580576, 0.06952169306093244, 0.2605439151162047, 0.0090962962883463, 0.14359153426603802, 0.05587724862841298, 0.0012994708983351856, 0.006497354491675928, 0.3144719573971149, 0.08251640204428429, 0.0014625571049006897, 0.8833844913600165, 0.0029251142098013793, 0.09652876892344552, 0.0029251142098013793, 0.0014625571049006897, 0.0014625571049006897, 0.0014625571049006897, 0.005850228419602759, 0.0014625571049006897, 0.003983889363437262, 0.9813647465267122, 0.0013279631211457542, 0.0026559262422915083, 0.0026559262422915083, 0.0013279631211457542, 0.0013279631211457542, 0.0013279631211457542, 0.0013279631211457542, 0.0013279631211457542, 0.02489613865688458, 0.9498834441395962, 0.005745262766973364, 0.007660350355964485, 0.0038301751779822426, 0.0019150875889911213, 0.0019150875889911213, 0.0019150875889911213, 0.0019150875889911213, 0.0019150875889911213, 0.06889396515773774, 0.015309770035052832, 0.007654885017526416, 0.007654885017526416, 0.8343824669103793, 0.03827442508763208, 0.007654885017526416, 0.015309770035052832, 0.007654885017526416, 0.007654885017526416, 0.16218533653801448, 0.060068643162227583, 0.008581234737461084, 0.26430202991380136, 0.004290617368730542, 0.36556059981584216, 0.10125856990204078, 0.00514874084247665, 0.0008581234737461083, 0.028318074633621574, 0.27459579716368565, 0.07126322521596357, 0.00038730013704328024, 0.0011619004111298407, 0.00038730013704328024, 0.006196802192692484, 0.644080127902975, 0.0007746002740865605, 0.00038730013704328024, 0.0007746002740865605, 0.8288066081766973, 0.0002315749114771437, 0.0002315749114771437, 0.0004631498229542874, 0.0002315749114771437, 0.16928126028979204, 0.0002315749114771437, 0.0002315749114771437, 0.0002315749114771437, 0.0002315749114771437, 0.9891160716766918, 0.0010378972420531918, 0.0010378972420531918, 0.0010378972420531918, 0.0010378972420531918, 0.0010378972420531918, 0.0020757944841063836, 0.0010378972420531918, 0.0010378972420531918, 0.0010378972420531918, 0.0013033746686786877, 0.4640013820496128, 0.0016292183358483595, 0.06712379543695242, 0.07820248012072126, 0.1704162379297384, 0.13750602754560154, 0.0003258436671696719, 0.07624741811770322, 0.002932593004527047, 0.013177751760427237, 0.003294437940106809, 0.5336989462973031, 0.2152366120869782, 0.15154414524491322, 0.027453649500890076, 0.002196291960071206, 0.001098145980035603, 0.05161286106167334, 0.002196291960071206, 0.1487058600177603, 0.0004285471470252458, 0.0004285471470252458, 0.0917090894634026, 0.0004285471470252458, 0.002142735735126229, 0.002142735735126229, 0.002142735735126229, 0.7512431487352559, 0.0004285471470252458, 0.0960326014923196, 0.134282874968074, 0.0024415068176013455, 0.003255342423468461, 0.03906410908162153, 0.7186168399806627, 0.0008138356058671152, 0.0016276712117342304, 0.0008138356058671152, 0.0016276712117342304, 0.993055427409533, 0.0009731067392548093, 0.0009731067392548093, 0.0009731067392548093, 0.00048655336962740467, 0.0019462134785096187, 0.0009731067392548093, 0.00048655336962740467, 0.00048655336962740467, 0.00048655336962740467, 0.5854986532763158, 0.003136599928265977, 0.06064093194647556, 0.13382826360601502, 0.037639199139191726, 0.12023633058352913, 0.018819599569595863, 0.0010455333094219923, 0.0010455333094219923, 0.037639199139191726, 0.0007432264369898433, 0.0007432264369898433, 0.0007432264369898433, 0.0007432264369898433, 0.0007432264369898433, 0.2296569690298616, 0.7632935507885691, 0.0007432264369898433, 0.0007432264369898433, 0.0007432264369898433, 0.02383860330654063, 0.8956503813743122, 0.0034055147580772326, 0.0034055147580772326, 0.0034055147580772326, 0.006811029516154465, 0.0034055147580772326, 0.04767720661308126, 0.0034055147580772326, 0.0034055147580772326, 0.254249517238578, 0.1295694655158138, 0.10675220114824911, 0.13853339080307137, 0.0016298045976831925, 0.36426132758219354, 0.0016298045976831925, 0.0016298045976831925, 0.0008149022988415963, 0.0016298045976831925, 0.24175811177114523, 0.0494505228622797, 0.0032967015241519805, 0.00439560203220264, 0.06483512997498894, 0.446153606268568, 0.006593403048303961, 0.0989010457245594, 0.08021973708769818, 0.00439560203220264, 0.007944205549329604, 0.007944205549329604, 0.007944205549329604, 0.007944205549329604, 0.007944205549329604, 0.007944205549329604, 0.007944205549329604, 0.06355364439463683, 0.8897510215249157, 0.007944205549329604, 0.0006899424190326655, 0.0006899424190326655, 0.0006899424190326655, 0.0006899424190326655, 0.7306490217555928, 0.001379884838065331, 0.0006899424190326655, 0.0006899424190326655, 0.26355800407047825, 0.0006899424190326655, 0.0021883969841424923, 0.5339688641307682, 0.0021883969841424923, 0.1991441255569668, 0.0010941984920712462, 0.1849195451600406, 0.0010941984920712462, 0.07331129896877349, 0.0010941984920712462, 0.0010941984920712462, 0.0008020559596458673, 0.0008020559596458673, 0.9031150105612465, 0.0008020559596458673, 0.0008020559596458673, 0.0008020559596458673, 0.0008020559596458673, 0.0016041119192917346, 0.0008020559596458673, 0.08983026748033714, 0.6955622605487507, 0.039265611482590765, 0.0018697910229805127, 0.052354148643454355, 0.0037395820459610253, 0.15145307286142154, 0.0018697910229805127, 0.0037395820459610253, 0.0018697910229805127, 0.046744775574512815, 0.0007295075252948376, 0.0007295075252948376, 0.9914007268756843, 0.0014590150505896752, 0.0014590150505896752, 0.0007295075252948376, 0.0007295075252948376, 0.0007295075252948376, 0.0007295075252948376, 0.0007295075252948376, 0.007174855101318071, 0.0035874275506590355, 0.7712969233916926, 0.007174855101318071, 0.0035874275506590355, 0.0035874275506590355, 0.0035874275506590355, 0.1901336601849289, 0.0035874275506590355, 0.0035874275506590355, 0.0072114218237782695, 0.028845687295113078, 0.0072114218237782695, 0.6273936986687094, 0.23076549836090463, 0.07932564006156097, 0.0072114218237782695, 0.0072114218237782695, 0.0072114218237782695, 0.0072114218237782695, 0.002938187252387702, 0.001469093626193851, 0.002938187252387702, 0.002938187252387702, 0.001469093626193851, 0.001469093626193851, 0.001469093626193851, 0.001469093626193851, 0.001469093626193851, 0.9813545422974924, 0.002931493786725586, 0.75046240940175, 0.005862987573451172, 0.1495061831230049, 0.05862987573451172, 0.002931493786725586, 0.008794481360176758, 0.005862987573451172, 0.002931493786725586, 0.005862987573451172, 0.6643719620049289, 0.04662259382490729, 0.0023311296912453647, 0.11655648456226822, 0.004662259382490729, 0.10956309548853213, 0.009324518764981459, 0.004662259382490729, 0.0023311296912453647, 0.037298075059925835, 0.0015939449536697462, 0.0031878899073394923, 0.0015939449536697462, 0.0015939449536697462, 0.9834640364142334, 0.0015939449536697462, 0.0015939449536697462, 0.0015939449536697462, 0.0015939449536697462, 0.0015939449536697462, 0.002875822054770041, 0.005751644109540082, 0.002875822054770041, 0.005751644109540082, 0.7505895562949807, 0.0977779498621814, 0.002875822054770041, 0.002875822054770041, 0.12941199246465185, 0.002875822054770041, 0.1864939363012951, 0.0021812156292549136, 0.0010906078146274568, 0.8026873515658082, 0.0010906078146274568, 0.0021812156292549136, 0.0010906078146274568, 0.0010906078146274568, 0.0010906078146274568, 0.0021812156292549136, 0.0008279966746845752, 0.17719128838249912, 0.0008279966746845752, 0.0016559933493691505, 0.0016559933493691505, 0.8139207312149375, 0.0008279966746845752, 0.0008279966746845752, 0.0008279966746845752, 0.0008279966746845752, 0.14370077435364884, 0.7823708825920881, 0.01596675270596098, 0.01596675270596098, 0.01596675270596098, 0.6505117625521937, 0.08131397031902421, 0.008131397031902422, 0.002710465677300807, 0.013552328386504036, 0.1572070092834468, 0.008131397031902422, 0.008131397031902422, 0.002710465677300807, 0.07318257328712179, 0.17453546020902555, 0.09485622837447041, 0.0037942491349788167, 0.0037942491349788167, 0.0037942491349788167, 0.5274006297620555, 0.0037942491349788167, 0.07967923183455515, 0.0037942491349788167, 0.11003322491438568, 0.2375626459221797, 0.13392440982348378, 0.40930100675464715, 0.12149482407516043, 0.0008753229400227697, 0.027310075728710415, 0.06827518932177604, 0.00035012917600910786, 0.00017506458800455393, 0.0007002583520182157, 0.003567190451247858, 0.003567190451247858, 0.003567190451247858, 0.003567190451247858, 0.9702758027394174, 0.003567190451247858, 0.003567190451247858, 0.003567190451247858, 0.003567190451247858, 0.003567190451247858, 0.00024997575768627016, 0.00024997575768627016, 0.00024997575768627016, 0.04374575759509728, 0.00024997575768627016, 0.00024997575768627016, 0.00024997575768627016, 0.00024997575768627016, 0.9544074428461795, 0.00024997575768627016, 0.41562932836977945, 0.11257674674102854, 0.04523172860130611, 0.001005149524473469, 0.15479302676891424, 0.001005149524473469, 0.0005025747622367345, 0.0683501676641959, 0.20102990489469383, 0.001005149524473469, 0.001400585166710072, 0.001400585166710072, 0.9481961578627187, 0.00700292583355036, 0.029412288500911512, 0.001400585166710072, 0.005602340666840288, 0.001400585166710072, 0.001400585166710072, 0.001400585166710072, 0.037982123197530346, 0.07596424639506069, 0.023373614275403288, 0.002921701784425411, 0.09349445710161315, 0.011686807137701644, 0.03213871962867952, 0.14608508922127056, 0.5551233390408281, 0.020451912490977876, 0.0021841074927522676, 0.0021841074927522676, 0.004368214985504535, 0.7775422674198074, 0.20093788933320864, 0.0021841074927522676, 0.0021841074927522676, 0.0021841074927522676, 0.0021841074927522676, 0.0021841074927522676, 0.0031755775316671658, 0.0031755775316671658, 0.0015877887658335829, 0.0031755775316671658, 0.004763366297500749, 0.0031755775316671658, 0.0031755775316671658, 0.0015877887658335829, 0.9764900909876534, 0.0015877887658335829, 0.011899299219248495, 0.09519439375398796, 0.01784894882887274, 0.005949649609624247, 0.6544614570586672, 0.029748248048121238, 0.005949649609624247, 0.16064053945985468, 0.01784894882887274, 0.0014767957929007645, 0.002953591585801529, 0.6616045152195426, 0.3219414828523667, 0.002953591585801529, 0.002953591585801529, 0.0014767957929007645, 0.0014767957929007645, 0.0014767957929007645, 0.0014767957929007645, 0.0018551883383387932, 0.13913912537540948, 0.0018551883383387932, 0.6289088466968509, 0.20221552887892846, 0.014841506706710346, 0.0018551883383387932, 0.0018551883383387932, 0.0018551883383387932, 0.0018551883383387932, 0.0008918424453801047, 0.0008918424453801047, 0.7277434354301654, 0.0008918424453801047, 0.19888086531976334, 0.0008918424453801047, 0.0008918424453801047, 0.0017836848907602093, 0.06599634095812774, 0.0008918424453801047, 0.7872636920211821, 0.005831582903860608, 0.002915791451930304, 0.005831582903860608, 0.008747374355790913, 0.1487053640484455, 0.02915791451930304, 0.005831582903860608, 0.002915791451930304, 0.008747374355790913, 0.011062573670772445, 0.0020113770310495357, 0.0010056885155247679, 0.0020113770310495357, 0.0030170655465743034, 0.7874541076558932, 0.08246645827303097, 0.0030170655465743034, 0.10459160561457585, 0.0030170655465743034, 0.0022745556037207723, 0.0011372778018603862, 0.1228260026009217, 0.6414246802492578, 0.12055144699720093, 0.10349227996929514, 0.0034118334055811585, 0.0011372778018603862, 0.0011372778018603862, 0.0034118334055811585, 0.002768637420236384, 0.0018457582801575893, 0.08767351830748549, 0.19841901511694085, 0.7013881464598839, 0.002768637420236384, 0.0018457582801575893, 0.0009228791400787946, 0.0009228791400787946, 0.002768637420236384, 0.0008573939861383536, 0.0008573939861383536, 0.0008573939861383536, 0.0017147879722767072, 0.9902900539897984, 0.0008573939861383536, 0.0017147879722767072, 0.0008573939861383536, 0.0008573939861383536, 0.0008573939861383536, 0.0016083650333967084, 0.3227452500349395, 0.1908593172964094, 0.10668821388198166, 0.12384410757154654, 0.0021444867111956112, 0.0016083650333967084, 0.12277186421594874, 0.07237642650285188, 0.05522053281328699, 0.006231696573071792, 0.006231696573071792, 0.7696145267743664, 0.006231696573071792, 0.003115848286535896, 0.003115848286535896, 0.003115848286535896, 0.003115848286535896, 0.015579241432679481, 0.18071920061908198, 0.0026760002629391154, 0.005352000525878231, 0.0026760002629391154, 0.005352000525878231, 0.008028000788817347, 0.0026760002629391154, 0.008028000788817347, 0.7947720780929173, 0.1578840155134078, 0.010704001051756461, 0.0006637511656766145, 0.8863290565668394, 0.10929769194808253, 0.00022125038855887153, 0.00022125038855887153, 0.00022125038855887153, 0.00022125038855887153, 0.0022125038855887154, 0.00022125038855887153, 0.00022125038855887153, 0.9629395282325661, 0.0018032575435066782, 0.0009016287717533391, 0.0018032575435066782, 0.0009016287717533391, 0.0009016287717533391, 0.023442348065586815, 0.004508143858766696, 0.0009016287717533391, 0.0009016287717533391, 0.9896777259991236, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0012203177879150722, 0.0056412266135829345, 0.06205349274941228, 0.0028206133067914673, 0.0056412266135829345, 0.0028206133067914673, 0.7079739400046583, 0.0028206133067914673, 0.0028206133067914673, 0.20308415808898564, 0.0056412266135829345, 0.0006340561332459595, 0.0006340561332459595, 0.7811571561590221, 0.0006340561332459595, 0.002536224532983838, 0.0006340561332459595, 0.0006340561332459595, 0.001268112266491919, 0.21177474850415048, 0.0006340561332459595, 0.0008338821253240365, 0.001667764250648073, 0.0008338821253240365, 0.9914858470102794, 0.0008338821253240365, 0.0008338821253240365, 0.0008338821253240365, 0.0008338821253240365, 0.0008338821253240365, 0.0008338821253240365, 0.28519180241563047, 0.08056889154930737, 0.0012140517904690152, 0.08399031023153823, 0.09591009144705219, 0.09944187847387115, 0.2964493735636159, 0.04348512776770836, 0.012913096316806797, 0.0007725784121166459, 0.09757846285614813, 0.042579692882682825, 0.005322461610335353, 0.13128738638827203, 0.4541833907486168, 0.0913689243107569, 0.09846553979120402, 0.0017741538701117844, 0.0008870769350558922, 0.07717569334986261, 0.0017595184601258927, 0.21378149290529597, 0.002639277690188839, 0.6642182186975245, 0.10996990375786829, 0.002639277690188839, 0.0008797592300629464, 0.0008797592300629464, 0.0008797592300629464, 0.0017595184601258927, 0.19455437404279094, 0.13487511820144402, 0.0011935851168269384, 0.02029094698605795, 0.015516606518750198, 0.42610988670721694, 0.08474454329471262, 0.029839627920673458, 0.0011935851168269384, 0.09190605399567425, 0.001863304902945468, 0.07173723876340052, 0.002794957354418202, 0.005589914708836404, 0.5860093919763497, 0.000931652451472734, 0.24129798493143811, 0.001863304902945468, 0.08664367798696426, 0.001863304902945468, 0.008248969398529107, 0.9568804502293765, 0.0027496564661763693, 0.0054993129323527385, 0.0027496564661763693, 0.0054993129323527385, 0.0027496564661763693, 0.0054993129323527385, 0.0027496564661763693, 0.0027496564661763693, 0.11812538179330494, 0.2080033896795152, 0.0025679430824631505, 0.5212924457400195, 0.0025679430824631505, 0.015407658494778905, 0.06163063397911562, 0.05135886164926302, 0.007703829247389452, 0.012839715412315754, 0.0018582304365550172, 0.19139773496516677, 0.014865843492440137, 0.08362036964497577, 0.09476975226430587, 0.0037164608731100343, 0.020440534802105188, 0.007432921746220069, 0.5779096657686104, 0.0018582304365550172, 0.0005523278366663169, 0.0005523278366663169, 0.0005523278366663169, 0.0005523278366663169, 0.8848291943394396, 0.0005523278366663169, 0.0005523278366663169, 0.0005523278366663169, 0.11101789516992969, 0.0005523278366663169, 0.002897447686164775, 0.0014487238430823876, 0.002897447686164775, 0.002897447686164775, 0.0014487238430823876, 0.0014487238430823876, 0.0014487238430823876, 0.0014487238430823876, 0.0014487238430823876, 0.9836834894529412, 0.014828042276569578, 0.9341666634238833, 0.004942680758856526, 0.004942680758856526, 0.004942680758856526, 0.004942680758856526, 0.004942680758856526, 0.004942680758856526, 0.004942680758856526, 0.019770723035426103, 0.24363651061858238, 0.009554372965434604, 0.002388593241358651, 0.5087703604093926, 0.045383271585814365, 0.016720152689510555, 0.03344030537902111, 0.07643498372347683, 0.06210342427532492, 0.002388593241358651, 0.0017148864943527869, 0.0017148864943527869, 0.16291421696351477, 0.0068595459774111476, 0.0034297729887055738, 0.0017148864943527869, 0.7099630086620538, 0.0017148864943527869, 0.0017148864943527869, 0.10975273563857836, 0.0008070472244828218, 0.991861038889388, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0008070472244828218, 0.0020685148688475955, 0.004137029737695191, 0.0020685148688475955, 0.0020685148688475955, 0.0020685148688475955, 0.0020685148688475955, 0.0020685148688475955, 0.0020685148688475955, 0.0020685148688475955, 0.9825445627026078, 0.0014044459849096422, 0.0028088919698192843, 0.1769601940986149, 0.009831121894367495, 0.02808891969819284, 0.0028088919698192843, 0.03932448757746998, 0.046346717502018187, 0.0014044459849096422, 0.6909874245755439, 0.01477024769972869, 0.1083151497980104, 0.009846831799819126, 0.777899712185711, 0.009846831799819126, 0.01477024769972869, 0.004923415899909563, 0.009846831799819126, 0.004923415899909563, 0.04431074309918607, 0.058228713094063564, 0.005822871309406357, 0.733681784985201, 0.023291485237625427, 0.163040396663378, 0.005822871309406357, 0.005822871309406357, 0.005822871309406357, 0.005822871309406357, 0.7236036729614496, 0.020930684755083254, 0.0029900978221547502, 0.035881173865857006, 0.03887127168801176, 0.12259401070834476, 0.0059801956443095005, 0.03887127168801176, 0.0029900978221547502, 0.008970293466464252, 0.000694863845455886, 0.10214498528201524, 0.34048328427338415, 0.079214478381971, 0.000694863845455886, 0.000694863845455886, 0.3912083449916638, 0.001389727690911772, 0.000694863845455886, 0.08199393376379455, 0.005721339009545305, 0.01144267801909061, 0.692282020154982, 0.13731213622908733, 0.005721339009545305, 0.005721339009545305, 0.09726276316227019, 0.028606695047726526, 0.005721339009545305, 0.02288535603818122, 0.0010941040051079035, 0.002188208010215807, 0.7800961556419352, 0.2111620729858254, 0.002188208010215807, 0.0010941040051079035, 0.0010941040051079035, 0.0010941040051079035, 0.0010941040051079035, 0.0010941040051079035, 0.0012356250861082414, 0.11058844520668759, 0.5078419103904872, 0.025948126808273065, 0.2773978318313002, 0.0006178125430541207, 0.0012356250861082414, 0.0012356250861082414, 0.0006178125430541207, 0.07413750516649448, 0.00042566898219869817, 0.00042566898219869817, 0.0008513379643973963, 0.00042566898219869817, 0.00042566898219869817, 0.9960654183449537, 0.00042566898219869817, 0.00042566898219869817, 0.00042566898219869817, 0.00042566898219869817, 0.0009948932563281913, 0.0009948932563281913, 0.0009948932563281913, 0.0009948932563281913, 0.0009948932563281913, 0.0019897865126563825, 0.0009948932563281913, 0.0009948932563281913, 0.0009948932563281913, 0.9909136833028785, 0.0024823019651698426, 0.0024823019651698426, 0.0012411509825849213, 0.0012411509825849213, 0.0024823019651698426, 0.0012411509825849213, 0.0012411509825849213, 0.0012411509825849213, 0.9854738801724274, 0.0012411509825849213, 0.4841131519834532, 0.011355740602081003, 0.2850888561680336, 0.10758070044076738, 0.0017930116740127898, 0.0011953411160085265, 0.0011953411160085265, 0.07889251365656275, 0.0011953411160085265, 0.02749284566819611, 0.2996996232632507, 0.2570377551830727, 0.001066546702004451, 0.0031996401060133527, 0.001066546702004451, 0.002133093404008902, 0.004266186808017804, 0.001066546702004451, 0.001066546702004451, 0.42768522750378485, 0.0020430020948348517, 0.004086004189669703, 0.008172008379339407, 0.14301014663843964, 0.7395667583302163, 0.0020430020948348517, 0.08580608798306379, 0.0020430020948348517, 0.004086004189669703, 0.004086004189669703, 0.2163331591651876, 0.11763115529607077, 0.0013520822447824225, 0.0013520822447824225, 0.0013520822447824225, 0.19469984324866885, 0.3853434397629904, 0.0013520822447824225, 0.0013520822447824225, 0.07571660570781566, 0.003663199264017168, 0.003663199264017168, 0.003663199264017168, 0.9707478049645496, 0.003663199264017168, 0.003663199264017168, 0.003663199264017168, 0.003663199264017168, 0.003663199264017168, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.0005505083985734557, 0.995319184620808, 0.0005505083985734557, 0.009888778100090742, 0.009888778100090742, 0.6971588560563973, 0.0692214467006352, 0.004944389050045371, 0.04449950145040834, 0.0024721945250226855, 0.1532760605514065, 0.0024721945250226855, 0.004944389050045371, 0.0032308416265746927, 0.2261589138602285, 0.24554396361967662, 0.061385990904919155, 0.0032308416265746927, 0.0032308416265746927, 0.45554866934703164, 0.0032308416265746927, 0.0032308416265746927, 0.0032308416265746927, 0.0006509873480539975, 0.7134821334671814, 0.28057554701127296, 0.001301974696107995, 0.001301974696107995, 0.0006509873480539975, 0.0006509873480539975, 0.0006509873480539975, 0.0006509873480539975, 0.0006509873480539975, 0.004609764014679734, 0.24662237478536578, 0.002304882007339867, 0.004609764014679734, 0.002304882007339867, 0.002304882007339867, 0.6661109001212215, 0.06684157821285615, 0.002304882007339867, 0.002304882007339867, 0.003889923147266511, 0.16726669533245997, 0.003889923147266511, 0.06612869350353069, 0.7390853979806371, 0.005834884720899766, 0.0019449615736332555, 0.0019449615736332555, 0.007779846294533022, 0.0019449615736332555, 0.9194199559163228, 0.07405369937819754, 0.000961736355561007, 0.0014426045333415106, 0.000961736355561007, 0.001923472711122014, 0.0004808681777805035, 0.0004808681777805035, 0.0004808681777805035, 0.0004808681777805035, 0.0018889356245084061, 0.0037778712490168123, 0.0018889356245084061, 0.0018889356245084061, 0.0018889356245084061, 0.0018889356245084061, 0.0018889356245084061, 0.0018889356245084061, 0.9822465247443711, 0.0018889356245084061, 0.002849246572734691, 0.04202638694783669, 0.35758044487820373, 0.16027011971632638, 0.07835428075020401, 0.0470125684501224, 0.05912186638424484, 0.005698493145469382, 0.06980654103199993, 0.17665328750955084, 0.9492372711801882, 0.00452017748181042, 0.00452017748181042, 0.00452017748181042, 0.00452017748181042, 0.00452017748181042, 0.00452017748181042, 0.00452017748181042, 0.01356053244543126, 0.005846729682875596, 0.005846729682875596, 0.1315514178647009, 0.1140112288160741, 0.014616824207188989, 0.6840673728964447, 0.005846729682875596, 0.002923364841437798, 0.002923364841437798, 0.032157013255815776, 0.0004184818445701628, 0.0004184818445701628, 0.0004184818445701628, 0.0004184818445701628, 0.9976607174552681, 0.0004184818445701628, 0.0004184818445701628, 0.002800320724836897, 0.004200481087255345, 0.6748772946856921, 0.15681796059086622, 0.0014001603624184485, 0.0014001603624184485, 0.12321411189282346, 0.012601443261766036, 0.0014001603624184485, 0.019602245073858278, 0.004081762965320668, 0.004081762965320668, 0.004081762965320668, 0.004081762965320668, 0.004081762965320668, 0.004081762965320668, 0.004081762965320668, 0.008163525930641336, 0.9673778227809984, 0.01869965036202965, 0.012466433574686435, 0.09973146859749148, 0.006233216787343217, 0.09349825181014826, 0.006233216787343217, 0.01869965036202965, 0.5921555947976057, 0.1495972028962372, 0.006233216787343217, 0.0009359713559959879, 0.7712403973406939, 0.0009359713559959879, 0.0009359713559959879, 0.0009359713559959879, 0.0009359713559959879, 0.2208892400150531, 0.0009359713559959879, 0.0009359713559959879, 0.0009359713559959879, 0.16825172888376239, 0.28513592810831395, 0.0003075899979593462, 0.0006151799959186924, 0.4573863269655478, 0.0858176094306576, 0.0003075899979593462, 0.0009227699938780386, 0.0006151799959186924, 0.0006151799959186924, 0.03399121662259851, 0.03399121662259851, 0.03399121662259851, 0.03399121662259851, 0.03399121662259851, 0.06798243324519702, 0.7817979823197658, 0.03399121662259851, 0.03399121662259851, 0.0005283925982716405, 0.0005283925982716405, 0.0005283925982716405, 0.0015851777948149217, 0.9939064773489559, 0.0005283925982716405, 0.0005283925982716405, 0.0005283925982716405, 0.0005283925982716405, 0.007380603443543097, 0.007380603443543097, 0.007380603443543097, 0.007380603443543097, 0.007380603443543097, 0.10332844820960335, 0.007380603443543097, 0.007380603443543097, 0.8487693960074562, 0.003602282042199859, 0.3638304862621858, 0.07204564084399719, 0.5295354602033793, 0.003602282042199859, 0.003602282042199859, 0.003602282042199859, 0.014409128168799436, 0.003602282042199859, 0.003602282042199859, 0.0007177212526573421, 0.0007177212526573421, 0.0007177212526573421, 0.0007177212526573421, 0.0007177212526573421, 0.994043934930419, 0.0007177212526573421, 0.0007177212526573421, 0.0007177212526573421, 0.0007177212526573421, 0.0028381458578464022, 0.0056762917156928045, 0.0028381458578464022, 0.0056762917156928045, 0.0028381458578464022, 0.9678077375256232, 0.0028381458578464022, 0.0028381458578464022, 0.0028381458578464022, 0.0056762917156928045, 0.0008514694688726811, 0.0008514694688726811, 0.32526133710936417, 0.664997655189564, 0.004257347344363406, 0.0008514694688726811, 0.0008514694688726811, 0.0008514694688726811, 0.0008514694688726811, 0.0008514694688726811, 0.01993001814442472, 0.001022052212534601, 0.21923019958867193, 0.0015330783188019015, 0.0005110261062673006, 0.0005110261062673006, 0.7542745328505356, 0.0005110261062673006, 0.0005110261062673006, 0.0015330783188019015, 0.7692319200697406, 0.0016722433044994362, 0.0016722433044994362, 0.0016722433044994362, 0.0016722433044994362, 0.12876273444645658, 0.0033444866089988724, 0.0016722433044994362, 0.0016722433044994362, 0.08695665183397068, 0.08511406675787901, 0.07134561478233976, 0.0012516774523217502, 0.11640600306592278, 0.6258387261608751, 0.08761742166252251, 0.0025033549046435005, 0.007510064713930501, 0.0012516774523217502, 0.0025033549046435005, 0.0008352228766040012, 0.0008352228766040012, 0.9914095545289495, 0.0008352228766040012, 0.0008352228766040012, 0.0008352228766040012, 0.0008352228766040012, 0.0008352228766040012, 0.0008352228766040012, 0.0008352228766040012, 0.2046111174338887, 0.0064955910296472606, 0.0064955910296472606, 0.009743386544470892, 0.0064955910296472606, 0.0032477955148236303, 0.022734568603765413, 0.009743386544470892, 0.0032477955148236303, 0.721010604290846, 0.0017887963282113585, 0.005366388984634075, 0.0017887963282113585, 0.0017887963282113585, 0.007155185312845434, 0.0017887963282113585, 0.0017887963282113585, 0.0017887963282113585, 0.0643966678156089, 0.9087085347313701, 0.008446090758713071, 0.006756872606970457, 0.005067654455227843, 0.9560974738863197, 0.006756872606970457, 0.006756872606970457, 0.0016892181517426143, 0.0033784363034852286, 0.0033784363034852286, 0.0033784363034852286, 0.006678573974700371, 0.006678573974700371, 0.006678573974700371, 0.006678573974700371, 0.006678573974700371, 0.006678573974700371, 0.006678573974700371, 0.955036078382153, 0.0012634076903805154, 0.0008422717935870104, 0.0008422717935870104, 0.05558993837674268, 0.0012634076903805154, 0.0008422717935870104, 0.937027370365549, 0.0004211358967935052, 0.0004211358967935052, 0.0008422717935870104, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.0012229020547814989, 0.9893277623182326, 0.022166677869269118, 0.11452783565789043, 0.003694446311544853, 0.007388892623089706, 0.014777785246179412, 0.7351948159974258, 0.09236115778862132, 0.003694446311544853, 0.003694446311544853, 0.003694446311544853, 0.002193045501953244, 0.004386091003906488, 0.002193045501953244, 0.002193045501953244, 0.004386091003906488, 0.002193045501953244, 0.002193045501953244, 0.002193045501953244, 0.9737122028672404, 0.002193045501953244, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.002072991559647521, 0.982597999272925, 0.002072991559647521, 0.0005752744589871009, 0.0319277324737841, 0.05206233853833263, 0.006903293507845211, 0.11908181301032988, 0.0005752744589871009, 0.062129641570606894, 0.1705688770896754, 0.0011505489179742018, 0.555427490152046, 0.009282680817086593, 0.009282680817086593, 0.03713072326834637, 0.03713072326834637, 0.1392402122562989, 0.7054837420985811, 0.009282680817086593, 0.03713072326834637, 0.009282680817086593, 0.009282680817086593, 0.0018290329809890807, 0.0018290329809890807, 0.7325277088861268, 0.25240655137649315, 0.007316131923956323, 0.0009145164904945404, 0.0009145164904945404, 0.0009145164904945404, 0.0009145164904945404, 0.0009145164904945404, 0.0020888368974761877, 0.0010444184487380939, 0.0010444184487380939, 0.0010444184487380939, 0.0010444184487380939, 0.0010444184487380939, 0.9880198525062369, 0.0010444184487380939, 0.0020888368974761877, 0.0010444184487380939, 0.24008011877263272, 0.02286477321644121, 0.010830682049893205, 0.11673068431551566, 0.12876477548206366, 0.34056478001330853, 0.0018051136749822009, 0.09025568374911004, 0.0042119319082918015, 0.04332272819957282, 0.0004329725937303872, 0.21345548870908088, 0.0008659451874607744, 0.0004329725937303872, 0.6451291646582769, 0.0004329725937303872, 0.0004329725937303872, 0.0004329725937303872, 0.1381182573999935, 0.0004329725937303872, 0.00022632253556728168, 0.00022632253556728168, 0.00022632253556728168, 0.00022632253556728168, 0.00022632253556728168, 0.00022632253556728168, 0.9980823818517123, 0.00022632253556728168, 0.00022632253556728168, 0.00022632253556728168, 0.21501537792288236, 0.11326291856323138, 0.001381255104429651, 0.061696061331191085, 0.001381255104429651, 0.001381255104429651, 0.08517739810649515, 0.12201086755795251, 0.000460418368143217, 0.3987223068120259, 0.0031006034074475613, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.0015503017037237807, 0.9828912801608769, 0.06047000083847815, 0.1959228027166692, 0.004837600067078252, 0.10884600150926067, 0.002418800033539126, 0.10884600150926067, 0.5079480070432165, 0.002418800033539126, 0.002418800033539126, 0.004837600067078252, 0.8597037083752139, 0.09236486123039489, 0.003552494662707496, 0.007104989325414992, 0.003552494662707496, 0.003552494662707496, 0.010657483988122487, 0.007104989325414992, 0.003552494662707496, 0.007104989325414992, 0.005279452644078111, 0.0013198631610195277, 0.9793384654764895, 0.003959589483058583, 0.0013198631610195277, 0.0026397263220390554, 0.0013198631610195277, 0.0013198631610195277, 0.0013198631610195277, 0.0013198631610195277, 0.000681100709174738, 0.000340550354587369, 0.6664570439274812, 0.000340550354587369, 0.000340550354587369, 0.000340550354587369, 0.3306743943043353, 0.000340550354587369, 0.000340550354587369, 0.000340550354587369, 0.7608070527694565, 0.222375195772167, 0.0018925548576354638, 0.0047313871440886595, 0.0009462774288177319, 0.0018925548576354638, 0.0018925548576354638, 0.0018925548576354638, 0.0009462774288177319, 0.0028388322864531957, 0.007774425647628068, 0.09329310777153682, 0.007774425647628068, 0.8668484597105296, 0.007774425647628068, 0.007774425647628068, 0.003887212823814034, 0.003887212823814034, 0.003887212823814034, 0.003887212823814034, 0.0011578102767354948, 0.07525766798780716, 0.24661358894466037, 0.669214339953116, 0.0011578102767354948, 0.0011578102767354948, 0.0011578102767354948, 0.0011578102767354948, 0.0011578102767354948, 0.0023156205534709895, 0.029343707090277976, 0.022007780317708482, 0.14671853545138988, 0.7482645308020884, 0.007335926772569494, 0.007335926772569494, 0.014671853545138988, 0.022007780317708482, 0.0037492286727931045, 0.03749228672793104, 0.20620757700362075, 0.644867331720414, 0.011247686018379313, 0.007498457345586209, 0.0037492286727931045, 0.011247686018379313, 0.007498457345586209, 0.06748611611027588, 0.20558986687136, 0.1016201341964151, 0.011160592773016686, 0.08341074598780891, 0.05697776310434834, 0.33951698014756027, 0.12100432164428618, 0.0017621988588973714, 0.0011747992392649143, 0.0781241494111168, 0.000394300171420789, 0.0001971500857103945, 0.0001971500857103945, 0.0001971500857103945, 0.9981708839517274, 0.0001971500857103945, 0.0001971500857103945, 0.0001971500857103945, 0.0001971500857103945, 0.0001971500857103945, 0.16598626452718668, 0.0008490346011620802, 0.6197952588483185, 0.09254477152666674, 0.0004245173005810401, 0.0008490346011620802, 0.11801580956152914, 0.0004245173005810401, 0.0004245173005810401, 0.0004245173005810401, 0.994697713331884, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.0004998480971517005, 0.8472707205833753, 0.0006433338804733297, 0.00032166694023666486, 0.00032166694023666486, 0.00032166694023666486, 0.14861012638933915, 0.0012866677609466594, 0.00032166694023666486, 0.00032166694023666486, 0.00032166694023666486, 0.8636777971084639, 0.0022145584541242666, 0.055363961353106665, 0.004429116908248533, 0.006643675362372799, 0.006643675362372799, 0.0022145584541242666, 0.033218376811864, 0.024360142995366933, 0.0022145584541242666, 0.002467380992255109, 0.8685181092737984, 0.004934761984510218, 0.004934761984510218, 0.10363000167471458, 0.002467380992255109, 0.002467380992255109, 0.004934761984510218, 0.002467380992255109, 0.002467380992255109, 0.005234052153098713, 0.0026170260765493564, 0.0026170260765493564, 0.9552145179405152, 0.013085130382746783, 0.005234052153098713, 0.007851078229648069, 0.005234052153098713, 0.0026170260765493564, 0.18936164194982416, 0.12799888202684317, 0.0007989942698304818, 0.11649336454128424, 0.0003195977079321927, 0.15868026198833368, 0.3675373641220216, 0.00015979885396609635, 0.0003195977079321927, 0.03835172495186313, 0.0898061505324451, 0.03987244027789056, 0.06893833132158649, 0.10471173568305839, 0.11104660937206905, 0.3189795222231245, 0.010433909605429306, 0.0007452792575306647, 0.17104158960328755, 0.08458919572973045, 0.9873141228755696, 0.0010933711216783717, 0.0010933711216783717, 0.0010933711216783717, 0.0010933711216783717, 0.0010933711216783717, 0.0010933711216783717, 0.0021867422433567434, 0.0010933711216783717, 0.0010933711216783717, 0.0022184479033450404, 0.004436895806690081, 0.0022184479033450404, 0.0022184479033450404, 0.0022184479033450404, 0.0022184479033450404, 0.0022184479033450404, 0.0022184479033450404, 0.9783355253751628, 0.0022184479033450404, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.0007660148827847585, 0.9935213029718317, 0.0007660148827847585, 0.001954933515702084, 0.003909867031404168, 0.001954933515702084, 0.001954933515702084, 0.001954933515702084, 0.001954933515702084, 0.001954933515702084, 0.001954933515702084, 0.001954933515702084, 0.9813766248824463, 0.0704030830272514, 0.00782256478080571, 0.023467694342417134, 0.01564512956161142, 0.6570954415876797, 0.00782256478080571, 0.00782256478080571, 0.04693538868483427, 0.023467694342417134, 0.1408061660545028, 0.0008590523279322015, 0.003436209311728806, 0.0008590523279322015, 0.13057595384569462, 0.3470571404846094, 0.001718104655864403, 0.0008590523279322015, 0.0025771569837966042, 0.06786513390664392, 0.44327100121301594, 0.0011938247812857698, 0.9962467799829748, 0.0005969123906428849, 0.00029845619532144245, 0.00029845619532144245, 0.00029845619532144245, 0.00029845619532144245, 0.00029845619532144245, 0.00029845619532144245, 0.00029845619532144245, 0.03500909600042255, 0.03500909600042255, 0.03500909600042255, 0.03500909600042255, 0.03500909600042255, 0.03500909600042255, 0.0700181920008451, 0.7001819200084509, 0.03500909600042255, 0.15714943984946658, 0.7134584569165783, 0.0031429887969893315, 0.0031429887969893315, 0.0031429887969893315, 0.0031429887969893315, 0.0031429887969893315, 0.10686161909763728, 0.0031429887969893315, 0.006285977593978663, 0.0009368914209072472, 0.0004684457104536236, 0.0004684457104536236, 0.0004684457104536236, 0.0004684457104536236, 0.0004684457104536236, 0.9949786890034966, 0.0004684457104536236, 0.0004684457104536236, 0.0004684457104536236, 0.05478547360848385, 0.05478547360848385, 0.00913091226808064, 0.00913091226808064, 0.6665565955698868, 0.00913091226808064, 0.00913091226808064, 0.14609459628929025, 0.03652364907232256, 0.0006938167500889637, 0.0006938167500889637, 0.0013876335001779274, 0.002081450250266891, 0.7493220900960808, 0.0006938167500889637, 0.0006938167500889637, 0.0006938167500889637, 0.2428358625311373, 0.0006938167500889637, 0.00512419340929007, 0.00512419340929007, 0.015372580227870207, 0.01024838681858014, 0.8813612663978919, 0.00512419340929007, 0.00512419340929007, 0.07173870773006097, 0.00512419340929007, 0.00512419340929007, 0.00045194096496768525, 0.0009038819299353705, 0.24585588494242078, 0.00045194096496768525, 0.17625697633739726, 0.00045194096496768525, 0.00045194096496768525, 0.00045194096496768525, 0.5739650255089603, 0.00045194096496768525, 0.002250216191999928, 0.2857774563839908, 0.001125108095999964, 0.001125108095999964, 0.001125108095999964, 0.001125108095999964, 0.001125108095999964, 0.1305125391359958, 0.5738051289599816, 0.001125108095999964, 0.0031666480631557478, 0.08312451165783838, 0.23274863264194745, 0.18049893959987762, 0.0007916620157889369, 0.07995786359468263, 0.0015833240315778739, 0.0015833240315778739, 0.0007916620157889369, 0.4148308962734029, 0.12364137304761916, 0.00824275820317461, 0.00824275820317461, 0.7088772054730165, 0.049456549219047664, 0.06594206562539688, 0.00824275820317461, 0.00824275820317461, 0.00824275820317461, 0.01648551640634922, 0.0032280695726980006, 0.132350852480618, 0.0016140347863490003, 0.0016140347863490003, 0.0016140347863490003, 0.004842104359047, 0.0016140347863490003, 0.8489822976195741, 0.0016140347863490003, 0.0016140347863490003, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.0011716299198172154, 0.9911989121653643, 0.10453683265871344, 0.15680524898807016, 0.008711402721559453, 0.004355701360779726, 0.004355701360779726, 0.030489909525458086, 0.004355701360779726, 0.6315766973130603, 0.004355701360779726, 0.05226841632935672, 0.00024765879443266054, 0.00024765879443266054, 0.00024765879443266054, 0.00024765879443266054, 0.00024765879443266054, 0.00024765879443266054, 0.00024765879443266054, 0.9980649415636219, 0.00024765879443266054, 0.00024765879443266054, 0.0016883141993906323, 0.0016883141993906323, 0.7209101631397999, 0.0033766283987812645, 0.04558448338354707, 0.08610402416892225, 0.03883122658598454, 0.0033766283987812645, 0.02194808459207822, 0.07597413897257846, 0.0024685536041703484, 0.0037028304062555225, 0.0012342768020851742, 0.0024685536041703484, 0.1567531538648171, 0.8269654573970667, 0.0012342768020851742, 0.0024685536041703484, 0.0012342768020851742, 0.0012342768020851742, 0.5276232244121398, 0.11886792143924431, 0.04980237024727232, 0.09396673631560816, 0.0009396673631560815, 0.09067790054456187, 0.11463941830504194, 0.0009396673631560815, 0.00046983368157804076, 0.001879334726312163, 0.0018738280126108084, 0.9800120505954527, 0.0018738280126108084, 0.0018738280126108084, 0.0018738280126108084, 0.0018738280126108084, 0.0037476560252216167, 0.0018738280126108084, 0.0018738280126108084, 0.0018738280126108084, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.0009344862905418214, 0.9924244405554142, 0.006368147545431894, 0.003184073772715947, 0.003184073772715947, 0.003184073772715947, 0.003184073772715947, 0.003184073772715947, 0.006368147545431894, 0.003184073772715947, 0.003184073772715947, 0.9711425006783639, 0.1643679074664847, 0.06832548310371521, 0.0006445800292803321, 0.0006445800292803321, 0.0006445800292803321, 0.0012891600585606643, 0.0006445800292803321, 0.0012891600585606643, 0.0006445800292803321, 0.7612490145800722, 0.0062410868785177055, 0.002080362292839235, 0.0062410868785177055, 0.00416072458567847, 0.002080362292839235, 0.00416072458567847, 0.00416072458567847, 0.002080362292839235, 0.002080362292839235, 0.9694488284630836, 0.2995410786271272, 0.14147705668056496, 0.09415542048051392, 0.10440031079186518, 0.0009757038391763101, 0.08195912249081005, 0.12342653565580322, 0.020977632542290667, 0.0009757038391763101, 0.13220787020839, 0.0034711318379054197, 0.0034711318379054197, 0.9719169146135176, 0.0034711318379054197, 0.0034711318379054197, 0.0034711318379054197, 0.0034711318379054197, 0.006942263675810839, 0.002326542678977606, 0.002326542678977606, 0.9422497849859304, 0.002326542678977606, 0.004653085357955212, 0.002326542678977606, 0.002326542678977606, 0.002326542678977606, 0.03257159750568648, 0.009306170715910424, 0.0004122759019236369, 0.0004122759019236369, 0.0004122759019236369, 0.0004122759019236369, 0.17686636192524022, 0.8196044930241901, 0.0004122759019236369, 0.0004122759019236369, 0.0004122759019236369, 0.0004122759019236369, 0.002787137838386016, 0.005574275676772032, 0.002787137838386016, 0.002787137838386016, 0.002787137838386016, 0.9727111055967196, 0.002787137838386016, 0.002787137838386016, 0.002787137838386016, 0.002787137838386016, 0.0002154801789478513, 0.0002154801789478513, 0.0002154801789478513, 0.9981041888864471, 0.0002154801789478513, 0.0002154801789478513, 0.0002154801789478513, 0.0002154801789478513, 0.0002154801789478513, 0.0002154801789478513, 0.000584904930334009, 0.8168197352114435, 0.0002924524651670045, 0.0002924524651670045, 0.0002924524651670045, 0.0002924524651670045, 0.0002924524651670045, 0.18044317100804175, 0.000584904930334009, 0.0002924524651670045, 0.009268338779318722, 0.30276573345774493, 0.04325224763682071, 0.0030894462597729076, 0.0030894462597729076, 0.006178892519545815, 0.006178892519545815, 0.11430951161159758, 0.5035797403429839, 0.006178892519545815, 0.6622428501709767, 0.031703115167759525, 0.08806420879933202, 0.06340623033551905, 0.0035225683519732806, 0.13385759737498468, 0.0035225683519732806, 0.007045136703946561, 0.0035225683519732806, 0.007045136703946561, 0.3627205540386883, 0.4366080743058285, 0.0016792618242531866, 0.0008396309121265933, 0.0008396309121265933, 0.07892530573989977, 0.01763224915465846, 0.008396309121265932, 0.09068013850967208, 0.0016792618242531866, 0.0019827273276053032, 0.0019827273276053032, 0.0019827273276053032, 0.0019827273276053032, 0.0019827273276053032, 0.0019827273276053032, 0.0019827273276053032, 0.9814500271646251, 0.0039654546552106065, 0.0019827273276053032, 0.007231530741614589, 0.0036157653708072945, 0.0036157653708072945, 0.0036157653708072945, 0.0036157653708072945, 0.0036157653708072945, 0.0036157653708072945, 0.9654093540055476, 0.0036157653708072945, 0.0036157653708072945, 0.0030037551445301317, 0.001802253086718079, 0.0006007510289060263, 0.002403004115624105, 0.0012015020578120526, 0.9852316874058832, 0.0012015020578120526, 0.0012015020578120526, 0.0012015020578120526, 0.0006007510289060263, 0.002745065752636479, 0.002745065752636479, 0.002745065752636479, 0.002745065752636479, 0.002745065752636479, 0.002745065752636479, 0.002745065752636479, 0.979988473691223, 0.002745065752636479], "Term": ["access", "access", "access", "access", "access", "access", "access", "access", "access", "access", "act", "act", "act", "act", "act", "act", "act", "act", "act", "act", "action", "action", "action", "action", "action", "action", "action", "action", "action", "action", "actually", "actually", "actually", "actually", "actually", "actually", "actually", "actually", "actually", "actually", "administration", "administration", "administration", "administration", "administration", "administration", "administration", "administration", "administration", "administration", "advice", "advice", "advice", "advice", "advice", "advice", "advice", "advice", "advice", "advice", "africa", "africa", "africa", "africa", "africa", "africa", "africa", "africa", "africa", "africa", "ago", "ago", "ago", "ago", "ago", "ago", "ago", "ago", "ago", "ago", "air", "air", "air", "air", "air", "air", "air", "air", "air", "air", "america", "america", "america", "america", "america", "america", "america", "america", "america", "america", "american", "american", "american", "american", "american", "american", "american", "american", "american", "american", "americans", "americans", "americans", "americans", "americans", "americans", "americans", "americans", "americans", "americans", "amid", "amid", "amid", "amid", "amid", "amid", "amid", "amid", "amid", "amid", "amp", "amp", "amp", "amp", "amp", "amp", "amp", "answer", "answer", "answer", "answer", "answer", "answer", "answer", "answer", "answer", "answer", "april", "april", "april", "april", "april", "april", "april", "april", "april", "ask", "ask", "ask", "ask", "ask", "ask", "ask", "ask", "ask", "ask", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "avoid", "avoid", "avoid", "avoid", "avoid", "avoid", "avoid", "avoid", "avoid", "avoid", "away", "away", "away", "away", "away", "away", "away", "away", "away", "away", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "bad", "believe", "believe", "believe", "believe", "believe", "believe", "believe", "believe", "believe", "believe", "biden", "biden", "biden", "biden", "biden", "biden", "biden", "biden", "biden", "biden", "black", "black", "black", "black", "black", "black", "black", "black", "black", "black", "blame", "blame", "blame", "blame", "blame", "blame", "blame", "blame", "blame", "blame", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "borisjohnson", "break", "break", "break", "break", "break", "break", "break", "break", "break", "break", "bring", "bring", "bring", "bring", "bring", "bring", "bring", "bring", "bring", "bring", "business", "business", "business", "business", "business", "business", "business", "business", "business", "business", "buy", "buy", "buy", "buy", "buy", "buy", "buy", "buy", "buy", "buy", "california", "california", "california", "california", "california", "california", "california", "california", "california", "california", "cancel", "cancel", "cancel", "cancel", "cancel", "cancel", "cancel", "cancel", "cancel", "cancel", "care", "care", "care", "care", "care", "care", "care", "care", "care", "care", "case", "case", "case", "case", "case", "case", "case", "case", "case", "case", "catch", "catch", "catch", "catch", "catch", "catch", "catch", "catch", "catch", "catch", "change", "change", "change", "change", "change", "change", "change", "change", "change", "change", "check", "check", "check", "check", "check", "check", "check", "check", "check", "check", "china", "china", "china", "china", "china", "china", "china", "china", "china", "china", "chinese", "chinese", "chinese", "chinese", "chinese", "chinese", "chinese", "chinese", "chinese", "chinese", "city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "clean", "clean", "clean", "clean", "clean", "clean", "clean", "clean", "clean", "clean", "click", "click", "click", "click", "click", "click", "click", "click", "click", "click", "close", "close", "close", "close", "close", "close", "close", "close", "close", "close", "come", "come", "come", "come", "come", "come", "come", "come", "come", "come", "common", "common", "common", "common", "common", "common", "common", "common", "common", "common", "community", "community", "community", "community", "community", "community", "community", "community", "community", "community", "company", "company", "company", "company", "company", "company", "company", "company", "company", "company", "conference", "conference", "conference", "conference", "conference", "conference", "conference", "conference", "conference", "conference", "continue", "continue", "continue", "continue", "continue", "continue", "continue", "continue", "continue", "continue", "copy", "copy", "copy", "copy", "copy", "copy", "copy", "copy", "copy", "corona", "corona", "corona", "corona", "corona", "corona", "corona", "corona", "corona", "corona", "coronaalert", "coronaalert", "coronaalert", "coronaalert", "coronaalert", "coronaalert", "coronaalert", "coronaalert", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronaoutbreak", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavid19", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronavirus", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronaviruschallenge", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusinindia", "coronavirusoutbreak", "coronavirusoutbreak", "coronavirusoutbreak", "coronavirusoutbreak", "coronavirusoutbreak", "coronavirusoutbreak", "coronavirusoutbreak", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronaviruspandemic", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirussa", "coronavirustruth", "coronavirustruth", "coronavirustruth", "coronavirustruth", "coronavirustruth", "coronavirustruth", "coronavirustruth", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusuk", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdate", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirusupdates", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "coronavirususa", "cost", "cost", "cost", "cost", "cost", "cost", "cost", "cost", "cost", "cost", "cough", "cough", "cough", "cough", "cough", "cough", "cough", "cough", "cough", "cough", "country", "country", "country", "country", "country", "country", "country", "country", "country", "country", "county", "county", "county", "county", "county", "county", "county", "county", "county", "county", "coverage", "coverage", "coverage", "coverage", "coverage", "coverage", "coverage", "covid", "covid", "covid", "covid", "covid", "covid", "covid", "covid", "covid", "covid", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid19india", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019", "covid2019uk", "covid2019uk", "covid2019uk", "covid2019uk", "covid2019uk", "covid2019uk", "covid2019uk", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid_19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "covid\u30fc19", "crazy", "crazy", "crazy", "crazy", "crazy", "crazy", "crazy", "crazy", "crazy", "crazy", "crisis", "crisis", "crisis", "crisis", "crisis", "crisis", "crisis", "crisis", "crisis", "crisis", "cure", "cure", "cure", "cure", "cure", "cure", "cure", "cure", "cure", "cure", "day", "day", "day", "day", "day", "day", "day", "day", "day", "day", "dead", "dead", "dead", "dead", "dead", "dead", "dead", "dead", "dead", "dead", "deadly", "deadly", "deadly", "deadly", "deadly", "deadly", "deadly", "deadly", "deadly", "deadly", "dear", "dear", "dear", "dear", "dear", "dear", "dear", "dear", "dear", "dear", "death", "death", "death", "death", "death", "death", "death", "death", "death", "death", "deliver", "deliver", "deliver", "deliver", "deliver", "deliver", "deliver", "deliver", "deliver", "department", "department", "department", "department", "department", "department", "department", "department", "department", "die", "die", "die", "die", "die", "die", "die", "die", "die", "die", "disease", "disease", "disease", "disease", "disease", "disease", "disease", "disease", "disease", "disease", "doctor", "doctor", "doctor", "doctor", "doctor", "doctor", "doctor", "doctor", "doctor", "doctor", "economic", "economic", "economic", "economic", "economic", "economic", "economic", "economic", "economic", "economic", "economy", "economy", "economy", "economy", "economy", "economy", "economy", "economy", "economy", "economy", "elderly", "elderly", "elderly", "elderly", "elderly", "elderly", "elderly", "elderly", "elderly", "elderly", "email", "email", "email", "email", "email", "email", "email", "email", "email", "email", "emergency", "emergency", "emergency", "emergency", "emergency", "emergency", "emergency", "emergency", "emergency", "emergency", "end", "end", "end", "end", "end", "end", "end", "end", "end", "end", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "enjoy", "essential", "essential", "essential", "essential", "essential", "essential", "essential", "essential", "essential", "essential", "event", "event", "event", "event", "event", "event", "event", "event", "event", "event", "face", "face", "face", "face", "face", "face", "face", "face", "face", "face", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook", "facebook", "failure", "failure", "failure", "failure", "failure", "failure", "failure", "failure", "failure", "failure", "family", "family", "family", "family", "family", "family", "family", "family", "family", "family", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "feel", "feel", "feel", "feel", "feel", "feel", "feel", "feel", "feel", "feel", "fight", "fight", "fight", "fight", "fight", "fight", "fight", "fight", "fight", "fight", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "flattenthecurve", "florida", "florida", "florida", "florida", "florida", "florida", "florida", "florida", "florida", "florida", "flu", "flu", "flu", "flu", "flu", "flu", "flu", "flu", "flu", "flu", "food", "food", "food", "food", "food", "food", "food", "food", "food", "food", "football", "football", "football", "football", "football", "football", "football", "football", "football", "football", "forward", "forward", "forward", "forward", "forward", "forward", "forward", "forward", "forward", "foxnews", "foxnews", "foxnews", "foxnews", "foxnews", "foxnews", "foxnews", "foxnews", "foxnews", "france", "france", "france", "france", "france", "france", "france", "france", "france", "france", "free", "free", "free", "free", "free", "free", "free", "free", "free", "free", "fuck", "fuck", "fuck", "fuck", "fuck", "fuck", "fuck", "fuck", "fuck", "fuck", "fun", "fun", "fun", "fun", "fun", "fun", "fun", "fun", "fun", "funny", "funny", "funny", "funny", "funny", "funny", "funny", "funny", "funny", "funny", "game", "game", "game", "game", "game", "game", "game", "game", "game", "game", "germany", "germany", "germany", "germany", "germany", "germany", "germany", "germany", "germany", "germany", "global", "global", "global", "global", "global", "global", "global", "global", "global", "global", "god", "god", "god", "god", "god", "god", "god", "god", "god", "god", "good", "good", "good", "good", "good", "good", "good", "good", "good", "good", "gop", "gop", "gop", "gop", "gop", "gop", "gop", "gop", "gop", "gop", "gov", "gov", "gov", "gov", "gov", "gov", "gov", "gov", "gov", "gov", "government", "government", "government", "government", "government", "government", "government", "government", "government", "government", "governor", "governor", "governor", "governor", "governor", "governor", "governor", "governor", "governor", "governor", "govt", "govt", "govt", "govt", "govt", "govt", "govt", "govt", "govt", "govt", "great", "great", "great", "great", "great", "great", "great", "great", "great", "great", "guidance", "guidance", "guidance", "guidance", "guidance", "guidance", "guidance", "guidance", "guidance", "guidance", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "hand", "happen", "happen", "happen", "happen", "happen", "happen", "happen", "happen", "happen", "happen", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard", "health", "health", "health", "health", "health", "health", "health", "health", "health", "health", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthcare", "healthy", "healthy", "healthy", "healthy", "healthy", "healthy", "healthy", "healthy", "healthy", "healthy", "hear", "hear", "hear", "hear", "hear", "hear", "hear", "hear", "hear", "hear", "help", "help", "help", "help", "help", "help", "help", "help", "help", "help", "high", "high", "high", "high", "high", "high", "high", "high", "high", "high", "hit", "hit", "hit", "hit", "hit", "hit", "hit", "hit", "hit", "hit", "hoax", "hoax", "hoax", "hoax", "hoax", "hoax", "hoax", "hoax", "hoax", "hoax", "home", "home", "home", "home", "home", "home", "home", "home", "home", "home", "hope", "hope", "hope", "hope", "hope", "hope", "hope", "hope", "hope", "hope", "hospital", "hospital", "hospital", "hospital", "hospital", "hospital", "hospital", "hospital", "hospital", "hospital", "house", "house", "house", "house", "house", "house", "house", "house", "house", "house", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "hygiene", "idea", "idea", "idea", "idea", "idea", "idea", "idea", "idea", "idea", "idea", "impact", "impact", "impact", "impact", "impact", "impact", "impact", "impact", "impact", "impact", "increase", "increase", "increase", "increase", "increase", "increase", "increase", "increase", "increase", "increase", "india", "india", "india", "india", "india", "india", "india", "india", "india", "india", "industry", "industry", "industry", "industry", "industry", "industry", "industry", "industry", "industry", "industry", "infection", "infection", "infection", "infection", "infection", "infection", "infection", "infection", "infection", "infection", "information", "information", "information", "information", "information", "information", "information", "information", "information", "information", "insurance", "insurance", "insurance", "insurance", "insurance", "insurance", "insurance", "insurance", "iran", "iran", "iran", "iran", "iran", "iran", "iran", "iran", "iran", "iran", "isolate", "isolate", "isolate", "isolate", "isolate", "isolate", "isolate", "isolate", "isolate", "isolate", "isolation", "isolation", "isolation", "isolation", "isolation", "isolation", "isolation", "isolation", "isolation", "isolation", "italian", "italian", "italian", "italian", "italian", "italian", "italian", "italian", "italian", "italian", "italy", "italy", "italy", "italy", "italy", "italy", "italy", "italy", "italy", "italy", "japan", "japan", "japan", "japan", "japan", "japan", "japan", "japan", "japan", "japan", "job", "job", "job", "job", "job", "job", "job", "job", "job", "job", "joebiden", "joebiden", "joebiden", "joebiden", "joebiden", "joebiden", "joebiden", "joebiden", "joebiden", "kill", "kill", "kill", "kill", "kill", "kill", "kill", "kill", "kill", "kill", "know", "know", "know", "know", "know", "know", "know", "know", "know", "know", "korea", "korea", "korea", "korea", "korea", "korea", "korea", "korea", "korea", "korea", "late", "late", "late", "late", "late", "late", "late", "late", "late", "late", "lead", "lead", "lead", "lead", "lead", "lead", "lead", "lead", "lead", "lead", "leader", "leader", "leader", "leader", "leader", "leader", "leader", "leader", "leader", "leader", "leadership", "leadership", "leadership", "leadership", "leadership", "leadership", "leadership", "leadership", "leadership", "leadership", "league", "league", "league", "league", "league", "league", "league", "league", "league", "league", "leave", "leave", "leave", "leave", "leave", "leave", "leave", "leave", "leave", "leave", "let", "let", "let", "let", "let", "let", "let", "let", "let", "let", "like", "like", "like", "like", "like", "like", "like", "like", "like", "like", "listen", "listen", "listen", "listen", "listen", "listen", "listen", "listen", "listen", "listen", "live", "live", "live", "live", "live", "live", "live", "live", "live", "live", "local", "local", "local", "local", "local", "local", "local", "local", "local", "local", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "lockdown", "long", "long", "long", "long", "long", "long", "long", "long", "long", "long", "look", "look", "look", "look", "look", "look", "look", "look", "look", "look", "lot", "lot", "lot", "lot", "lot", "lot", "lot", "lot", "lot", "lot", "love", "love", "love", "love", "love", "love", "love", "love", "love", "love", "maga", "maga", "maga", "maga", "maga", "maga", "maga", "maga", "maga", "maga", "make", "make", "make", "make", "make", "make", "make", "make", "make", "make", "man", "man", "man", "man", "man", "man", "man", "man", "man", "man", "mar", "mar", "mar", "mar", "mar", "mar", "mar", "mar", "mar", "mar", "march", "march", "march", "march", "march", "march", "march", "march", "march", "march", "market", "market", "market", "market", "market", "market", "market", "market", "market", "market", "mask", "mask", "mask", "mask", "mask", "mask", "mask", "mask", "mask", "mask", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe", "medical", "medical", "medical", "medical", "medical", "medical", "medical", "medical", "medical", "medical", "medicine", "medicine", "medicine", "medicine", "medicine", "medicine", "medicine", "medicine", "medicine", "medicine", "meeting", "meeting", "meeting", "meeting", "meeting", "meeting", "meeting", "meeting", "meeting", "meeting", "message", "message", "message", "message", "message", "message", "message", "message", "message", "message", "million", "million", "million", "million", "million", "million", "million", "million", "million", "million", "mind", "mind", "mind", "mind", "mind", "mind", "mind", "mind", "mind", "mind", "minister", "minister", "minister", "minister", "minister", "minister", "minister", "minister", "minister", "minister", "monday", "monday", "monday", "monday", "monday", "monday", "monday", "monday", "monday", "monday", "money", "money", "money", "money", "money", "money", "money", "money", "money", "money", "month", "month", "month", "month", "month", "month", "month", "month", "month", "month", "msnbc", "msnbc", "msnbc", "msnbc", "msnbc", "music", "music", "music", "music", "music", "music", "music", "music", "music", "music", "nature", "nature", "nature", "nature", "nature", "nature", "nature", "nature", "nature", "nature", "need", "need", "need", "need", "need", "need", "need", "need", "need", "need", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "negative", "new", "new", "new", "new", "new", "new", "new", "new", "new", "new", "news", "news", "news", "news", "news", "news", "news", "news", "news", "news", "nhs", "nhs", "nhs", "nhs", "nhs", "nhs", "nhs", "nhs", "nhs", "nhs", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "nigeria", "non", "non", "non", "non", "non", "non", "non", "non", "non", "non", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "ny", "ny", "ny", "ny", "ny", "ny", "ny", "ny", "ny", "offer", "offer", "offer", "offer", "offer", "offer", "offer", "offer", "offer", "offer", "office", "office", "office", "office", "office", "office", "office", "office", "office", "office", "official", "official", "official", "official", "official", "official", "official", "official", "official", "official", "oh", "oh", "oh", "oh", "oh", "oh", "oh", "oh", "oh", "oh", "old", "old", "old", "old", "old", "old", "old", "old", "old", "old", "online", "online", "online", "online", "online", "online", "online", "online", "online", "online", "open", "open", "open", "open", "open", "open", "open", "open", "open", "open", "order", "order", "order", "order", "order", "order", "order", "order", "order", "order", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "outbreak", "page", "page", "page", "page", "page", "page", "page", "page", "page", "page", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pakistan", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "pandemic", "panic", "panic", "panic", "panic", "panic", "panic", "panic", "panic", "panic", "panic", "paper", "paper", "paper", "paper", "paper", "paper", "paper", "paper", "paper", "paper", "past", "past", "past", "past", "past", "past", "past", "past", "past", "past", "patient", "patient", "patient", "patient", "patient", "patient", "patient", "patient", "patient", "patient", "pay", "pay", "pay", "pay", "pay", "pay", "pay", "pay", "pay", "pay", "people", "people", "people", "people", "people", "people", "people", "people", "people", "people", "place", "place", "place", "place", "place", "place", "place", "place", "place", "place", "plan", "plan", "plan", "plan", "plan", "plan", "plan", "plan", "plan", "plan", "play", "play", "play", "play", "play", "play", "play", "play", "play", "play", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "political", "political", "political", "political", "political", "political", "political", "political", "political", "political", "poor", "poor", "poor", "poor", "poor", "poor", "poor", "poor", "poor", "poor", "population", "population", "population", "population", "population", "population", "population", "population", "population", "population", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive", "post", "post", "post", "post", "post", "post", "post", "post", "post", "post", "potus", "potus", "potus", "potus", "potus", "potus", "potus", "potus", "potus", "potus", "ppl", "ppl", "ppl", "ppl", "ppl", "ppl", "ppl", "ppl", "ppl", "ppl", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "president", "president", "president", "president", "president", "president", "president", "president", "president", "president", "press", "press", "press", "press", "press", "press", "press", "press", "press", "press", "prevent", "prevent", "prevent", "prevent", "prevent", "prevent", "prevent", "prevent", "prevent", "prevent", "price", "price", "price", "price", "price", "price", "price", "price", "price", "price", "priority", "priority", "priority", "priority", "priority", "priority", "priority", "priority", "priority", "probably", "probably", "probably", "probably", "probably", "probably", "probably", "probably", "probably", "probably", "protect", "protect", "protect", "protect", "protect", "protect", "protect", "protect", "protect", "protect", "protection", "protection", "protection", "protection", "protection", "protection", "protection", "protection", "protection", "protection", "provide", "provide", "provide", "provide", "provide", "provide", "provide", "provide", "provide", "provide", "public", "public", "public", "public", "public", "public", "public", "public", "public", "public", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "quarantine", "question", "question", "question", "question", "question", "question", "question", "question", "question", "question", "rate", "rate", "rate", "rate", "rate", "rate", "rate", "rate", "rate", "rate", "read", "read", "read", "read", "read", "read", "read", "read", "read", "read", "real", "real", "real", "real", "real", "real", "real", "real", "real", "real", "remain", "remain", "remain", "remain", "remain", "remain", "remain", "remain", "remain", "remain", "remember", "remember", "remember", "remember", "remember", "remember", "remember", "remember", "remember", "remember", "remote", "remote", "remote", "remote", "remote", "remote", "remote", "remote", "remote", "report", "report", "report", "report", "report", "report", "report", "report", "report", "report", "research", "research", "research", "research", "research", "research", "research", "research", "research", "research", "respond", "respond", "respond", "respond", "respond", "respond", "respond", "respond", "respond", "respond", "response", "response", "response", "response", "response", "response", "response", "response", "response", "response", "responsible", "responsible", "responsible", "responsible", "responsible", "responsible", "responsible", "responsible", "responsible", "responsible", "result", "result", "result", "result", "result", "result", "result", "result", "result", "result", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "rise", "rise", "rise", "rise", "rise", "rise", "rise", "rise", "rise", "rise", "risk", "risk", "risk", "risk", "risk", "risk", "risk", "risk", "risk", "risk", "roll", "roll", "roll", "roll", "roll", "roll", "roll", "roll", "roll", "room", "room", "room", "room", "room", "room", "room", "room", "room", "room", "safe", "safe", "safe", "safe", "safe", "safe", "safe", "safety", "safety", "safety", "safety", "safety", "safety", "safety", "safety", "safety", "safety", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sanitizer", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "sarscov2", "save", "save", "save", "save", "save", "save", "save", "save", "save", "save", "say", "say", "say", "say", "say", "say", "say", "say", "say", "say", "scare", "scare", "scare", "scare", "scare", "scare", "scare", "scare", "scare", "school", "school", "school", "school", "school", "school", "school", "school", "school", "second", "second", "second", "second", "second", "second", "second", "second", "second", "security", "security", "security", "security", "security", "security", "security", "security", "security", "security", "self", "self", "self", "self", "self", "self", "self", "self", "self", "self", "sense", "sense", "sense", "sense", "sense", "sense", "sense", "sense", "sense", "sense", "service", "service", "service", "service", "service", "service", "service", "service", "service", "service", "share", "share", "share", "share", "share", "share", "share", "share", "share", "share", "shit", "shit", "shit", "shit", "shit", "shit", "shit", "shit", "shit", "shit", "shut", "shut", "shut", "shut", "shut", "shut", "shut", "shut", "shut", "shut", "sign", "sign", "sign", "sign", "sign", "sign", "sign", "sign", "sign", "sign", "simple", "simple", "simple", "simple", "simple", "simple", "simple", "simple", "simple", "simple", "slow", "slow", "slow", "slow", "slow", "slow", "slow", "slow", "slow", "slow", "small", "small", "small", "small", "small", "small", "small", "small", "small", "small", "soap", "soap", "soap", "soap", "soap", "soap", "soap", "soap", "social", "social", "social", "social", "social", "social", "social", "social", "social", "social", "soon", "soon", "soon", "soon", "soon", "soon", "soon", "soon", "soon", "soon", "sorry", "sorry", "sorry", "sorry", "sorry", "sorry", "sorry", "sorry", "sorry", "sorry", "south", "south", "south", "south", "south", "south", "south", "south", "south", "south", "spain", "spain", "spain", "spain", "spain", "spain", "spain", "spain", "spain", "spain", "spread", "spread", "spread", "spread", "spread", "spread", "spread", "spread", "spread", "spread", "st", "st", "st", "st", "st", "st", "st", "st", "st", "st", "staff", "staff", "staff", "staff", "staff", "staff", "staff", "staff", "staff", "staff", "stand", "stand", "stand", "stand", "stand", "stand", "stand", "stand", "stand", "stand", "start", "start", "start", "start", "start", "start", "start", "start", "start", "start", "state", "state", "state", "state", "state", "state", "state", "state", "state", "state", "stay", "stay", "stay", "stay", "stay", "stay", "stay", "stay", "stay", "stay", "stop", "stop", "stop", "stop", "stop", "stop", "stop", "stop", "stop", "stop", "story", "story", "story", "story", "story", "story", "story", "story", "story", "story", "strong", "strong", "strong", "strong", "strong", "strong", "strong", "strong", "strong", "strong", "stupid", "stupid", "stupid", "stupid", "stupid", "stupid", "stupid", "stupid", "stupid", "stupid", "supply", "supply", "supply", "supply", "supply", "supply", "supply", "supply", "supply", "supply", "support", "support", "support", "support", "support", "support", "support", "support", "support", "support", "talk", "talk", "talk", "talk", "talk", "talk", "talk", "talk", "talk", "talk", "tax", "tax", "tax", "tax", "tax", "tax", "tax", "tax", "tax", "tax", "team", "team", "team", "team", "team", "team", "team", "team", "team", "team", "tech", "tech", "tech", "tech", "tech", "tech", "tech", "tech", "technology", "technology", "technology", "technology", "technology", "technology", "technology", "technology", "technology", "technology", "tell", "tell", "tell", "tell", "tell", "tell", "tell", "tell", "tell", "tell", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "thank", "thank", "thank", "thank", "thank", "thank", "thank", "thank", "thank", "thank", "thing", "thing", "thing", "thing", "thing", "thing", "thing", "thing", "thing", "thing", "think", "think", "think", "think", "think", "think", "think", "think", "think", "think", "thread", "thread", "thread", "thread", "thread", "thread", "thread", "thread", "thread", "thread", "threat", "threat", "threat", "threat", "threat", "threat", "threat", "threat", "threat", "threat", "till", "till", "till", "till", "till", "till", "till", "till", "till", "time", "time", "time", "time", "time", "time", "time", "time", "time", "time", "today", "today", "today", "today", "today", "today", "today", "today", "today", "today", "toilet", "toilet", "toilet", "toilet", "toilet", "toilet", "toilet", "toilet", "toilet", "toilet", "toll", "toll", "toll", "toll", "toll", "toll", "toll", "toll", "toll", "toll", "total", "total", "total", "total", "total", "total", "total", "total", "total", "total", "touch", "touch", "touch", "touch", "touch", "touch", "touch", "touch", "touch", "touch", "town", "town", "town", "town", "town", "town", "town", "town", "town", "town", "travel", "travel", "travel", "travel", "travel", "travel", "travel", "travel", "travel", "travel", "trump", "trump", "trump", "trump", "trump", "trump", "trump", "trump", "trump", "trump", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpdemic", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "trumpvirus", "try", "try", "try", "try", "try", "try", "try", "try", "try", "try", "tuesday", "tuesday", "tuesday", "tuesday", "tuesday", "tuesday", "tuesday", "tuesday", "tuesday", "uk", "uk", "uk", "uk", "uk", "uk", "uk", "uk", "uk", "uk", "university", "university", "university", "university", "university", "university", "university", "university", "university", "university", "update", "update", "update", "update", "update", "update", "update", "update", "update", "update", "usa", "usa", "usa", "usa", "usa", "usa", "usa", "usa", "usa", "usa", "use", "use", "use", "use", "use", "use", "use", "use", "use", "use", "usual", "usual", "usual", "usual", "usual", "usual", "usual", "usual", "usual", "usual", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "vaccine", "video", "video", "video", "video", "video", "video", "video", "video", "video", "video", "viral", "viral", "viral", "viral", "viral", "viral", "viral", "viral", "viral", "viral", "virus", "virus", "virus", "virus", "virus", "virus", "virus", "virus", "virus", "virus", "visit", "visit", "visit", "visit", "visit", "visit", "visit", "visit", "visit", "visit", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "want", "want", "want", "want", "want", "want", "want", "want", "want", "want", "war", "war", "war", "war", "war", "war", "war", "war", "war", "war", "wash", "wash", "wash", "wash", "wash", "wash", "wash", "wash", "wash", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "washyourhands", "watch", "watch", "watch", "watch", "watch", "watch", "watch", "watch", "watch", "watch", "water", "water", "water", "water", "water", "water", "water", "water", "water", "water", "way", "way", "way", "way", "way", "way", "way", "way", "way", "way", "wear", "wear", "wear", "wear", "wear", "wear", "wear", "wear", "website", "website", "website", "website", "website", "website", "website", "website", "website", "website", "week", "week", "week", "week", "week", "week", "week", "week", "week", "week", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "work", "work", "work", "work", "work", "work", "work", "work", "work", "work", "world", "world", "world", "world", "world", "world", "world", "world", "world", "world", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worldwide", "worth", "worth", "worth", "worth", "worth", "worth", "worth", "worth", "worth", "worth", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhan", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "wuhanvirus", "year", "year", "year", "year", "year", "year", "year", "year", "year", "year", "york", "york", "york", "york", "york", "york", "york", "york", "york"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 9, 1, 6, 5, 8, 7, 3, 10, 4]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el878030360836667602109931664", ldavis_el878030360836667602109931664_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el878030360836667602109931664", ldavis_el878030360836667602109931664_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el878030360836667602109931664", ldavis_el878030360836667602109931664_data);
            })
         });
}
</script>



In the above analysis, we extracted the models for each stage. In addition, we visualized the dynamic model. We can see the relationship among the different topics and how important of a specific (top) word in a topic.
