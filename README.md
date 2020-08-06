# covid-tweets

Analysis of COVID tweets by physicians using topic modeling and concept extraction.

## Requirements

The pipeline runs in Python 3.7. The following modules are required:

* [`twarc`](https://github.com/DocNow/twarc), for tweet hydration and retrieval
* [`gensim`](https://radimrehurek.com/gensim/), for NLP preprocessing and LDA wrapper
* [**MALLET**](https://radimrehurek.com/gensim/models/wrappers/ldamallet.html), for MALLET LDA which must be installed separately from `gensim`.
* [**MetaMap**](https://metamap.nlm.nih.gov) Java toolkit for concept annotation.
* [`PyMetaMap`](https://github.com/AnthonyMRios/pymetamap), a wrapper around MetaMap that can be accessed in Python.

## Pipeline Summary

### 1. Extract Medical Tweets

The **`extract_medical_tweets.py`** script takes a gzipped TSV file containing tweet IDs, such as [this one hosted by Zenodo](https://zenodo.org/record/3884334#.XuOdQC3Mw_U). It hydrates the tweets using Twarc, filters them using regular expressions to find tweets likely authored by physicians and medical professionals, and saves only the relevant tweets to JSON and CSV. This script may take hours to days to run, but can be parallelized using the `--num_workers` and `--worker` command-line arguments.

### 2. Augment Tweets with Threads

The **`retrieve_tweet_threads.ipynb`** notebook loads the completed set of tweets from the batched output files of the previous step. It then retrieves both upstream and downstream tweets using the `in_reply_to_status_id_str` field of each tweet, looking for tweets that are in reply to a tweet by the same user. It writes out a CSV file called `thread_annotated_tweets.csv`, which contains a `thread_id` field that groups together tweets in the same thread.

### 3. Combine n-gram counts

This combines the n-gram counts from the previous batches of counts into two files: `relevant_word_counts.pkl` (n-gram counts for doctor tweets) and `irrelevant_word_counts.pkl` (n-gram counts for non-doctor tweets).

### 4. Annotate Concepts

The **`annotate_concepts.py`** script uses `PyMetaMap` to extract concepts for the tweets in `thread_annotated_tweets.csv`. Like Step 1, this step takes hours to days, can be parallelized using the `--start` and `--end` command-line arguments.

### 5. Clean Up Concepts

The **`clean_up_concepts.ipynb`** notebook handles concatenating the concepts from above, and filtering them for relevance as measured by their enrichment in the doctor tweet set compared to the non-doctor tweet set.

-----------------------

At this stage, one should organize the files into a pipeline directory that will be used for the input and output of the subsequent steps. The pipeline directory should be initialized with a `level_0` subdirectory, which contains the following files (or symlinks to them):

* `tweets.csv` - a CSV file containing tweets. This can be the `thread_annotated_tweets.csv` file generated in step 2, or it can be the output of `merge_threads.py` (which merges tweets from the same thread into single rows).
* `relevant_word_counts.pkl/irrelevant_word_counts.pkl` - the outputs of step 3 above.
* `concepts.csv` - a CSV file containing concepts. If threads are not merged, use the output of step 5; otherwise, pass the `--concepts` argument to `merge_threads.py` and it will output a merged concepts file to use instead.

The following steps can be run succinctly using the `execute_level.sh` script, as in:

```
./execute_level.sh [LEVEL: 0, 1, 2, etc.] [PIPELINE DIRECTORY] [PATH TO MALLET]
```

### 6. Build Topic Model

The **`build_topic_model.py`** script takes a set of tweets and builds a MALLET LDA topic model. This can be run with different values for the `--topics` parameter to see the effect of different sizes of model.

### 7. Compute Relevance

The **`compute_relevance.py`** script scores the relevance of each topic by reading the word counts for the current level and computing the enrichment of each concept (between the current tweet set and the tweets discarded in the previous level). The *relevance* of a topic is the sum of the total concept relevances in all tweets, weighted by the probability of each tweet being associated with that topic. This is output in a `relevance.json` file along with the most highly-associated tweets for each topic. The script also outputs a `concepts_with_relevance.csv` file, which gives the concept enrichment scores used to compute topic relevance.

### 8. Filter Topics

The **`filter_topics.py`** script uses the relevance scores computed in step 7 to identify the most relevant topics, then populates the next level directory with the necessary files: `tweets.csv` (containing a filtered set of tweets), `(ir)relevant_word_counts.pkl` (containing word counts for the filtered set versus the removed tweets), and `concepts.csv` (containing the concepts used in the filtered set of tweets).

Steps 6-8 can be repeated using `execute_level.sh` with increasing level numbers until the tweet set is sufficiently filtered.