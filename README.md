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

### 3. Annotate Concepts

The **`annotate_concepts.py`** script uses `PyMetaMap` to extract concepts for the tweets in `thread_annotated_tweets.csv`. Like Step 1, this step takes hours to days, can be parallelized using the `--start` and `--end` command-line arguments.

### 4. Clean Up Concepts

The **`clean_up_concepts.ipynb`** notebook handles concatenating the concepts from above, and filtering them for relevance as measured by their inverse frequency in the English language.
