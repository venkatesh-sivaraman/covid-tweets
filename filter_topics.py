"""
Takes a topic model and a set of clinical concepts, and determines the most
relevant topics using the average relevance of concepts in each topic.
"""

import numpy as np
import pandas as pd
import os
import json
import re
import argparse
import utils
import datetime
import pickle
import pprint
import tqdm
import json

def compute_relevance(input_dir, verbose=False):
    """
    Computes a relevance dictionary given the relevant_word_counts.pkl and
    irrelevant_word_counts.pkl files in the input directory.
    """
    with open(os.path.join(input_dir, "relevant_word_counts.pkl"), "rb") as file:
        doctor_info = pickle.load(file)

    with open(os.path.join(input_dir, "irrelevant_word_counts.pkl"), "rb") as file:
        non_doctor_info = pickle.load(file)

    doctor_tweet_count = doctor_info["tweet_count"]
    doctor_word_counts = doctor_info["word_counts"][0]
    non_doctor_tweet_count = non_doctor_info["tweet_count"]
    non_doctor_word_counts = non_doctor_info["word_counts"][0]

    relevance = {}
    for word, f in doctor_word_counts.items():
        if word.lower() in utils.FILTER_WORDS:
            continue
        non_doctor_f = non_doctor_word_counts.get(word, 0)
        relevance[word] = (f / doctor_tweet_count + 1e-3) / (non_doctor_f / non_doctor_tweet_count + 1e-3)

    if verbose:
        sorted_ratios = sorted(relevance.items(), key=lambda x: x[1])
        print("Most enriched terms:")
        for word, ratio in reversed(sorted_ratios[-20:]):
            print(word, ratio)
        print("\nLeast enriched:")
        for word, ratio in sorted_ratios[:20]:
            print(word, ratio)

    return relevance

def load_topics(topics_file, tweets_file, head=0):
    """
    Loads a topic probability dataframe and associates it with the tweet info.
    Returns both the merged topic dataframe and the tweets dataframe."""
    topics_df = pd.read_csv(topics_file, index_col="id", lineterminator='\n', dtype=utils.dtype_spec)
    tweets_df = utils.read_tweet_csv(tweets_file, index_by_id=True)
    if head:
        tweets_df = tweets_df.head(head)
    assert len(topics_df) == len(tweets_df)

    topics_df = pd.merge(topics_df, tweets_df, left_index=True, right_index=True)
    return topics_df, tweets_df

def load_concepts(concepts_file, thread_mapping=None, verbose=False):
    """
    Loads a dataframe of concepts.
    
    Args:
        concepts_file: Path to the concepts file.
        thread_mapping: If provided, a dataframe indexed by tweet ID that
            contains a thread_id column.
    """
    concepts_df = pd.read_csv(concepts_file)
    concepts_df = concepts_df[~pd.isna(concepts_df.trigger_word)]

    if thread_mapping is not None:
        if verbose: print("Associating concepts with thread IDs...")
        concepts_df = (pd.merge(concepts_df, thread_mapping,
                                how='inner', 
                                right_index=True,
                                left_on='tweet_id')[['thread_id',
                                                     'preferred_name',
                                                     'cui',
                                                     'trigger_word',
                                                     'relevance',
                                                     'semtypes']]
                      .rename(columns={"thread_id": "tweet_id"})
                      .reset_index(drop=True))

    return concepts_df
        
# Categories of concept that are deemed "useful" for clinical relevance
USEFUL_SEMTYPES = {
    "orch",    "phsu",    "dsyn",
    "patf",    "virs",    "neop",
    "diap",    "medd",    "fndg",
    "celc",    "blor",    "bpoc",
    "prog",    "bmod",    "topp",
    "mbrt",    "sosy",    "acty",
    "dora",
}

def filter_useful_concepts(df):
    """
    Returns a new dataframe with only concepts that are deemed 'useful' (i.e.
    not filter words, and in a useful UMLS semantic type category).
    """
    def is_useful_concept(row):
        if row.preferred_name.lower() in utils.FILTER_WORDS:
            return False

        categories = set(row.semtypes.replace("[", "").replace("]", "").split(","))
        if not categories & USEFUL_SEMTYPES:
            return False
        return True

    return df[df.apply(is_useful_concept, axis=1)]

def filter_topics(topics_df, tweets_df, concepts_df, relevance, start_num_topics=100, end_num_topics=20, thread_mapping=None, verbose=False):
    """
    Filters the tweets in the given topics dataframe by the relevance of the
    concepts that each tweet contains.
    """

    # First compute enrichment ratio for all concepts
    def concept_enrichment(concept_row):
        trigger = " ".join(re.split(r"\W", concept_row.trigger_word))
        return (relevance.get(trigger, 0))
    concepts_df["enrichment"] = concepts_df.apply(concept_enrichment, axis=1)

    if verbose:
        print("Most relevant concepts:")
        print(concepts_df.drop_duplicates("trigger_word").sort_values(by="enrichment", ascending=False).head(10)[["cui", "preferred_name", "trigger_word", "enrichment"]])

    # Don't double count the same concept
    old_count = len(concepts_df)
    concepts_df = concepts_df.drop_duplicates(["tweet_id", "preferred_name"])
    if verbose: print("Dropped {} duplicate concepts".format(old_count - len(concepts_df)))

    concept_relevances = filter_useful_concepts(concepts_df).groupby('tweet_id').agg({'enrichment': 'sum'})
    tweets_with_concept_counts = pd.merge(topics_df, concept_relevances, left_index=True, right_index=True, how='left')
    if thread_mapping is not None:
        # Count number of tweets in each thread
        tweets_with_concept_counts = pd.merge(tweets_with_concept_counts, thread_mapping.thread_id.value_counts().rename("num_tweets"), left_index=True, right_index=True, how='left')
    tweets_with_concept_counts.enrichment = tweets_with_concept_counts.enrichment.fillna(0)

    # Compute topic importances
    if verbose: print("Computing topic importances...")
    topic_relevances = np.zeros(start_num_topics)
    topic_counts = np.zeros(start_num_topics)
    topic_counter = tqdm.tqdm(range(len(topic_counts))) if verbose else range(len(topic_counts))
    for i in topic_counter:
        tweets = tweets_with_concept_counts[tweets_with_concept_counts.top_topic == i]
        topic_relevances[i] += tweets.enrichment.sum()
        if thread_mapping is not None:
            topic_counts[i] += tweets.num_tweets.sum()            
        else:
            topic_counts[i] += len(tweets)

    topic_importances = topic_relevances / topic_counts
    relevant_topics = set(topic_importances.argsort()[-end_num_topics:].tolist())
    if verbose: print("Most relevant topics:", relevant_topics)

    # Generate the output dataframe
    relevant_tweets_df = tweets_df.loc[topics_df[topics_df.top_topic.isin(relevant_topics)].index]
    if verbose: print("{} relevant tweets".format(len(relevant_tweets_df)))

    return relevant_tweets_df

def compute_relevant_ngram_counts(tweets_df, relevant_tweets_df, verbose=False, min_count=0):
    """
    Computes the ngram counts for relevant and irrelevant tweets, and returns them
    as two dictionaries containing the keys "tweet_count" and "word_counts".
    """
    irrelevant_tweets_df = tweets_df.loc[tweets_df.index.difference(relevant_tweets_df.index)]
    return (
        {
            "tweet_count": len(relevant_tweets_df),
            "word_counts": utils.collect_df_ngram_counts(relevant_tweets_df, min_count=min_count, verbose=verbose)
        },
        {
            "tweet_count": len(irrelevant_tweets_df),
            "word_counts": utils.collect_df_ngram_counts(irrelevant_tweets_df, min_count=min_count, verbose=verbose)
        },
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Filter topics by relevance.'))
    parser.add_argument('tweets', type=str,
                        help='Path to a directory containing a tweets.csv file, as well as word count pickles')
    parser.add_argument('model', type=str,
                        help='Path to a topic model directory (containing tweet_topics.csv)')
    parser.add_argument('concepts', type=str,
                        help='Path to a concepts CSV file')
    parser.add_argument('output', type=str,
                        help='Path to an output directory')
    parser.add_argument('--start_topics', type=int, help='Number of topics in initial data', default=100,
                        dest='start_num_topics')
    parser.add_argument('--end_topics', type=int, help='Number of topics to filter to', default=20,
                        dest='end_num_topics')
    parser.add_argument('--head', type=int, help='Number of tweets limited to in the topic model', default=0,
                        dest='head')
    parser.add_argument('--min_count', type=int,
                        help=('Minimum number of occurrences of ngram in '
                        'relevant tweets (higher saves more memory)'),
                        default=0, dest='min_count')
    parser.add_argument('--thread_mapping', type=str, default=None,
                        help=('Original thread_annotated_tweets path if using merged threads'),
                        dest='thread_mapping')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        dest='verbose')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.verbose: print("Computing relevance...")
    relevance = compute_relevance(args.tweets, verbose=args.verbose)
    if args.verbose: print("Loading topics and tweets...")
    topics_df, tweets_df = load_topics(
        os.path.join(args.model, "tweet_topics.csv"),
        os.path.join(args.tweets, "tweets.csv"),
        head=args.head)

    if args.thread_mapping:
        if args.verbose: print("Loading thread mapping...")
        thread_mapping = utils.read_tweet_csv(args.thread_mapping, index_by_id=True)
    else:
        thread_mapping = None

    if args.verbose: print("Loading concepts...")
    concepts_df = filter_useful_concepts(
        load_concepts(args.concepts, thread_mapping=thread_mapping, verbose=args.verbose)
    )

    filtered_df = filter_topics(topics_df,
                                tweets_df,
                                concepts_df,
                                relevance,
                                start_num_topics=args.start_num_topics,
                                end_num_topics=args.end_num_topics,
                                thread_mapping=thread_mapping,
                                verbose=args.verbose)

    if args.verbose: print("Writing tweets...")
    tweets_df_no_index = filtered_df.reset_index()
    tweets_df_no_index.id = tweets_df_no_index.id.astype(str)
    utils.write_tweet_csv(tweets_df_no_index, os.path.join(args.output, "tweets.csv"))

    if args.verbose: print("Computing word counts...")
    relevant_info, irrelevant_info = compute_relevant_ngram_counts(
        tweets_df,
        filtered_df,
        min_count=args.min_count,
        verbose=args.verbose
    )
    with open(os.path.join(args.output, "relevant_word_counts.pkl"), "wb") as file:
        pickle.dump(relevant_info, file)
    with open(os.path.join(args.output, "irrelevant_word_counts.pkl"), "wb") as file:
        pickle.dump(irrelevant_info, file)

    if args.verbose: print("Done.")