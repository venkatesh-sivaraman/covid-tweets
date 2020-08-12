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
import matplotlib.pyplot as plt
from kneed import KneeLocator
from compute_relevance import load_topics


def find_knee(topic_importances, window_len=9, plot=False):
    """
    Interprets the given list of topic importances as a knee plot, then finds 
    the point of most increasing growth. Returns a tuple (top_topics, bottom_topics),
    where top_topics is the portion that meets the relevance cutoff (in decreasing
    relevance order), and bottom_topics is the portion that does not (in increasing
    relevance order).
    """
    importances = np.pad(np.array(sorted(topic_importances)), window_len // 2, mode='edge')
    window = np.ones(window_len) / window_len
    importances_lp = np.convolve(importances, window, mode='valid')

    kneedle = KneeLocator(np.arange(len(importances_lp)), importances_lp, S=1.0, curve='convex', direction='increasing')
    if plot:
        plt.figure()
        plt.plot(sorted(topic_importances), label="Original")
        plt.plot(importances_lp, label="Smoothed")
        plt.vlines(kneedle.knee, 0, max(topic_importances), label="Knee")
        plt.legend()
        plt.show()

    sorted_topics = np.array(topic_importances).argsort()
    top_topics = list(reversed(sorted_topics[kneedle.knee:]))
    bottom_topics = sorted_topics[:20].tolist()
    return top_topics, bottom_topics

def filter_topics(topics_df, tweets_df, relevance_info, fraction=0.2, start_num_topics=100, end_num_topics=20, verbose=False):
    """
    Filters the tweets in the given topics dataframe by the relevance given in
    the relevance info dictionary. Both topics_df and tweets_df should be indexed
    by the tweet ID.
    """

    topic_importances = np.array([t["relevance"] for t in relevance_info])
    # relevant_topics, irrelevant_topics = find_knee(topic_importances) # , plot=verbose
    # if verbose: print("Most relevant topics:", relevant_topics)

    # # Generate the output dataframe
    # relevant_topic_probs = topics_df[["prob_topic_{}".format(i) for i in relevant_topics]]
    # relevant_sum = relevant_topic_probs.sum(axis=1)
    # irrelevant_topic_probs = topics_df[["prob_topic_{}".format(i) for i in irrelevant_topics]]
    # irrelevant_sum = irrelevant_topic_probs.sum(axis=1)

    # relevance_ratios = relevant_sum / (relevant_sum.sum() / len(relevant_sum))
    # irrelevance_ratios = irrelevant_sum / (irrelevant_sum.sum() / len(irrelevant_sum))
    
    # if verbose:
    #     print("With relevant and against irrelevant:",
    #           len(relevance_ratios[relevance_ratios >= 1].index
    #               .intersection(irrelevance_ratios[irrelevance_ratios < 1].index)))
    #     print("With relevant and with irrelevant:",
    #           len(relevance_ratios[relevance_ratios >= 1].index
    #               .intersection(irrelevance_ratios[irrelevance_ratios >= 1].index)))
    #     print("Against relevant and with irrelevant:",
    #           len(relevance_ratios[relevance_ratios < 1].index
    #               .intersection(irrelevance_ratios[irrelevance_ratios >= 1].index)))
    #     print("Against relevant and against irrelevant:",
    #           len(relevance_ratios[relevance_ratios < 1].index
    #               .intersection(irrelevance_ratios[irrelevance_ratios < 1].index)))

    # relevant_indexes = (relevance_ratios[relevance_ratios >= 1].index
    #                     .intersection(irrelevance_ratios[irrelevance_ratios < 1].index))
    # relevant_tweets_df = tweets_df.loc[relevant_indexes]

    def tweet_predicted_relevance(topic_probs):
        # Weight the topic probs by the relevance of each topic
        return np.sum(topic_importances * topic_probs.values)

    topics_df["predicted_relevance"] = topics_df[["prob_topic_{}".format(i) for i in range(100)]].apply(tweet_predicted_relevance, axis=1)
    relevant_indexes = topics_df.sort_values("predicted_relevance", ascending=False).head(int(len(topics_df) * fraction)).index
    relevant_tweets_df = tweets_df.loc[relevant_indexes]

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
    parser.add_argument('base_dir', type=str,
                        help='Path to a directory containing a tweets.csv file, word count pickles, and topic models')
    parser.add_argument('output', type=str,
                        help='Path to an output directory')
    parser.add_argument('--start_topics', type=int, help='Number of topics in initial data', default=100,
                        dest='start_num_topics')
    parser.add_argument('--end_topics', type=int, help='Number of topics to filter to', default=20,
                        dest='end_num_topics')
    parser.add_argument('--fraction', type=float, help='Take the most relevant fraction of the dataset', default=0.2,
                        dest='fraction')
    parser.add_argument('--head', type=int, help='Number of tweets limited to in the topic model', default=0,
                        dest='head')
    parser.add_argument('--min_count', type=int,
                        help=('Minimum number of occurrences of ngram in '
                        'relevant tweets (higher saves more memory)'),
                        default=2, dest='min_count')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        dest='verbose')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    model_dir = os.path.join(args.base_dir, str(args.start_num_topics))
    with open(os.path.join(model_dir, "relevance.json"), "r") as file:
        relevance_info = json.load(file)

    if args.verbose: print("Loading topics and tweets...")
    topics_df, tweets_df = load_topics(
        os.path.join(model_dir, "tweet_topics.csv"),
        os.path.join(args.base_dir, "tweets.csv"),
        head=args.head,
        return_tweets=True)

    filtered_df = filter_topics(topics_df,
                                tweets_df,
                                relevance_info,
                                fraction=args.fraction,
                                start_num_topics=args.start_num_topics,
                                end_num_topics=args.end_num_topics,
                                verbose=args.verbose)
    del topics_df

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

    if args.verbose: print("Filtering concepts...")
    concepts_df = pd.read_csv(os.path.join(args.base_dir, "concepts.csv"))
    concepts_df = concepts_df[concepts_df.tweet_id.isin(filtered_df.index)]
    concepts_df[['tweet_id',
                 'preferred_name',
                 'trigger',
                 'trigger_word',
                 'cui',
                 'score',
                 'semtypes']].to_csv(os.path.join(args.output, "concepts.csv"))

    if args.verbose: print("Done.")