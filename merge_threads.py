"""
An optional step to merge together tweets with the same thread ID. The tweets
in the same thread are saved as one row with each tweet separated by two
newlines. Tweets are ordered chronologically. The ID of the tweet is marked as
the thread ID.
"""

import argparse
import pandas as pd
import os
import datetime
import utils

def earliest_date(series):
    """Converts the strings in the series to dates and returns the earliest."""
    dates = [datetime.datetime.strptime(val,'%a %b %d %H:%M:%S +0000 %Y') for val in series]
    return datetime.datetime.strftime(min(dates), '%a %b %d %H:%M:%S +0000 %Y')

def join_text(series):
    """Joins the strings in the series together using double newlines."""
    return "\n\n".join(series)

def arbitrary(series):
    """Returns an arbitrary value from the series (non-null preferred)."""
    if len(series) == 0:
        return None
    if not all(v == series.iloc[0] for v in series):
        return next((v for v in series if v), None)
    return series.iloc[0]

def merge_threads(tweets_df):
    """
    Merges tweets with the same thread ID together, separating them with two
    newlines each.
    """
    # Sort by id
    sorted_df = tweets_df.sort_values(by="id")

    # Groupby thread ID
    grouped = sorted_df.groupby("thread_id").agg({
        "full_text": join_text,
        "created_at": earliest_date,
        "name": arbitrary,
        "screen_name": arbitrary,
        "bio": arbitrary,
        "user_id": arbitrary
    })

    # Use thread_id as the new tweet ID
    grouped = grouped.reset_index().rename(columns={"thread_id": "id"})
    
    return grouped

def merge_concepts(concepts_df, tweets_df):
    """
    Replaces the contents of the tweet_id column in the given concepts dataframe
    with the thread ID from the tweets dataframe.
    """
    return (pd.merge(concepts_df,
                     tweets_df,
                     how='inner',
                     right_on='id',
                     left_on='tweet_id')[['thread_id',
                                         'preferred_name',
                                         'trigger',
                                         'trigger_word',
                                         'cui',
                                         'score',
                                         'semtypes']]
            .rename(columns={"thread_id": "tweet_id"})
            .reset_index(drop=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Merge tweets with the same thread ID.'))
    parser.add_argument('tweets', type=str,
                        help='Path to a CSV file containing tweets')
    parser.add_argument('out', type=str,
                        help='Path to a directory in which to write')
    parser.add_argument('--concepts', type=str, default=None,
                        help='Concepts CSV path to also merge (writes in same directory)',
                        dest='concepts')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    print("Reading tweets...")
    tweets_df = utils.read_tweet_csv(args.tweets)
    print("Merging {} tweets...".format(len(tweets_df)))
    grouped = merge_threads(tweets_df)
    utils.write_tweet_csv(grouped, os.path.join(args.out, "merged_threads.csv"))

    # Write a simple thread mapping
    tweets_df[['id', 'thread_id']].set_index('id').to_csv(os.path.join(args.out, "thread_mapping.csv"))

    print("Done merging {} threads.".format(len(grouped)))
    if args.concepts:
        print("Merging concepts...")
        concepts_df = merge_concepts(pd.read_csv(args.concepts, dtype={"tweet_id": str}), tweets_df)
        concepts_df.to_csv(os.path.join(args.out, "merged_concepts.csv"))
        print("Done writing concepts.")

