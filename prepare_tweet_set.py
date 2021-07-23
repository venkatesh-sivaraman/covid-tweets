import pandas as pd
import os
import argparse
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re
import tqdm
import datetime
import pickle
import utils

def deduplicate_tweets(tweets_df):
    """
    Removes tweets designated as duplicates by converting them to lowercased
    lists of words.

    Example:
    'My article is featured on the @ResearchGate #COVID19 community page. Read it here:  https://t.co/6FSvcNSVAt'
    is converted to
    'my article is featured on the researchgate covid19 community page read it here https t co 6fsvcnsvat'
    """
    tweets_df["semistandardized"] = (tweets_df.full_text
                                     .apply(utils.preprocess_tweet_text)
                                     .apply(strip_multiple_whitespaces)
                                     .apply(lambda x: x.lower())
                                     .apply(lambda x: ' '.join(re.split(r"\W+", x))))
    tweets_df["id_num"] = tweets_df["id"].astype(int)
    return (tweets_df
            .sort_values("id_num")
            .drop_duplicates("semistandardized")
            .drop("semistandardized", axis=1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Preprocess tweets and concepts, and set up a pipeline run directory.'))
    parser.add_argument('base_dir', type=str,
                        help='Path to a base directory where the pipeline will be run.')
    parser.add_argument('tweets', type=str,
                        help='Path to CSV file containing tweets (optionally pre-merged)')
    parser.add_argument('concepts', type=str,
                        help='Path to CSV file containing concepts (optionally also pre-merged)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        dest='verbose')
    parser.add_argument('-ld', '--lower-date', type=str, default=None,
                        dest='date_lower', help='Earliest date to allow in filtered set')
    parser.add_argument('-ud', '--upper-date', type=str, default=None,
                        dest='date_upper', help='Latest date to allow in filtered set (inclusive)')
    parser.add_argument('--min_count', type=int,
                        help=('Minimum number of occurrences of ngram in '
                        'relevant tweets (higher saves more memory)'),
                        default=2, dest='min_count')

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        os.mkdir(args.base_dir)
    level_dir = os.path.join(args.base_dir, "level_0")
    if not os.path.exists(level_dir):
        os.mkdir(level_dir)

    tweets_df = utils.read_tweet_csv(args.tweets)

    if args.date_lower or args.date_upper:
        days = pd.to_datetime(tweets_df.created_at, format=utils.CREATED_AT_FORMAT)
        if args.date_lower:
            date_cutoff = datetime.datetime.strptime(args.date_lower, '%Y-%m-%d').date()
            days = days[days.dt.date >= date_cutoff]
            tweets_df = tweets_df.loc[days.index]
        if args.date_upper:
            date_cutoff = datetime.datetime.strptime(args.date_upper, '%Y-%m-%d').date()
            days = days[days.dt.date <= date_cutoff]
            tweets_df = tweets_df.loc[days.index]
        if args.verbose: print("{} tweets match date filter".format(len(tweets_df)))

    if args.verbose: print("Deduplicating...")
    old_count = len(tweets_df)
    tweets_df = deduplicate_tweets(tweets_df)
    if args.verbose: print("Removed {} duplicate tweets".format(old_count - len(tweets_df)))

    if "standardized_text" in tweets_df.columns:
        if args.verbose: print("Tweets CSV already has standardized_text, skipping LDA preprocessing.")
    else:
        if args.verbose: print("Preprocessing for LDA...")
        texts = []
        for i in (tqdm.tqdm(range(len(tweets_df))) if args.verbose else range(len(tweets_df))):
            texts.append(' '.join(utils.preprocess_for_lda(tweets_df.iloc[i].full_text)))
        tweets_df["standardized_text"] = texts

        old_count = len(tweets_df)
        tweets_df = tweets_df[tweets_df.standardized_text.str.len() > 0]
        print("Removed {} tweets with no non-filtered words".format(old_count - len(tweets_df)))

    utils.write_tweet_csv(tweets_df, os.path.join(level_dir, "tweets.csv"))

    # Collect word counts
    if args.verbose: print("Collecting ngram counts...")
    word_counts = utils.collect_df_ngram_counts(tweets_df, min_count=args.min_count, verbose=args.verbose)
    with open(os.path.join(level_dir, "relevant_word_counts.pkl"), "wb") as file:
        pickle.dump({
            "tweet_count": len(tweets_df),
            "word_counts": word_counts
        }, file)

    if args.verbose: print("Filtering concepts...")
    concepts_df = pd.read_csv(args.concepts)
    concepts_df = concepts_df[concepts_df.tweet_id.isin(tweets_df.index)]
    concepts_df[['tweet_id',
                 'preferred_name',
                 'trigger',
                 'trigger_word',
                 'cui',
                 'score',
                 'semtypes']].to_csv(os.path.join(level_dir, "concepts.csv"))

    print("Please copy or symlink the irrelevant_word_counts.pkl files into the directory {} before starting the pipeline.".format(level_dir))