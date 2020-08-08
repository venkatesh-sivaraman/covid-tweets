import pandas as pd
import os
import argparse
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re
import tqdm
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
                                     .apply(strip_multiple_whitespaces)
                                     .apply(lambda x: x.lower())
                                     .apply(lambda x: ' '.join(re.split(r"\W+", x))))
    return (tweets_df
            .sort_values("created_at")
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

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        os.mkdir(args.base_dir)
    level_dir = os.path.join(args.base_dir, "level_0")
    if not os.path.exists(level_dir):
        os.mkdir(level_dir)

    tweets_df = utils.read_tweet_csv(args.tweets)
    if args.verbose: print("Deduplicating...")
    old_count = len(tweets_df)
    tweets_df = deduplicate_tweets(tweets_df)
    if args.verbose: print("Removed {} duplicate tweets".format(old_count - len(tweets_df)))

    if args.verbose: print("Preprocessing for LDA...")
    texts = []
    for i in (tqdm.tqdm(range(len(tweets_df))) if args.verbose else range(len(tweets_df))):
        texts.append(' '.join(utils.preprocess_for_lda(tweets_df.iloc[i].full_text)))
    tweets_df["standardized_text"] = texts

    old_count = len(tweets_df)
    tweets_df = tweets_df[tweets_df.standardized_text.str.len() == 0]
    print("Removed {} tweets with no non-filtered words".format(old_count - len(tweets_df)))

    utils.write_tweet_csv(tweets_df, os.path.join(level_dir, "tweets.csv"))

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

    print(("Please copy or symlink the relevant_word_counts.pkl and "
           "irrelevant_word_counts.pkl files into the directory {}.").format(level_dir))