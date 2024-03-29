import numpy as np
import pandas as pd
import os
import json
import re
import nltk
import gzip
import tqdm
import twarc
import pickle
import argparse
import utils
import datetime

nltk.download("stopwords")
from nltk.corpus import stopwords

# Degree titles that can be found in the user name or bio
all_degree_titles = re.compile("|".join([
    r"\bM\.?D\.?\b",
    r"\bM\.?B\.?B\.?S\.?\b",
    r"\bB\.?M\.?B\.?S\.?\b",
    r"\bD\.?C\.?M\.?\b",
    r"\bM\.?P\.?H\.?\b"]))

# Degree titles that can be found in the screen name
screen_name_titles = re.compile("|".join([r"MD$", r"MBBS$", r"BMBS$", r"DCM$", r"MPH$", r"^Dr[^a-z]"]), flags=re.I)

# Keywords to search in the user bio
keywords = re.compile("|".join([
    r"physicians?\b", r"doctors?\b",
    r"public\s+health\b", r"nurse\b"
    r"cardiolog", r"pulmonolog",
    r"gastrointerolog", r"pediatric",
    r"\Winternist", r"surgeon", r"surgery\b",
    r"epidemiolog", r"specialists?\b",
    r"virolog", r"radiolog", r"\Woncolog"]), flags=re.I)

TWEET_DATE_FORMAT = '%Y-%m-%d'

def is_doctor_tweet(tweet):
    """
    Determines if the given tweet is authored by a likely physician.

    tweet: A dictionary in the standard tweet CSV format, containing the following keys:
        - full_text
        - name
        - screen_name
        - bio
        - created_at
        - user_id
        - id
        - place
        - geo
        - is_quote
        - reply_to_id
        - reply_to_user

    Returns: True if the tweet is authored by a likely physician, False otherwise
    """
    if all_degree_titles.search('\n'.join([tweet["name"], tweet["bio"]])):
        return True
    elif screen_name_titles.search(tweet["screen_name"]):
        return True
    elif keywords.search(tweet["bio"]):
        return True
    return False

# Read the tweet IDs that need to be hydrated
def read_tweet_ids(filename, num_workers, worker_index, skip_batches, batch_size=100000, date_lower=None, date_upper=None):
    """
    Reads tweet IDs from the given gzipped TSV file.

    filename: The file path from which to read.

    Yields: tuples (batch_index, batch), where batch_index is an integer and
        batch is a list of tweet ID strings.
    """
    current_batch = []
    batch_index = 0
    known_accept_dates = set()
    known_reject_dates = set()
    if date_lower is not None: date_lower = datetime.datetime.strptime(date_lower, TWEET_DATE_FORMAT)
    if date_upper is not None: date_upper = datetime.datetime.strptime(date_upper, TWEET_DATE_FORMAT)
    with gzip.open(filename, 'rt') as file:
        for i, line in enumerate(file):
            if i % 1000000 == 0:
                print("Line {} of file".format(i))
            comps = line.strip().split("\t")
            try:
                id_str = str(int(comps[0]))
            except:
                continue
            else:
                if i % num_workers != worker_index:
                    continue
                
                # Filter date if applicable
                if date_lower is not None or date_upper is not None and len(comps) >= 2:
                    if comps[1] in known_reject_dates:
                        continue
                    elif comps[1] not in known_accept_dates:
                        try:
                            date = datetime.datetime.strptime(comps[1], TWEET_DATE_FORMAT)
                        except:
                            pass
                        else:
                            reject = ((date_lower is not None and date < date_lower) or
                                      (date_upper is not None and date > date_upper))
                            if reject:
                                known_reject_dates.add(comps[1])
                                continue
                            else:
                                known_accept_dates.add(comps[1])
                
                current_batch.append(id_str)
                if len(current_batch) == batch_size:
                    if batch_index >= skip_batches:
                        yield (batch_index, current_batch)
                    batch_index += 1
                    current_batch = []
    if current_batch:
        if batch_index >= skip_batches:
            yield (batch_index, current_batch)
        batch_index += 1
        current_batch = []

def hydrate_worker(credentials_path, tweet_ids_filename, output_directory, num_workers, worker_index, skip_batches, batch_size=100000, date_lower=None, date_upper=None):
    """
    Performs hydration of incremented batches of tweet IDs and filters them for likely doctor
    status.

    credentials_path: Absolute path to a credentials file, in the format written by the twarc
        module.
    tweet_ids_filename: Path to a .tsv.gz file containing tweet IDs in the first column.
    output_directory: Path to output directory.
    num_workers: Number of parallel workers being used.
    worker_index: Index of this worker.
    skip_batches: Number of batches to skip from the beginning (in case some batches have already
        been completed in a prior run).
    batch_size: Number of tweets to process in each batch.
    """

    # Load Twarc object
    t = utils.load_twarc(credentials_path)

    for batch_index, batch in read_tweet_ids(tweet_ids_filename, num_workers, worker_index,
                                             skip_batches, batch_size=batch_size,
                                             date_lower=date_lower, date_upper=date_upper):
        print("Worker {}, batch index {}".format(worker_index, batch_index))
        path_suffix = "_worker{}_batch{}".format(worker_index, batch_index)
        json_file = open(os.path.join(output_directory, "doctor_tweets" + path_suffix + ".json"), "w")
        csv_data = []
        non_doctor_ngrams = [{}, {}, {}] # unigrams, bigrams, trigrams
        doctor_ngrams = [{}, {}, {}]

        tweets_analyzed = 0
        english_text_tweets = 0
        for i, json_tweet in tqdm.tqdm(enumerate(t.hydrate(batch)), mininterval=1, total=len(batch)):
            tweets_analyzed += 1

            # Ignore non-English tweets and tweets without text
            if json_tweet["lang"] != "en" or not json_tweet["full_text"]:
                continue
            english_text_tweets += 1

            tweet = utils.json_to_tweet(json_tweet)
            ngram_set = non_doctor_ngrams
            if is_doctor_tweet(tweet):
                json_file.write("{}\n".format(json.dumps(json_tweet)))
                csv_data.append(tweet)
                ngram_set = doctor_ngrams
            
            # n-gram collection - first preprocess tweet text
            tokens = [t for t in re.split(r"\W", utils.preprocess_for_metamap(tweet["full_text"]).lower()) if t]
            for n, ngram_counter in enumerate(ngram_set):
                utils.collect_ngrams(ngram_counter, tokens, n=n + 1)

        print("Worker {}, loaded {} doctor tweets ({} tweets analyzed, {} with English text)".format(
            worker_index, len(csv_data), tweets_analyzed, english_text_tweets))
        df = pd.DataFrame(csv_data)
        utils.write_tweet_csv(df, os.path.join(output_directory, "doctor_tweets" + path_suffix + ".csv"))
        json_file.close()

        # Write n-grams
        with open(os.path.join(output_directory, "word_counts" + path_suffix + ".pkl"), "wb") as file:
            pickle.dump({
                "total_tweets_count": tweets_analyzed,
                "english_text_tweets": english_text_tweets,
                "doctor_tweets_count": len(csv_data),
                "non_doctor_ngrams": non_doctor_ngrams,
                "doctor_ngrams": doctor_ngrams
            }, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Hydrate tweets and save if they are from a '
                                     'likely physician.'))
    parser.add_argument('creds', type=str,
                        help='Absolute path to twarc credentials (like /Users/USER/.twarc)')
    parser.add_argument('ids', type=str,
                        help='Path to a .tsv.gz file containing tweet IDs in the first column')
    parser.add_argument('out', type=str,
                        help='Path to the output directory')
    parser.add_argument('--num_workers', type=int, help='Number of parallel workers', default=1,
                        dest='num_workers')
    parser.add_argument('--worker', type=int, help='Index of this worker (0 to num_workers)',
                        default=0, dest='worker')
    parser.add_argument('--batch_size', type=int, help='Number of tweet IDs to process per batch',
                        default=100000, dest='batch_size')
    parser.add_argument('--skip_batches', type=int, help='Number of batches to skip upfront',
                        default=0, dest='skip_batches')
    parser.add_argument('-ld', '--lower-date', type=str, default=None,
                        dest='date_lower', help='Earliest date to allow in filtered set (YYYY-MM-DD)')
    parser.add_argument('-ud', '--upper-date', type=str, default=None,
                        dest='date_upper', help='Latest date to allow in filtered set (YYYY-MM-DD, inclusive)')

    args = parser.parse_args()
    hydrate_worker(args.creds, args.ids, args.out, args.num_workers,
                   args.worker, args.skip_batches, batch_size=args.batch_size,
                   date_lower=args.date_lower, date_upper=args.date_upper)
