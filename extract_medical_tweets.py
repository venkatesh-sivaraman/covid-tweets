import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
import re
import nltk
import shutil
import pickle
import time
import gzip
import datetime
import twarc
import requests
import multiprocessing as mp
import argparse

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

def json_to_tweet(tweet):
    """
    Converts a tweet JSON from the Twarc API to the standard list of fields
    for this analysis.

    json_tweet: A dictionary in the Twarc API format.

    Returns: A dictionary in the standard format for writing to CSV for this analysis.
    """
    return {
        "full_text": tweet["full_text"],
        "name": tweet["user"]["name"],
        "screen_name": tweet["user"]["screen_name"],
        "bio": tweet["user"]["description"],
        "created_at": tweet["created_at"],
        "user_id": tweet["user"]["id_str"],
        "id": tweet["id"],
        "place": tweet["place"],
        "geo": tweet["geo"],
        "is_quote": tweet["is_quote_status"],
        "reply_to_id": tweet["in_reply_to_status_id_str"],
        "reply_to_user": tweet["in_reply_to_user_id_str"]
    }

# Read the tweet IDs that need to be hydrated
def read_tweet_ids(filename, num_workers, worker_index, skip_batches, batch_size=100000):
    """
    Reads tweet IDs from the given gzipped TSV file.

    filename: The file path from which to read.

    Yields: tuples (batch_index, batch), where batch_index is an integer and
        batch is a list of tweet ID strings.
    """
    current_batch = []
    batch_index = 0
    with gzip.open(filename, 'rt') as file:
        for i, line in enumerate(file):
            comps = line.strip().split("\t")
            try:
                id_str = str(int(comps[0]))
            except:
                continue
            else:
                if i % num_workers != worker_index:
                    continue
                current_batch.append(id_str)
                if len(current_batch) == batch_size:
                    if batch_index >= skip_batches:
                        yield (batch_index, current_batch)
                    batch_index += 1
                    current_batch = []

def hydrate_worker(credentials_path, tweet_ids_filename, output_directory, num_workers, worker_index, skip_batches, batch_size=100000):
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

    # Get Twarc credentials from .twarc
    with open(credentials_path, "r") as file:
        lines = file.readlines()
        keys = [line.strip().split(" = ")[1] for line in lines[1:] if line.strip()]
    t = twarc.Twarc(*keys, app_auth=True)

    for batch_index, batch in read_tweet_ids(tweet_ids_filename, num_workers, worker_index,
                                             skip_batches, batch_size=batch_size):
        print("Worker {}, batch index {}".format(worker_index, batch_index))
        json_file = open(os.path.join(output_directory, "doctor_tweets_worker{}_batch{}.json".format(worker_index, batch_index)), "w")
        csv_data = []
        for i, json_tweet in enumerate(t.hydrate(batch)):
            if i % 5000 == 0:
                print("Worker {}, tweet {}".format(worker_index, i))

            # Ignore non-English tweets and tweets without text
            if json_tweet["lang"] != "en" or not json_tweet["full_text"]:
                continue
            tweet = json_to_tweet(json_tweet)
            if is_doctor_tweet(tweet):
                json_file.write("{}\n".format(json.dumps(json_tweet)))
                csv_data.append(tweet)
        print("Worker {}, loaded {} doctor tweets".format(worker_index, len(csv_data)))
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(output_directory, "doctor_tweets_worker{}_batch{}.csv".format(worker_index, batch_index)),
                  line_terminator="\n")
        json_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Hydrate tweets and save if they are from a '
                                     'likely physician.'))
    parser.add_argument('creds', type=str,
                        help='Absolute path to twarc credentials (like /Users/USER/.twarc)')
    parser.add_argument('ids', type=str,
                        help='Path to a .tsv.gz file containing tweet IDs in the first column')
    parser.add_argument('out', type=str,
                        help='Path to the output directory')
    parser.add_argument('--num_workers', type=int, help='Number of parallel workers', default=3,
                        dest='num_workers')
    parser.add_argument('--worker', type=int, help='Index of this worker (0 to num_workers)',
                        default=0, dest='worker')
    parser.add_argument('--batch_size', type=int, help='Number of tweet IDs to process per batch',
                        default=100000, dest='batch_size')
    parser.add_argument('--skip_batches', type=int, help='Number of batches to skip upfront',
                        default=0, dest='skip_batches')

    args = parser.parse_args()
    hydrate_worker(args.creds, args.ids, args.out, args.num_workers, args.worker, args.skip_batches, batch_size=args.batch_size)
