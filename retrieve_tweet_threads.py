#!/usr/bin/env python
# coding: utf-8

# # Tweet Thread Augmentation
# 
# Many tweets are part of **threads**, which consist of multiple tweets in a linked-list sequence of replies to one another. Since some of the tweets may not have contained the original coronavirus keywords, this step pulls tweets in threads for which at least one tweet is in the dataset. This consists of three steps:
# 
# 1. Extract **upstream tweets**, which are explicitly linked to in the `in_reply_to_status_id_str` field of the tweet. We can do this using a simple hydrate command with Twarc.
# 2. Extract **downstream tweets**, which are not explicitly linked. Instead, we look for tweets in each user's timeline within a two-day window on either side of their tweets in the dataset, and recursively find the tweets that link back to the original dataset.
# 3. Join these two sets together with the original tweet dataset, and assign each tweet a **thread ID**.

import json
import time
import argparse
import pandas as pd
import os
import datetime
import utils


def extract_upstream_tweets(t, df, intermediate_dir):
    """
    Returns a dataframe containing all messages that were replied to by a tweet
    in the dataset by the same user. Also writes the tweets to JSON and CSV in
    the intermediates directory.
    """

    reply_ids = df[~pd.isna(df.reply_to_id) & (df.reply_to_user == df.user_id)].reply_to_id.unique().tolist()
    print("{} reply IDs".format(len(reply_ids)))

    # Recursively extract replies
    seen_ids = set()
    reply_ids = list(set(reply_ids))
    hydrated_replies = []
    i = 0

    while reply_ids:
        print("Round {}, {} tweets to hydrate".format(i, len(reply_ids)))
        new_replies = list(t.hydrate(reply_ids))
        hydrated_replies += new_replies
        seen_ids |= set([tweet["id_str"] for tweet in new_replies])
        # Mark tweets that are in reply to a message by the same user for the next round
        reply_ids = [tweet["in_reply_to_status_id_str"] for tweet in new_replies
                    if tweet["in_reply_to_status_id_str"] is not None and
                    tweet["in_reply_to_status_id_str"] not in seen_ids and
                    tweet["in_reply_to_user_id_str"] == tweet["user"]["id_str"]]
        i += 1

    # Write upstream tweets as JSON
    print("Writing JSON...")
    upstream_tweets = []
    with open(os.path.join(intermediate_dir, "all_upstream_tweets.json"), "w") as file:
        for item in hydrated_replies:
            tweet = json.dumps(item)
            file.write(tweet + "\n")
            upstream_tweets.append(utils.json_to_tweet(item))

    print("Writing CSV...")
    upstream_df = pd.DataFrame(upstream_tweets)
    upstream_df.to_csv(os.path.join(intermediate_dir, "all_upstream_tweets.csv"),
                    line_terminator='\n')
    print("Wrote {} upstream tweets.".format(len(hydrated_replies)))
    return upstream_df


# # Downstream Tweets
# 
# Next use user timelines to find tweets that reply to tweets in the dataset. To do this efficiently, we find a unique set of users and find a consensus date window in which to search for tweets. We assume that replies occur within two days of the original tweet.

# First establish a list of reference IDs for each date
def get_date(tweet, day_delta=0):
    date = datetime.datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
    if day_delta != 0:
        date = date + datetime.timedelta(days=day_delta)
    return datetime.date.strftime(date, '%Y-%m-%d')

def get_user_dates(df):
    """Computes a set of search dates for each user in the dataset."""
    user_dates = {}

    tweets_with_replies = df[~pd.isna(df.reply_to_id) & (df.reply_to_user == df.user_id)]
    for i, tweet in tweets_with_replies.iterrows():
        user = tweet["user_id"]
        min_date = get_date(tweet, -2)
        max_date = get_date(tweet, 2)
        if user in user_dates:
            user_dates[user] = (min(min_date, user_dates[user][0]),
                            min(max_date, user_dates[user][1]))
        else:
            user_dates[user] = (min_date, max_date)

    print("{} users".format(len(user_dates)))
    return user_dates
    
def make_date_bounds_functions(df):
    """
    Creates functions to estimate the smallest and largest tweet IDs for a
    given day.
    """
    min_ids = {}
    max_ids = {}
    for i, tweet in df.iterrows():
        if i % 100000 == 0:
            print(i)
        id_num = int(tweet["id"])
        date = get_date(tweet)
        min_ids[date] = min(min_ids.get(date, 1e30), id_num)
        max_ids[date] = max(max_ids.get(date, 0), id_num)

    print(sorted(min_ids.items())[-5:], sorted(max_ids.items())[-5:])

    # Fill in reference IDs for dates that aren't in the set. To do this, we'll create two
    # rough, conservative linear models for tweet IDs over time by estimating the increment
    # in the minimum and maximum tweet IDs per day.

    # Note that we're going to give each tweet a two-day interval on either side, so the
    # exactness of this estimate isn't important except to improve the performance of the
    # tweet scraper.

    available_days = sorted(min_ids.keys())
    series = [available_days[0]]
    current = series[-1]

    min_id_items = []
    max_id_items = []

    date_index = 0
    while current != available_days[-1]:
        date = datetime.datetime.strptime(current, '%Y-%m-%d')
        current = datetime.date.strftime(date + datetime.timedelta(days=1), '%Y-%m-%d')
        if current in min_ids:
            min_id_items.append((date_index, min_ids[current]))
        if current in max_ids:
            max_id_items.append((date_index, max_ids[current]))
        series.append(current)
        date_index += 1

    assert len(min_id_items) >= 2 and len(max_id_items) >= 2, "Must have at least two days' worth of tweets to estimate tweet IDs"
    min_inc_per_day = (min_id_items[-1][1] - min_id_items[0][1]) / (min_id_items[-1][0] - min_id_items[0][0])
    max_inc_per_day = (max_id_items[-1][1] - max_id_items[0][1]) / (max_id_items[-1][0] - max_id_items[0][0])
    earliest_date = datetime.datetime.strptime(available_days[0], '%Y-%m-%d')

    def get_min_id(date_str):
        """Estimates the minimum tweet ID for the given date string, in the format YYYY-MM-DD."""
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        days = (date - earliest_date).days
        return min_ids[available_days[0]] + days * min_inc_per_day

    def get_max_id(date_str):
        """Estimates the maximum tweet ID for the given date string, in the format YYYY-MM-DD."""
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        days = (date - earliest_date).days
        return max_ids[available_days[0]] + days * max_inc_per_day

    print("Estimate:", get_min_id(series[0]), "Actual:", min_ids[series[0]])
    return get_min_id, get_max_id


def extract_timeline_tweets(t, df, intermediate_dir):
    """
    Extracts tweets in the users' timelines based on time windows around the
    tweets in the dataset.
    """

    current_batch = []
    current_csv = []
    batch_idx = 0

    user_dates = get_user_dates(df)
    user_items = sorted(user_dates.items())
    get_min_id, get_max_id = make_date_bounds_functions(df)

    i = 0
    for i, (user_id, (min_date, max_date)) in enumerate(user_items):
        if i % 100 == 0:
            # Periodically sleep to appease the Twitter rate limiting gods
            print(i)
            time.sleep(20)
        i += 1
        
        # Compute the boundary tweet IDs needed to search the Twitter timeline for this user
        min_id = int(get_min_id(min_date))
        max_id = int(get_max_id(max_date))
        for tweet in t.timeline(user_id=user_id, max_id=max_id, since_id=min_id):
            current_batch.append(tweet)
            current_csv.append(utils.json_to_tweet(tweet))

        if i % 1000 == 0:
            print("Writing")
            with open(os.path.join(intermediate_dir, "timeline_tweets_{}.json".format(batch_idx)), "w") as file:
                for item in current_batch:
                    file.write(json.dumps(item) + "\n")

            batch_df = pd.DataFrame(current_csv)
            batch_df.to_csv(os.path.join(intermediate_dir, "timeline_tweets_{}.csv".format(batch_idx)),
                        line_terminator="\n")
            batch_idx += 1
            current_batch = []
            current_csv = []
            
    # Write out the stragglers
    print("Writing last batch")
    with open(os.path.join(intermediate_dir, "timeline_tweets_{}.json".format(batch_idx)), "w") as file:
        for item in current_batch:
            file.write(json.dumps(item) + "\n")

    batch_df = pd.DataFrame(current_csv)
    batch_df.to_csv(os.path.join(intermediate_dir, "timeline_tweets_{}.csv".format(batch_idx)), line_terminator='\n')
    batch_idx += 1
    current_batch = []
    current_csv = []


# # Putting It All Together
# 
# We want to read all the timeline tweets, as well as the original tweet sets and upstream tweets, and put together a directed acyclic graph of tweet replies. This DAG will contain many individual components that are disconnected from each other, corresponding to tweet groups by individual users. We find all tweets in reply graphs that contain at least one tweet in the original dataset, and label those as belonging to a single "thread."
def get_all_threaded_tweets(df, upstream_df, intermediate_dir):
    """
    Combines all the tweets collected so far, including the original dataset,
    upstream tweets, and timeline tweets.
    """

    batch_idx = 0
    path = os.path.join(intermediate_dir, "timeline_tweets_{}.csv".format(batch_idx))
    timelines = None
    while os.path.exists(path):
        print("Reading {}...".format(os.path.basename(path)))
        sub_df = pd.read_csv(path, dtype=utils.dtype_spec, lineterminator='\n')
        if timelines is None:
            timelines = sub_df
        else:
            timelines = pd.concat([timelines, sub_df])
        batch_idx += 1
        path = os.path.join(intermediate_dir, "timeline_tweets_{}.csv".format(batch_idx))
    timelines = timelines.loc[:, ~timelines.columns.str.contains('^Unnamed')].reset_index(drop=True)    

    # Let's build a set of all the threaded tweets we know about.

    # Uncomment to load initial tweets from a previous run
    # df = pd.read_csv(os.path.join(intermediate_dir, "initial_doctor_tweets.csv"), dtype=utils.dtype_spec, index_col=0, lineterminator='\n')

    all_threaded_tweets = pd.concat([
        timelines,
        upstream_df,
        df
    ])

    print("{} tweets total".format(len(all_threaded_tweets)))
    return all_threaded_tweets


# 6/22/20: The old approach to threads may have been incorrect because some tweets have multiple
# replies, so the graph is more of a DAG than a linked list. We need to find the correct tweets
# to concatenate in these cases.

def write_thread_annotated_tweets(df, all_threaded_tweets, output_dir, time_period=datetime.timedelta(days=1)):
    downstream_replies = {}
    upstream_replies = {}

    dedup_tweets = all_threaded_tweets.drop_duplicates("id")
    dedup_tweets["id_num"] = dedup_tweets["id"].astype(int)
    dedup_tweets = dedup_tweets.sort_values("id_num", ascending=False).reset_index()
    print("Removed {} tweets".format(len(all_threaded_tweets) - len(dedup_tweets)))

    tweet_ids = {row["id"]: i for i, row in dedup_tweets.iterrows()}

    missing_tweets = 0
    for i, row in dedup_tweets.iterrows():
        if i % 100000 == 0: print(i, len(downstream_replies), len(upstream_replies))
        if row["reply_to_id"] and row["reply_to_user"] == row["user_id"]:
            if row["reply_to_id"] not in tweet_ids:
                missing_tweets += 1
                continue
            reply_index = tweet_ids[row["reply_to_id"]]
            downstream_replies.setdefault(reply_index, set()).add(i)
            upstream_replies.setdefault(i, set()).add(reply_index)
    print("Missing tweets: {}".format(missing_tweets))
    
    # For each tweet in the original dataframe, grab tweets in the thread within an hour of each tweet and call
    # them a thread.

    thread_ids = {}
    current_thread = 0

    df["id_num"] = df["id"].astype(int)
    sorted_original_tweets = df.sort_values(by="id_num", ascending=True).reset_index()

    for i, row in sorted_original_tweets.iterrows():
        if i % 100000 == 0:
            print(i)
        if row["id"] in thread_ids:
            continue
        thread_ids[row["id"]] = current_thread

        if time_period:
            timestamp = datetime.datetime.strptime(row.created_at, "%a %b %d %H:%M:%S +0000 %Y")
            lower_time_bound = timestamp - time_period
            upper_time_bound = timestamp + time_period
        
        # Grab upstream tweets
        curr = tweet_ids[row["id"]]
        while curr in upstream_replies:
            curr = list(upstream_replies[curr])[0]
            tweet = dedup_tweets.iloc[curr]
            
            if time_period:
                curr_ts = datetime.datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S +0000 %Y")
                if curr_ts < lower_time_bound or curr_ts > upper_time_bound:
                    break

            thread_ids[tweet["id"]] = current_thread
        
        # Downstream tweets
        queue = [tweet_ids[row["id"]]]
        while queue:
            curr = queue.pop(0)
            if curr not in downstream_replies:
                continue
            tweet = dedup_tweets.iloc[curr]
            
            if time_period:
                curr_ts = datetime.datetime.strptime(tweet.created_at, "%a %b %d %H:%M:%S +0000 %Y")
                if curr_ts < lower_time_bound or curr_ts > upper_time_bound:
                    continue

            thread_ids[tweet["id"]] = current_thread
            
            queue += list(downstream_replies[curr])

        current_thread += 1
        
    print("{} thread IDs".format(len(thread_ids)))

    present_tweets = dedup_tweets[dedup_tweets.id.isin(thread_ids)]
    present_tweets["thread_id"] = present_tweets.id.map(lambda id: thread_ids[id])
    present_tweets = present_tweets.sort_values(by="id", ascending=True)
    present_tweets = present_tweets.loc[:, ~present_tweets.columns.str.contains('^Unnamed|^index$|^level_')].reset_index(drop=True)    

    utils.write_tweet_csv(present_tweets, os.path.join(output_dir, "thread_annotated_tweets.csv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Augment dataset with tweet threads.'))
    parser.add_argument('creds', type=str,
                        help='Absolute path to twarc credentials (like /Users/USER/.twarc)')
    parser.add_argument('input', type=str,
                        help='Path to directory containing raw hydrated tweets')
    parser.add_argument('intermediates', type=str,
                        help='Path to intermediates directory (into which output thread_annotated_tweets.csv file will be written)')
    parser.add_argument('pipeline', type=str,
                        help='Path to pipeline directory (which will be automatically created and organized)')

    args = parser.parse_args()

    credentials_path = args.creds
    input_dir = args.input
    intermediate_dir = args.intermediates
    if not os.path.exists(intermediate_dir):
        os.mkdir(intermediate_dir)
        
    # Path to pipeline directory (will be automatically organized as the output for all later runs)
    pipeline_dir = args.pipeline
    if not os.path.exists(pipeline_dir):
        os.mkdir(pipeline_dir)
    output_dir = os.path.join(pipeline_dir, "raw")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load input tweets
    filenames = [os.path.join(input_dir, path) for path in os.listdir(input_dir)
                    if path.endswith(".csv") and not path.startswith(".")]
    df = pd.concat([pd.read_csv(filename, dtype=utils.dtype_spec, lineterminator='\n')
                    for filename in filenames])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)
    print(len(df), "tweets")
    df.to_csv(os.path.join(intermediate_dir, "initial_doctor_tweets.csv"), line_terminator='\n')

    # Load Twarc object
    t = utils.load_twarc(credentials_path)

    print("\n========= Upstream Tweets\n")
    upstream_df = extract_upstream_tweets(t, df, intermediate_dir)
    print("\n========= Timeline Tweets\n")
    extract_timeline_tweets(t, df, intermediate_dir)
    print("\n========= Combining\n")
    all_threaded_tweets = get_all_threaded_tweets(df, upstream_df, intermediate_dir)
    write_thread_annotated_tweets(df, all_threaded_tweets, output_dir)
    print("Done.")