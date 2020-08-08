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

def load_relevance(input_dir, return_counts=False, verbose=False):
    """
    Computes a relevance dictionary given the relevant_word_counts.pkl and
    irrelevant_word_counts.pkl files in the input directory.
    """
    with open(os.path.join(input_dir, "relevant_word_counts.pkl"), "rb") as file:
        doctor_info = pickle.load(file)

    with open(os.path.join(input_dir, "irrelevant_word_counts.pkl"), "rb") as file:
        non_doctor_info = pickle.load(file)

    doctor_tweet_count = doctor_info["tweet_count"]
    doctor_word_counts = doctor_info["word_counts"]
    non_doctor_tweet_count = non_doctor_info["tweet_count"]
    non_doctor_word_counts = non_doctor_info["word_counts"]

    relevance = {}
    for n, word_count_set in enumerate(doctor_word_counts):
        for word, f in word_count_set.items():
            if word.lower() in utils.FILTER_WORDS:
                continue
            non_doctor_f = non_doctor_word_counts[n].get(word, 0)
            relevance[word] = (f / doctor_tweet_count + 1e-3) / (non_doctor_f / non_doctor_tweet_count + 1e-3)

    if verbose:
        sorted_ratios = sorted(relevance.items(), key=lambda x: x[1])
        print("Most enriched terms:")
        for word, ratio in reversed(sorted_ratios[-20:]):
            print(word, ratio)
        print("\nLeast enriched:")
        for word, ratio in sorted_ratios[:20]:
            print(word, ratio)

    return (doctor_word_counts, non_doctor_word_counts, relevance) if return_counts else relevance

def load_topics(topics_file, tweets_file, head=0, return_tweets=False):
    """
    Loads a topic probability dataframe and associates it with the tweet info.
    Returns both the merged topic dataframe and the tweets dataframe.
    """
    topics_df = utils.read_tweet_csv(topics_file, index_by_id=True)
    tweets_df = utils.read_tweet_csv(tweets_file, index_by_id=True)
    if head:
        tweets_df = tweets_df.head(head)
    assert len(topics_df) == len(tweets_df)

    topics_df = pd.merge(topics_df, tweets_df, left_index=True, right_index=True)
    return (topics_df, tweets_df) if return_tweets else topics_df

def load_concepts(concepts_file, relevance, verbose=False):
    """
    Loads a dataframe of concepts and computes the concept relevances.
    
    Args:
        concepts_file: Path to the concepts file.
    """
    concepts_df = pd.read_csv(concepts_file)
    concepts_df = concepts_df[~pd.isna(concepts_df.trigger_word)]

    # First compute enrichment ratio for all concepts
    def concept_enrichment(concept_row):
        trigger = " ".join(re.split(r"\W", concept_row.trigger_word))
        return (relevance.get(trigger, 0))
    concepts_df["relevance"] = concepts_df.apply(concept_enrichment, axis=1)

    if verbose:
        print("Most relevant concepts:")
        print(concepts_df.drop_duplicates("trigger_word").sort_values(by="relevance", ascending=False).head(10)[["cui", "preferred_name", "trigger_word", "relevance"]])

    # Don't double count the same concept
    old_count = len(concepts_df)
    concepts_df = concepts_df.drop_duplicates(["tweet_id", "preferred_name"]).drop_duplicates(["tweet_id", "trigger_word"])
    if verbose: print("Dropped {} duplicate concepts".format(old_count - len(concepts_df)))

    concept_relevances = concepts_df.groupby('tweet_id').agg({'relevance': 'sum'})
    return concepts_df, concept_relevances

def representative_keyphrases(topics_df, all_word_counts, tweet_count, num_topics=100, max_keyphrases=5, min_count=2, verbose=False):
    """
    Determines a set of representative keyphrases for each topic. Returns a list
    of lists where each inner list contains a keyphrase and an enrichment score
    for that keyphrase.
    """
    rep_keyphrases = []

    for topic in range(num_topics):
        single_topic_df = topics_df[topics_df.top_topic == topic]
        word_counts = utils.collect_df_ngram_counts(single_topic_df, min_count=min_count, verbose=False)
        relevance = {}
        for n, word_count_set in enumerate(word_counts):
            for word, f in word_count_set.items():
                comps = word.split()
                if {word, comps[0], comps[-1]} & utils.FILTER_WORDS:
                    continue
                reference_f = all_word_counts[n].get(word, 0) - f
                relevance[word] = (f / len(single_topic_df) + 1e-3) / (reference_f / (tweet_count - len(single_topic_df)) + 1e-3)
        sorted_ratios = sorted(relevance.items(), key=lambda x: x[1], reverse=True)
        
        keyphrases = sorted([(utils.preprocess_keyphrase(x[0]), x) for x in sorted_ratios[:max_keyphrases]],
                            key=lambda x: len(x[0]),
                            reverse=True)
        # Remove redundant keyphrases
        i = 0
        while i < len(keyphrases):
            current, _ = keyphrases[i]
            j = i + 1
            while j < len(keyphrases):
                candidate, _ = keyphrases[j]
                # If this is a subset, or the overlap is sufficiently large (at most one different
                # word present in each) then remove
                if candidate & current == candidate or len(candidate & current) >= max(len(candidate) - 1, len(current) - 1, 1):
                    del keyphrases[j]
                else:
                    j += 1
            i += 1
        
        if verbose: print("Topic", topic, ":", ", ".join(word for info, (word, ratio) in keyphrases))
        rep_keyphrases.append([(word, round(ratio, 3)) for tokens, (word, ratio) in keyphrases])

    return rep_keyphrases

def get_date(tweet):
    return datetime.date.strftime(datetime.datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')

def tweet_counts_by_date(topics_df, num_topics=100, verbose=False):
    """
    Computes the number of tweets posted on each day for each topic.
    """
    topic_dates = [{} for _ in range(num_topics)]
    topic_probs = topics_df[["prob_topic_{}".format(t) for t in range(num_topics)]]

    counter = tqdm.tqdm(range(len(topics_df))) if verbose else range(len(topics_df))
    for i in counter:
        probs = topic_probs.iloc[i].values
        day = get_date(topics_df.iloc[i])
        for t, p in zip(topic_dates, probs):
            t[day] = t.get(day, 0) + p

    return topic_dates

def compute_tweet_relevance(topics_df, concept_relevances, num_topics=100, verbose=False):
    """
    Computes the relevance of each topic as well as each tweet.
    """
    tweets_with_concept_counts = pd.merge(topics_df, concept_relevances, left_index=True, right_index=True, how='left')
    tweets_with_concept_counts.relevance = tweets_with_concept_counts.relevance.fillna(0)

    # Compute topic importances
    if verbose: print("Computing topic importances...")
    relevance_summary = []

    topic_counter = tqdm.tqdm(range(num_topics)) if verbose else range(num_topics)
    for i in topic_counter:
        label = "prob_topic_" + str(i)
        weights = tweets_with_concept_counts[label]
        weighted_rel = weights * tweets_with_concept_counts.relevance
        topic_relevance = weighted_rel.sum() / weights.sum()

        ranked_tweets = tweets_with_concept_counts.sort_values(label, ascending=False).head(1000)
        relevance_summary.append({
            "topic_num": i,
            "relevance": topic_relevance,
            "tweet_ids": [{
                "id": n,
                "prob": round(row[label], 3),
                "relevance": round(row["relevance"], 3)
            } for n, row in ranked_tweets.iterrows()]
        })

    return relevance_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Compute topic and tweet relevance.'))
    parser.add_argument('base_dir', type=str,
                        help='Path to a directory containing topic models, a tweets.csv file, word count pickles, and concepts')
    parser.add_argument('-wc', '--word-counts', type=str, dest='word_counts_path', default=None,
                        help='Path to a word counts pkl to use as the baseline for representative keyphrases (default is to use current level)')
    parser.add_argument('--head', type=int, help='Number of tweets limited to in the topic model', default=0,
                        dest='head')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        dest='verbose')

    args = parser.parse_args()

    if args.verbose: print("Computing relevance...")
    word_counts, _, relevance = load_relevance(args.base_dir, return_counts=True, verbose=args.verbose)

    if args.verbose: print("Loading concepts...")
    concepts_df, concept_relevances = load_concepts(os.path.join(args.base_dir, "concepts.csv"),
                                                    relevance, verbose=args.verbose)
    concepts_df.to_csv(os.path.join(args.base_dir, "concepts_with_relevance.csv"))

    for fname in sorted(os.listdir(args.base_dir)):
        try:
            num_topics = int(fname)
        except:
            continue
        else:
            if args.verbose: print("Loading {}-topic model...".format(num_topics))
            topics_df = load_topics(
                os.path.join(args.base_dir, fname, "tweet_topics.csv"),
                os.path.join(args.base_dir, "tweets.csv"),
                head=args.head)
            
            relevance_summary = compute_tweet_relevance(topics_df,
                                                        concept_relevances,
                                                        num_topics=num_topics,
                                                        verbose=args.verbose)

            baseline_tweet_count = len(topics_df)
            if args.word_counts_path:
                with open(args.word_counts_path, "rb") as file:
                    info = pickle.load(file)
                    baseline_tweet_count = info["tweet_count"]
                    word_counts = info["word_counts"]

            # Representative keyphrases
            keyphrases = representative_keyphrases(topics_df,
                                                   word_counts,
                                                   baseline_tweet_count,
                                                   num_topics=num_topics,
                                                   verbose=args.verbose)
            # Time series
            time_series = tweet_counts_by_date(topics_df,
                                               num_topics=num_topics,
                                               verbose=args.verbose)

            for item, kp, ts in zip(relevance_summary, keyphrases, time_series):
                item["keyphrases"] = kp
                item["time_series"] = ts

            with open(os.path.join(args.base_dir, fname, "relevance.json"), "w") as file:
                json.dump(relevance_summary, file)

    print("Done.")