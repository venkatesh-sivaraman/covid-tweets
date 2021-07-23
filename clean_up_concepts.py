# The concepts extracted by MetaMap contain many spurious concepts, often
# identified from very common words. This step produces a condensed dataframe 
# with just the concepts whose trigger words appear relatively *rarely* in the
# English language.

import pandas as pd
import os
import re
import pickle
import numpy as np
import argparse
import utils

def make_concept_enrichment_function(word_counts_dir):
    # Base level
    with open(os.path.join(word_counts_dir, "relevant_word_counts.pkl"), "rb") as file:
        doctor_info = pickle.load(file)

    with open(os.path.join(word_counts_dir, "irrelevant_word_counts.pkl"), "rb") as file:
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

    def concept_enrichment(concept_row):
        words = re.split(r"\W", concept_row.trigger_word)
        trigger = " ".join(words)
        if trigger in relevance:
            return (relevance.get(trigger, 0))
        word_rels = [relevance[word] for word in words if word in relevance]
        return np.mean(word_rels) if word_rels else 0.0

    return concept_enrichment

def filter_concepts(concepts_dir, concept_enrichment_fn):
    batch_index = 0
    path = os.path.join(concepts_dir, "concepts_{}.csv".format(batch_index))

    df = None
    while os.path.exists(path):
        sub_df = pd.read_csv(path)
        print("Processing {}, {} concepts so far".format(path, len(df) if df is not None else 0))

        filtered_concepts = sub_df[~pd.isna(sub_df.trigger)]

        # Extract the trigger word
        filtered_concepts["trigger_word"] = filtered_concepts["trigger"].str.extract(r"\d-\"([^\"]+)\"-")[0].str.lower()
        filtered_concepts = filtered_concepts[~pd.isna(filtered_concepts.trigger_word)]    

        # Filter by semtype and exclude certain words
        filtered_concepts = utils.filter_useful_concepts(filtered_concepts)

        # Compute enrichment
        filtered_concepts["enrichment"] = filtered_concepts.apply(concept_enrichment_fn, axis=1)

        # Filter for only concepts that are MORE enriched in doctor tweets than non-doctor tweets
        filtered_concepts = filtered_concepts[filtered_concepts["enrichment"] >= 1.0]

        # Concatenate concepts
        if df is None:
            df = filtered_concepts
        else:
            df = pd.concat([df, filtered_concepts])

        batch_index += 1
        path = os.path.join(concepts_dir, "concepts_{}.csv".format(batch_index))

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Pare down concept annotations by filtering for a base level of relevance.'))
    parser.add_argument('concepts', type=str,
                        help='Path to directory containing concepts_<num>.csv files')
    parser.add_argument('word_counts', type=str,
                        help='Path to directory containing relevant_word_counts.pkl and irrelevant_word_counts.pkl')
    parser.add_argument('out', type=str,
                        help='Path to the output directory')
    args = parser.parse_args()
    
    concepts_dir = args.concepts
    word_counts_dir = args.word_counts
    output_dir = args.out
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    concept_enrichment = make_concept_enrichment_function(word_counts_dir)
    df = filter_concepts(concepts_dir, concept_enrichment)

    unique_concepts = df.drop_duplicates('preferred_name').sort_values('enrichment', ascending=False)
    print("Most enriched:", unique_concepts.head(20)[['trigger_word', 'preferred_name', 'enrichment']])
    print("Least enriched:", unique_concepts.tail(20)[['trigger_word', 'preferred_name', 'enrichment']])

    df.to_csv(os.path.join(output_dir, "concepts.csv"))
    print("Done.")
