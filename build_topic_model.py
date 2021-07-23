# Adapted from code by Dheekshita Kumar and Julia Wu

import numpy as np
import pandas as pd
import os
import json
import re
import argparse
import utils
import datetime
import pprint
import tqdm
import json

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

def write_summary(lda_model, gensim_lda_model, lemmatized_data, id2word, corpus, output_dir, num_topics):
    """
    Writes a summary of the LDA model to a summary.json file in the given output
    directory. Takes two versions of the same model as input: one LdaModel instance,
    and one gensim-compatible model instance.
    """

    # Compute Perplexity
    bound = gensim_lda_model.log_perplexity(corpus)
    perplexity = 2 ** (-bound)

    # Compute Coherence
    coherence_model_lda = CoherenceModel(model=lda_model,
                                        texts=lemmatized_data,
                                        dictionary=id2word,
                                        coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()

    # Look at topic keywords and their weights.
    topic_words = sorted([{"topic_num": x[0], "weights": [(word, round(weight, 3)) for word, weight in x[1]]}
                          for x in lda_model.show_topics(
                              num_topics=num_topics,
                              num_words=25,
                              formatted=False)],
                         key=lambda x: x["topic_num"])

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "num_tweets": len(lemmatized_data),
            "log_perplexity_bound": bound,
            "perplexity": perplexity,
            "coherence": coherence_score,
            "topics": topic_words
        }, f, sort_keys=True, indent=2, separators=(',', ': '))

def write_tweet_topics(tweets_df, gensim_lda_model, id2word, corpus, output_dir, num_topics=100):
    """
    Writes the 'probability' of each topic for each tweet to a file in output_dir
    called tweet_topics.csv.

    Args:
        tweets_df: The dataframe of tweets.
        gensim_lda_model: A gensim-compatible model instance.
        id2word: a gensim Dictionary representing a bag of words.
        corpus: bag of words representations of tweets in the same order as tweets_df
        output_dir: Directory in which to write output
        num_topics: The number of topics in the model
    """

    print("Writing tweet topics...")
    file = open(os.path.join(output_dir, "tweet_topics.csv"), "w")
    file.write("id," + ",".join(["prob_topic_" + str(t) for t in range(num_topics)]) + ",top_topic\n")

    ids = tweets_df.id.values.tolist()
    for corpus_element, id_str in tqdm.tqdm(zip(corpus, ids), total=len(ids)):
        topic_dist = gensim_lda_model.get_document_topics(corpus_element, minimum_probability=0.0)
        probs = ["0" for _ in range(num_topics)]
        max_topic = -1
        max_prob = 1.0 / num_topics # don't count tweets that have a uniform probability for all topics
        for topic, prob in topic_dist:
            probs[topic] = "{:.3g}".format(prob)
            if prob > max_prob:
                max_topic = topic
                max_prob = prob
        file.write(id_str + "," + ",".join(probs) + "," + str(max_topic) + "\n")

    file.close()

def build(mallet_path, tweets_df, output_dir, num_topics=100, num_iterations=1000, verbose=False):
    """
    Builds a topic model from the given tweets. Writes the following files into
    the output directory:

    * summary.json: A JSON summary of the model, including top word
        weights
    * tweet_topics.csv: A CSV file indexed by tweet ID where each column represents
        the weight of a given topic in the tweet.
    * model.pkl: A pickle file that contains the LDA model.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Preprocessing tweets...")
    initial_data = tweets_df.standardized_text.values.tolist()
    lemmatized_data = [tweet.split() for tweet in tqdm.tqdm(initial_data)]

    # Create dictionary and term-frequency mapping
    print("Building corpus...")
    id2word = corpora.Dictionary(lemmatized_data)
    tf = [id2word.doc2bow(tweet) for tweet in lemmatized_data]
    id2word.save(os.path.join(output_dir, "dictionary.pkl"))

    if verbose:
        print('The corpus (token_id, #occurances in this doc) for example tweets are:')
        print('Note words are in order of token_id, not order of original tweet')
        for i in range(20):
            print(tf[i])

        print('The first 20 words in the dictionary are:')
        for i in range(20):
            print(i, id2word[i])

    # Build the LDA model
    start_time = datetime.datetime.now()
    print("Building model...")
    if verbose: print('Started at ', str(start_time))

    lda_model = gensim.models.wrappers.LdaMallet(
                    mallet_path=os.path.join(mallet_path, "bin", "mallet"),
                    corpus=tf,
                    num_topics=num_topics,
                    iterations=num_iterations,
                    id2word=id2word)

    end_time = datetime.datetime.now()
    print("Saving model...")
    lda_model.save(os.path.join(output_dir, "model.pkl"))

    if verbose:
        print('Elapsed time: {}'.format(str(end_time - start_time)))

    # Write outputs
    gensim_lda_model = malletmodel2ldamodel(lda_model)
    write_summary(lda_model, gensim_lda_model, lemmatized_data, id2word, tf, output_dir, num_topics=num_topics)
    write_tweet_topics(tweets_df, gensim_lda_model, id2word, tf, output_dir, num_topics=num_topics)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Build a MALLET LDA topic model.'))
    parser.add_argument('mallet', type=str,
                        help='Absolute path to MALLET directory')
    parser.add_argument('tweets', type=str,
                        help='Path to a CSV file containing tweets')
    parser.add_argument('out', type=str,
                        help='Path to an output directory to create')
    parser.add_argument('--topics', type=int, help='Number of topics', default=100,
                        dest='num_topics')
    parser.add_argument('--iter', type=int, help='Number of iterations to run LDA', default=1000,
                        dest='num_iterations')
    parser.add_argument('--head', type=int, help='Number of tweets to limit to', default=0,
                        dest='head')
    parser.add_argument('-v', '--verbose', help='Verbose mode', default=False,
                        dest='verbose', action='store_true')

    args = parser.parse_args()

    # Read tweets
    print("Reading tweets...")
    tweets_df = utils.read_tweet_csv(args.tweets)
    if args.head:
        tweets_df = tweets_df.head(args.head)
    if args.verbose: print("Read {} tweets".format(len(tweets_df)))

    build(args.mallet, tweets_df, args.out, num_topics=args.num_topics, num_iterations=args.num_iterations, verbose=args.verbose)
