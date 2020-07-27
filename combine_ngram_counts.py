"""
When run as a script, combines multiple pickles from the output of
extract_medical_tweets into an initial estimate of word relevances. Afterward,
the compute_relevance function can be used to refine the relevance values using
new tweet sets.

The output of the script returns word counts only for ngrams that are in the
relevant set of tweets with a frequency of at least min_count.
"""

import os
import pickle
import tqdm
import argparse

def combine_tweet_counts(word_count_dir, key):
    """
    Sums the number of tweets analyzed from all files whose names begin with
    "word_counts" and end with ".pkl" in the given directory. These files are 
    assumed to be the output of extract_medical_tweets.py.

    Args:
        word_count_dir: Path to a word counts directory.
        key: Used to index into the pickled objects to find the lists of ngrams
            (for example, "total_tweets_count" or "doctor_tweets_count").

    Returns:
        The total number of tweets analyzed.
    """
    print("Calculating tweet count...")
    paths = [p for p in os.listdir(word_count_dir)
             if p.startswith("word_counts") and p.endswith(".pkl")]
    total = 0
    for path in tqdm.tqdm(paths):
        with open(os.path.join(word_count_dir, path), "rb") as file:
            info = pickle.load(file)
        total += info[key]
    print("Total:", total)
    
    return total

def combine_ngram_counts(word_count_dir, key, n, min_count=0, reference_ngrams=None):
    """
    Combines the ngram counts from all files whose names begin with "word_counts"
    and end with ".pkl" in the given directory. These files are assumed to be
    the output of extract_medical_tweets.py.

    Args:
        word_count_dir: Path to a word counts directory.
        key: Used to index into the pickled objects to find the lists of ngrams
            (for example, "doctor_ngrams" or "nondoctor_ngrams")
        n: Index of the ngram set to use (usually 0, 1, or 2, for unigrams,
            bigrams, or trigrams).
        min_count: Minimum number of occurrences of an ngram required to store
            it.
        reference_ngrams: If not None, a set/dictionary of ngrams that are
            allowed to be included in the final set.

    Returns:
        A dictionary of ngrams to counts.
    """
    ngrams = {}

    print(key, n)
    paths = [p for p in os.listdir(word_count_dir)
             if p.startswith("word_counts") and p.endswith(".pkl")]
    for path in tqdm.tqdm(paths):
        with open(os.path.join(word_count_dir, path), "rb") as file:
            info = pickle.load(file)
        sub_ngrams = info[key][n]

        for ngram, count in sub_ngrams.items():
            if reference_ngrams and ngram not in reference_ngrams:
                continue
            ngrams[ngram] = ngrams.get(ngram, 0) + count

    if min_count > 0:
        ngrams = {w: f for w, f in ngrams.items() if f >= min_count}

    print("Final counts:", len(ngrams))

    return ngrams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Merge together multiple word count files.'))
    parser.add_argument('input', type=str,
                        help='Path to directory containing word_counts_*.pkl')
    parser.add_argument('output', type=str,
                        help='Path to an output directory')
    parser.add_argument('--min_count', type=int,
                        help=('Minimum number of occurrences of ngram in '
                        'relevant tweets (higher saves more memory)'),
                        default=0, dest='min_count')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Relevant tweets first (establishes a set of useful ngrams)
    relevant = [combine_ngram_counts(args.input, "doctor_ngrams", i, min_count=args.min_count)
                for i in range(3)]
    relevant_tweet_count = combine_tweet_counts(args.input, "doctor_tweets_count")
    with open(os.path.join(args.output, "relevant_word_counts.pkl"), "wb") as file:
        pickle.dump({
            "tweet_count": relevant_tweet_count,
            "word_counts": relevant
        }, file)

    # Irrelevant tweets
    irrelevant = [combine_ngram_counts(args.input, "non_doctor_ngrams",
                                       i, reference_ngrams=relevant[i])
                  for i in range(3)]
    irrelevant_tweet_count = combine_tweet_counts(args.input, "total_tweets_count") - relevant_tweet_count
    with open(os.path.join(args.output, "irrelevant_word_counts.pkl"), "wb") as file:
        pickle.dump({
            "tweet_count": irrelevant_tweet_count,
            "word_counts": irrelevant
        }, file)