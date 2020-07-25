import pandas as pd
import os
import re
import string
import sys
import argparse
import pymetamap
from pymetamap import MetaMap
import utils

# Number of tweets fed into PyMetaMap at a time
METAMAP_BATCH_SIZE = 100

def concept_to_dict(concept, tweet_ids):
    """
    Converts the given concept object from PyMetaMap to a dictionary suitable
    for storing in a dataframe.

    concept: A ConceptMMI or ConceptAA object.
    tweet_ids: List of tweet IDs from which the concepts are indexed.

    Returns: A dictionary containing the concept's salient fields.
    """
    if isinstance(concept, pymetamap.ConceptMMI):
        return {
            "index": concept.index,
            "tweet_id": tweet_ids[int(concept.index)],
            "score": concept.score,
            "preferred_name": concept.preferred_name,
            "cui": concept.cui,
            "semtypes": concept.semtypes,
            "trigger": concept.trigger,
            "location": concept.location,
            "pos_info": concept.pos_info,
            "tree_codes": concept.tree_codes
        }
    elif isinstance(concept, pymetamap.ConceptAA):
        return {
            "index": concept.index,
            "tweet_id": tweet_ids[int(concept.index)],
            "aa": concept.aa,
            "short_form": concept.short_form,
            "long_form": concept.long_form,
            "pos_info": concept.pos_info
        }
    return {
        "index": concept.index,
        "pos_info": concept.pos_info
    }

def filter_tweets(data, tweet_ids):
    """
    Removes IDs and tweets that contain at most four words, as well as tweets
    that are duplicated. These tweets are unlikely to contain useful information
    and slow down the MetaMap processing time.

    data: The list of tweet text strings.
    tweet_ids: The list of tweet IDs corresponding to those tweet texts.

    Returns: A tuple (data, tweet_ids)
    """

    new_tweet_ids = []
    new_data = []
    seen_text = set()
    for tid, d in zip(tweet_ids, data):
        # Remove duplicates
        if d in seen_text: continue

        # Remove short texts
        if len(d.split()) <= 4: continue

        seen_text.add(d)
        new_tweet_ids.append(tid)
        new_data.append(d)

    return new_data, new_tweet_ids

def metamap_worker(metamap_path, tweets_path, output_dir, batch_size, batch_start, batch_end=None):
    """
    Runs MetaMap on a subset of tweets in the given dataset.

    metamap_path: Path to the metamap executable. For example,
        '/path/to/MetaMap/public_mm/bin/metamap18'
    tweets_path: Path to a CSV file containing tweets in the standard format.
    output_dir: Path to a directory in which output batch files should be written.
    batch_size: Number of tweets worth of concepts to write out in each batch file.
    batch_start: Index of the tweet in the file to start with. Batches will be
        numbered automatically.
    batch_end: Index of the tweet in the file to end with (exclusive).
    """

    # Setup MetaMap instance
    mm = MetaMap.get_instance(metamap_path)

    # Read and preprocess tweets
    tweets = utils.read_tweet_csv(tweets_path)
    data = tweets.full_text.values.tolist()
    data = [utils.preprocess_for_metamap(tweet) for tweet in data]
    tweet_ids = tweets["id"].values.tolist()
    
    # Remove spurious and duplicated tweets
    data, tweet_ids = filter_tweets(data, tweet_ids)
    assert len(data) == len(tweet_ids)

    all_concepts = []
    batch_idx = batch_start // batch_size
    print("Starting batch index:", batch_idx)

    failed_batches = []

    for batch_start in range(batch_start, batch_end if batch_end else len(data), METAMAP_BATCH_SIZE):
        batch_tweets = data[batch_start:batch_start + METAMAP_BATCH_SIZE]
        batch_ids = list(range(batch_start, batch_start + len(batch_tweets)))

        try:
            try:
                # Extract concepts as a batch
                concepts, error = mm.extract_concepts(batch_tweets, batch_ids)
            except (TypeError, IndexError):
                # Try extracting concepts individually
                for i, tweet in enumerate(batch_tweets):
                    concepts, error = mm.extract_concepts([tweet], [i + batch_start])

            if error is not None:
                print(error)
            else:
                all_concepts.extend([concept_to_dict(concept, tweet_ids) for concept in concepts])

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise KeyboardInterrupt
            print("Failed batch", batch_start)
            failed_batches.append(batch_start)

        if batch_start % 1000 == 0:
            print("Batch {}, {} concepts extracted".format(batch_start, len(all_concepts)))

        if (batch_start + METAMAP_BATCH_SIZE) % batch_size == 0:
            df = pd.DataFrame(all_concepts)
            df.to_csv(os.path.join(output_dir, "concepts_{}.csv".format(batch_idx)))
            batch_idx += 1
            all_concepts = []

    # Write out the last (partial) batch
    if len(all_concepts) > 0:
        df = pd.DataFrame(all_concepts)
        df.to_csv(os.path.join(output_dir, "concepts_{}.csv".format(batch_idx)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract medical concepts from tweets using MetaMap.')
    parser.add_argument('mm', type=str,
                        help='Path to MetaMap executable')
    parser.add_argument('input', type=str,
                        help='Path to tweets CSV file')
    parser.add_argument('output', type=str,
                        help='Path to output directory')
    parser.add_argument('--start', type=int, help='Tweet index to start', default=0,
                        dest='start')
    parser.add_argument('--end', type=int, help='Tweet index to end (defaults to end)',
                        default=None, dest='end')
    parser.add_argument('--batch_size', type=int, help='Number of tweet IDs to write out per batch (default 5000)',
                        default=5000, dest='batch_size')
    args = parser.parse_args()

    metamap_worker(args.mm, args.input, args.output, args.batch_size,
                   args.start, args.end)