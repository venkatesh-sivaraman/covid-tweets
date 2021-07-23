#!/bin/bash

startdate=$1
enddate=$2
datadir=$3
wordcounts=$4
twarc=$5
metamap=$6
mallet=$7

mkdir -p $datadir

# Get latest tweets from Zenodo
python fetch_tweet_ids.py --out $datadir || exit 1

# Extract medical tweets
mkdir -p $datadir/raw-tweets
echo "================ HYDRATING MEDICAL TWEETS ================"
python extract_medical_tweets.py $twarc $datadir/full_dataset_clean.tsv.gz $datadir/raw-tweets

# Augment with threads
echo "================ AUGMENT WITH THREADS ================"
python retrieve_tweet_threads.py $twarc $datadir/raw-tweets $datadir/thread_intermediates $datadir/premerged

# Annotate concepts
echo "================ ANNOTATE CONCEPTS ================"
mkdir -p $datadir/raw-concepts
python annotate_concepts.py $metamap $datadir/premerged/thread_annotated_tweets.csv $datadir/raw-concepts
python clean_up_concepts.py $datadir/raw-concepts $wordcounts $datadir/premerged

# Merge threads
echo "================ MERGE THREADS ================"
python merge_threads.py $datadir/premerged/thread_annotated_tweets.csv $datadir/merged --concepts $datadir/premerged/concepts.csv

# Prepare for pipeline
echo "================ PREPROCESS ================"
python prepare_tweet_set.py $datadir $datadir/merged/merged_threads.csv $datadir/merged/merged_concepts.csv -v
ln -s "$( cd $wordcounts; pwd )/irrelevant_word_counts.pkl" $datadir/level_0/irrelevant_word_counts.pkl

for level in 0 1 2
do
    nextlevel=$(expr $level + 1)
    mkdir -p "$datadir/level_$nextlevel"
    # Build several topic models
    echo "================ LEVEL $level ================"
    python build_topic_model.py "$mallet" "$datadir/level_$level/tweets.csv" "$datadir/level_$level/100" --verbose --topics=100

    # Filter the tweets by the 100-topic model
    echo "================ RELEVANCE ================"
    if [[ $level == "0" ]]; then
        python compute_relevance.py "$datadir/level_$level" -v
    else
    python compute_relevance.py "$datadir/level_$level" -v -wc "$datadir/level_0/relevant_word_counts.pkl"
    fi
    echo "================ FILTER ================"
    python filter_topics.py "$datadir/level_$level" "$datadir/level_$nextlevel" --verbose
done