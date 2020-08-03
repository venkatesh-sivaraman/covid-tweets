#!/bin/bash

level=$1
datadir=$2
mallet=$3
concepts=$4
threadmapping=$5

nextlevel=$(expr $level + 1)

# Build several topic models
echo "================ 100 TOPICS ================"
python build_topic_model.py "$mallet" "$datadir/level_$level/tweets.csv" "$datadir/level_$level/100" --verbose --topics=100
echo "================ 50 TOPICS ================"
python build_topic_model.py "$mallet" "$datadir/level_$level/tweets.csv" "$datadir/level_$level/50" --verbose --topics=50
echo "================ 10 TOPICS ================"
python build_topic_model.py "$mallet" "$datadir/level_$level/tweets.csv" "$datadir/level_$level/10" --verbose --topics=10

# Filter the tweets by the 100-topic model
echo "================ FILTER ================"
python filter_topics.py "$datadir/level_$level" "$datadir/level_$level/100" "$concepts" "$datadir/level_$nextlevel" --thread_mapping="$threadmapping" --verbose