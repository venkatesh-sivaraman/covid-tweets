import re
import string

import gensim
from gensim.parsing.preprocessing import strip_multiple_whitespaces

# dtypes used for passing to the dtype kwarg when reading a tweet dataframe
dtype_spec = {'id': str, 'user_id': str, 'reply_to_id': str, 'reply_to_user': str}

def processHashtags(tweet):
    """
    Helper function for taking care of hashtags.

    This removes any '#' symbols from the tweet, even if the occur in the middle of
    words. No further action is taken.
    """
    #remove '#' symbols
    tweet = re.sub('#', '', tweet)
    return tweet

def preprocess_tweet_text(tweet):
    """
    Performs a range of preprocessing tasks on the given tweet text, including:
    - remove links
    - remove @ mentions
    - remove hashtag symbols
    """
    # Remove links
    # looks for [text].[text 2-5 chars long][might have / at end, might have /more text at end]
    link_regex = '\\S+\\.\\w{2,5}(\\/\\w*)'
    tweet = re.sub(link_regex,'', tweet)

    # Remove @ mentions, even the username
    tweet = re.sub('@\\w+', '', tweet)

    # Process hashtags
    tweet = processHashtags(tweet)

    return tweet

printable_chars = set(string.printable)
decimal_regex = r'\b[0-9.,]+\b'

def to_ascii(tweet):
    return ''.join(c for c in tweet if c in printable_chars)

def preprocess_for_metamap(tweet):
    """
    Processes a tweet for entry into MetaMap, by removing non-ASCII characters,
    numbers, and random other characters/newlines/spaces. Calls
    preprocess_tweet_text first.
    """

    tweet = preprocess_tweet_text(tweet)

    tweet = to_ascii(tweet)

    # Remove decimal numbers
    tweet = re.sub(decimal_regex, '', tweet)

    # Remove newlines and random other characters
    tweet = re.sub(r'(?:\n|\r|[()-])+', ' ', tweet)

    # Remove multiple spaces
    tweet = strip_multiple_whitespaces(tweet)

    return tweet
