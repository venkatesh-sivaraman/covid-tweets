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
html_regex = r'&\w+;'

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
    
    # Remove HTML symbols
    tweet = re.sub(html_regex, '', tweet)

    # Remove newlines and random other characters
    tweet = re.sub(r'(?:\n|\r|[()-])+', ' ', tweet)

    # Remove multiple spaces
    tweet = strip_multiple_whitespaces(tweet)

    return tweet

def load_twarc(credentials_path):
    """
    Initializes a Twarc object using the credentials at the given path. The
    credentials path should contain a consumer key, consumer secret, access token,
    and access secret as written by the Twitter API following Twarc setup.

    Returns: An authenticated Twarc object.
    """

    import twarc

    with open(credentials_path, "r") as file:
        lines = file.readlines()
        keys = [line.strip().split(" = ")[1] for line in lines[1:] if line.strip()]
    t = twarc.Twarc(*keys, app_auth=True)
    return t

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
