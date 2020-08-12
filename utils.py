import re
import string
import pandas as pd
import tqdm
import datetime

import gensim
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


### Tweet CSV files


# dtypes used for passing to the dtype kwarg when reading a tweet dataframe
dtype_spec = {'id': str, 'user_id': str, 'reply_to_id': str, 'reply_to_user': str}

def read_tweet_csv(path, index_by_id=False):
    """
    Reads a standard tweet CSV file and returns the dataframe.

    index_by_id: If true, set the dataframe index to be the tweet ID.
    """
    return pd.read_csv(path,
                       lineterminator='\n',
                       dtype=dtype_spec,
                       index_col="id" if index_by_id else 0)

def write_tweet_csv(df, path):
    """
    Writes the given dataframe to a CSV file at the given path.
    """
    df.to_csv(path, line_terminator="\n")

CREATED_AT_FORMAT = '%a %b %d %H:%M:%S +0000 %Y'

def get_date(tweet):
    return datetime.date.strftime(datetime.datetime.strptime(tweet['created_at'], CREATED_AT_FORMAT), '%Y-%m-%d')


### Preprocessing


def processHashtags(tweet):
    """
    Helper function for taking care of hashtags.

    This removes any '#' symbols from the tweet, even if the occur in the middle of
    words. No further action is taken.
    """
    #remove '#' symbols
    tweet = re.sub('#', '', tweet)
    return tweet

def remove_html(tweet):
    """
    Removes HTML symbols like &amp; and &gt; from the tweet.
    """
    return re.sub(r'&\w+;', ' ', tweet)

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

    # Remove HTML elements
    tweet = remove_html(tweet)

    return tweet


#### LDA Preprocessing


'''
Augmenting stopwords with words used to filter the tweets originally.
Since they show up in almost every tweet, they aren't useful for differentiating
between topics.

stopwords are all lowercase.
'''
COVID_STOPWORDS = set([
                       'coronavirus',
                       '2019ncov',
                       'coronaviruspandemic',
                       'coronaoutbreak',
                       'wuhanvirus',
                       'covid19',
                       'covid-19',
                       'ncov',
                       'ncov2019',
                       'corona',
                       'virus',
                       'covid',
                       'covidー',
                       'cov',
                       'sarscov',
                       'sarscov2',
                       'amp'])
FILTER_WORDS = STOPWORDS.union(COVID_STOPWORDS)

def decontract(tweet):
    '''
    helper function for splitting contractions.
    \'s is removed because we can't disambiguate between possession (Julia's)
    and is (Julia is ...)
    '''
    tweet = re.sub(r"\b([A-Za-z]+)'([A-Za-z]+)\b", r"\1\2", tweet)
    return tweet

# Source: https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(tokens):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(tokens)
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

def preprocess_for_lda(tweet, pos_tag=True):
    """
    Processes a tweet for entry into an LDA topic model. Removes hashtags and
    unnecessary characters, filters out stopwords, tokenizes the tweet into
    individual words, and lemmatizes the words.
    """
    tweet = preprocess_tweet_text(tweet)

    # Handle contractions
    tweet = decontract(tweet)

    # Remove punctuation
    tweet = strip_punctuation(tweet)

    # Remove multiple spaces
    tweet = strip_multiple_whitespaces(tweet)

    # Tokenize, cases everything to lowercase, removes emojis
    tokens = simple_preprocess(tweet)

    # Lemmatize tokens
    if pos_tag:
        # This uses pos-tags and is slower but more accurate
        words = lemmatize_sentence(tokens)
    else:
        words = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove stopwords
    words = [word for word in words if word not in FILTER_WORDS]

    return words

def preprocess_keyphrase(keyphrase):
    """
    Returns an item representing the given keyphrase, which is a set of words
    separated by spaces. The returned item is a set of lemmatized tokens, with
    stopwords excluded.
    """
    return set([lemmatizer.lemmatize(w)
                for w in keyphrase.split()
                if w not in FILTER_WORDS])


#### MetaMap Preprocessing


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


### Word Counts


def collect_ngrams(counter, tokens, n=1, unique=False):
    """
    Adds the n-grams from the given set of tokens to the counter dictionary.
    """
    seen = set()
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i + n])
        if unique:
            if ngram in seen: continue
            seen.add(ngram)
        counter[ngram] = counter.get(ngram, 0) + 1

def collect_df_ngram_counts(df, min_count=0, unique=False, verbose=False):
    """
    Computes the ngram counts in the given dataframe, and returns them as a
    list of dictionaries, one for each type of ngram (1-3). If unique is true,
    only counts one of each unique n-gram per tweet.
    """
    word_counts = [{}, {}, {}]
    counter = tqdm.tqdm(range(len(df))) if verbose else range(len(df))
    for i in counter:
        tweet = df.iloc[i]
        tokens = [t for t in re.split(r"\W", preprocess_for_metamap(tweet["full_text"]).lower()) if t]
        for n, count_set in enumerate(word_counts):
            collect_ngrams(count_set, tokens, n + 1, unique=unique)

    if min_count > 0:
        word_counts = [{w: f for w, f in wc_set.items() if f >= min_count}
                       for wc_set in word_counts]
    return word_counts


### Concepts


# Categories of concept that are deemed "useful" for clinical relevance
USEFUL_SEMTYPES = {
    "orch",    "phsu",    "dsyn",
    "patf",    "virs",    "neop",
    "diap",    "topp",    "medd",
    "mbrt",    "sosy",    "bmod"
}

def filter_useful_concepts(df):
    """
    Returns a new dataframe with only concepts that are deemed 'useful' (i.e.
    not filter words, and in a useful UMLS semantic type category).
    """
    def is_useful_concept(row):
        if row.preferred_name.lower() in FILTER_WORDS:
            return False

        categories = set(row.semtypes.replace("[", "").replace("]", "").split(","))
        if not categories & USEFUL_SEMTYPES:
            return False
        return True

    return df[df.apply(is_useful_concept, axis=1)]
