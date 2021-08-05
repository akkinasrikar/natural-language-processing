import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweets(tweet):
	stemmer = PorterStemmer()
	stopwords_english = stopwords.words('english')
	tweet = re.sub(r'\$\w*', '', tweet)
	tweet = re.sub(r'^RT[\s]+', '', tweet)
	tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	tweet = re.sub(r'#', '', tweet)
	tokenizer = TweetTokenizer(preserve_case=False, 
							   strip_handles=True,
	                           reduce_len=True)
	tweet_tokens = tokenizer.tokenize(tweet)
	tweet_clean=list(set([stemmer.stem(i) for i in tweet_tokens if (i not in stopwords_english and i not in string.punctuation)]))
	return tweet_clean
