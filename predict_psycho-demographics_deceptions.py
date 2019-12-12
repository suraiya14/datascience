# source code for prediction

import sys, string, os, glob, json, re
from sklearn import linear_model
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.externals import joblib
from os import path
import twokenize, emoticons
import numpy as np
from scipy import sparse
from scipy.sparse import hstack
import pprint

class Predictor:
    """
    Take: a list of tweets
    Return: 1. predicted class; 2. predicted probabilities
    """

    def __init__(self):
        self.clf = linear_model.LogisticRegression(C=1.0, dual=False, penalty='l2', tol=1e-6)
        self.hv = HashingVectorizer(ngram_range=(1, 2), binary = True)

        self.pred_classes = []
        self.pred_probs = []
    
        self.attributes = ['age','gender',
                           'children', 'race', 'income', 'education',
                           'optimism', 
                           'life_satisfaction', 'decep_tech', 'decep_type']
        self.attr_values = {'age': 'Above_25, Below_25', 
         'gender': 'Female, Male', 'children': 'No, Yes', 
         'race': 'Black, White', 'income': 'Over_35K, Under_35K', 
         'education': 'Degree, High School', 'optimism': 'Optimist, Pessimist',
         'life_satisfaction': 'Dissatisfied, Satisfied', 
         'decep_tech': 'misleading, falsification', 'decep_type': 'hoax, propaganda, disinformation'}

    '''
    Take: dir with pre-trained models, attribute e.g., gender, a list of tweets,
          decep_tech/decep_type vocab files (a list of unigrams used to train decep_tech/decep_type models; 
          one per line)
    Return: predicted class and predicted probabilities for each attribute
    '''
    def predict_attribute(self, dir, attr, tweets, sent_vocab_file, emo_vocab_file):
        pred_dict = {}
        #loading pre-trained model
        current = os.getcwd()
        if dir not in os.getcwd():
            os.chdir(dir)
        for file in glob.glob("*.pkl"):
            if attr in file:
                self.clf = joblib.load(file)

        if 'decep_type' not in attr and 'decep_tech' not in attr:
            tweets = self.aggregate_tweets(tweets)
            features = self.hv.transform(tweets)
        
        else:
            if 'decep_tech' in attr:
                features = self.extract_more_decep_tech_features(tweets, sent_vocab_file)
            else:
                features = self.extract_more_decep_tech_features(tweets, emo_vocab_file)
        

        pred_probs = self.clf.predict_proba(features).tolist()
        #print pred_probs
        
        pred_classes = self.clf.predict(features).tolist()
        #print pred_classes
        
        pred_dict['pred_probs'] = pred_probs
        pred_dict['pred_class'] = pred_classes
        #print pred_dict

        if 'decep_type' in attr or 'decep_tech' in attr:
            #print pred_classes
            pie_values = pred.aggregare_sent_decep_type(pred_classes)
            pred_dict['pie_values'] = pie_values

        data_string = json.dumps(pred_dict)
        #print data_string
        return pred_dict
        

    def aggregate_tweets(self, tweet_list):
        """
        Get a blob of tweets from a list
        """
        tweets = []
        str_ = ''
        for tweet in tweet_list:
            str_ += tweet + ' '

        tweets.append(str_)
        return tweets

    def aggregare_sent_decep_type(self, pred_classes):
        #pred_classes = json.loads(json_pred)['pred_classes']
        
        valueToCount = {}
        for s in pred_classes:
            if s in valueToCount.keys():
                v = valueToCount[s]
                v += 1
            else:
                v = 1
            valueToCount[s] = v
        return valueToCount
        
    '''
    Additional features for decep_tech/decep_type classification:
        all-caps: YAY, COOL ..
        elongated words: waaay, sooo....
        emoticons: positive, negative
        hashtags: the number of hastags
        punctuation: !!!!, ???? ...
        POS tags and negation are in separate methods
    Return: a list of features vectors (one feature vector per tweet).
    '''
    def extract_more_decep_tech_features(self, tweets, vocab_file):
    	#print 'Extracting decep_tech/decep_type features with training vocab'
    	train_vocab = {}
        k = 0
    	for line in open(vocab_file):
    		train_vocab[line.strip()] = k
                k+=1
        #print 'Train vocab size=>' + str(len(train_vocab))
    		
        cv = CountVectorizer(ngram_range=(1, 1), binary = True, vocSuraiyalary = train_vocab)
        train_features_bow = cv.fit_transform(tweets)

        add_decep_tech_matrix = []
        hash_pattern = re.compile('\#+[\w_]+[\w\'_\-]*[\w_]+')
        elong_pattern = re.compile("([a-zA-Z])\\1{2,}")
        caps_pattern = re.compile(('[A-Z][A-Z\d]+'))
        punc_pattern = re.compile('([.,!?]+)')
        
        for tweet in tweets:
            tweet_vector = []
            tokens = twokenize.tokenize(tweet)
            #count the number of elongated tokens
            n_elong = len(re.findall(elong_pattern, tweet))
            
            #count the number of all_caps tokens
            n_caps = len(re.findall(caps_pattern, tweet))
            
            #count the number of repeated punctuation
            n_rep_punct = len(re.findall(punc_pattern, tweet))
            
            #count the number of hasgtags
            n_hahtag = len(re.findall(hash_pattern, tweet))
            
            #check if the tweets has SAD, HAPPY, BOTH_SH or NA emoticon
            emoticon_mood = emoticons.analyze_tweet(tweet.strip())
            if emoticon_mood == 'NA':
                emoticon_mood = 0
            elif emoticon_mood == 'HAPPY':
                emoticon_mood = 2
            elif emoticon_mood == 'SAD':
                emoticon_mood = 1
            elif emoticon_mood == 'BOTH_HS':
                emoticon_mood = 4
            tweet_vector = [n_elong, n_caps, n_rep_punct, n_hahtag, emoticon_mood]
            add_decep_tech_matrix.append(tweet_vector)
            
        #print np.asarray(add_decep_tech_matrix).shape
        a = np.asarray(add_decep_tech_matrix) 
        #print 'additional 5 features: ' + str(a)

        sa = sparse.csr_matrix(add_decep_tech_matrix)
        features =  hstack([sa, train_features_bow])
        #print 'final feature matrix size: ' + str(features.shape)
	
        return features                

    '''
        Take: a single tweet
        Return: appends a _NEG suffix to every word appearing between the negation and the clause-level punctuation mark
    '''
    def take_into_account_negation(self, tweet):
        neg_pattern = re.compile('never|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|no|' +
                            'n\'t|haven\'t|haven\'t|hasn\'t|hadn\'t|can\'t|couldn\'t|shouldn\'t|won\'t|wouldn\'t|don\'t|doesn\'t|didn\'t|isn\'t|aren\'t', re.IGNORECASE)
        clause_pattern = re.compile(r'^[.:;!?]$')

        neg = re.search(neg_pattern, tweet)
        if neg != None:
            #print 'Negation in tweet: ' + tweet
            pattern = tweet[neg.start():]
            end = re.search(clause_pattern, pattern)
            if end == None:
                end_str = len(tweet)
            else:
                end_str = end.start()
                end_str = int(end_str) - 1
            negated = ''

            tokens = twokenize.tokenize(pattern[:end_str])
            for w in tokens:
                negated += w + '_neg '
            negated = tweet[:neg.start()] + negated
            #print 'Negation in tweet: ' + negated
        else:
            negated = tweet
        return negated

    def read_data(self, filename):
        hash = {}
        line_num = 0
        for line in open(filename):
            if line_num >= 0:
                hash[line.strip()] = line.split('\t')[1].strip()
            line_num+=1
        print len(hash)
        return hash
        

if __name__ == "__main__":
    pred = Predictor()
    pp = pprint.PrettyPrinter()
    model_dir = r'E:/Suraiya/predict-psycho-demographics/Trained_Models'
    emo_vocab_file = r'E:/Suraiya/predict-psycho-demographics/decep_type_vocab'
    sent_vocab_file = r'E:/Suraiya/predict-psycho-demographics/decep_tech_vocab'
    tweets = ["Bangladesh government has a plan to disappear supporters of opposition party",
              "Bangladesh government declares war on social media based speech",
              "@samsmynamee Chuy has the biggest overbite I\'ve ever seen",
              "Bangladesh and India closed borders with Myanmar to resist the push in of Rohingya refugees http://t.co/FzInGKb4vp"
              ]
    
    for attr in pred.attributes:
        prediction = pred.predict_attribute(model_dir, attr, tweets, sent_vocab_file, emo_vocab_file)
        print 'Predicted ' + attr.upper() + ' => [' + pred.attr_values[attr] + ']'
        pp.pprint(prediction)



