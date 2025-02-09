import re
from progress.bar import ChargingBar
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


class TextFeatureExtractor:

    def extract_features(self, df):
        ''' Extracts the features for the text.'''
        title_length = []
        title_capitals_count = []
        title_capitals_ratio = []
        title_non_letter_count = []
        title_non_letter_ratio = []
        title_sentiment = []
        title_sentiment_polarity = []
        title_word_count = []
        title_number_count = []
        bar = ChargingBar('Processing Text:\t\t', max=len(df['title']))
        for idx, title in df['title'].items():
            bar.next()
            title_length.append(self.text_length(title))
            title_capitals_count.append(self.capitals_count(title))
            title_capitals_ratio.append(self.capitals_ratio(title))
            title_non_letter_count.append(self.non_letter_count(title))
            title_non_letter_ratio.append(self.non_letter_ratio(title))
            title_sentiment.append(self.sentiment(title)[0])
            title_sentiment_polarity.append(self.sentiment(title)[1])
            title_word_count.append(self.word_count(title))
            title_number_count.append(self.number_count(title))
        bar.finish()
        df['title_length'] = title_length
        df['title_capitals_count'] = title_capitals_count
        df['title_capitals_ratio'] = title_capitals_ratio
        df['title_non_letter_count'] = title_non_letter_count
        df['title_non_letter_ratio'] = title_non_letter_ratio
        df['title_sentiment'] = title_sentiment
        df['title_sentiment_polarity'] = title_sentiment_polarity
        df['title_word_count'] = title_word_count
        df['title_number_count'] = title_number_count
        return df

    def text_length(self, text):
        ''' Calculates the length of text'''
        return len(text)

    def capitals_count(self, text):
        ''' Counts the number of capitals'''
        return sum(1 for c in text if c.isupper())

    def capitals_ratio(self, text):
        ''' Calculates the number of capitals compared to all letters'''
        return self.capitals_count(text) / self.text_length(text)

    def non_letter_count(self, text):
        ''' Calculates the number of non letters (so numbers and punctuation)'''
        letters_and_space_regex = re.compile('[^a-zA-Z ]')
        return len(re.findall(letters_and_space_regex, text))

    def non_letter_ratio(self, text):
        ''' Calculates the ratio of non letters'''
        return self.non_letter_count(text) / self.text_length(text)

    def sentiment(self, text):
        ''' Calculates the sentiment of the text using nltk sentiment analyzer'''
        lines_list = tokenize.sent_tokenize(text)
        sid = SentimentIntensityAnalyzer()
        sent = 0
        sent_pol = 0
        for sentence in lines_list:
            sent += sid.polarity_scores(sentence)['compound']
            sent_pol += abs(sid.polarity_scores(sentence)['compound'])
        return ((sent / len(lines_list)), (sent_pol / len(lines_list)))

    def word_count(self, text):
        ''' Counts the number of words'''
        return sum([i.strip(string.punctuation).isalpha() for i in text.split()])

    def number_count(self, text):
        ''' Counts the number of numbers'''
        return sum(character.isdigit() for character in text)
