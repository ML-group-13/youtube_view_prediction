import re
import string 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from feature_extraction.text_feature_extraction.clickbait_detector.src.detect import Predictor


class TextFeatureExtractor:

    def extract_features(self, df):
        title_length = []
        title_capitals_count = []
        title_capitals_ratio = []
        title_non_letter_count = []
        title_non_letter_ratio = []
        title_sentiment = []
        title_sentiment_polarity = []
        clickbait_predictions = []
        clickbait_predictor = Predictor()
        title_word_count = []
        title_number_count = []
        for idx, title in df['title'].items():
            title_length.append(self.text_length(title))
            title_capitals_count.append(self.capitals_count(title))
            title_capitals_ratio.append(self.capitals_ratio(title))
            title_non_letter_count.append(self.non_letter_count(title))
            title_non_letter_ratio.append(self.non_letter_ratio(title))
            title_sentiment.append(self.sentiment(title)[0])
            title_sentiment_polarity.append(self.sentiment(title)[1])
            clickbait_predictions.append(clickbait_predictor.predict(title))
            title_word_count.append(self.word_count(title))
            title_number_count.append(self.number_count(title))
        df['title_length'] = title_length
        df['title_capitals_count'] = title_capitals_count
        df['title_capitals_ratio'] = title_capitals_ratio
        df['title_non_letter_count'] = title_non_letter_count
        df['title_non_letter_ratio'] = title_non_letter_ratio
        df['title_sentiment'] = title_sentiment
        df['title_sentiment_polarity'] = title_sentiment_polarity
        df['clickbait_score'] = clickbait_predictions
        df['title_word_count'] = title_word_count
        df['title_number_count'] = title_number_count
        return df

    def text_length(self, text):
        return len(text)

    def capitals_count(self, text):
        return sum(1 for c in text if c.isupper())

    def capitals_ratio(self, text):
        return self.capitals_count(text) / self.text_length(text)

    def non_letter_count(self, text):
        letters_and_space_regex = re.compile('[^a-zA-Z ]')
        return len(re.findall(letters_and_space_regex, text))

    def non_letter_ratio(self, text):
        return self.non_letter_count(text) / self.text_length(text)

    def sentiment(self, text):
        lines_list = tokenize.sent_tokenize(text)
        sid = SentimentIntensityAnalyzer()
        sent = 0
        sent_pol = 0
        for sentence in lines_list:
            sent += sid.polarity_scores(sentence)['compound']
            sent_pol += abs(sid.polarity_scores(sentence)['compound'])
        return ((sent / len(lines_list)), (sent_pol / len(lines_list)))
		
    def word_count(self, text):
        return sum([i.strip(string.punctuation).isalpha() for i in text.split()]) 
		
    def number_count(self, text):
        return sum(character.isdigit() for character in text)


