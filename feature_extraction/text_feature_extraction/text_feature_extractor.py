import re
from progress.bar import ChargingBar

class TextFeatureExtractor:

    def extract_features(self, df):
        title_length = []
        title_capitals_count = []
        title_capitals_ratio = []
        title_non_letter_count = []
        title_non_letter_ratio = []
        bar = ChargingBar('Processing Text:\t\t', max=len(df['title']))
        for idx, title in df['title'].items():
            bar.next()
            title_length.append(self.text_length(title))
            title_capitals_count.append(self.capitals_count(title))
            title_capitals_ratio.append(self.capitals_ratio(title))
            title_non_letter_count.append(self.non_letter_count(title))
            title_non_letter_ratio.append(self.non_letter_ratio(title))
        bar.finish()
        df['title_length'] = title_length
        df['title_capitals_count'] = title_capitals_count
        df['title_capitals_ratio'] = title_capitals_ratio
        df['title_non_letter_count'] = title_non_letter_count
        df['title_non_letter_ratio'] = title_non_letter_ratio
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
