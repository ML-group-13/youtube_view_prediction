import re


class TextFeatureExtractor:

    def extract_features(self, df):
        title_length = []
        title_capitals_count = []
        title_capitals_ratio = []
        title_non_letter_count = []
        title_non_letter_ratio = []
        for title in df['title']:
            title_length.append(self.text_length(title))
            title_capitals_count.append(self.capitals_count(title))
            title_capitals_ratio.append(self.capitals_ratio(title))
            title_non_letter_count.append(self.non_letter_count(title))
            title_non_letter_ratio.append(self.non_letter_ratio(title))
        df['title_length'] = title_length
        df['title_capitals_count'] = title_capitals_count
        df['title_capitals_ratio'] = title_capitals_ratio
        df['title_non_letter_count'] = title_non_letter_count
        df['title_non_letter_ratio'] = title_non_letter_ratio
        return df

    def text_length(self, text):
        return len(text)

    def capitals_count(self, text):
        return sum(1 for c in input if c.isupper())

    def capitals_ratio(self, text):
        return self.capitals_count(text) / self.text_length(text)

    def non_letter_count(self, text):
        letters_and_space_regex = re.compile('[^a-zA-Z ]')
        return len(re.findall(letters_and_space_regex, input))

    def non_letter_ratio(self, text):
        return self.non_letter_count(text) / self.text_length(text)
