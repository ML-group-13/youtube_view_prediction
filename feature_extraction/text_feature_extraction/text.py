import re

input = "Was It Worth It?"

length_text = len(input)
capitals_count = sum(1 for c in input if c.isupper())
capitals_ratio = capitals_count / length_text
regex = re.compile('[^a-zA-Z ]')
non_letter_count = len(re.findall(regex, input))
non_letter_ratio = non_letter_count / length_text

print(length_text)
print(capitals_count)
print(non_letter_count)
print(capitals_ratio)
print(non_letter_ratio)
