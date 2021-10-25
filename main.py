import re
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)

df = pd.read_csv("C:/Users/Daniel/PycharmProjects/UCD_DA_Project/Tweets.csv")
print(df.head())

comment = df['text'][5]
comment_full = df['text']
print(comment_full)

print(comment)

regex = r"@\w+"

re.findall(regex, comment)

# utilising iterator
temp_comment = [re.findall(regex, i) for i in df['text']]

print(temp_comment)

# isolating a single entry from the list temp_comment
temp_temp = temp_comment[3][0]

print(temp_temp)

temp_temp.replace(r"@\w+", "")

#defining a regex function that can be re-used at a later point

temp_string = re.sub(regex, "", comment)

print(temp_string)

# defining custom function to replace all matches with empty string

def custom_funk(text):
    inner_regex = re.sub(regex, "", text)

    # using iterator for applying custom function throughout all entries of tweets
    for i in comment_full:
        return inner_regex


df['cleaned_tweets']=[custom_funk(i) for i in comment_full]
df['cleaned_tweets'][0:10]