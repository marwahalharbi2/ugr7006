#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import re
re1 = re.compile(r'  +')
import html
import string
from wordcloud import WordCloud #pip3 install wordcloud in anaconda prompt if needed
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


df = pd.read_csv("AllSaudiCitiesReviewsBooking.csv")
df.head()


# In[3]:


print(df.shape)


# In[4]:


print(df.columns)


# In[5]:


# Distribution of Positive and Negative Reviews
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
count_neg = df['negative_review'].count()
count_Pos= df['positive_review'].count()

values = (count_neg), (count_Pos)
ax.pie(values, 
 labels = ['Number of Positive Reviews', 'Number of Negative Reviews'],
 colors=['gold', 'lightcoral'],
 shadow=True,
 startangle=90, 
 autopct='%1.2f%%')
ax.axis('equal')
plt.title('Positive Reviews Vs. Negative Reviews');


# In[6]:


# Violin Plot of the Customer Ratings for the top 10 reviewers’ country of origin


# In[7]:


contry_series = df.Country.value_counts()[:10]
country_df = contry_series.to_frame()
country_df.columns = ['Count'] 
# This result in one colomn! why?

country_df


# In[8]:


fig = px.bar(x=country_df.index, y=country_df['Count'], 
             title='Customer Ratings for the top 10 reviewers’ country of origin')
fig.show() 


# In[9]:


df.Country.value_counts()[:10].index.tolist()


# In[10]:


top10_list = df.Country.value_counts()[:10].index.tolist()
top10 = df[df.Country.isin(top10_list)]
fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax = sns.violinplot(x = 'Country', 
 y = 'Reviewer_rating',
 data = top10, 
 order = top10_list,
 linewidth = 2) 
plt.suptitle('Distribution of Ratings by Country') 
plt.xticks(rotation=90);


# In[11]:


# Distribution of Review Tags Count for each Trip Type


# In[12]:


#Define tag list
tag_list_people = ['Group','Couple','Family','friends','Solo']

#Count for each review tag
tag_counts_people = []
for tag in tag_list_people:
    counts_people = df['Review_tags'].str.count(tag).sum()
    tag_counts_people.append(counts_people)
    
#Convert to a dataframe
trip_type_people = pd.DataFrame({'Trip Type':tag_list_people,'Counts':tag_counts_people}).sort_values('Counts',ascending = False)
    
    
#Visualize the trip type counts from Review_tags
fig_people = px.bar(trip_type_people, x='Trip Type', y='Counts', title='Review Tags Counts for each Trip Type')
fig_people.show()    


# In[13]:


#Define tag list
tag_list_trip_type = ['Business','Leisure']

#Count for each review tag
tag_counts_trip_type = []
for tag in tag_list_trip_type:
    counts_trip_type = df['Review_tags'].str.count(tag).sum()
    tag_counts_trip_type.append(counts_trip_type)
    
#Convert to a dataframe
trip_type_trip_type = pd.DataFrame({'Trip Type':tag_list_trip_type,'Counts':tag_counts_trip_type}).sort_values('Counts',ascending = False)
    
    
#Visualize the trip type counts from Review_tags
fig_trip_type = px.bar(trip_type_trip_type, x='Trip Type', y='Counts', title='Review Tags Counts for each Trip Type')
fig_trip_type.show()    


# In[14]:


#Define tag list
tag_list_nights = ['Stayed 1 night','Stayed 2 night','Stayed 3 night','Stayed 4 night','Stayed 5 night','Stayed 6 night']

#Count for each review tag
tag_counts_nights = []
for tag in tag_list_nights:
    counts_nights = df['Review_tags'].str.count(tag).sum()
    tag_counts_nights.append(counts_nights)
    
#Convert to a dataframe
trip_type_nights = pd.DataFrame({'Days':tag_list_nights,'Counts':tag_counts_nights}).sort_values('Counts',ascending = False)
    
    
#Visualize the trip type counts from Review_tags
fig_nights = px.pie(trip_type_nights,values='Counts', title='Review Tags Counts for each Trip Type', names='Days')
fig_nights.show()    


# In[15]:


df.Review_tags=df.Review_tags.str.split(',')
New_df=pd.DataFrame({'Trip Type':np.concatenate(df.Review_tags.values),'reviewer_name':df.reviewer_name.repeat(df.Review_tags.apply(len))})

New_df.groupby('Trip Type').reviewer_name.agg(['count']).sort_values('count', ascending = False)


# ### Data cleaning:

# In[16]:


print(df.info())


# In[17]:


#change the column type of Review_date object  to convert it into datetime64
df["Review_date"] =  pd.to_datetime(df['Review_date'])
print(df.info())


# In[18]:


print(df.shape)


# In[19]:


df = df.dropna(how='all', subset=['negative_review', 'positive_review'])
print(df.shape)


# In[20]:


# Data Pre-processing: 
# find out why  replace is not working?


# In[21]:


#df.replace(['check out','checking out'], 'checkout',regex=True)


# In[22]:


# # fix replace issue
# df2 = df.replace(to_replace='check in', value='checkin',regex=True)
# df2[df2['negative_review'].str.contains('check in')]['negative_review']


# In[23]:


df = df.replace(['check in','checking in'], 'checkin',regex=True)
df = df.replace(['check out','checking out'], 'checkout',regex=True)
#df = df.replace(['’'], '',regex=True)

df


# ### Data Pre-processing:

# In[24]:


# Lower case all comments 


# In[25]:


df['Overall_review'] = df['Overall_review'].fillna('').str.lower()
df['negative_review'] = df['negative_review'].fillna('').str.lower()
df['positive_review'] = df['positive_review'].fillna('').str.lower()

df.head()


# In[26]:


contractions = [
"ain't", "am not" , "are not",
"aren't", "are not" , "am not",
"can't", "cannot",
"can't've", "cannot have",
"could've", "could have",
"couldn't", "could not",
"couldn't've", "could not have",
"didn't", "did not",
"doesn't", "does not",
"don't", "do not",
"hadn't", "had not",
"hadn't've", "had not have",
"hasn't", "has not",
"haven't", "have not",
"isn't", "is not",
"wasn't", "was not"
]


# In[27]:


def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    punc = str.maketrans('', '', string.punctuation)
    return text.translate(punc)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    
    return re.sub(r'\d+', '', str(text))


def remove_whitespaces(text):
    return text.strip()


def text2words(text):
    return word_tokenize(text)


def remove_stopwords(words, stop_words):
    return [word for word in words.split() if word not in stop_words]


def lemmatize_words(words):
    """Lemmatize words in text"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])


def normalize_text(text):
    text = replace_numbers(text)
    text = remove_special_chars(text)
    text = remove_punctuation(text)
    text = remove_whitespaces(text)
    #words = text2words(text)
    #print(words)
    stop_words = stopwords.words('english')
    final_stop_words = set([word for word in stop_words if word not in contractions])
    words = remove_stopwords(text, final_stop_words)
    #print(words)
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)

    return ''.join(words)


# In[28]:


normalize_text(df['Overall_review'][0])


# In[29]:


df['Overall_review'][0]


# In[30]:


df['Overall_review'] = df['Overall_review'].apply(normalize_text)


# In[31]:


df['negative_review'] = df['negative_review'].apply(normalize_text)


# In[32]:


df['positive_review'] = df['positive_review'].apply(normalize_text)


# In[33]:


df['positive_review'] 


# In[34]:


# Create Word Cloud for Positive Reviews & Negative Reviews


# In[35]:


#instantiate a CountVectorizer object
# utilize the new STOP_WORDS list
cv_positive=CountVectorizer( stop_words=STOPWORDS, ngram_range=(2, 3))

# fit transform our text and create a dataframe with the result
corpus_positive = [' '.join(df['positive_review'].tolist())]
X_positive = cv_positive.fit_transform(corpus_positive)
X_positive = X_positive.toarray()

bow_positive=pd.DataFrame(X_positive, columns = cv_positive.get_feature_names())
bow_positive
#bow.index=speakers


# In[36]:


bow_positive.iloc[0].sort_values(ascending=False).to_dict()


# In[37]:


# create a pandas Series of the top 4000 most frequent words
#text_positive=bow_positive.iloc[0].sort_values(ascending=False)[:4000]

# create a dictionary Note: you could pass the pandas Series directoy into the wordcloud object
text2_dict_positive=bow_positive.iloc[0].sort_values(ascending=False).to_dict()

# create the WordCloud object
wordcloud_positive = WordCloud(width = 800, height = 800,min_word_length =3,
                background_color ='white',
                min_font_size = 10, stopwords = STOPWORDS,
                      collocations=True)

# generate the word cloud
wordcloud_positive.generate_from_frequencies(text2_dict_positive)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[38]:


#instantiate a CountVectorizer object
# utilize the new STOP_WORDS list
cv_negative=CountVectorizer( stop_words=STOPWORDS, ngram_range=(2, 3))

# fit transform our text and create a dataframe with the result
corpus_negative = [' '.join(df['negative_review'].tolist())]
X_negative = cv_negative.fit_transform(corpus_negative)
X_negative = X_negative.toarray()

bow_negative=pd.DataFrame(X_negative, columns = cv_negative.get_feature_names())
bow_negative
#bow.index=speakers


# In[39]:


bow_negative.iloc[0].sort_values(ascending=False).to_dict()


# In[40]:


# create a pandas Series of the top 4000 most frequent words
#text_negative=bow_negative.iloc[0].sort_values(ascending=False)[:4000]

# create a dictionary Note: you could pass the pandas Series directoy into the wordcloud object
text2_dict_negative=bow_negative.iloc[0].sort_values(ascending=False).to_dict()

# create the WordCloud object
wordcloud_negative = WordCloud(width = 800, height = 800,min_word_length =3,
                background_color ='white',
                min_font_size = 10, stopwords = STOPWORDS,
                      collocations=True)

# generate the word cloud
wordcloud_negative.generate_from_frequencies(text2_dict_negative)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# ### Parts of Speech (POS) tagging 

# In[41]:


#texts = df['Overall_review'].tolist()
#tagged_texts = pos_tag_sents(map(word_tokenize, texts))


# In[42]:


def pos_tagging(text):    
    tagged = nltk.pos_tag(text.split())
    return tagged


# In[43]:


pos_tagging(df['Overall_review'][0])


# In[44]:


df['Overall_review_POS'] = df['Overall_review'].apply(pos_tagging)


# In[45]:


df['negative_review_POS'] = df['negative_review'].apply(pos_tagging)


# In[46]:


df['positive_review_POS'] = df['positive_review'].apply(pos_tagging)


# In[47]:


df['Overall_review_POS']


# ### Named Entity Recognition

# In[48]:


# process the text and print Named entities
#https://pythonprogramming.net/named-entity-recognition-nltk-tutorial/

# df["tokenized_positive_review"] = df["positive_review"].apply(nltk.word_tokenize)
# tokenized = df['tokenized_positive_review'].tolist()

# # function
# def get_named_entity():
#     try:
#         for i in tokenized:
#             print(i)
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             namedEnt = nltk.ne_chunk(tagged, binary=True)
#             namedEnt.draw()
#     except:
#         pass
# get_named_entity()


# In[49]:


#!python -m spacy download en_core_web_sm


# In[50]:


# https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
# ModuleNotFoundError: No module named 'spacy' ===> pip install -U spacy in anaconda prompt ===> python -m spacy download en_core_web_sm

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[51]:


ner = spacy.load('en_core_web_sm')
def get_named_entity(text):
    doc = ner(text)
    return [(X.text, X.label_) for X in doc.ents]


# In[52]:


get_named_entity(df['Overall_review'][8])


# In[53]:


displacy.render(ner(df['Overall_review'][8]), jupyter=True, style='ent')


# In[54]:


df['Overall_review_NER'] = df['Overall_review'].apply(get_named_entity)


# In[55]:


df['positive_review_NER'] = df['positive_review'].apply(get_named_entity)


# In[56]:


df['negative_review_NER'] = df['negative_review'].apply(get_named_entity)


# In[57]:


df


# In[ ]:




