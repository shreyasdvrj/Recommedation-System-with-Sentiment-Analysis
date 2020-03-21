import pandas as pd
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
metadata['overview'] = metadata['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

##Working on your imput
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]
get_recommendations('Toy Story')

##SEntiment analysis to our recommendation system
import nltk
dataset = ["Food is good and not too expensive. Serving is just right for adult. Ambient is nice too.",
           "Used to be good. Chicken soup was below average, bbq used to be good.",
           "Food was good, standouts were the spicy beef soup and seafood pancake! ",
           "Good office lunch or after work place to go to with a big group as they have a lot of private areas with large tables",
           "As a Korean person, it was very disappointing food quality and very pricey for what you get. I won't go back there again. ",
           "Not great quality food for the price. Average food at premium prices really.",
           "Fast service. Prices are reasonable and food is decent.",
           "Extremely long waiting time. Food is decent but definitely not worth the wait.",
           "Clean premises, tasty food. My family favourites are the clear Tom yum soup, stuffed chicken wings, chargrilled squid.",
           "really good and authentic Thai food! in particular, we loved their tom yup clear soup with sliced fish. it's so well balanced that we can have it just on its own. "
           ]

nltk.download('vader_lexicon')
def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score
nltk_results = [nltk_sentiment(row) for row in dataset]
results_df = pd.DataFrame(nltk_results)
text_df = pd.DataFrame(dataset, columns = ['text'])
nltk_df = text_df.join(results_df)
review=input("Give your review on our recommendations : ")
score=nltk_sentiment(review)
max_key = max(score, key=score.get) 
if max_key=='pos':
  print("We are glad you liked it.")
elif max_key=='neg':
  print("Sorry to disappoint you. We will try harder.")
elif max_key=='neu':
  print("Cool.")
else:
  print("Can't understand, sorry.")