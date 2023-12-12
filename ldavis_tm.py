import pandas as pd

# Read data from CSV file
df = pd.read_csv('C:/Users/Madanjit/Review_afterpre/kids.csv')

# Assuming your text data is in a column named 'Text'
text_data = df['Review Text']
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Tokenize text
df['Tokens'] = df['Review Text'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['Tokens'] = df['Tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

# Apply stemming (or lemmatization)
stemmer = PorterStemmer()
df['Tokens'] = df['Tokens'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])
from gensim import corpora

# Create a dictionary and DTM
dictionary = corpora.Dictionary(df['Tokens'])
dtm = [dictionary.doc2bow(tokens) for tokens in df['Tokens']]
from gensim.models.ldamodel import LdaModel

# Define the number of topics
num_topics = 45

# Run LDA
lda_model = LdaModel(dtm, num_topics=num_topics, id2word=dictionary, passes=15)
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Create an interactive visualization
vis_data = gensimvis.prepare(lda_model, dtm, dictionary)
pyLDAvis.display(vis_data)
# Save the visualization as an HTML file
pyLDAvis.save_html(vis_data, 'lda_visualization3.html')


