import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

st.set_page_config(page_title="Flipkart Sentiment Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Flipkart Product Review - Sentiment Analysis System")
st.markdown("### Analyzing Indian Customer Reviews using NLP")

df = pd.read_csv("flipkart_reviews_dataset.csv")
df = df[['review', 'rating']].dropna()
df.columns = ['review', 'rating']

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['review'].apply(get_sentiment)

st.markdown("## 📊 Dataset Overview")
st.dataframe(df.head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Positive Reviews", len(df[df['sentiment']=='Positive']))
col3.metric("Negative Reviews", len(df[df['sentiment']=='Negative']))

st.markdown("## 📈 Sentiment Distribution")
fig1, ax1 = plt.subplots()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
df['sentiment'].value_counts().plot(kind='bar', color=colors, ax=ax1)
ax1.set_title("Sentiment Count")
ax1.set_xlabel("Sentiment")
ax1.set_ylabel("Number of Reviews")
st.pyplot(fig1)

st.markdown("## 🥧 Sentiment Percentage")
fig2, ax2 = plt.subplots()
df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

st.markdown("## ☁️ Word Cloud")
text = " ".join(df['review'].astype(str).tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.imshow(wc, interpolation='bilinear')
ax3.axis('off')
st.pyplot(fig3)

st.markdown("## ✍️ Test Any Review Live")
user_input = st.text_area("Type any review here:")
if st.button("Analyze Sentiment"):
    if user_input:
        result = get_sentiment(user_input)
        if result == 'Positive':
            st.success(f"✅ Sentiment: {result}")
        elif result == 'Negative':
            st.error(f"❌ Sentiment: {result}")
        else:
            st.warning(f"😐 Sentiment: {result}")