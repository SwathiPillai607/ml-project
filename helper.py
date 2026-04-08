from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from transformers import pipeline
import streamlit as st

# --- MODEL LOADING (CACHED) ---
@st.cache_resource  
def load_sentiment_model():
    """Loads the Transformers pipeline once and caches it in memory."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- ANALYSIS FUNCTIONS ---

def get_sentiment_analysis(selected_user, df):
    # FIX: Call the cached loader function here
    sentiment_analyzer = load_sentiment_model()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # 1. Clean: Remove system notifications and media
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # 2. Limit: Analyze the last 30 messages to keep the app fast
    messages = temp['message'].tail(30).tolist()
    
    if not messages:
        return None

    return sentiment_analyzer(messages)

def train_user_prediction_model(df):
    # 1. Cleaning: Remove system messages and media
    df = df[df['user'] != 'group_notification']
    df = df[~df['message'].str.contains('<Media omitted>')]
    
    # Filter users with > 5 messages to ensure the model has enough data
    user_counts = df['user'].value_counts()
    valid_users = user_counts[user_counts > 5].index
    df = df[df['user'].isin(valid_users)]
    
    df = df.dropna(subset=['user', 'message'])

    # 2. Encode Target
    le = LabelEncoder()
    y = le.fit_transform(df['user'])
    
    # 3. Vectorize (Using N-grams to pick up phrases)
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X = tfidf.fit_transform(df['message']).toarray()
    
    # 4. Train-Test Split (with stratify to keep user proportions even)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. XGBoost Model
    model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        eval_metric='mlogloss',
        tree_method='hist' 
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return round(accuracy * 100, 2), model, tfidf, le

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    text = temp['message'].astype(str).str.cat(sep=" ")
    if not text.strip():
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(text)

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'User Name', 'user': 'Percent (%)'})
    return x, df_percent

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline