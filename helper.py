from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight 
import numpy as np
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
    sentiment_analyzer = load_sentiment_model()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    messages = temp['message'].tail(30).tolist()
    
    if not messages:
        return None

    return sentiment_analyzer(messages)

def train_user_prediction_model(df):
    # 1. Light Cleaning
    df = df[df['user'] != 'group_notification']
    df = df[~df['message'].str.contains('<Media omitted>')]
    df['message'] = df['message'].astype(str).str.strip()
    df = df[df['message'].str.len() > 1] # Remove single-character junk
    
    # 2. Filter: Only need 3 messages to avoid the crash
    user_counts = df['user'].value_counts()
    valid_users = user_counts[user_counts >= 3].index 
    df = df[df['user'].isin(valid_users)].copy()

    if df.empty or len(df['user'].unique()) < 2:
        return 0, None, None, None, []

    # 3. Encode & Vectorize (No stop_words to keep vocabulary full)
    le = LabelEncoder()
    y = le.fit_transform(df['user'])
    
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(df['message']).toarray()
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. XGBoost with "Scale Position Weight" logic
    # We use 'balanced' weights so we DON'T have to delete any data
    model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, eval_metric='mlogloss')
    
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Get common words for the app bypass
    all_words = " ".join(df['message']).lower().split()
    common_words = [word for word, count in Counter(all_words).most_common(20)]
    
    accuracy = round(model.score(X_test, y_test) * 100, 2)
    return accuracy, model, tfidf, le, common_words

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
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    if 'index' in df_percent.columns:
        df_percent.columns = ['User Name', 'Percent (%)']
    else:
        df_percent.columns = ['User Name', 'Percent (%)']
    return x, df_percent

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline = timeline.sort_values(['year', 'month_num'])

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline