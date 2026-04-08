from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def create_wordcloud(selected_user, df):
    # Filter by user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # 1. Remove "group_notification" and "Media omitted"
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # 2. Ensure all messages are strings and join them
    text = temp['message'].astype(str).str.cat(sep=" ")

    # 3. Handle case where there's no text to show
    if not text.strip():
        return None

    # 4. Generate WordCloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(text)
    return df_wc

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def train_user_prediction_model(df):
    # 1. Clean data for ML (Filtering noise)
    # We only want actual messages, not 'Media omitted' or 'group_notification'
    df = df[df['user'] != 'group_notification']
    df = df[df['message'].astype(str).str.strip() != '<Media omitted>']
    df = df.dropna(subset=['user', 'message'])

    # 2. Encode Target (Converting User Names to Numbers)
    le = LabelEncoder()
    y = le.fit_transform(df['user'])
    
    # 3. Vectorize Features (Converting Text to Numbers)
    # We use TF-IDF to focus on unique words used by different people
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(df['message']).toarray()
    
    # 4. Train-Test Split (Standard 80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. XGBoost Model Configuration
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # 6. Training
    model.fit(X_train, y_train)
    
    # 7. Accuracy Score
    accuracy = model.score(X_test, y_test)
    
    # Return everything needed for the App's session state
    return round(accuracy * 100, 2), model, tfidf, le

def most_busy_users(df):
    # Get top 5 users
    x = df['user'].value_counts().head()
    # Calculate percentage
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'User Name', 'user': 'Percent (%)'})
    return x, df_percent

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
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # Extract emojis from each message
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df