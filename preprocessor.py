import re
import pandas as pd

def preprocess(data):
    # Flexible pattern for date/time
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:AM|PM|am|pm)?\s?-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    # Clean up the date string and convert
    df['message_date'] = df['message_date'].str.replace(' - ', '', regex=False)
    
    # Using errors='coerce' will prevent crashing if a line is weird
    df['date'] = pd.to_datetime(df['message_date'], errors='coerce')
    
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]: 
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message', 'message_date'], inplace=True)
    
    # Drop any rows where the date failed to parse
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    # Optional but helpful for other charts:
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    return df