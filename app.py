import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="WhatsApp Chat Analyzer")

st.sidebar.title("WhatsApp Chat Analyzer")

# 2. File Uploader
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp Chat file (.txt)")

if uploaded_file is not None:
    # Read and Preprocess
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # 3. Validation Check: Ensure the file was parsed correctly
    if df.empty:
        st.error("The uploaded file could not be parsed. Please ensure it is a valid WhatsApp export (.txt) with the correct date/time format.")
    else:
        # Fetch unique users for dropdown
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis for:", user_list)

        # --- Sidebar Button logic using session state to prevent reload issues ---
        if st.sidebar.button("Show Analysis"):
            st.session_state['analysis_clicked'] = True

        # Only run the analysis if the button was clicked OR if it was already active
        if st.session_state.get('analysis_clicked'):
            
            # --- LEVEL 1: TOP STATISTICS (KPI CARDS) ---
            st.title("Top Statistics")
            
            num_messages = df.shape[0] if selected_user == 'Overall' else df[df['user'] == selected_user].shape[0]
            
            # Calculate words
            if selected_user != 'Overall':
                temp_df = df[df['user'] == selected_user]
            else:
                temp_df = df
                
            words = []
            for message in temp_df['message']:
                words.extend(str(message).split())
                
            num_media = temp_df[temp_df['message'] == '<Media omitted>\n'].shape[0]
            num_links = temp_df[temp_df['message'].str.contains('http')].shape[0]

            # Displaying Metrics in 4 columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", num_messages)
            with col2:
                st.metric("Total Words", len(words))
            with col3:
                st.metric("Media Shared", num_media)
            with col4:
                st.metric("Links Shared", num_links)

            st.markdown("---")

            # --- LEVEL 2: TABBED DASHBOARD ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 User Activity",
                "⏳ Timeline",
                "💬 Content Analysis",
                "🤖 ML Insights",
            ])

            with tab1:
                if selected_user == 'Overall':
                    st.header("Most Busy Users")
                    x, new_df = helper.most_busy_users(df)
                    fig, ax = plt.subplots()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        ax.bar(x.index, x.values, color='#25D366')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col_b:
                        st.dataframe(new_df)
                else:
                    st.info("User-specific activity analysis is shown in the Timeline tab.")

            with tab2:
                st.header("Monthly Activity Timeline")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green', marker='o')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with tab3:
                st.header("Wordcloud & Emojis")
                col_x, col_y = st.columns(2)
                with col_x:
                    st.subheader("Wordcloud")
                    df_wc = helper.create_wordcloud(selected_user, df)
                    if df_wc is not None:
                        fig, ax = plt.subplots()
                        ax.imshow(df_wc)
                        plt.axis("off")
                        st.pyplot(fig)
                    else:
                        st.write("No words found for this user.")
                with col_y:
                    st.subheader("Most Common Emojis")
                    emoji_df = helper.emoji_helper(selected_user, df)
                    if not emoji_df.empty:
                        st.dataframe(emoji_df)
                    else:
                        st.write("No emojis found.")

            with tab4:
                st.header("🤖 Advanced Machine Learning Insights")
                
                # --- Sentiment Analysis (BERT) ---
                st.subheader("Message Sentiment Analysis (BERT)")
                with st.spinner("Analyzing tone of recent messages..."):
                    results = helper.get_sentiment_analysis(selected_user, df)
                    if results:
                        sentiments = [res['label'] for res in results]
                        pos_count = sentiments.count('POSITIVE')
                        neg_count = sentiments.count('NEGATIVE')
                        
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric("Positive Tone Count", pos_count)
                            st.metric("Negative Tone Count", neg_count)
                        with col_s2:
                            fig_s, ax_s = plt.subplots()
                            ax_s.pie([pos_count, neg_count], labels=['Positive', 'Negative'], 
                                     autopct='%1.1f%%', colors=['#25D366', '#FF4B4B'])
                            st.pyplot(fig_s)
                    else:
                        st.info("Not enough text data to perform sentiment analysis.")
                
                st.markdown("---")
                
                # --- User Prediction (XGBoost) ---
                st.subheader("User Identification Model (XGBoost)")
                if st.button("Train / Refresh Model", key="train_button"):
                    with st.spinner("Training XGBoost model... this may take a moment."):
                        accuracy, model, vectorizer, le = helper.train_user_prediction_model(df)
                        st.session_state['accuracy'] = accuracy
                        st.session_state['model'] = model
                        st.session_state['vectorizer'] = vectorizer
                        st.session_state['le'] = le
                        st.session_state['model_trained'] = True
                        st.success("Model Training Complete!")

                if st.session_state.get('model_trained'):
                    st.metric("Model Accuracy", f"{st.session_state['accuracy']}%")
                    st.markdown("---")
                    st.subheader("Predict User from Message")
                    sample_message = st.text_input("Type a message to identify the sender:", key="predict_input")
                    if sample_message:
                        features = st.session_state['vectorizer'].transform([sample_message])
                        prediction = st.session_state['model'].predict(features)
                        predicted_user = st.session_state['le'].inverse_transform(prediction)[0]
                        st.success(f"The model predicts this sender is: **{predicted_user}**")
                else:
                    st.info("Click the 'Train' button above to initialize the Machine Learning model.")