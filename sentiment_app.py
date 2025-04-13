# Sentiment Analysis Tool - Implementation

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tweepy
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1. DATA COLLECTION MODULE
class DataCollector:
    def __init__(self, twitter_credentials=None):
        self.twitter_credentials = twitter_credentials
        if twitter_credentials:
            auth = tweepy.OAuthHandler(twitter_credentials['consumer_key'], 
                                       twitter_credentials['consumer_secret'])
            auth.set_access_token(twitter_credentials['access_token'], 
                                 twitter_credentials['access_token_secret'])
            self.twitter_api = tweepy.API(auth)
    
    def collect_twitter_data(self, query, count=100):
        """Collect tweets based on query"""
        if not hasattr(self, 'twitter_api'):
            raise ValueError("Twitter credentials not provided")
        
        tweets = []
        for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query, 
                                  tweet_mode='extended', lang='en').items(count):
            tweets.append({
                'text': tweet.full_text,
                'created_at': tweet.created_at,
                'user': tweet.user.screen_name,
                'source': 'twitter'
            })
        return pd.DataFrame(tweets)
    
    def scrape_reviews(self, url, review_element_selector):
        """Simple web scraper for review sites"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        reviews = []
        for element in soup.select(review_element_selector):
            reviews.append({
                'text': element.text.strip(),
                'source': 'web_scrape'
            })
        return pd.DataFrame(reviews)
    
    def load_csv_data(self, file_path):
        """Load data from CSV file"""
        return pd.read_csv(file_path)

# 2. TEXT PREPROCESSING MODULE
class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
    
    def clean_text(self, text):
        """Clean the text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove user mentions and hashtags for Twitter data
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove non-alphanumeric characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if enabled
            if self.remove_stopwords:
                tokens = [word for word in tokens if word not in stop_words]
            
            # Lemmatize if enabled
            if self.lemmatize:
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            # Join tokens back into text
            cleaned_text = ' '.join(tokens)
            return cleaned_text
        return ''
    
    def preprocess_dataframe(self, df, text_column):
        """Preprocess text in the dataframe"""
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        return df

# 3. SENTIMENT ANALYSIS ENGINE
class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx] if self.labels is not None else 0
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=16, epochs=4, learning_rate=2e-5):
        """Train the sentiment analysis model"""
        # Create datasets
        train_dataset = self.SentimentDataset(train_texts, train_labels, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.SentimentDataset(val_texts, val_labels, self.tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_dataloader = None
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     labels=labels)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader:
                self.model.eval()
                val_loss = 0
                predictions = []
                true_labels = []
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, 
                                           attention_mask=attention_mask, 
                                           labels=labels)
                        
                        val_loss += outputs.loss.item()
                        
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        true = labels.cpu().numpy()
                        
                        predictions.extend(preds)
                        true_labels.extend(true)
                
                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Validation loss: {avg_val_loss:.4f}")
                print(classification_report(true_labels, predictions))
                
                self.model.train()
    
    def predict(self, texts, batch_size=16):
        """Predict sentiment for a list of texts"""
        self.model.eval()
        predictions = []
        
        # Create dataset with no labels
        dataset = self.SentimentDataset(texts, None, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        # Convert numerical predictions to labels
        labeled_predictions = [self.label_map[pred] for pred in predictions]
        return labeled_predictions

    def save_model(self, path):
        """Save the model to disk"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path):
        """Load the model from disk"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)

# 4. VISUALIZATION DASHBOARD
class SentimentVisualizer:
    def __init__(self):
        self.color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    
    def run_dashboard(self, data):
        """Run the Streamlit dashboard"""
        st.title("Sentiment Analysis Dashboard")
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Date filter if date column exists
        if 'created_at' in data.columns:
            min_date = data['created_at'].min().date()
            max_date = data['created_at'].max().date()
            date_range = st.sidebar.date_input("Date Range", 
                                            [min_date, max_date], 
                                            min_value=min_date, 
                                            max_value=max_date)
            if len(date_range) == 2:
                start_date, end_date = date_range
                data = data[(data['created_at'].dt.date >= start_date) & 
                           (data['created_at'].dt.date <= end_date)]
        
        # Source filter if source column exists
        if 'source' in data.columns:
            sources = data['source'].unique()
            selected_sources = st.sidebar.multiselect("Sources", sources, default=sources)
            if selected_sources:
                data = data[data['source'].isin(selected_sources)]
        
        # Display overall sentiment distribution
        st.header("Overall Sentiment Distribution")
        sentiment_counts = data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                    color='Sentiment', color_discrete_map=self.color_map)
        st.plotly_chart(fig)
        
        # Display sentiment over time if date column exists
        if 'created_at' in data.columns:
            st.header("Sentiment Over Time")
            data['date'] = data['created_at'].dt.date
            sentiment_time = data.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(sentiment_time, x='date', y='count', color='sentiment',
                         color_discrete_map=self.color_map)
            st.plotly_chart(fig)
        
        # Word cloud section
        st.header("Word Frequency Analysis")
        sentiment_filter = st.selectbox("Select Sentiment", ['all'] + list(self.color_map.keys()))
        
        if sentiment_filter == 'all':
            filtered_data = data
        else:
            filtered_data = data[data['sentiment'] == sentiment_filter]
        
        # Display sample of the data
        st.header("Sample Data")
        st.dataframe(filtered_data[['cleaned_text', 'sentiment']].head(10))

# 5. MAIN APPLICATION
def main():
    # Example usage
    st.title("Sentiment Analysis Tool")
    st.write("This application analyzes sentiment from various data sources.")
    
    # Upload data or use sample data
    st.header("1. Data Input")
    data_source = st.radio("Select data source:", 
                         ["Upload CSV", "Use sample data", "Collect from Twitter", "Web scraping"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
        else:
            st.info("Please upload a CSV file.")
            return
            
    elif data_source == "Use sample data":
        # Create some sample data
        sample_texts = [
            "I absolutely love this product! It's amazing.",
            "This service is okay, nothing special.",
            "Terrible experience, would not recommend to anyone.",
            "Not bad, but could be better.",
            "The best purchase I've made all year!",
            "Completely disappointed with the quality.",
            "Average product for the price.",
            "Outstanding customer service, very impressed!",
            "Waste of money, don't buy it.",
            "Pretty satisfied with my purchase."
        ]
        sample_dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
        sample_sources = ["twitter", "review", "twitter", "review", "twitter", 
                         "review", "twitter", "review", "twitter", "review"]
        
        data = pd.DataFrame({
            'text': sample_texts,
            'created_at': sample_dates,
            'source': sample_sources
        })
        st.success("Sample data loaded!")
        
    elif data_source == "Collect from Twitter":
        st.warning("Note: Twitter API credentials are required for this feature.")
        query = st.text_input("Enter search query:")
        count = st.slider("Number of tweets to collect:", 10, 1000, 100)
        
        # Would need to handle Twitter API credentials securely in a real app
        if st.button("Collect Tweets") and query:
            st.info(f"This would collect {count} tweets about '{query}'")
            # For demo purposes, create sample tweets
            sample_tweets = [
                f"I think {query} is amazing! #awesome",
                f"Not sure about {query}, seems overrated",
                f"Absolutely hate {query}, waste of time",
                f"{query} changed my life for the better",
                f"Mixed feelings about {query}"
            ]
            data = pd.DataFrame({
                'text': sample_tweets * 20,
                'created_at': pd.date_range(start="2023-01-01", periods=100, freq='H'),
                'source': "twitter"
            })
            st.success("Sample Twitter data created!")
        else:
            st.info("Enter a search query and click 'Collect Tweets'")
            return
            
    elif data_source == "Web scraping":
        url = st.text_input("Enter URL to scrape:")
        selector = st.text_input("CSS Selector for review elements:", 
                              value=".review-text")
        
        if st.button("Scrape Reviews") and url:
            st.info(f"This would scrape reviews from {url}")
            # For demo purposes, create sample scraped reviews
            sample_reviews = [
                "Great value for money, highly recommend!",
                "Product arrived damaged, disappointed.",
                "Exactly what I was looking for.",
                "Shipping took forever, but product is decent.",
                "Perfect fit, very happy with purchase."
            ]
            data = pd.DataFrame({
                'text': sample_reviews * 20,
                'created_at': pd.date_range(start="2023-01-01", periods=100, freq='H'),
                'source': "web_scrape"
            })
            st.success("Sample scraped data created!")
        else:
            st.info("Enter a URL and click 'Scrape Reviews'")
            return
    
    # Preprocess data
    st.header("2. Text Preprocessing")
    text_column = st.selectbox("Select text column:", data.columns)
    
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    lemmatize = st.checkbox("Lemmatize text", value=True)
    
    preprocessor = TextPreprocessor(remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    
    if st.button("Preprocess Data"):
        with st.spinner("Preprocessing text data..."):
            data = preprocessor.preprocess_dataframe(data, text_column)
            st.success("Preprocessing complete!")
            
            # Show sample of preprocessed data
            st.subheader("Sample of preprocessed data:")
            st.dataframe(data[[text_column, 'cleaned_text']].head())
    else:
        st.info("Click 'Preprocess Data' to clean the text data.")
        return
    
    # Analyze sentiment
    st.header("3. Sentiment Analysis")
    
    analysis_method = st.radio("Select analysis method:", 
                             ["Demo (pre-trained model)", "Train new model"])
    
    if analysis_method == "Demo (pre-trained model)":
        # For demo, we'll assign random sentiments
        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                # Simulated sentiment analysis
                sentiments = np.random.choice(['positive', 'neutral', 'negative'], 
                                             size=len(data), 
                                             p=[0.6, 0.3, 0.1])
                data['sentiment'] = sentiments
                st.success("Sentiment analysis complete!")
                
                # Show sample with sentiments
                st.subheader("Sample results:")
                st.dataframe(data[[text_column, 'cleaned_text', 'sentiment']].head(10))
                
                # Visualize results
                visualizer = SentimentVisualizer()
                visualizer.run_dashboard(data)
    
    elif analysis_method == "Train new model":
        st.info("In a real application, this would allow you to train a custom BERT model.")
        st.warning("Training deep learning models requires significant computational resources.")
        
        label_column = st.selectbox("Select label column (if available):", 
                                  ['None'] + list(data.columns))
        
        if label_column != 'None':
            if st.button("Train Model"):
                st.info("Training simulation started (this would take hours in a real scenario)")
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.1)  # Simulate training time
                    progress_bar.progress(i + 1)
                
                # Simulated results after training
                sentiments = np.random.choice(['positive', 'neutral', 'negative'], 
                                             size=len(data), 
                                             p=[0.5, 0.3, 0.2])
                data['sentiment'] = sentiments
                
                st.success("Model training complete!")
                visualizer = SentimentVisualizer()
                visualizer.run_dashboard(data)
        else:
            st.error("Label column is required for training. Please select a column with sentiment labels.")
    
if __name__ == "__main__":
    main()
