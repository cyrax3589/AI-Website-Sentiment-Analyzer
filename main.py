import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askstring
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import textwrap
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import re

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
sentiment_transformer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

#BERT analyzer
def get_transformer_sentiment(text):
    try:
        result = sentiment_transformer(text[:512])[0]  # Truncate to prevent token length issues
        score = float(result['score'])
        label = float(result['label'].split()[0])  # Convert '1/2/3/4/5
        
        # # More strict thresholds for BERT model
        if label >= 4.0:
            return "POSITIVE", score * 0.8  # Reduce positive confidence
        elif label <= 2.5:
            return "NEGATIVE", score * 1.2  # Increase negative confidence
        else:
            return "NEUTRAL", score * 0.5
    except Exception as e:
        return "NEUTRAL", 0.0

#VADER analyzer
def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.1:
        return "POSITIVE", scores['compound']
    elif scores['compound'] <= -0.02:
        return "NEGATIVE", scores['compound']
    else:
        return "NEUTRAL", scores['compound']

#Combined analyzer
def get_combined_sentiment(text):
    vader_sentiment, vader_score = get_vader_sentiment(text)
    transformer_sentiment, transformer_score = get_transformer_sentiment(text)
    
    # Give more weight to negative sentiments
    if vader_sentiment == "NEGATIVE" or transformer_sentiment == "NEGATIVE":
        return "NEGATIVE", min(vader_score, transformer_score)
    elif vader_sentiment == transformer_sentiment:
        return vader_sentiment, (vader_score + transformer_score) / 2
    else:
        # Bias towards negative sentiment when models disagree
        if abs(vader_score) > transformer_score:
            return vader_sentiment, vader_score
        else:
            return transformer_sentiment, transformer_score

# Scrape and save to CSV
def scrape_and_save_to_csv(url):
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize the driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        
        # Wait for JavaScript content to load
        time.sleep(5)

        # Get page content
        reviews = []
        elements = driver.find_elements(By.CSS_SELECTOR, 'p, span, div')
        
        for element in elements:
            text = element.text.strip()
            if 50 < len(text) < 1000:
                reviews.append(text)

        data = []
        for review_text in reviews:
            wrapped_texts = textwrap.wrap(review_text, width=500)
            for wrapped_text in wrapped_texts:
                sentiment, score = get_combined_sentiment(wrapped_text)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                user_id = "Unknown"
                location = "Unknown"

                data.append([wrapped_text, sentiment, "Web", timestamp, user_id, location, score])

        driver.quit()

        if data:
            save_to_csv(data)
        else:
            messagebox.showwarning("Warning", "No valid review data found on the page!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to scrape URL: {e}")
        if 'driver' in locals():
            driver.quit()

# Save to CSV
def save_to_csv(data):
    csv_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if csv_filename:
        with open(csv_filename, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Text", "Sentiment", "Source", "Date/Time", "User ID", "Location", "Confidence Score"])
            writer.writerows(data)
        messagebox.showinfo("Success", f"Data saved to {csv_filename}")


def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        messagebox.showwarning("Warning", "No file selected!")
        return
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sentiment_counts = df["Sentiment"].value_counts()
        colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "blue"}
        pie_colors = [colors.get(label, "gray") for label in sentiment_counts.index]

        exp = [0.1] + [0] * (len(sentiment_counts) - 1)
        
        axes[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%",
            colors=pie_colors, shadow=True, startangle=140, explode=[0.1] + [0] * (len(sentiment_counts) - 1),
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={"edgecolor": "black", "linewidth": 1.2})

        axes[0].set_title("Sentiment Distribution")

        if "Source" in df.columns:
            sentiment_by_source = df.groupby(["Source", "Sentiment"]).size().unstack().fillna(0)
            entiment_by_source = sentiment_by_source.reindex(["NEGATIVE", "NEUTRAL", "POSITIVE"], axis=1, fill_value=0)

            sentiment_by_source.plot(kind='bar', stacked=False, ax=axes[1], 
                                    color=['red', 'green', 'blue'], edgecolor='black', width=0.7)
            
            axes[1].set_title("Sentiment Count by Source", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Source", fontsize=12, fontweight='bold')
            axes[1].set_ylabel("Count", fontsize=12, fontweight='bold')
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
            axes[1].tick_params(axis='x', rotation=90)
        else:
            axes[1].text(0.5, 0.5, "No 'Source' column found in CSV", fontsize=12, ha='center')
            axes[1].set_title("Sentiment Count by Source")
        
        if "Text" in df.columns:
            text_data = " ".join(df["Text"].astype(str)).lower()
            text_data = re.sub(r'[^a-zA-Z\s]', '', text_data)  
            
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(width=500, height=800, background_color="white",
                                  stopwords=stopwords, max_words=500, colormap='viridis').generate(text_data)

            axes[2].imshow(wordcloud, interpolation="bilinear")
            axes[2].axis("off")
            axes[2].set_title("Most Frequent Words in Reviews", fontsize=14, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, "No 'Text' column found in CSV", fontsize=12, ha='center')

        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Success", "CSV Analysis Completed!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {e}")


# GUI Setup
app = tk.Tk()
app.title("AI Website Sentiment Analyzer")
app.geometry("1080x720")
app.configure(bg="#1e1e2e")
ttk.Style().configure("TButton", font=("Arial", 12, "bold"), padding=10)
ttk.Style().configure("TLabel", font=("Arial", 14, "bold"), foreground="white", background="#282c34")

def scrape_url():
    url = askstring("Enter URL", "Paste the URL to scrape:")
    if url:
        scrape_and_save_to_csv(url)

frame = tk.Frame(app, bg="#282c34")
frame.pack(expand=True)
title_label = ttk.Label(frame, text="AI Sentiment Analyzer", font=("Arial", 18, "bold"))
title_label.pack(pady=20)
scrape_button = tk.Button(app, text="Scrape Website & Save CSV", command=scrape_url, bg="lightblue", font=("Arial", 14))
scrape_button.pack(pady=10)

load_csv_button = tk.Button(app, text="Load and Analyze CSV", command=load_csv, bg="lightgreen", font=("Arial", 14))
load_csv_button.pack(pady=10)

app.mainloop()

