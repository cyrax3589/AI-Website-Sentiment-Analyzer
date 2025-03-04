Overview

The AI Website Sentiment Analyzer is a Python-based tool that scrapes textual data from websites, analyzes sentiment using both VADER and BERT-based sentiment models, and visualizes the results. It enables users to extract and evaluate text data from web pages, store it in CSV format, and generate insights via sentiment distribution charts and word clouds.

Features

Web Scraping: Extracts textual data from web pages using Selenium.

Sentiment Analysis:

VADER (for short, general text analysis)

BERT Transformer (for more nuanced sentiment detection)

Combined sentiment analysis for more accurate results

CSV Export: Saves extracted and analyzed data to a CSV file.

Data Visualization:

Sentiment distribution pie chart

Sentiment breakdown by source

Word cloud of frequently used terms

GUI Interface: Provides an easy-to-use Tkinter-based UI for user interaction.

Installation

Prerequisites

Ensure you have the following installed:

Python 3.7+

pip

Required Dependencies

Install the necessary Python libraries using:

pip install -r requirements.txt

Dependencies:

pandas

matplotlib

wordcloud

tkinter

beautifulsoup4

nltk

transformers

torch

selenium

webdriver-manager

How to Use

Run the Application

python sentiment_analyzer.py

Steps:

Click "Scrape Website & Save CSV" to enter a URL and extract reviews/text.

Click "Load and Analyze CSV" to visualize sentiment analysis results.

Folder Structure

![image](https://github.com/user-attachments/assets/d0bd8136-5ae0-4a0f-b594-d4bf05772e00)

Notes

The script runs Selenium in headless mode to avoid opening a browser.

Text exceeding 512 characters is truncated for BERT analysis due to model limitations.

Ensure Google Chrome and ChromeDriver are installed and updated.

License

This project is licensed under the MIT License.
