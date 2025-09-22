# AI Website Sentiment Analyzer  

The **AI Website Sentiment Analyzer** is a Python-based tool that scrapes textual data from websites, performs sentiment analysis using **VADER** and **BERT-based models**, and visualizes the results. It allows users to extract and evaluate text from web pages, save it in CSV format, and gain insights through sentiment charts and word clouds.  

---

## Features  

- **Web Scraping**: Extracts text from web pages using Selenium.  
- **Sentiment Analysis**:  
  - **VADER** – quick sentiment detection for short/general text.  
  - **BERT Transformer** – deeper, context-aware sentiment analysis.  
  - **Combined sentiment analysis** for improved accuracy.  
- **CSV Export**: Saves scraped and analyzed data to CSV.  
- **Data Visualization**:  
  - Sentiment distribution pie chart.  
  - Sentiment breakdown by source.  
  - Word cloud of frequently used terms.  
- **GUI Interface**: Easy-to-use Tkinter-based user interface.  

---

## Folder Structure

project/
│── sentiment_analyzer.py

│── requirements.txt

│── README.md

│── data/

│   └── scraped_data.csv

│── visuals/

│   ├── sentiment_pie.png

│   ├── sentiment_breakdown.png

│   └── wordcloud.png
