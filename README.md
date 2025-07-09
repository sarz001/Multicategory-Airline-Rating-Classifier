# Multicategory Airline Rating Classifier

This project implements a Naive Bayes-based Natural Language Processing (NLP) system for performing **multi-aspect sentiment analysis** and **rating prediction** on airline, airport, seat, and lounge reviews. Given a textual review, the system predicts ratings (1–10 scale) across several service quality dimensions.

---

## Overview

This project performs **fine-grained rating prediction** based on user-generated review content using a probabilistic model. It focuses on:

- Airline service experience  
- Cabin staff and seating  
- Inflight food, entertainment, and value for money  

The system preprocesses the review text and then applies a Naive Bayes classifier to predict multiple category-specific ratings.

---

## Features

- **Multi-category prediction** for the following aspects:
  - Overall Experience  
  - Seat Comfort  
  - Cabin Staff  
  - Food & Beverages  
  - Inflight Entertainment  
  - Value for Money  

- **Text Preprocessing and Tokenization**:
  - Removal of punctuation and conversion to lowercase  
  - Stopword removal using NLTK  
  - Handling of negations and compound sentences (e.g., "but", "and")  
  - Phrase detection for expressions like "not comfortable", "very helpful"  

- **Naive Bayes Classification**:
  - Prior and conditional probability estimation  
  - Laplace (add-one) smoothing for unseen words  
  - Use of logarithmic scoring to prevent numerical underflow  
  - Category-wise MAP (maximum a posteriori) estimation  

- **Robustness Features**:
  - Graceful handling of missing or invalid ratings  
  - Smoothing for words not seen in training  
  - Resistance to noise and short reviews  

---

## Dataset Structure

The system expects a dataset named `airline.csv` with the following columns:

| content | overall_rating | seat_comfort_rating | cabin_staff_rating | food_beverages_rating | inflight_entertainment_rating | value_money_rating |
|---------|----------------|---------------------|--------------------|------------------------|-------------------------------|---------------------|
| "Service was great but seats were uncomfortable" | 7 | 5 | 9 | 7 | 6 | 6 |

- `content`: Free-text review  
- Other columns: Integer ratings from 1 to 10  

---

## Dependencies

Install the required packages:

```bash
pip install pandas numpy nltk
