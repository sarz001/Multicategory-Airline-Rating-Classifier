# ✈️ Multicategory Airline Rating Classifier

A Naive Bayes-based Natural Language Processing (NLP) system for performing **multi-aspect sentiment analysis** and **rating prediction** on airline, airport, seat, and lounge reviews. Given a textual review, the system predicts ratings (1–10 scale) across several service quality dimensions.

---

## 📌 Overview

This project performs **fine-grained rating prediction** based on review content using a probabilistic model. It focuses on:

- Airline service experience
- Cabin staff and seating
- Inflight food, entertainment, and value for money

The system processes review text using custom preprocessing and then applies a **Naive Bayes classifier** to predict multiple category-specific ratings.

---

## 🎯 Features

- 🔹 **Multi-category prediction** for the following aspects:
  - Overall Experience
  - Seat Comfort
  - Cabin Staff
  - Food & Beverages
  - Inflight Entertainment
  - Value for Money

- 🧠 **Text Preprocessing & Custom Tokenization**:
  - Punctuation removal, lowercasing
  - Stopword removal (using NLTK)
  - Handles negations and compound sentences (e.g., clauses joined by *"but"*, *"and"*)
  - Captures multi-word phrases like *"not comfortable"*, *"very helpful"*

- 📊 **Naive Bayes Model**:
  - Prior and conditional probability computation
  - Laplace (add-one) smoothing
  - Logarithmic scoring to prevent underflow
  - Category-wise MAP (maximum a posteriori) estimation

- 💡 **Handles edge cases**:
  - Missing ratings
  - Unseen words
  - Short and noisy reviews

---

## 📁 Dataset Structure

The project expects a CSV file named `airline.csv` with the following structure:

| content | overall_rating | seat_comfort_rating | cabin_staff_rating | food_beverages_rating | inflight_entertainment_rating | value_money_rating |
|---------|----------------|---------------------|--------------------|------------------------|-------------------------------|---------------------|
| "Service was great but seats were uncomfortable" | 7 | 5 | 9 | 7 | 6 | 6 |

Each numeric column should contain ratings from 1 to 10.

---

## 🛠 Dependencies

Make sure you have the following installed:

```bash
pip install pandas numpy nltk
