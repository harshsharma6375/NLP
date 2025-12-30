# Optimized NLP Conversation Analytics

A high-performance, lightweight NLP pipeline designed for analyzing customer support transcripts. It extracts key insights like specific customer queries, sentiment, intents, and empathy *without* needing heavy GPU resources.

## 🚀 Key Features

*   **⚡ Lightweight & Fast**: Optimized codebase using `TextBlob` and `scikit-learn`. No heavy model downloads.
*   **Customer Query Extraction**: Automatically isolates and lists specific questions asked by the customer.
*   **Named Entity Recognition (NER)**: Extracts Brands, Products, and Dates using `spaCy`.
*   **Product Detection**: Identifies tech products (iPhone, Galaxy, etc.) using hybrid matching.
*   **Intent Recognition**: Uses TF-IDF + SVM for accurate intent classification.
*   **Empathy Detection**: Analyzes agent responses for empathetic language patterns.

## 🛠️ Tech Stack

*   **Python**: Core orchestration
*   **spaCy**: NER & Pattern Matching
*   **TextBlob**: Efficient Sentiment Analysis
*   **Scikit-Learn**: Intent Classification

## ⚡ How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Pipeline**
    ```bash
    python src/pipeline.py
    ```

3.  **View Output**
    Results are saved to `outputs/analysis_result.json`.

## 📊 Output Structure

The system produces a clean JSON output focusing on actionable insights:

```json
{
    "summary": {
        "overall_sentiment": "Negative",
        "primary_intent": "Complaint",
        "empathy_score": 0.8,
        "products_detected": ["iPhone 14", "Samsung Galaxy S23"]
    },
    "customer_queries": [
        "but it has not been delivered yet?",
        "How do I return this?"
    ],
    "details": { ... }
}
```

---
*Built for speed, reliability, and ease of use.*
