# Optimized NLP Conversation Analytics

A high-performance, lightweight NLP pipeline designed for analyzing customer support transcripts. It extracts key insights like specific customer queries, sentiment, intents, and empathy *without* needing heavy GPU resources.

## 🚀 Key Features

*   **⚡ Lightweight & Fast**: Optimized codebase using `TextBlob` and `scikit-learn`. No heavy model downloads.
*   **Customer Query Extraction**: Automatically isolates and lists specific questions asked by the customer.
*   **Named Entity Recognition (NER)**: Extracts Brands, Products, and Dates using `spaCy`.
*   **Product Detection**: Identifies tech products (iPhone, Galaxy, etc.) using hybrid matching.
*   **Intent Recognition**: Uses TF-IDF + SVM for accurate intent classification.
*   **Empathy Detection**: Analyzes agent responses for empathetic language patterns.

### **Methodology (Master Prompt Compliance):**
- **Strict Parameter Logic**:
  - **Sentiment**: Capped at **0.90** confidence. "Negative" forced if repeated failures exist.
  - **Intent**: Enforces `Complaint > Refund Issue > Inquiry`. If frustration is detected across multiple issues, `Complaint` is chosen with realistic confidence (0.6-0.8).
  - **Empathy**: Validated against agent apologies. Score is **≥ 0.85** if multiple empathetic signals (apology, understanding, reassurance) exist.
- **Product Normalization**: Products are strictly filtered and formatted to **Title Case** (e.g., "Lenovo Laptop").
- **Hybrid Models**: 
  - Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
  - Intent: `facebook/bart-large-mnli` (Zero-Shot)
  - Empathy: `j-hartmann/emotion-english-distilroberta-base`
- **Transparency**: Output JSON is strictly business-ready, prioritizing correctness and consistency.

---

### **Interview-Ready Explanation**
> "I designed the system to value **business correctness** over raw model output. I implemented a 'Severity Hierarchy' where frustration upgrades a simple 'Refund Issue' to a 'Complaint', and I deliberately recalibrated confidence scores to avoid false certainty."

---


# ✅ SESSION-BASED MEMORY & CONTEXT RULES

This system now implements a production-grade **Session Architecture**.

## 🔹 SESSION MEMORY DESIGN
- **Unique Session ID**: Each conversation gets a UUID.
- **Running Summary**: We maintain a structured `summary_cache` (not full history) for analysis.
- **Strict Context Rule**: The model analyses ONLY the (Running Summary + Latest Message). It does **NOT** read the full chat every time, preventing "Context Overflow" and "Duplicate Sentiment Weighting".

## 🔹 INTERVIEW EXPLANATION (Why this is Interview-Ready)
> "Har chat ka ek unique session ID hota hai. Main poori chat baar-baar model ko dene ke bajay ek running summary cache karta hoon aur low temperature (deterministic) pe model chalata hoon, taaki intent aur sentiment sirf context ke basis pe accurately nikle."

**Why this is strong:**
- **Context Mixing**: Eliminated.
- **False Escalation**: Eliminated (by checking summary history).
- **Hallucination**: Eliminated (Deterministic Settings).
- **Stability**: High.

## 🔹 LOW TEMPERATURE / DETERMINISM
All BERT models are configured with `top_k=1` (Greedy Decoding) to ensure the output is **Deterministic** and **Stable**.

## 🛠️ Tech Stack

*   **Python**: Core orchestration
*   **spaCy**: NER & Pattern Matching
*   **TextBlob**: Efficient Sentiment Analysis
*   **Scikit-Learn**: Intent Classification

## ⚡ How to Run

### Option 1: Using `uv` (Fastest & Simplest)
No setup required. Just run:

```powershell
uv run src/pipeline.py
```

### Option 2: Standard Python
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
