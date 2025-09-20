# AI-Powered-BBC-News-Explorer
Explore BBC News with AI — this project integrates semantic search (BERT embeddings), automatic summarization (BART), and text classification (Logistic Regression) into a smart Streamlit interface.
An interactive AI web app built with Streamlit that enables:
* **Semantic Search:** Find the most relevant BBC news articles to your query (supports Arabic & English).
* **Auto Classification:** Classify new articles into categories (trained on BBC dataset with Logistic Regression).
* **Summarization:** Generate concise summaries of long news articles.
* **Translation Layer:** Added deep-translator so that Arabic inputs are automatically translated before classification by Logistic Regression.


AI-Powered BBC News Explorer/
│
├── app.py                     # Streamlit app (main interface)
├── requirements.txt           # Dependencies to install
├── bbc-news-data.csv          # Dataset (BBC News, cleaned)
├── classifier_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer used with model
├── notebook/                  # Colab notebook used for training
│   └── bbc_nlp_pipeline.ipynb
└── images/                    # Screenshots for demo
    ├── ui_home.png
    ├── search_example.png
    └── classification_result.png

   ## Models & Components
* **Sentence-BERT (MiniLM-L12-v2)** → used for semantic search (supports multilingual queries).  
* **Summarizer (BART-large-cnn)** → used for generating summaries (English).  
* **Logistic Regression Classifier** → trained on BBC News dataset (English-only categories).  
* **TF-IDF Vectorizer** → converts cleaned text into numerical features.  
* **Deep Translator** → supports Arabic input by auto-translating before classification.  

 
 ## How to Run Locally
1. **Clone repository:**
   ```bash
   git clone https://github.com/<your-username>/AI-Powered-BBC-News-Explorer.git
   cd AI-Powered-BBC-News-Explorer

python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Mac/Linux
```
2. **Install requirements:**
```pip install -r requirements.txt```
4. Run the app:
```streamlit run app.py```

## Example Queries
1. Arabic:
"من فاز في مباراة كرة القدم الأخيرة؟" → gets translated → classified as Spo
"أسعار النفط وتأثيرها على الاقتصاد" → classified as Business

2. English:
"Prime minister announced new education policies" → classified as Politics
"Stock market faced a huge drop due to oil prices" → classified as Business

## Results
Logistic Regression achieved ~97% accuracy on the test split.
Confusion matrix & classification report are included in the notebook (notebook/bbc_nlp_pipeline.ipynb).
Summarization reduces long articles into 3–4 sentences for quick reading.

## Features Roadmap
  Add video/audio content support via speech-to-text (future).
  Enhance Arabic summarization with multilingual models.
  Deploy on HuggingFace Spaces or Streamlit Cloud for public demo.
  ## Screenshots
  Main UI of the application.

  
  
Classification result example.


