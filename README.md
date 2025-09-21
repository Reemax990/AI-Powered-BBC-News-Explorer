# AI-Powered-BBC-News-Explorer
An interactive AI web app built with Streamlit that enables:
* **Semantic Search:** Find the most relevant BBC news articles to your query (supports Arabic & English).
* **Auto Classification:** Classify new articles into categories (trained on BBC dataset with Logistic Regression).
* **Summarization:** Generate concise summaries of long news articles.
* **Translation Layer:** Added deep-translator so that Arabic inputs are automatically translated before classification by Logistic Regression.

## Project Structure
### AI-Powered BBC News Explorer
* app.py                     # Streamlit app (main interface)
*  requirements.txt           # Dependencies to install
*  bbc-news-data.csv          # Dataset (BBC News, cleaned)
*   classifier_model.pkl       # Trained Logistic Regression model
*   tfidf_vectorizer.pkl       # TF-IDF vectorizer used with model
  ###   notebook
*    bbc_nlp_pipeline.ipynb
### images                  
* ui_home.png
*  classification_result.png

   ## Models & Components
* **Sentence-BERT (MiniLM-L12-v2)** → used for semantic search (supports multilingual queries).  
* **Summarizer (BART-large-cnn)** → used for generating summaries (English).  
* **Logistic Regression Classifier** → trained on BBC News dataset (English-only categories).  
* **TF-IDF Vectorizer** → converts cleaned text into numerical features.  
* **Deep Translator** → supports Arabic input by auto-translating before classification.  

 
 ## How to Run Locally
1. **Clone repository:**
   ```bash git clone https://github.com/<your-username>/AI-Powered-BBC-News-Explorer.git cd AI-Powered-BBC-News-Explorer```
2. **Create virtual environment (recommended):**
```python -m venv .venv .venv\Scripts\activate   # On Windows source .venv/bin/activate  # On Mac/Linux```
4. **Install requirements:**
```pip install -r requirements.txt```
5. **Run the app:**
```streamlit run app.py```

## Example Queries
1. Arabic:
  
  
  "من فاز في مباراة كرة القدم الأخيرة؟" → gets translated → classified as Spo
"أسعار النفط وتأثيرها على الاقتصاد" → classified as Business


2.  English:
* "Prime minister announced new education policies" → classified as Politics
* "Stock market faced a huge drop due to oil prices" → classified as Business

## Results
* Logistic Regression achieved ~97% accuracy on the test split.
* Confusion matrix & classification report are included in the notebook (notebook/bbc_nlp_pipeline.ipynb).
* Summarization reduces long articles into 3–4 sentences for quick reading.

## Features Roadmap
 1. Add video/audio content support via speech-to-text (future).
 2. Enhance Arabic summarization with multilingual models.
 3. Deploy on HuggingFace Spaces or Streamlit Cloud for public demo.
  ## Screenshots

   ### 🖥️ Web App Interface
* ![Interface](images/UI1.PNG)

#### Semantic Search
*   ![Interface](images/search.PNG)

  #### Summarization
*    ![Interface](images/summary.PNG)

#### Classification
*    ![Interface](images/classify.PNG)


  ### 📊 Model Accuracy (Logistic Regression)
* ![Accuracy](images/LR.png)

