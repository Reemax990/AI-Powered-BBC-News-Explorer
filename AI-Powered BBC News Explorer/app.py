import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import joblib
from deep_translator import GoogleTranslator

#.venv\Scripts\activate >> for term
#streamlit run app.py
def translate_if_arabic(text):
    try:
        # لو النص فيه حروف عربية يترجم
        if any("\u0600" <= c <= "\u06FF" for c in text):
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text 
# =====================
# تحميل النماذج
# =====================
# موديل البحث الذكي (Multilingual MiniLM)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# موديل التلخيص (انجليزي)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# موديل التصنيف (joblib) مدرب مسبقا
clf = joblib.load("classifier_model.pkl")      # LR model
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =====================
# البيانات
# =====================
# لازم تكون محضرة embeddings من قبل
import pandas as pd
df = pd.read_csv("bbc-news-data.csv", encoding="latin-1")
embeddings = model.encode(df["content"].tolist(), convert_to_tensor=True)

# =====================
# واجهة المستخدم
# =====================
st.title("AI-Powered BBC News Explorer")

# البحث الذكي
st.header("🔍 Intelligent Search")
query = st.text_input("اكتب استفسارك هنا:")
if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = similarities.argsort()[-3:][::-1]
    results = df.iloc[top_idx][["title", "category", "content"]].copy()
    results["similarity"] = similarities[top_idx]

    for _, row in results.iterrows():
        st.subheader(f"{row['title']} ({row['category']})")
        st.write(row["content"][:300] + "...")
        st.caption(f"Similarity Score: {row['similarity']:.4f}")

        if st.button(f"تلخيص {row['title']}"):
            summary = summarizer(row["content"], max_length=130, min_length=30, do_sample=False)
            st.success(summary[0]['summary_text'])

# التصنيف الآلي
st.header("📝 Auto Classification")
new_text = st.text_area("اكتب نص جديد للتصنيف:")
if st.button("صنّف النص"):
    if new_text.strip() != "":
        processed_text = translate_if_arabic(new_text)
        new_embedding = vectorizer.transform([processed_text])
        prediction = clf.predict(new_embedding)[0]
        st.success(f"✅ تم تصنيف النص تحت: **{prediction}**")
    else:
        st.warning("⚠️ رجاءً اكتب نص أولاً.")
