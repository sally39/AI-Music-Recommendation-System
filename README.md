# 🎵 AI-Powered Music Recommendation System  

A simple machine learning web app that recommends songs based on **content similarity** using lyrics and metadata.

---

## 🚀 Features  
- Personalized song recommendations  
- Works without user history (no cold-start issue)  
- Fast and scalable  
- Displays similarity scores  
- Interactive UI with Streamlit  

---

## 🧠 Tech Stack  
- Python  
- Pandas, NumPy, Scikit-learn  
- TF-IDF Vectorization  
- Nearest Neighbors Algorithm  
- Streamlit  

---

## 📌 How It Works
Converts song data into numerical vectors using TF-IDF
Uses cosine similarity to find similar songs
Recommends top matching songs with scores

---

## 📊 Dataset

Spotify Million Song Dataset

---
## ⚙️ Installation  & Usage

```bash
git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system
pip install -r requirements.txt

python train_model.py
streamlit run app.py
