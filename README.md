🎵 AI-Powered Music Recommendation System

A machine learning-based web app that recommends songs using content-based filtering. It analyzes lyrics and metadata to suggest similar songs—even for new or less popular tracks.

🚀 Features
🎯 Personalized recommendations
🧠 No user history required (solves cold-start problem)
⚡ Fast and scalable
📊 Similarity scores
🌐 Simple Streamlit UI
🧠 Tech Stack
Python
Pandas, NumPy, Scikit-learn
TF-IDF Vectorization
Nearest Neighbors
Streamlit
⚙️ Setup
git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system
pip install -r requirements.txt
▶️ Run
# Train model
python train_model.py

# Launch app
streamlit run app.py
📌 How It Works
Converts song text into vectors using TF-IDF
Finds similar songs using cosine similarity
Returns top recommendations with match scores
📊 Dataset
Spotify Million Song Dataset
