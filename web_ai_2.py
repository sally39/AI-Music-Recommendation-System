import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import base64

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Music Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        height: 3rem;
        overflow: hidden;
    }
    .recommendation-card {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .album-cover {
        width: 100%;
        aspect-ratio: 1;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .button-container {
        display: flex;
        justify-content: center;
    }
    .similarity-score {
        color: #1DB954;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def train_model():
    """Train a memory-efficient content-based recommendation model."""
    try:
        # Check if model files already exist to avoid retraining
        if os.path.exists('df.pkl') and os.path.exists('nn_model.pkl') and os.path.exists('tfidf_matrix.pkl'):
            return
        
        df = pd.read_csv('spotify_millsongdata.csv')  # Load dataset
        
        # Handle missing values
        text_columns = ['lyrics', 'text']
        available_text_col = next((col for col in text_columns if col in df.columns), None)
        
        if available_text_col:
            # Fill missing text values
            df[available_text_col] = df[available_text_col].fillna('')
            feature_column = available_text_col
        elif 'artist' in df.columns and 'song' in df.columns:
            # Create features from metadata if no lyrics available
            df['features'] = df['artist'] + " " + df['song']
            feature_column = 'features'
        else:
            raise ValueError("No suitable text column found for feature extraction!")
        
        # Create feature vectors using TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[feature_column])
        
        # Nearest Neighbors Model
        nn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
        nn_model.fit(tfidf_matrix)
        
        # Save model & dataset
        pickle.dump(df, open('df.pkl', 'wb'))
        pickle.dump(nn_model, open('nn_model.pkl', 'wb'))
        pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
        pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    
    except Exception as e:
        st.error(f"Error training model: {e}")

def generate_album_cover(artist, song):
    """Generate SVG-based album cover based on artist and song name"""
    # Generate color based on hash of artist+song
    hue = sum(ord(c) for c in (artist + song)) % 360
    
    svg = f'''
    <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
        <rect width="150" height="150" fill="hsl({hue}, 70%, 30%)" />
        <circle cx="75" cy="75" r="30" fill="hsl({hue}, 70%, 20%)" />
        <circle cx="75" cy="75" r="10" fill="black" />
        <text x="75" y="130" font-family="Arial" font-size="12" fill="white" text-anchor="middle">{artist}</text>
    </svg>
    '''
    
    # Convert SVG to base64 for embedding
    svg_bytes = svg.encode('utf-8')
    encoded = base64.b64encode(svg_bytes).decode('utf-8')
    img_data = f"data:image/svg+xml;base64,{encoded}"
    
    return img_data

def recommend(song, n_recommendations=5):
    """Recommend songs using Nearest Neighbors model."""
    try:
        # Find the song in the dataset
        if song not in music['song'].values:
            st.warning(f"Song '{song}' not found in database. Please select another song.")
            return [], [], []
            
        index = music[music['song'] == song].index[0]
        song_vector = tfidf_matrix[index:index+1]
        
        # Get recommendations
        distances, indices = nn_model.kneighbors(song_vector, n_neighbors=n_recommendations+1)
        
        # Skip the first result (which is the song itself)
        recommended_music_names = [music.iloc[i].song for i in indices[0][1:]]
        recommended_music_artists = [music.iloc[i].artist for i in indices[0][1:]]
        
        # Calculate similarity scores (convert distances to similarity)
        similarity_scores = [round((1 - dist) * 100) for dist in distances[0][1:]]
        
        return recommended_music_names, recommended_music_artists, similarity_scores
        
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], [], []

# Main application
def main():
    st.markdown('<h1 class="main-header">AI-Powered Music Recommender</h1>', unsafe_allow_html=True)
    
    # Load or train model
    train_model()
    
    global music, nn_model, tfidf_matrix
    try:
        music = pickle.load(open('df.pkl', 'rb'))
        nn_model = pickle.load(open('nn_model.pkl', 'rb'))
        tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model files not found. Please check if training was successful.")
        return
    
    # Song selection
    selected_song = st.selectbox(
        "Type or select a song from the dropdown", 
        options=sorted(music['song'].values),
        index=0
    )
    
    # Get artist of selected song
    selected_artist = ""
    if selected_song in music['song'].values:
        selected_artist = music[music['song'] == selected_song]['artist'].values[0]
    
    # Button for recommendations
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_recommendations = st.button('Show Recommendations', key="show_recs")
    
    # Display recommendations
    if show_recommendations:
        with st.spinner("Generating recommendations..."):
            recommended_music_names, recommended_music_artists, similarity_scores = recommend(selected_song)
            
            if recommended_music_names:
                st.subheader(f"Songs similar to '{selected_song}' by {selected_artist}")
                
                # Display recommendations in a grid
                cols = st.columns(len(recommended_music_names))
                
                for i, (col, name, artist, score) in enumerate(zip(cols, recommended_music_names, recommended_music_artists, similarity_scores)):
                    with col:
                        # Generate album cover
                        album_img = generate_album_cover(artist, name)
                        
                        # Create recommendation card
                        st.markdown(f'''
                        <div class="recommendation-card">
                            <img src="{album_img}" class="album-cover" alt="{name} album cover">
                            <div class="recommendation-title">{name}</div>
                            <div>{artist}</div>
                            <div class="similarity-score">{score}% match</div>
                        </div>
                        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()