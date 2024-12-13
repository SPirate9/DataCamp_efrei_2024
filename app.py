import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
import html
import re

# Global variables for model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@st.cache_resource
def get_model_and_tokenizer():
    """Load RoBERTa sentiment analysis model"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def clean_comment(text):
    """Clean and preprocess comments"""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+:\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def analyze_sentiment(text):
    """Analyze sentiment of a given text"""
    tokenizer, model = get_model_and_tokenizer()
    text = clean_comment(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=514)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(scores).item()
    sentiment_score = scores[0][sentiment].item()
    sentiment_map = {0: 0, 1: 3, 2: 5}
    note = sentiment_map[sentiment]
    return {
        "sentiment": ["Négatif", "Neutre", "Positif"][sentiment],
        "score": sentiment_score,
        "note": note
    }

def analyze_comments_dataframe(df, column):
    """Analyze sentiment for an entire DataFrame"""
    results = []
    for comment in df[column].astype(str):
        analysis = analyze_sentiment(comment)
        results.append({
            "Commentaire": comment,
            "Sentiment": analysis["sentiment"],
            "Score de confiance": analysis["score"],
            "Note sur 5": analysis["note"]
        })
    return pd.DataFrame(results)

def fetch_comments(video_id="16duP6ga_Q8"):
    """Fetch comments from a YouTube video"""
    youtube = build("youtube", "v3", developerKey="AIzaSyAUnpA_084X_LrgZP_bDIe-m6XzD6GW08g")
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    response = request.execute()
    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]
        cleaned_comment = clean_comment(comment["textDisplay"])
        analysis = analyze_sentiment(cleaned_comment)
        comments.append({
            "Commentaire": cleaned_comment,
            "Likes": comment["likeCount"],
            "Sentiment": analysis["sentiment"],
            "Score de confiance": analysis["score"],
            "Note sur 5": analysis["note"]
        })
    return comments

def display_sentiment_visualizations(results_df):
    """Create and display sentiment visualizations"""
    avg_note = results_df['Note sur 5'].mean()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_note,
        title={"text": "Note Moyenne"},
        gauge={
            "axis": {"range": [0, 5]},
            "bar": {"color": "green"},
            "steps": [
                {"range": [0, 2], "color": "red"},
                {"range": [2, 4], "color": "yellow"},
                {"range": [4, 5], "color": "green"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

    st.markdown("### Distribution des Sentiments")
    sentiment_counts = results_df['Sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['red', 'gray', 'green'], ax=ax)
    ax.set(ylabel="Nombre de commentaires", xlabel="Sentiments")
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 1, str(count), ha='center', fontsize=10)
    st.pyplot(fig)

# Main application
def main():
    st.title("Analyse des Sentiments Pokémon TCG Pocket")

    # Tabs for navigation
    tabs = st.tabs(["Explications", "Dashboard Tableau", "Google Play & Apple Store (Roberta)", 
                    "Google Play & Apple Store (Logistic Regression)", "YouTube", "Analyse de Sentiment"])
    
    with tabs[0]:
        st.subheader("Explications")
        st.write("Bienvenue dans l'application d'analyse des sentiments. Naviguez à travers les onglets pour explorer les différentes fonctionnalités.")
    
    with tabs[1]:
        st.subheader("Dashboard Tableau")
        st.write("Fonctionnalité en développement.")
    
    with tabs[2]:
        st.subheader("Google Play & Apple Store (Roberta)")
        st.write("Fonctionnalité en développement.")
    
    with tabs[3]:
        st.subheader("Google Play & Apple Store (Logistic Regression)")
        st.write("Fonctionnalité en développement.")
    
    with tabs[4]:
        st.subheader("YouTube")
        st.write("Analyse des commentaires de la vidéo YouTube spécifiée.")
        
        if st.button("Analyser les Commentaires YouTube"):
            try:
                comments = fetch_comments()  # Fetch comments for the predefined video ID
                comments_df = pd.DataFrame(comments)
                st.dataframe(comments_df)
                display_sentiment_visualizations(comments_df)
                st.download_button(
                    label="Télécharger les résultats",
                    data=comments_df.to_csv(index=False).encode('utf-8'),
                    file_name="youtube_comments_sentiment.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
    
    with tabs[5]:
        st.subheader("Analyse de Sentiment")
    
    # Choix entre texte ou CSV
    option = st.radio("Choisissez une option :", ["Texte", "Fichier CSV"], horizontal=True)
    
    if option == "Texte":
        st.subheader("Analyse de Sentiment Individuel")
        user_input = st.text_area("Saisissez un texte à analyser :")
        if st.button("Analyser le Sentiment"):
            try:
                result = analyze_sentiment(user_input)
                st.markdown(f"""
                ### Résultats de l'Analyse
                - **Texte analysé :** {user_input}
                - **Sentiment :** {result['sentiment']}
                - **Score de confiance :** {result['score']:.2f}
                - **Note sur 5 :** {result['note']}
                """)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
    
    elif option == "Fichier CSV":
        st.subheader("Analyse de Fichier CSV")
        uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            column = st.selectbox("Sélectionnez la colonne à analyser", df.columns)
            
            if st.button("Analyser le Fichier CSV"):
                with st.spinner("Analyse en cours..."):
                    try:
                        results_df = analyze_comments_dataframe(df, column)
                        
                        st.dataframe(results_df)
                        
                        # Download option
                        st.download_button(
                            label="Télécharger les résultats",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations
                        display_sentiment_visualizations(results_df)
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {e}")

if __name__ == "__main__":
    main()