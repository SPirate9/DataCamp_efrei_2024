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

# Caching model loading to improve performance
@st.cache_resource
def load_sentiment_model():
    """Load RoBERTa sentiment analysis model"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Caching text cleaning to avoid redundant processing
@st.cache_data
def clean_comment(text):
    """Clean and preprocess comments"""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+:\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@st.cache_data
def analyze_sentiment(text, tokenizer, model):
    """Analyze sentiment of a given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=514)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(scores).item()
    sentiment_score = scores[0][sentiment].item()
    
    # Mapping sentiment to rating
    sentiment_map = {0: 0, 1: 3, 2: 5}
    note = sentiment_map[sentiment]
    
    return {
        "sentiment": ["Négatif", "Neutre", "Positif"][sentiment],
        "score": sentiment_score,
        "note": note
    }

@st.cache_data
def analyze_comments_dataframe(df, column, tokenizer, model):
    """Analyze sentiment for an entire DataFrame"""
    results = []
    for comment in df[column].astype(str):
        analysis = analyze_sentiment(comment, tokenizer, model)
        results.append({
            "Commentaire": comment,
            "Sentiment": analysis["sentiment"],
            "Score de confiance": analysis["score"],
            "Note sur 5": analysis["note"]
        })
    return pd.DataFrame(results)

def display_sentiment_visualizations(results_df):
    """Create and display sentiment visualizations"""
    # Average rating gauge
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

    # Sentiment distribution bar plot
    st.markdown("### Distribution des Sentiments")
    sentiment_counts = results_df['Sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['red', 'gray', 'green'], ax=ax)
    ax.set(ylabel="Nombre de commentaires", xlabel="Sentiments")
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 1, str(count), ha='center', fontsize=10)
    st.pyplot(fig)

def main():
    st.title("Analyse des Sentiments Pokémon TCG Pocket")
    
    # Load sentiment model
    tokenizer, model = load_sentiment_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choisissez une option", 
        [
            "Analyse de Texte", 
            "Analyse de Fichier CSV", 
            "Analyse YouTube", 
            "Dashboard Tableau"
        ])

    if app_mode == "Analyse de Texte":
        st.header("Analyse de Sentiment Individuel")
        user_input = st.text_area("Saisissez un texte à analyser :")
        
        if st.button("Analyser le Sentiment"):
            if user_input:
                result = analyze_sentiment(user_input, tokenizer, model)
                st.markdown(f"""
                ### Résultats de l'Analyse
                - **Texte analysé :** {user_input}
                - **Sentiment :** {result['sentiment']}
                - **Score de confiance :** {result['score']:.2f}
                - **Note sur 5 :** {result['note']}
                """)

    elif app_mode == "Analyse de Fichier CSV":
        st.header("Analyse de Sentiment par Fichier CSV")
        uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            column = st.selectbox("Sélectionnez la colonne à analyser", df.columns)
            
            if st.button("Analyser"):
                with st.spinner("Analyse en cours..."):
                    results_df = analyze_comments_dataframe(df, column, tokenizer, model)
                    
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

    # Add other modes (YouTube, Dashboard) similarly

if __name__ == "__main__":
    main()