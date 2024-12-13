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
        st.write("""
        Ce projet vise à analyser les commentaires des utilisateurs sur plusieurs plateformes :
        - **Google Play & Apple Store :** Analyse des avis laissés par les utilisateurs des applications mobiles.
        - **Dashboard Tableau :** Un tableau de bord interactif pour regrouper et visualiser les données de manière dynamique.
        - **YouTube :** Extraction et analyse des commentaires des vidéos YouTube sur Pokémon TCG Pocket.
        - **Analyse de Sentiment :** Un onglet dédié à l'analyse de sentiment où vous pouvez soit saisir un texte pour analyse, soit télécharger un fichier CSV pour obtenir une analyse sur l'ensemble des commentaires présents dans ce fichier.
         """)
        st.write("""
        Les données ont été extraites à l'aide de Python et visualisées avec **Streamlit**. 
        Cette application permet ainsi une exploration interactive des commentaires utilisateurs sur diverses plateformes, avec un focus particulier sur l'analyse de sentiment via **Roberta**, un modèle de traitement du langage naturel.
        """)
    with tabs[1]:
        st.subheader("Dashboard Tableau")
        st.write("Visualisez ici un tableau de bord Tableau intégré représentant l'analyse des notes Apple store & Play store.")
        # Code HTML du tableau Tableau Public
        tableau_html = """
    <div class='tableauPlaceholder' id='viz1733948073609' style='position: relative'>
      <noscript>
        <a href='#'>
          <img alt='Tableau de bord 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Po&#47;Pokemon_review&#47;Tableaudebord1&#47;1_rss.png' style='border: none' />
        </a>
      </noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='Pokemon_review&#47;Tableaudebord1' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Po&#47;Pokemon_review&#47;Tableaudebord1&#47;1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='fr-FR' />
        <param name='filter' value='publish=yes' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1733948073609');
      var vizElement = divElement.getElementsByTagName('object')[0];
      if (divElement.offsetWidth > 800) {
        vizElement.style.width = '100%';
        vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
      } else if (divElement.offsetWidth > 500) {
        vizElement.style.width = '100%';
        vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
      } else {
        vizElement.style.width = '100%';
        vizElement.style.height = '1377px';
      }
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
        st.components.v1.html(tableau_html, height=1500)
    
    with tabs[2]:
        st.subheader("Google Play & Apple Store (Roberta)")
        st.write("Cet onglet affichera des informations récupérées sur Google Play et Apple Store. La dernière colonne (note) est la note calculée par notre modèle RoBERTa en fonction du sentiment détecté dans les avis. Ce modèle est plus précis pour l'analyse de sentiment que notre modèle de régression logistique, mais il peut être un peu plus lent en raison de sa taille et de sa complexité.")
        all_reviews_notees = pd.read_csv('data_source/samples_roberta.csv', sep=';') 
        st.write("### Aperçu des avis notés :")
        st.dataframe(all_reviews_notees.dropna())
        tableau_html = "<div class='tableauPlaceholder' id='viz1734095771528' style='position: relative'><noscript><a href='#'><img alt='Tableau de bord 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ro&#47;Roberta_app_play_store&#47;Tableaudebord1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Roberta_app_play_store&#47;Tableaudebord1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ro&#47;Roberta_app_play_store&#47;Tableaudebord1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='fr-FR' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1734095771528');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
        st.components.v1.html(tableau_html, height=1500)
    
    with tabs[3]:
        st.subheader("Google Play & Apple Store (Logistic Regression)")
        st.write("Cet onglet affichera des informations récupérées sur Google Play et Apple Store (~109000 avis). La dernière colonne (notes_calculees) est la note qu'a calculé notre modèle de regression logistique. Ce modèle est légèrement moins efficaces que le modele roberta mais il est beaucoup plus rapide.")
        all_reviews_notees = pd.read_csv('data_source/all_reviews_notees 2.csv', sep=';')
        st.write("### Aperçu des avis notés :")
        st.dataframe(all_reviews_notees)
        tableau_html = "<div class='tableauPlaceholder' id='viz1734012329999' style='position: relative'><noscript><a href='#'><img alt='Tableau de bord 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Cl&#47;Classeur2_17340108242920&#47;Tableaudebord1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Classeur2_17340108242920&#47;Tableaudebord1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Cl&#47;Classeur2_17340108242920&#47;Tableaudebord1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='fr-FR' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1734012329999');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1427px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"

        st.components.v1.html(tableau_html, height=1500)
    
    with tabs[4]:
        st.subheader("YouTube")
        st.write("Extraction des commentaires de la vidéo officielle du jeu Pokémon sur YouTube. Les commentaires sont ensuite analysés à l'aide de RoBERTa.")

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
        st.subheader("Analyse de Sentiment Individuel")
        user_input = st.text_area("Saisissez un texte à analyser :")
        if st.button("Analyser le Sentiment"):
            try:
                if user_input.strip():
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

        st.markdown("---")
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
