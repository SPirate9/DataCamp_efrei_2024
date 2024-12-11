import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import html
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Charger le modèle Roberta au début
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Fonction pour analyser un texte
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(scores).item()  # 0: Négatif, 1: Neutre, 2: Positif
    sentiment_score = scores[0][sentiment].item()

    # Conversion du sentiment en note sur 5
    note = {0: 0, 1: 3, 2: 5}[sentiment]

    return sentiment, sentiment_score, note

# Fonction pour nettoyer les commentaires
def clean_comment(text):
    text = html.unescape(text)  # Décoder les entités HTML (par exemple, &#39; -> ')
    text = re.sub(r"<.*?>", "", text)  # Supprimer les balises HTML
    text = re.sub(r"\d+:\d+", "", text)  # Supprimer les horodatages (par exemple, 1:37)
    text = re.sub(r"\s+", " ", text)  # Réduire les espaces multiples
    return text.strip()

# Fonction pour afficher les résultats de l'analyse de sentiment
def display_sentiment_results(text, sentiment, score, note):
    sentiment_label = ["Négatif", "Neutre", "Positif"][sentiment]
    st.write("### Résultats")
    st.write(f"**Texte analysé :** {text}")
    st.write(f"**Sentiment détecté :** {sentiment_label}")
    st.write(f"**Score de confiance :** {score:.2f}")
    st.write(f"**Note sur 5 :** {note}")

# Fonction pour extraire les commentaires YouTube
def fetch_comments(video_id):
    youtube = build("youtube", "v3", developerKey="AIzaSyAUnpA_084X_LrgZP_bDIe-m6XzD6GW08g")
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=500)
    response = request.execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]
        cleaned_comment = clean_comment(comment["textDisplay"])  # Nettoyer le commentaire
        sentiment, score, note = analyze_sentiment(cleaned_comment)  # Utiliser la fonction existante
        sentiment_label = ["Négatif", "Neutre", "Positif"][sentiment]  # Label du sentiment
        comments.append({
            "Commentaire": cleaned_comment,
            "Likes": comment["likeCount"],
            "Sentiment": sentiment_label,
            "Score de confiance": score,
            "Note sur 5": note
        })
    return comments

# Application principale
st.title("Analyse des Sentiments des Joueurs pour Pokémon TCG Pocket")

# Onglets
tabs = st.tabs(["Explications", "Dashboard Tableau", "Google Play & Apple Store", "YouTube", "Analyse de Sentiment"])

# Onglet 2 : Google Play & Apple Store
with tabs[2]:
    st.header("Analyse des Stores")
    st.write("Cet onglet affichera des informations récupérées sur Google Play et Apple Store")
    all_reviews_notees = pd.read_csv('all_reviews_notees 1.csv') 

    st.write("### Aperçu des avis notés :")
    st.dataframe(all_reviews_notees) 

# Onglet 3 : Dashboard Tableau
with tabs[1]:
    st.header("Dashboard Tableau")
    st.write("Visualisez ici un tableau de bord Tableau intégré.")
    st.write("Travaux en cours...")

# Onglet 4 : Explications
with tabs[0]:
    st.header("Explications")
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

# Extraction et affichage des commentaires YouTube
with tabs[3]:
    st.header("Commentaires YouTube")
    st.write("Analyse des commentaires de la vidéo YouTube sur Pokémon TCG Pocket.")
    
    try:
        comments_df = pd.DataFrame(fetch_comments("16duP6ga_Q8"))
        if not comments_df.empty:
            st.write(f"Commentaires extraits : {len(comments_df)}")
            st.dataframe(comments_df)
        else:
            st.write("Aucun commentaire trouvé.")
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des commentaires : {e}")

with tabs[4]:  # Onglet "Analyse de Sentiment"
    st.header("Analyse de Sentiment avec Roberta")
    
    # Option pour l'utilisateur : Choisir entre texte libre ou fichier CSV
    option = st.radio("Choisissez une option :", ("Entrer un texte", "Charger un fichier CSV"))
    
    if option == "Entrer un texte":
        # Zone de saisie pour l'utilisateur
        user_input = st.text_area("Saisissez un texte ici :", "")

        if user_input.strip():  # Si un texte est saisi
            sentiment, score, note = analyze_sentiment(user_input)
            display_sentiment_results(user_input, sentiment, score, note)
    
    elif option == "Charger un fichier CSV":
        uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Aperçu des 10 premières lignes du fichier chargé :")
                st.dataframe(df.head(10))

                columns = df.columns.tolist()
                selected_column = st.selectbox("Étape 1 : Sélectionnez la colonne contenant les commentaires :", columns)

                if st.button("Confirmer la colonne sélectionnée"):
                    if selected_column:
                        st.write(f"Analyse de la colonne confirmée : **{selected_column}**")
                        comments = df[selected_column].astype(str)

                        results = []
                        progress_bar = st.progress(0)

                        for idx, comment in enumerate(comments):
                            sentiment, score, note = analyze_sentiment(comment)
                            sentiment_label = ["Négatif", "Neutre", "Positif"][sentiment]
                            results.append({
                                "Commentaire": comment,
                                "Sentiment": sentiment_label,
                                "Score de confiance": score,
                                "Note sur 5": note
                            })
                            progress_bar.progress((idx + 1) / len(comments))

                        results_df = pd.DataFrame(results)

                        st.subheader("Tableau des Commentaires avec Analyse de Sentiment")
                        st.dataframe(results_df)

                        st.download_button(
                            label="Télécharger le fichier avec l'analyse de sentiment",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name="commentaires_avec_analyse.csv",
                            mime="text/csv"
                        )

                        # Étape 3 : Dataviz
                        st.subheader("Visualisation des Résultats")

                        # Moyenne des notes
                        avg_note = results_df['Note sur 5'].mean()
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg_note,
                            title={"text": "Note Moyenne"},
                            gauge={"axis": {"range": [0, 5]}, "bar": {"color": "green"},
                                    "steps": [{"range": [0, 2], "color": "red"},
                                                {"range": [2, 4], "color": "yellow"},
                                                {"range": [4, 5], "color": "green"}]}
                        ))
                        st.plotly_chart(fig)

                        # Répartition des sentiments
                        st.write("Répartition des Sentiments")
                        sentiment_counts = results_df['Sentiment'].value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['red', 'gray', 'green'], ax=ax)
                        ax.set(title="Distribution des Sentiments", ylabel="Nombre de commentaires", xlabel="Sentiments")
                        for i, count in enumerate(sentiment_counts.values):
                            ax.text(i, count + 1, str(count), ha='center', fontsize=10)
                        st.pyplot(fig)
                        
                    else:
                        st.error("Veuillez sélectionner une colonne avant de confirmer.")
            except Exception as e:
                st.error(f"Erreur lors du chargement ou de l'analyse du fichier CSV : {e}")
