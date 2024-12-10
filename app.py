import praw
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import html
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 

# Connexion à Reddit via l'API
reddit = praw.Reddit(client_id='g4Qn1BhPN4eXIZxhs302gQ',
                     client_secret='-Qu81qyQb_x2r2OGi9bYoEEP3u2g_g',
                     user_agent='votre_user_agent')

# Liste des URLs des posts Reddit
post_urls = [
    'https://www.reddit.com/r/Farfa/comments/1ggbdlg/my_honest_review_to_pokemon_tcg_pocket_as_a_guy/?tl=fr',
    'https://www.reddit.com/r/PokemonTCG/comments/1giugtx/what_are_yall_thoughts_on_pkmn_tcg_pocket/',
    'https://www.reddit.com/r/gachagaming/comments/1go9pxd/pok%C3%A9mon_tcg_pocket_first_impressions/',
    'https://www.reddit.com/r/PTCGP/comments/1fvoqag/pok%C3%A9mon_tcg_pocket_a_freetoplay_game_done_right/',
    'https://www.reddit.com/r/jeuxvideo/comments/1gkn0kj/pok%C3%A9mon_pocket_le_jeu_mobile_pok%C3%A9mon_fait/',
    'https://www.reddit.com/r/iosgaming/comments/1gfbocv/pok%C3%A9mon_tcg_pocket/',
    'https://www.reddit.com/r/nintendo/comments/1gjfqdz/pok%C3%A9mon_tcg_pocket_surpasses_12m_in_four_days/'
    'https://www.reddit.com/r/Games/comments/1gfao81/pok%C3%A9mon_trading_card_game_pocket_is_available_now/'
]

# Fonction pour récupérer les commentaires d'un post
def get_comments_from_post(post_url):
    post = reddit.submission(url=post_url)
    post.comments.replace_more(limit=None)  # Charger tous les commentaires

    comments = []
    for comment in post.comments.list():
        comment_info = {
            'score': comment.score,  # Le nombre de votes (score)
            'body': comment.body
        }
        comments.append(comment_info)
    
    return comments

# Récupérer les commentaires pour chaque URL
def fetch_reddit_data():
    all_comments = []
    for url in post_urls:
        comments = get_comments_from_post(url)
        all_comments.extend(comments)
        print(f"Commentaires récupérés pour {url}: {len(comments)}")
    return pd.DataFrame(all_comments)

# Application principale
st.title("Analyse Multi-Plateformes")

# Onglets
tabs = st.tabs(["Reddit", "Google Play & Apple Store", "Dashboard Power BI", "Explications", "YouTube Test", "Analyse de Sentiment"])

# Onglet 1 : Reddit
with tabs[0]:
    st.header("Commentaires Reddit")
    st.write("Données extraites des discussions Reddit sur Pokémon TCG Pocket.")
    
    df_comments = fetch_reddit_data()
    st.write("Aperçu des commentaires :", df_comments.head())
    
    score_filter = st.slider("Filtrer par score", min_value=min(df_comments["score"]), 
                              max_value=max(df_comments["score"]), value=3)
    filtered_data = df_comments[df_comments["score"] >= score_filter]
    st.write("Commentaires filtrés :", filtered_data)
    st.bar_chart(filtered_data["score"].value_counts())

# Onglet 2 : Google Play & Apple Store
with tabs[1]:
    st.header("Analyse des Stores")
    st.write("Cet onglet affichera des informations récupérées sur Google Play et Apple Store (données à intégrer).")
    # Exemple de placeholder :
    st.write("Travaux en cours...")

# Onglet 3 : Dashboard Power BI
with tabs[2]:
    st.header("Dashboard Power BI")
    st.write("Visualisez ici un tableau de bord Power BI intégré.")
    # Vous pouvez intégrer un lien ou une iframe Power BI ici.
    st.write("Travaux en cours...")

# Onglet 4 : Explications
with tabs[3]:
    st.header("Explications")
    st.write("""
    Ce projet vise à analyser les commentaires des utilisateurs sur plusieurs plateformes :
    - **Reddit :** pour collecter des commentaires et votes d'utilisateurs passionnés.
    - **Google Play & Apple Store :** pour analyser les avis sur les applications mobiles.
    - **Power BI :** pour regrouper et visualiser les données de manière interactive.
    """)
    st.write("Les données ont été extraites à l'aide de Python et visualisées avec Streamlit.")

api_key = "AIzaSyAUnpA_084X_LrgZP_bDIe-m6XzD6GW08g"
youtube = build("youtube", "v3", developerKey=api_key)

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


with tabs[4]:
    st.header("YouTube Test")
    st.write("""
    Analyse des commentaires d'une vidéo YouTube officielle de Pokémon.
    Vous pouvez observer les données collectées et visualisées dynamiquement.
    """)

    # Vidéo cible
    video_id = "16duP6ga_Q8"  # Remplacez par l'ID de la vidéo à analyser

    # Fonction pour récupérer tous les commentaires avec pagination
    def fetch_all_comments(video_id):
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()
        
        while response:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "Commentaire": comment["textDisplay"],
                    "Likes": comment["likeCount"]
                })

            # Passer à la page suivante s'il y a un token
            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response["nextPageToken"]
                )
                response = request.execute()
            else:
                break

        return comments

    # Fonction pour nettoyer les commentaires
    def clean_comment(text):
        text = html.unescape(text)  # Décoder les entités HTML (&#39; -> ')
        text = re.sub(r"<.*?>", "", text)  # Supprimer les balises HTML
        text = re.sub(r"\d+:\d+", "", text)  # Supprimer les horodatages (1:37)
        text = re.sub(r"\s+", " ", text)  # Supprimer les espaces multiples
        return text.strip()  # Supprimer les espaces superflus

    # Extraction des commentaires
    st.subheader("Commentaires extraits")
    try:
        all_comments = fetch_all_comments(video_id)
        
        # Nettoyage des commentaires
        for comment in all_comments:
            comment["Commentaire"] = clean_comment(comment["Commentaire"])
        
        # Vérification des doublons : suppression des commentaires identiques
        comments_df = pd.DataFrame(all_comments)
        comments_df = comments_df.drop_duplicates(subset="Commentaire", keep="first")  # Supprime les doublons basés sur le texte du commentaire

        # Affichage des commentaires dans Streamlit
        if not comments_df.empty:
            st.write(f"Nombre total de commentaires extraits (sans doublons) : {len(comments_df)}")
            st.dataframe(comments_df)  # Afficher sous forme de tableau interactif
        else:
            st.write("Aucun commentaire trouvé pour cette vidéo.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'extraction des commentaires : {e}")


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
    if sentiment == 0:
        note = 0
    elif sentiment == 1:
        note = 3
    else:
        note = 5

    return sentiment, sentiment_score, note

with tabs[5]:  # Onglet "Analyse de Sentiment"
    st.header("Analyse de Sentiment avec Roberta")
    st.write("Entrez un texte et obtenez son sentiment ainsi qu'une note associée.")

    # Zone de saisie pour l'utilisateur
    user_input = st.text_area("Saisissez un texte ici :", "")

    if user_input.strip():  # Si un texte est saisi
        sentiment, score, note = analyze_sentiment(user_input)
        sentiment_label = ["Négatif", "Neutre", "Positif"][sentiment]

        # Résultats de l'analyse
        st.write("### Résultats")
        st.write(f"**Texte analysé :** {user_input}")
        st.write(f"**Sentiment détecté :** {sentiment_label}")
        st.write(f"**Score de confiance :** {score:.2f}")
        st.write(f"**Note sur 5 :** {note}")