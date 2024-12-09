import praw
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# Connexion à Supabase
url = "https://uxylpimtlwyonwartkrn.supabase.co"  # Remplace avec ton URL Supabase
key = "YeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV4eWxwaW10bHd5b253YXJ0a3JuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM3NzQ3OTcsImV4cCI6MjA0OTM1MDc5N30.qrOkhFljAqhioQ5bM6gSn-XZQu78IE6KrzhzlrZU4fU"  # Remplace avec ta clé Supabase
supabase: Client = create_client(url, key)

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
    'https://www.reddit.com/r/iosgaming/comments/1gfbocv/pok%C3%A9mon_tcg_pocket/'
]

# Fonction pour récupérer les commentaires d'un post
def get_comments_from_post(post_url):
    post = reddit.submission(url=post_url)
    post.comments.replace_more(limit=None)  # Charger tous les commentaires

    comments = []
    for comment in post.comments.list():
        comment_info = {
            'author': comment.author.name if comment.author else 'Anonyme',
            'score': comment.score,  # Le nombre de votes (score)
            'body': comment.body
        }
        comments.append(comment_info)
    
    return comments

# Récupérer les commentaires pour chaque URL
def insert_comments_into_supabase(comments):
    for comment in comments:
        data = {
            'author': comment['author'],
            'score': comment['score'],
            'body': comment['body']
        }
        # Insert dans la table "comments" de Supabase
        supabase.table('comments').insert(data).execute()

# Fonction pour récupérer les commentaires stockés depuis Supabase
def fetch_comments_from_supabase():
    response = supabase.table('comments').select('*').execute()
    return response.data

# Récupérer les commentaires pour chaque URL
def fetch_reddit_data():
    all_comments = []
    for url in post_urls:
        comments = get_comments_from_post(url)
        all_comments.extend(comments)
    return all_comments

# Application principale
st.title("Analyse Multi-Plateformes")

# Onglets
tabs = st.tabs(["Reddit", "Google Play & Apple Store", "Dashboard Power BI", "Explications"])

# Onglet 1 : Reddit
with tabs[0]:
    st.header("Commentaires Reddit")
    st.write("Données extraites des discussions Reddit sur Pokémon TCG Pocket.")

    # Vérifier si les commentaires sont déjà dans Supabase
    comments_from_db = fetch_comments_from_supabase()
    if comments_from_db:
        df_comments = pd.DataFrame(comments_from_db)
        st.write("Commentaires récupérés depuis Supabase :", df_comments.head())
    else:
        st.write("Aucun commentaire trouvé dans la base de données. Récupération depuis Reddit...")
        comments = fetch_reddit_data()
        insert_comments_into_supabase(comments)
        df_comments = pd.DataFrame(comments)
        st.write("Commentaires récupérés et insérés dans Supabase :", df_comments.head())

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
