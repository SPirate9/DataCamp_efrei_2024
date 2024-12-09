import praw
import pandas as pd
import streamlit as st


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
all_comments = []
for url in post_urls:
    comments = get_comments_from_post(url)
    all_comments.extend(comments)
    print(f"Commentaires récupérés pour {url}: {len(comments)}")

# Créer un DataFrame avec les données
df_comments = pd.DataFrame(all_comments)

st.title('Commentaires Reddit sur Pokémon TCG Pocket')
st.write(df_comments.head())
score_filter = st.slider('Filtrer par score', min_value=1, max_value=5, value=3)
filtered_data = df_comments[df_comments['score'] >= score_filter]
st.write(filtered_data)
st.subheader('Graphique des scores')
st.bar_chart(filtered_data['score'].value_counts())

