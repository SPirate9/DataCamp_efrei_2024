from google_play_scraper import reviews, Sort, app
import praw
from datetime import datetime
from app_store_scraper import AppStore
import praw

# ID de l'application "Pokémon TCG Live"
app_id = 'jp.pokemon.pokemontcgp'

# Récupérer les avis (jusqu'à 200 récents)
result, _ = reviews(
    app_id,
    lang='fr',  # Avis en anglais
    country='fr',  # Pays : Royaume-Uni
    sort=Sort.NEWEST,  # Trier par les plus récents
    count=1  # Nombre d'avis à récupérer
)

# Afficher les avis récupérés
for review in result:
    print(f"Auteur : {review['userName']}")
    print(f"Note : {review['score']}")
    print(f"Avis : {review['content']}")
    print(f"Date : {review['at']}")
    print(f"Nombre de J'aime : {review.get('thumbsUpCount', 'N/A')}")
    print(f"URL de l'avis : {review.get('url', 'N/A')}")
    print(f"ID de l'avis : {review.get('reviewId', 'N/A')}")
    print(f"Version de l'application : {review.get('appVersionName', 'N/A')}")
    print(f"Réponse du développeur : {review.get('developerResponse', 'Aucune réponse')}")
    print(f"Date de réponse du développeur : {review.get('repliedAt', 'N/A')}")
    print(f"Avis traduit : {review.get('translatedContent', 'Aucun avis traduit')}")
    print("-" * 50)



# Connexion à Reddit via l'API
reddit = praw.Reddit(client_id='g4Qn1BhPN4eXIZxhs302gQ',
                     client_secret='-Qu81qyQb_x2r2OGi9bYoEEP3u2g_g',
                     user_agent='votre_user_agent')

# URL du post spécifique
post_url = 'https://www.reddit.com/r/Farfa/comments/1ggbdlg/my_honest_review_to_pokemon_tcg_pocket_as_a_guy/?tl=fr'

# Récupérer le post spécifique via l'URL
post = reddit.submission(url=post_url)

# Fonction pour enregistrer dans un fichier texte
def save_to_txt(file_name, text):
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(text + "\n")

# Extraire les informations du post
post_info = f"Titre du post : {post.title}\n"
post_info += f"Auteur du post : {post.author.name if post.author else 'Anonyme'}\n"
post_info += f"Date de création (UTC) : {datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
post_info += f"Score (nombre de upvotes) : {post.score}\n"
post_info += f"Contenu du post : \n{post.selftext}\n"
post_info += f"URL du post : {post.url}\n"
post_info += f"Nombre de commentaires : {post.num_comments}\n"
post_info += "="*50 + "\n"

# Sauvegarder les informations du post
save_to_txt("post_comments.txt", post_info)

# Extraire tous les commentaires (remplacer 'load more' par des vrais commentaires)
post.comments.replace_more(limit=None)  # `limit=None` permet de charger tous les commentaires sans restriction

# Fonction récursive pour afficher toutes les réponses d'un commentaire et les enregistrer
def save_comments_recursive(comment, level=1):
    # Formatage du commentaire
    comment_text = f"{'  ' * level}Auteur : {comment.author.name if comment.author else 'Anonyme'}\n"
    comment_text += f"{'  ' * level}Date du commentaire (UTC) : {datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    comment_text += f"{'  ' * level}Score du commentaire : {comment.score}\n"
    comment_text += f"{'  ' * level}Contenu du commentaire : \n{'  ' * level}{comment.body}\n"
    comment_text += f"{'  ' * level}Nombre de réponses : {len(comment.replies)}\n"
    
    # Sauvegarder le commentaire dans le fichier
    save_to_txt("post_comments.txt", comment_text)

    # Si le commentaire a des réponses, on les sauvegarde aussi
    if len(comment.replies) > 0:
        for reply in comment.replies:
            save_comments_recursive(reply, level + 1)

# Itérer à travers tous les commentaires récupérés
for i, comment in enumerate(post.comments.list(), start=1):
    save_comments_recursive(comment)  # Sauvegarder le commentaire et ses réponses



# Définir l'application à scraper (id de l'app et pays)
app = AppStore(country="fr", app_name="le-jcc-pokemon-pocket", app_id=6479970832)

# Nombre d'avis à récupérer par appel
reviews_per_page = 200

# Initialiser une liste pour stocker tous les avis
all_reviews = []

# Demander plusieurs pages d'avis
for page in range(1, 11):  # Par exemple, pour récupérer 10 pages d'avis
    app.review(how_many=reviews_per_page)
    
    # Si les avis sont récupérés, on les ajoute à la liste
    if app.reviews:
        all_reviews.extend(app.reviews)
    else:
        break  # Sortir de la boucle si aucun avis n'est trouvé

# Afficher les avis dans la console
for review in all_reviews:
    print(f"Auteur : {review['userName']}")
    print(f"Note : {review['rating']}")
    print(f"Commentaire : {review['review']}")
    print("-" * 50)

print(f"Nombre total d'avis récupérés : {len(all_reviews)}")



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

# Parcourir les posts et récupérer leurs commentaires
all_comments = []
for url in post_urls:
    comments = get_comments_from_post(url)
    all_comments.extend(comments)
    print(f"Commentaires récupérés pour {url}: {len(comments)}")

# Sauvegarder les résultats dans un fichier texte
with open("comments_output.txt", "w", encoding="utf-8") as f:
    for comment in all_comments:
        f.write(f"Auteur : {comment['author']}\n")
        f.write(f"Score : {comment['score']}\n")
        f.write(f"Commentaire : {comment['body']}\n")
        f.write("="*50 + "\n")

print(f"Nombre total de commentaires récupérés: {len(all_comments)}")
print("Les commentaires ont été sauvegardés dans 'comments_output.txt'.")



