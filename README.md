# DataCamp Groupe 9

## Projet DataCamp du groupe 9

### Membres du groupe :

- **Saad SHAHZAD**
- **Joseph DESTAT GUILLOT**
- **Yanis MOHELLIBI**
- **Anira José MENDES PEREIRA**

### Commande d'installation
pip install -r requirements.txt

### Sinon:
pip install pandas scikit-learn nltk numpy transformers wordcloud matplotlib google-play-scraper app_store_scraper praw streamlit google-api-python-client torch seaborn plotly

### Lien du l'application Streamlit :
 
https://pokemon-comments-viewer.streamlit.app/
 
### Contenu du dossier compressé :

- un fichier `Code_Webscraping_RegresisionLogistique.ipynb` sur le scraping des données et la création du modèle de régression logistique
- un fichier `app.py` sur l'application streamlit, et sur le modèle ROBERTA
- un dossier data_source contenant les fichiers d'entrées en csv

### Trois tableaux de bords :
- un sur les notes données par les utilisateurs des avis applestore et playstore
- lien : https://public.tableau.com/app/profile/jos.mendes.pereira/viz/Pokemon_review/Tableaudebord1?publish=yes
- un sur les notes prédites par le modèle de regression logistique à partir uniquement des commentaires des avis applestore et  playstore, sans prendre en compte les notes des utilisateurs, uniquement en prenant en compte le sentiment du commentaire, avec un nuage de mots.
- lien : https://public.tableau.com/app/profile/jos.mendes.pereira/viz/Classeur2_17340108242920/Tableaudebord1?publish=yes
- un sur les avis applestore et  playstore dont le sentiment a été prédit par le modèle ROBERTA
- lien : https://public.tableau.com/app/profile/jos.mendes.pereira/viz/Roberta_app_play_store/Tableaudebord1?publish=yes