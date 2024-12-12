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

### Contenu du dossier compressé :

- un fichier `Code_Webscraping_RegresisionLogistique.ipynb` sur le scraping des données et la création du modèle de régression logistique
- un fichier `app.py` sur l'application streamlit, et sur le modèle ROBERTA

### Trois tableaux de bords :

- un sur les notes données par les utilisateurs des avis applestore et playstore 
- un sur les notes prédites par le modèle de regression logistique à partir uniquement des commentaires des avis applestore et 	playstore, sans prendre en compte les notes des utilisateurs, uniquement en prenant en compte le sentiment du commentaire
- un sur les avis applestore et  playstore dont le sentiment a été prédit par le modèle ROBERTA
- un sur les avis YouTube dont le sentiment a été prédit par le modèle ROBERTA, avec en plus un nuage de mots sur les commentaires 	des avis google
 
