import os
import re
from dotenv import load_dotenv
import groq  # Assure-toi que cette lib est bien installée (via requirements.txt)

# Dossier contenant les fichiers d’entretien
INPUT_FOLDER = "data_4"
THEMES_FILE = "thèmes_globaux.txt"
TOP10_FILE = "top_10.txt"

def load_config():
    """Charge la clé API depuis .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La clé API GROQ_API_KEY n'est pas définie dans le fichier .env.")
    return api_key

def initialize_groq_client(api_key):
    """Initialise le client Groq."""
    return groq.Groq(api_key=api_key)

def extraire_themes():
    themes = []
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
                contenu = f.read()
                # Cherche ** thème ** avec expression régulière
                themes += re.findall(r"\*\*(.*?)\*\*", contenu)
    return [theme.strip() for theme in themes if theme.strip()]

def sauvegarder_themes(themes):
    with open(THEMES_FILE, "w", encoding="utf-8") as f:
        for theme in themes:
            f.write(f"{theme}\n")

def appeler_llama(client, prompt):
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant en analyse sociologique."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def generer_top10(client, themes, max_themes=300):
    # Réduction à 500 thèmes max, dédoublonnés
    themes_reduits = list(dict.fromkeys(themes))[:max_themes]

    prompt = (
        "Voici une liste de thèmes extraits d'entretiens sociologiques :\n"
        + "\n".join(themes_reduits) +
        "\n\nPeux-tu identifier les 10 thèmes les plus récurrents, "
        "en regroupant ceux qui se ressemblent sémantiquement ? "
        "Présente-les sous forme de liste numérotée avec un court commentaire sur chacun."
    )
    return appeler_llama(client, prompt)

def proposer_analyse(client, themes, max_themes=30):
    # Encore plus court ici, on veut juste des exemples représentatifs
    themes_reduits = list(dict.fromkeys(themes))[:max_themes]

    prompt = (
        "Voici une liste de thèmes sociologiques issus d'entretiens :\n"
        + "\n".join(themes_reduits) +
        "\n\nPropose une manière d'analyser ces thèmes dans une perspective sociologique, "
        "en tenant compte des approches qualitatives."
    )
    return appeler_llama(client, prompt)


if __name__ == "__main__":
    api_key = load_config()
    client = initialize_groq_client(api_key)

    themes = extraire_themes()
    sauvegarder_themes(themes)

    top10 = generer_top10(client, themes)
    with open(TOP10_FILE, "w", encoding="utf-8") as f:
        f.write("Top 10 des thèmes récurrents :\n\n" + top10 + "\n\n")

        analyse = proposer_analyse(client, themes)
        f.write("Proposition d'analyse sociologique :\n\n" + analyse)

    print("Analyse terminée. Résultats dans 'thèmes_globaux.txt' et 'top_10.txt'.")
