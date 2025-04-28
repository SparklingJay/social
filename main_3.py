import os
from dotenv import load_dotenv
import groq
import requests
from bs4 import BeautifulSoup
from collections import Counter

# --- Fonctions ---

def load_config():
    """Charge la clé API et l'URL du Google Doc depuis le fichier .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    google_doc_url = os.getenv("GOOGLE_DOC_URL")

    if not api_key:
        raise ValueError("La clé API GROQ_API_KEY n'est pas définie dans le fichier .env.")
    if not google_doc_url:
        raise ValueError("L'URL du Google Doc GOOGLE_DOC_URL n'est pas définie dans le fichier .env.")
    
    return api_key, google_doc_url

def initialize_groq_client(api_key):
    """Initialise le client Groq."""
    return groq.Groq(api_key=api_key)

def fetch_text_from_google_doc(url):
    """Récupère le texte brut d'un Google Doc public via l'URL."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Erreur lors de la récupération du document : {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

def split_text(text, max_chunk_size=12000):
    """Découpe le texte pour éviter de dépasser la limite de tokens."""
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def detect_themes(client, model_name, text_chunk):
    """Demande à Groq d'identifier les thématiques sociologiques dans un extrait."""
    prompt = (
        "Voici un entretien sociologique.\n"
        "Identifie de façon claire les grandes thématiques sociologiques abordées dans ce texte :\n\n"
        + text_chunk
    )
    print(f"\n--- Modèle utilisé : {model_name} ---")
    print(f"--- Prompt envoyé (extrait) ---\n{prompt[:500]}...\n")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def parse_themes(themes_text):
    """Transforme la réponse Groq en liste de thématiques."""
    lines = themes_text.strip().split("\n")
    themes = []
    for line in lines:
        line = line.strip("-•1234567890. ").strip()
        if line:
            themes.append(line.lower())
    return themes

def save_themes_to_file(themes, output_path):
    """Sauvegarde une liste de thématiques dans un fichier .txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        for theme in themes:
            f.write(f"{theme}\n")

def process_google_doc(url, client, model_name, output_directory, file_name):
    """Traite un Google Doc pour extraire et sauvegarder les thématiques."""
    text = fetch_text_from_google_doc(url)
    chunks = split_text(text)

    all_themes = []

    for idx, chunk in enumerate(chunks):
        print(f"\n--- Résultat pour le morceau {idx+1}/{len(chunks)} ---\n")
        themes_text = detect_themes(client, model_name, chunk)
        themes_list = parse_themes(themes_text)
        all_themes.extend(themes_list)

    output_path = os.path.join(output_directory, f"{file_name}_themes.txt")
    save_themes_to_file(all_themes, output_path)

    print(f"\nThématiques sauvegardées dans {output_path}")

# --- Programme principal ---

def main():
    api_key, google_doc_url = load_config()
    client = initialize_groq_client(api_key)
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"

    file_name = "entretien"  # Nom pour le fichier de sortie (modifiable)
    output_directory = "./output"

    os.makedirs(output_directory, exist_ok=True)

    process_google_doc(google_doc_url, client, model_name, output_directory, file_name)

if __name__ == "__main__":
    main()
