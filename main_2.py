import os
from dotenv import load_dotenv
import groq
import PyPDF2
from collections import Counter

# --- Fonctions ---

def load_api_key():
    """Charge la clé API depuis le fichier .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La clé API GROQ_API_KEY n'est pas définie dans le fichier .env")
    return api_key

def initialize_groq_client(api_key):
    """Initialise le client Groq."""
    return groq.Groq(api_key=api_key)

def extract_text_from_pdf(file_path):
    """Extrait le texte brut d'un fichier PDF."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text(text, max_chunk_size=12000):
    """Découpe le texte pour éviter de dépasser la limite de tokens."""
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i+max_chunk_size]
        chunks.append(chunk)
    return chunks

def detect_themes(client, model_name, text_chunk):
    """Demande à Groq de détecter les thématiques sociologiques dans un extrait."""
    prompt = (
        "Voici un extrait d'entretien sociologique.\n"
        "Identifie et liste les grandes thématiques sociologiques abordées dans ce texte :\n\n"
        + text_chunk
    )
    print(f"\n--- Modèle utilisé : {model_name} ---")
    print(f"--- Prompt envoyé (extrait) ---\n{prompt[:500]}...\n")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def process_single_pdf(file_path, client, model_name):
    """Traite un seul PDF pour extraire et sauvegarder ses thématiques."""
    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)

    all_themes = []

    for idx, chunk in enumerate(chunks):
        print(f"\n--- Résultat pour le morceau {idx+1}/{len(chunks)} ---\n")
        themes_text = detect_themes(client, model_name, chunk)
        themes_list = parse_themes(themes_text)
        all_themes.extend(themes_list)

    return all_themes

def parse_themes(themes_text):
    """Transforme la réponse de Groq (texte brut) en une liste de thématiques."""
    lines = themes_text.strip().split("\n")
    themes = []
    for line in lines:
        line = line.strip("-•1234567890. ").strip()
        if line:
            themes.append(line.lower())  # On met en minuscules pour normaliser
    return themes

def save_themes_to_file(themes, output_path):
    """Sauvegarde une liste de thématiques dans un fichier .txt."""
    with open(output_path, "w", encoding="utf-8") as f:
        for theme in themes:
            f.write(f"{theme}\n")

def process_all_pdfs(data_directory, output_directory, client, model_name):
    """Traite tous les fichiers PDF et sauvegarde les thématiques individuelles et globales."""
    all_detected_themes = []

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(data_directory):
        if filename.lower().endswith(".pdf"):
            print(f"\n=== Traitement de : {filename} ===")
            file_path = os.path.join(data_directory, filename)
            themes = process_single_pdf(file_path, client, model_name)
            all_detected_themes.extend(themes)

            # Sauvegarde les thématiques individuelles
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_themes.txt")
            save_themes_to_file(themes, output_file_path)

    return all_detected_themes

def identify_recurrent_themes(all_themes, min_occurrences=2):
    """Identifie les thématiques les plus fréquentes parmi tous les entretiens."""
    counter = Counter(all_themes)
    recurrent = [theme for theme, count in counter.items() if count >= min_occurrences]
    return recurrent

# --- Programme principal ---

def main():
    api_key = load_api_key()
    client = initialize_groq_client(api_key)
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"

    data_directory = "./data_2"
    output_directory = "./output"

    all_themes = process_all_pdfs(data_directory, output_directory, client, model_name)

    # Identifier les thématiques récurrentes
    recurrent_themes = identify_recurrent_themes(all_themes, min_occurrences=2)

    # Sauvegarder les thématiques récurrentes
    recurrent_themes_file = os.path.join(output_directory, "themes_recurrents.txt")
    save_themes_to_file(recurrent_themes, recurrent_themes_file)

    print("\n=== Analyse terminée ===")
    print(f"Thématiques récurrentes sauvegardées dans {recurrent_themes_file}")

if __name__ == "__main__":
    main()
