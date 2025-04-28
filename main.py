import os
from dotenv import load_dotenv
import groq
import PyPDF2

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
    """Découpe le texte en morceaux plus petits pour rester sous les limites de tokens."""
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i+max_chunk_size]
        chunks.append(chunk)
    return chunks

def detect_themes(client, model_name, text_chunk):
    """Utilise Groq pour détecter les thématiques principales dans un morceau de texte."""
    prompt = (
        "Voici un extrait d'entretien.\n"
        "Identifie entre 0 et 10 thématiques pour une analyse sociologique (de manière synthétique, sous forme de liste) :\n\n"
        + text_chunk
    )
    print(f"\n--- Modèle utilisé : {model_name} ---")
    print(f"--- Prompt envoyé ---\n{prompt[:500]}...\n")  # Limite l'affichage à 500 caractères
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def process_pdfs_in_directory(directory_path, client, model_name):
    """Traite tous les PDF du répertoire pour détecter les thématiques."""
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"\n=== Thématiques détectées pour : {filename} ===\n")
            text = extract_text_from_pdf(file_path)
            chunks = split_text(text)

            for idx, chunk in enumerate(chunks):
                print(f"\n--- Résultat pour le morceau {idx+1}/{len(chunks)} ---\n")
                themes = detect_themes(client, model_name, chunk)
                print(themes)
                print("\n" + "-"*50)

# --- Programme principal ---

def main():
    api_key = load_api_key()
    client = initialize_groq_client(api_key)
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
    data_directory = "./data"

    process_pdfs_in_directory(data_directory, client, model_name)

if __name__ == "__main__":
    main()
