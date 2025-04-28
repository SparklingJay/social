import os
from dotenv import load_dotenv
import groq
import fitz  # PyMuPDF
from collections import Counter

# --- Fonctions ---

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

def extract_text_from_pdf(pdf_path):
    """Extrait le texte brut d'un PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    if not text.strip():
        raise Exception(f"Le fichier {pdf_path} est vide ou illisible.")
    return text

def split_text(text, max_chunk_size=12000):
    """Découpe le texte pour éviter de dépasser la limite de tokens."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def detect_themes(client, model_name, text_chunk):
    """Demande à Groq d'identifier les thématiques sociologiques dans un extrait."""
    prompt = (
        "Voici un entretien sociologique extrait d'un fichier PDF.\n"
        "Identifie clairement les grandes thématiques sociologiques abordées dans ce texte :\n\n"
        + text_chunk
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def parse_themes(themes_text):
    """Transforme la réponse Groq en liste de thématiques propres."""
    lines = themes_text.strip().split("\n")
    themes = []
    for line in lines:
        line = line.strip("-•1234567890. ").strip()
        if line:
            themes.append(line.lower())
    return themes

def save_themes_to_file(themes, output_path):
    """Sauvegarde une liste de thématiques dans un fichier .txt."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, theme in enumerate(themes, 1):
            f.write(f"{idx}. {theme}\n")

def process_single_pdf(pdf_path, client, model_name, output_directory):
    """Traite un PDF pour extraire et retourner ses thématiques."""
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    all_themes = []

    for idx, chunk in enumerate(chunks):
        print(f"🔎 Analyse du morceau {idx + 1}/{len(chunks)} pour {os.path.basename(pdf_path)}...")
        themes_text = detect_themes(client, model_name, chunk)
        themes_list = parse_themes(themes_text)
        all_themes.extend(themes_list)

    # Sauvegarder les thèmes de ce fichier PDF
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_directory, f"{file_name}_themes.txt")
    save_themes_to_file(all_themes, output_path)
    print(f"✅ Thématiques sauvegardées pour {file_name}")

    return all_themes

def process_all_pdfs(data_directory, output_directory):
    """Traite tous les PDF d'un dossier et produit un fichier global de thématiques."""
    api_key = load_config()
    client = initialize_groq_client(api_key)
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"

    global_themes = []

    for filename in os.listdir(data_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_directory, filename)
            try:
                themes = process_single_pdf(pdf_path, client, model_name, output_directory)
                global_themes.extend(themes)
            except Exception as e:
                print(f"❌ Erreur avec {filename}: {str(e)}")

    # Résumer les thèmes globaux
    theme_counter = Counter(global_themes)
    sorted_themes = theme_counter.most_common()

    global_output_path = os.path.join(output_directory, "themes_globaux.txt")
    with open(global_output_path, "w", encoding="utf-8") as f:
        for theme, count in sorted_themes:
            f.write(f"{theme}: {count}\n")

    print(f"\n🌍 Fichier global des thématiques enregistré : {global_output_path}")

# --- Programme principal ---

def main():
    data_directory = "./data"
    output_directory = "./output"

    os.makedirs(output_directory, exist_ok=True)
    process_all_pdfs(data_directory, output_directory)

if __name__ == "_main_":
    main()