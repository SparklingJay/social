import os
import time
import json
from dotenv import load_dotenv
import groq
import fitz  # PyMuPDF
from collections import defaultdict

# --- Fonctions ---

def load_config():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La clé API GROQ_API_KEY n'est pas définie dans le fichier .env.")
    return api_key

def initialize_groq_client(api_key):
    return groq.Groq(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    if not text.strip():
        raise Exception(f"Le fichier {pdf_path} est vide ou illisible.")
    return text

def split_text(text, max_chunk_size=20000):
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def detect_themes(client, model_name, text_chunk):
    prompt = (
        "Voici un extrait d'entretien sociologique.\n"
        "Identifie les **thématiques sociologiques** abordées, et pour chaque thème, liste les **verbatims associés**, "
        "sous forme d’un dictionnaire JSON valide comme ceci :\n"
        "{\n"
        "  \"amitié\": [\"verbatim 1\", \"verbatim 2\"],\n"
        "  \"rapport à l'école\": [\"verbatim 3\"]\n"
        "}\n\n"
        f"{text_chunk}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def parse_themes_and_verbatims(response_text):
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_part = response_text[start:end]
        theme_dict = json.loads(json_part)
        return theme_dict
    except Exception as e:
        print("❌ Erreur lors du parsing JSON :", e)
        return {}

def process_single_pdf(pdf_path, client, model_name):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    aggregated = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        print(f"🔍 Chunk {idx+1}/{len(chunks)} : {os.path.basename(pdf_path)}")
        try:
            response_text = detect_themes(client, model_name, chunk)
            print("🧾 Réponse Groq brute :\n", response_text)
            themes = parse_themes_and_verbatims(response_text)

            for theme, verbatims in themes.items():
                aggregated[theme].extend(verbatims)

        except Exception as e:
            print(f"⚠️ Erreur dans le chunk {idx+1} : {e}")
        time.sleep(5)

    return aggregated

def merge_theme_dicts(global_dict, new_dict, source_file):
    for theme, verbatims in new_dict.items():
        if source_file not in global_dict[theme]:
            global_dict[theme][source_file] = []
        global_dict[theme][source_file].extend(verbatims)

def process_all_pdfs(data_directory, output_directory):
    api_key = load_config()
    client = initialize_groq_client(api_key)
    model_name = "qwen-qwq-32b"

    final_result = defaultdict(dict)

    for filename in os.listdir(data_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_directory, filename)
            print(f"\n📂 Traitement de : {filename}")
            try:
                theme_data = process_single_pdf(pdf_path, client, model_name)
                merge_theme_dicts(final_result, theme_data, filename)
            except Exception as e:
                print(f"❌ Erreur avec {filename}: {e}")

    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, "themes_entretiens.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Résultat sauvegardé dans : {output_path}")

# --- Programme principal ---

def main():
    data_directory = "./data_5"
    output_directory = "./output_6"
    process_all_pdfs(data_directory, output_directory)

if __name__ == "__main__":
    main()
