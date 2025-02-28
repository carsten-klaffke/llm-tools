import yaml
from flask import Flask, request, jsonify
import requests
import chromadb

# Flask App initialisieren
app = Flask(__name__)

# CORS aktivieren, falls die API auch von externen Clients genutzt wird
from flask_cors import CORS
CORS(app)

# 📌 Konfiguration aus YAML-Datei laden
def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()

AZURE_OPENAI_API_KEY = config["azure_openai"]["api_key"]
AZURE_OPENAI_ENDPOINT = config["azure_openai"]["endpoint"]
MODEL_NAME = config["azure_openai"]["model_name"]


chroma_client = chromadb.HttpClient(host="localhost", port=8000)


def get_or_create_collection(user_id: str):
    """ Holt oder erstellt eine ChromaDB-Collection für den angegebenen Benutzer. """
    return chroma_client.get_or_create_collection(name=f"user_{user_id}")


def generate_embedding(text: str, language: str):
    """
    Erstellt ein Embedding für den gegebenen Text über Azure OpenAI.
    Setzt die Sprache als Präfix vor den Text.
    """
    tagged_text = f"[{language.upper()}] {text}"  # Sprache als Tag voranstellen

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    payload = {"input": tagged_text}
    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Fehler beim Abrufen des Embeddings: {response.status_code}, {response.text}")


def create_or_update_embedding(user_id: str, node_id: str, text: str, language: str = "de"):
    """
    Erstellt ein Embedding für den gegebenen Text mit Azure OpenAI und speichert es in ChromaDB.
    """
    collection = get_or_create_collection(user_id)
    embedding = generate_embedding(text, language)  # Sprache wird jetzt in `generate_embedding` verarbeitet

    # Falls die ID existiert, vorher löschen
    existing = collection.get(ids=[node_id])
    if existing and existing["ids"]:
        collection.delete(ids=[node_id])

    collection.add(
        ids=[node_id],
        embeddings=[embedding],
        documents=[text],  # Speichert den ursprünglichen Text
        metadatas=[{"language": language}]  # Sprache als Metadaten speichern
    )

    print(f"Embedding für Node {node_id} mit Sprache '{language}' aktualisiert.")


def match_embedding(user_id: str, query_text: str, language: str, top_k: int = 5, score_threshold: float = 1.0):
    """
    Erstellt ein Embedding für die Suchanfrage mit Azure OpenAI und sucht die ähnlichsten gespeicherten Knoten.
    Berücksichtigt die Sprache der Suchanfrage.
    """
    collection = get_or_create_collection(user_id)
    query_embedding = generate_embedding(query_text, language)  # Sprache wird jetzt übergeben

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances", "documents", "metadatas"]
    )

    matching_ids = []

    if "distances" in results:
        for i, distance in enumerate(results["distances"][0]):
            if distance <= score_threshold:
                matching_ids.append(results["ids"][0][i])

    print(f"Beste Übereinstimmungen für '{query_text}' ({language}): {matching_ids}")
    return matching_ids


@app.route("/create_embedding", methods=["POST"])
def create_embedding_service():
    """ Erstellt ein Embedding für einen Knoten und speichert es in ChromaDB. """
    data = request.json
    user_id = data.get("user_id")
    node_id = data.get("node_id")
    text = data.get("text")
    lang = data.get("language")

    if not all([user_id, node_id, text, lang]):
        return jsonify({"error": "user_id, node_id, text und language sind erforderlich"}), 400

    create_or_update_embedding(user_id, node_id, text, lang)

    return {
        "model_version": MODEL_NAME
    }


@app.route("/match_embedding", methods=["POST"])
def match_embedding_service():
    """ Erstellt ein Embedding für eine Suchanfrage und sucht die ähnlichsten gespeicherten Knoten. """
    data = request.json
    user_id = data.get("user_id")
    query_text = data.get("query_text")
    lang = data.get("language")
    top_k = data.get("top_k", 5)

    if not all([user_id, query_text, lang]):
        return jsonify({"error": "user_id, query_text und language sind erforderlich"}), 400

    matching_ids = match_embedding(user_id, query_text, lang, top_k)

    return jsonify({"matches": matching_ids})


if __name__ == "__main__":
    app.run(host="0.0.0.0")