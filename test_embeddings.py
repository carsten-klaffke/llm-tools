import pytest
from embeddings import create_or_update_embedding, match_embedding, generate_embedding, get_or_create_collection
import chromadb

# Testbenutzer
TEST_USER = "test_user"

# Knoten zum Sonnensystem
SOLAR_SYSTEM_NODES = {
    "earth": "Die Erde ist der dritte Planet von der Sonne.",
    "mars": "Mars ist der vierte Planet und hat eine dünne Atmosphäre.",
    "jupiter": "Jupiter ist der größte Planet im Sonnensystem.",
    "saturn": "Saturn ist bekannt für seine beeindruckenden Ringe.",
    "venus": "Venus ist der zweite Planet und hat eine extrem dichte Atmosphäre."
}

# Themenfremde Knoten
OTHER_NODES = {
    "dog": "Hunde sind domestizierte Tiere und treue Begleiter des Menschen.",
    "car": "Autos sind motorisierte Fahrzeuge, die für den Transport genutzt werden."
}

@pytest.fixture(scope="module", autouse=True)
def setup_and_clear_chromadb():
    """
    Löscht alle Collections aus der Test-ChromaDB, bevor die Tests ausgeführt werden.
    Erstellt danach die Test-Embeddings für das Sonnensystem.
    """
    # ChromaDB-Client
    global chroma_client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Test-Embeddings erneut anlegen
    for node_id, text in SOLAR_SYSTEM_NODES.items():
        create_or_update_embedding(TEST_USER, node_id, text)

    for node_id, text in OTHER_NODES.items():
        create_or_update_embedding(TEST_USER, node_id, text)

def test_generate_embedding():
    """
    Testet die Erzeugung eines Embeddings.
    """
    test_text = "Die Sonne ist der Mittelpunkt des Sonnensystems."
    embedding = generate_embedding(test_text)
    assert isinstance(embedding, list), "Embedding sollte eine Liste sein"
    assert len(embedding) > 0, "Embedding sollte nicht leer sein"


def test_create_embedding():
    """
    Testet das Speichern von Embeddings in ChromaDB.
    """
    collection = get_or_create_collection(TEST_USER)

    # Überprüfe, ob alle Knoten vorhanden sind
    for node_id in SOLAR_SYSTEM_NODES.keys():
        results = collection.get(ids=[node_id])
        assert len(results["documents"]) == 1, f"Knoten {node_id} sollte in ChromaDB existieren"


def test_match_relevant_embedding():
    """
    Testet das Abrufen eines ähnlichen Embeddings für relevante Fragen.
    """
    test_queries = {
        "planet_question": "Welcher Planet hat Ringe?",  # Sollte Saturn matchen
        "red_planet": "Welcher Planet ist als roter Planet bekannt?",  # Sollte Mars matchen
        "biggest_planet": "Welcher Planet ist der größte?",  # Sollte Jupiter matchen
    }

    expected_matches = {
        "planet_question": "saturn",
        "red_planet": "mars",
        "biggest_planet": "jupiter"
    }

    for query, expected_node in expected_matches.items():
        matching_ids = match_embedding(TEST_USER, test_queries[query], top_k=1, score_threshold=0.1)
        assert expected_node in matching_ids, f"'{test_queries[query]}' sollte mit '{expected_node}' matchen"


def test_match_irrelevant_embedding():
    """
    Testet das Verhalten für Fragen, die nichts mit dem Sonnensystem zu tun haben.
    """
    irrelevant_queries = [
        "Was ist das beste Hundefutter?",  # Sollte nicht mit Planeten matchen
        "Welches Auto ist das schnellste?",  # Sollte nicht mit Planeten matchen
    ]

    for query in irrelevant_queries:
        matching_ids = match_embedding(TEST_USER, query, top_k=1)

        # Prüfen, ob ein Planeten-Knoten in den Ergebnissen ist
        matched_planets = any(node in SOLAR_SYSTEM_NODES.keys() for node in matching_ids)

        assert not matched_planets, f"'{query}' sollte keine relevanten Planeten-Knoten finden"