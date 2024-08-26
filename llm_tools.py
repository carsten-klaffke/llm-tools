import ssl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP_WORDS
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.parse
from transformers import pipeline, AutoTokenizer, MarianMTModel, MarianTokenizer
import sentencepiece

nltk.download('punkt')

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Umgehen der SSL-Zertifikatsüberprüfung
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Flask-App initialisieren
app = Flask(__name__)

# CORS-Konfiguration
CORS(app, resources={r"/*": {"origins": "*"}})  # Erlaubt CORS-Anfragen von allen Quellen

# Pool-Größe (Anzahl der Modelle im Pool)
POOL_SIZE = 5

# Spacy-Modelle Map
spacy_models_map = {
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'ar': 'xx_ent_wiki_sm',
    'cn': 'xx_ent_wiki_sm',
    'hi': 'xx_ent_wiki_sm',
    'id': 'xx_ent_wiki_sm',
    'ja': 'ja_core_news_sm',
    'ko': 'xx_ent_wiki_sm',
    'nl': 'nl_core_news_sm',
    'pt': 'pt_core_news_sm',
    'ru': 'ru_core_news_sm',
    'es': 'es_core_news_sm',
    'tr': 'xx_ent_wiki_sm',
}

# MarianMT-Modelle Map
marian_models_map = {
    'de': './models/de',
    'fr': './models/fr',
    'ar': './models/ar',
    'cn': './models/cn',
    'hi': './models/hi',
    'id': './models/id',
    'ja': './models/ja',
    'ko': './models/ko',
    'nl': './models/nl',
    'pt': './models/pt',
    'ru': './models/ru',
    'es': './models/es',
    'tr': './models/tr',
}

nlp_en = spacy.load('en_core_web_sm')

from collections import OrderedDict
import threading


class ModelPoolManager:
    def __init__(self, name, models_map, loader_fn, pool_size=POOL_SIZE):
        self.name = name
        self.models_map = models_map  # Map der Sprachen und Modellnamen
        self.loader_fn = loader_fn  # Funktion zum Laden der Modelle
        self.pool_size = pool_size  # Maximale Pool-Größe
        self.model_pool = OrderedDict()  # Ein OrderedDict für alle Sprachen (key: Sprache, value: Modell)
        self.pool_lock = threading.Lock()  # Für Thread-Sicherheit

    def get_model(self, language):
        with self.pool_lock:
            if language not in self.models_map:
                raise ValueError(f"Unsupported language: {language}")

            if language in self.model_pool:
                # Modell aus dem Pool holen und als zuletzt verwendet markieren
                model = self.model_pool.pop(language)
                self.model_pool[language] = model  # Setzt es ans Ende (zuletzt verwendet)
                return model
            else:
                # Ein neues Modell laden, wenn es nicht im Pool ist
                model = self.loader_fn(self.models_map[language])
                self._add_model_to_pool(language, model)
                return model

    def _add_model_to_pool(self, language, model):
        # Prüfen, ob die Pool-Größe überschritten wird
        if len(self.model_pool) >= self.pool_size:
            # Ältestes Modell (Least Recently Used) entfernen
            oldest_language, oldest_model = self.model_pool.popitem(last=False)

        # Das neue Modell zum Pool hinzufügen
        self.model_pool[language] = model


# Loader-Funktionen für SpaCy und MarianMT
def load_spacy_model(model_name):
    return spacy.load(model_name)


def load_marian_model(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return (model, tokenizer)


# SpaCy und MarianMT Pool Manager initialisieren
spacy_pool_manager = ModelPoolManager('Spacy-Pool', spacy_models_map, load_spacy_model)
marian_pool_manager = ModelPoolManager('MarianMt-Pool', marian_models_map, load_marian_model)

STOP_WORDS = {
    'en': EN_STOP_WORDS,
    'de': DE_STOP_WORDS,
    'fr': FR_STOP_WORDS,
}

EXTENDED_STOP_WORDS = set(stopwords.words('english')).union({
    'one', 'two', 'first', 'second', 'new', 'like', 'using', 'used', 'also', 'many', 'make', 'get', 'us', 'however',
    'within'
})


# Ersetze Googletrans durch MarianMT für die lokale Übersetzung
def translate_text_if_needed(text: str, language: str, model, tokenizer):
    if language == 'en':
        return text  # Keine Übersetzung erforderlich

    # Tokenisiere den Eingabetext
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generiere die Übersetzung
    translated = model.generate(**inputs)

    # Dekodiere die Ausgabe in Text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return translated_text


def get_synonyms(word, pos=None, max_synonyms=5):
    synonyms = set()

    # Suche nach Synonymen basierend auf der angegebenen Wortart (pos) in WordNet
    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            if lemma.name() != word:  # Das ursprüngliche Wort nicht einschließen
                synonyms.add(lemma.name())

    # Optional: Die Anzahl der Synonyme begrenzen
    return list(synonyms)[:max_synonyms]


def extract_nouns_with_synonyms(text: str, language: str):
    if language == 'en':
        # Für Englisch direkt das immer verfügbare Modell nutzen
        nlp = nlp_en
    else:
        # Modell aus dem Pool holen
        nlp = spacy_pool_manager.get_model(language)

    # Verarbeite den Text
    doc = nlp(text)
    nouns_and_names = [token.text for token in doc if token.pos_ == 'NOUN' or token.ent_type_ == 'PERSON']
    return nouns_and_names


def extract_nouns_with_synonyms(text: str, language: str):
    if language == 'en':
        # Für Englisch direkt das immer verfügbare Modell nutzen
        nlp = nlp_en
    else:
        # Modell aus dem Pool holen
        nlp = spacy_pool_manager.get_model(language)

    # Verarbeite den Text
    doc = nlp(text)

    # Extrahiere alle Substantive und potenzielle Eigennamen
    nouns_and_names = [token.text for token in doc if token.pos_ == 'NOUN' or token.ent_type_ == 'PERSON']

    # Hinzufügen von potenziellen Eigennamen (großgeschriebene Wörter, die keine Stop-Wörter sind)
    potential_names = [token.text for token in doc if token.is_title and not token.is_stop and token.is_alpha]
    nouns_and_names.extend(potential_names)

    # Normalisiere die Substantive und Namen, aber behalte die Originalsprache bei
    normalized_nouns_and_names = normalize_keywords_in_language(nouns_and_names, language)

    # Füge Synonyme hinzu
    nouns_with_synonyms = normalized_nouns_and_names.copy()
    all_synonyms = []
    for noun in normalized_nouns_and_names:
        synonyms = get_synonyms(noun)
        all_synonyms.extend(synonyms)

    # Kombiniere die Nomen und die Synonyme, Synonyme hinten anstellen
    final_nouns = nouns_with_synonyms + all_synonyms

    # Begrenze die Liste auf maximal 100 Einträge
    final_nouns = final_nouns[:100]

    return final_nouns


def extract_keywords_with_synonyms(text: str, language: str, top_n_keywords: int = 10, num_topics: int = 3,
                                   num_words: int = 5):
    marian_model, tokenizer = marian_pool_manager.get_model(language)

    # Übersetzen
    translated_text = translate_text_if_needed(text, language, marian_model, tokenizer)

    if language == 'en':
        # Für Englisch direkt das immer verfügbare Modell nutzen
        nlp = nlp_en
    else:
        # Modell aus dem Pool holen
        nlp = spacy_pool_manager.get_model(language)

    doc = nlp(translated_text)

    # Extrahiere Keywords basierend auf Bedeutung und Häufigkeit
    keywords = {}
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space or token.text.lower() in STOP_WORDS.get('en', set()):
            continue
        if token.lemma_ not in keywords:
            keywords[token.lemma_] = 0
        keywords[token.lemma_] += 1

    sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
    top_keywords = [keyword for keyword, freq in sorted_keywords[:top_n_keywords]]
    # Verwenden von TF-IDF zur Bestimmung wichtiger Begriffe
    tfidf = TfidfVectorizer(stop_words='english')

    # Überprüfen, ob nach der Vorverarbeitung genügend sinnvolle Wörter vorhanden sind
    if not translated_text.strip():
        return []  # Leere Liste zurückgeben, wenn der Text leer ist

    try:
        tfidf_matrix = tfidf.fit_transform([translated_text])
        tfidf_keywords = tfidf.get_feature_names_out()[:top_n_keywords]
    except ValueError as e:
        print(f"TF-IDF Error: {e}")
        return []  # Leere Liste zurückgeben, wenn TF-IDF keinen gültigen Text findet

    # Tokenisierung des Textes für LDA
    tokens = word_tokenize(translated_text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in EXTENDED_STOP_WORDS]

    # Prüfen, ob genügend Tokens vorhanden sind
    if len(tokens) == 0:
        return []  # Rückgabe einer leeren Liste, wenn keine gültigen Tokens vorhanden sind

    # Gensim LDA
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    if len(corpus[0]) == 0:
        return []  # Rückgabe einer leeren Liste, wenn kein gültiges Korpus erstellt wurde

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, iterations=500,
                         random_state=42)
    topics = lda_model.print_topics(num_words=num_words)

    # Extrahiere die Wörter aus den Themen
    topic_words = []
    for topic in topics:
        topic = topic[1]
        words = [word.split('*')[-1].replace('"', '').strip() for word in topic.split('+')]
        topic_words.extend(words)

    # Kombiniere alle extrahierten Wörter und filtere Stop-Wörter erneut
    combined_keywords = list(set(top_keywords + list(tfidf_keywords) + topic_words))
    # Füge Synonyme hinzu
    keywords_with_synonyms = combined_keywords.copy()
    all_synonyms = []
    for keyword in combined_keywords:
        synonyms = get_synonyms(keyword)
        all_synonyms.extend(synonyms)

    # Kombiniere die Keywords und die Synonyme, Synonyme hinten anstellen
    final_keywords = keywords_with_synonyms + all_synonyms
    # Begrenze die Liste auf maximal 100 Einträge
    final_keywords = final_keywords[:100]

    # Normalisierung der Keywords
    final_keywords = normalize_keywords(final_keywords)

    return final_keywords

# Funktion zur Normalisierung der Keywords
def normalize_keywords(keywords):
    normalized = []
    for keyword in keywords:
        # Verarbeiten des Keywords mit spaCy
        doc = nlp_en(keyword.lower())

        # Alle Token (nicht nur noun_chunks) durchlaufen
        for token in doc:
            # Nimm Substantive oder relevante Verben auf
            if not token.is_stop and token.is_alpha:
                if token.pos_ in ['NOUN', 'PROPN', 'VERB']:  # Erfasse Nomen, Eigennamen und Verben
                    normalized.append(token.lemma_)

    # Entfernen von Duplikaten und leeren Strings
    return list(set(filter(None, normalized)))


# Funktion zur Normalisierung der Keywords unter Beibehaltung der Originalsprache
def normalize_keywords_in_language(keywords, language):
    if language == 'en':
        # Für Englisch direkt das immer verfügbare Modell nutzen
        nlp = nlp_en
    else:
        # Modell aus dem Pool holen
        nlp = spacy_pool_manager.get_model(language)

    normalized = []
    for keyword in keywords:
        doc = nlp(keyword.lower())
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and chunk.root.is_alpha:
                normalized.append(chunk.lemma_)

    # Entfernen von Duplikaten und leeren Strings
    return list(set(filter(None, normalized)))


def chunk_sentences(sentences, tokenizer, max_token_length=1024, max_chunks=4):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Tokenisiere den Satz, um seine Token-Länge zu bestimmen
        token_ids = tokenizer.encode(sentence, truncation=False)
        token_length = len(token_ids)

        # Falls der aktuelle Chunk mit dem Satz die maximale Länge überschreiten würde, Chunk abschließen
        if current_length + token_length > max_token_length:
            # Nutze ".join" mit Sätzen, um doppelte Leerzeichen zu vermeiden
            chunks.append(" ".join(current_chunk).strip())  # Entferne Leerzeichen am Ende des Chunks
            current_chunk = []
            current_length = 0

            # Überprüfe, ob das Chunk-Limit erreicht wurde
            if len(chunks) >= max_chunks:
                break

        # Füge den Satz zum aktuellen Chunk hinzu
        current_chunk.append(sentence)
        current_length += token_length

    # Füge den letzten Chunk hinzu, falls noch ein Rest übrig ist und das Chunk-Limit nicht überschritten wurde
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(" ".join(current_chunk).strip())  # Entferne Leerzeichen am Ende des Chunks

    return chunks


def summarize_large_text(text, max_token_length=1024, max_chunks=4):

    try:
        # Tokenisiere den Text in einzelne Sätze
        sentences = sent_tokenize(text)

        # Teile die Sätze in Chunks auf, mit Limit für die Anzahl der Chunks
        chunks = chunk_sentences(sentences, tokenizer, max_token_length, max_chunks)

        # Übersetze und fasse nur die Chunks zusammen, die innerhalb der max_chunks-Grenze liegen
        summaries = []
        for chunk in chunks:
            # Führe die Zusammenfassung des übersetzten Chunks durch
            summary_chunk = summarize_text(chunk, max_length=256)  # "en", da wir nun englischen Text haben
            summaries.append(summary_chunk)

        # Alle Teilsummen zusammenfügen und erneut zusammenfassen, falls nötig
        combined_summary = " ".join(summaries).strip()  # Entferne überflüssige Leerzeichen am Ende
        # Optional: Final zusammenfassen, um eine kurze Zusammenfassung der Teilsummen zu erhalten
        final_summary = summarize_text(combined_summary, max_length=256)
        return final_summary

    except Exception as e:
        return f"Error while summarizing: {str(e)}"

def summarize_text(text, max_length=256):
    try:
        # Tokenisierung des Textes mit Begrenzung
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        token_length = len(tokens.input_ids[0])

        # Falls der Text mehr als 1024 Token hat, kürze ihn
        if token_length > 1024:
            text = tokenizer.decode(tokens.input_ids[0][:1024], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)

        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        token_length = len(tokens.input_ids[0])

        if token_length <= max_length:
            return text

        # Generiere die Zusammenfassung mit max_length = 256
        summary = summarizer(text, max_length=max_length, do_sample=False)
        summary_text = summary[0]['summary_text']
    except Exception as e:
        summary_text = f"Could not generate summary: {str(e)}"

    return summary_text


@app.route('/extract_nouns', methods=['POST', 'OPTIONS'])
def extract_nouns_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'}), 200
    elif request.method == 'POST':
        data = request.get_json()
        text = urllib.parse.unquote(data.get('text', ''))
        language = data.get('language', 'en')

        try:
            nouns = extract_nouns_with_synonyms(text, language)
            return jsonify({"nouns": nouns})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


@app.route('/extract_keywords', methods=['POST', 'OPTIONS'])
def extract_keywords_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'}), 200
    elif request.method == 'POST':
        data = request.get_json()
        text = urllib.parse.unquote(data.get('text', ''))
        language = data.get('language', 'en')

        try:
            keywords = extract_keywords_with_synonyms(text, language)
            return jsonify({"keywords": keywords})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    try:
        # Holen der Anfrage-Daten
        data = request.get_json()
        text = urllib.parse.unquote(data.get('text', ''))
        language = data.get('language', "en")

        # Überprüfen, ob der Text existiert
        if not text:
            return jsonify({"error": "No text provided."}), 400

        # Tokenisiere den Text, um die Token-Länge zu prüfen
        tokens = tokenizer.encode(text, truncation=False)
        token_length = len(tokens)

        if token_length <= 256:
            summary = None
        else:
            marian_model, marian_tokenizer = marian_pool_manager.get_model(language)
            translated_text = translate_text_if_needed(text, language, marian_model, marian_tokenizer)
            summary = summarize_large_text(translated_text)

        # Rückgabe der Zusammenfassung als JSON-Antwort
        return jsonify({"summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
