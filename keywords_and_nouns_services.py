import ssl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP_WORDS
from googletrans import Translator
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.parse

# Umgehen der SSL-Zertifikatsüberprüfung
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Flask-App initialisieren
app = Flask(__name__)

# CORS-Konfiguration
CORS(app, resources={r"/*": {"origins": "*"}})  # Erlaubt CORS-Anfragen von allen Quellen

# Laden der spaCy-Modelle für die unterstützten Sprachen einmalig
nlp_models = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_lg'),
    'fr': spacy.load('fr_core_news_sm'),
    'ar': spacy.load('xx_ent_wiki_sm'),
    'cn': spacy.load('xx_ent_wiki_sm'),
    'hi': spacy.load('xx_ent_wiki_sm'),
    'id': spacy.load('xx_ent_wiki_sm'),
    'ja': spacy.load('ja_core_news_sm'),
    'ko': spacy.load('xx_ent_wiki_sm'),
    'nl': spacy.load('nl_core_news_sm'),
    'pt': spacy.load('pt_core_news_sm'),
    'ru': spacy.load('ru_core_news_sm'),
    'es': spacy.load('es_core_news_sm'),
    'tr': spacy.load('xx_ent_wiki_sm'),
}

STOP_WORDS = {
    'en': EN_STOP_WORDS,
    'de': DE_STOP_WORDS,
    'fr': FR_STOP_WORDS,
}

EXTENDED_STOP_WORDS = set(stopwords.words('english')).union({
    'one', 'two', 'first', 'second', 'new', 'like', 'using', 'used', 'also', 'many', 'make', 'get', 'us', 'however', 'within'
})

# Laden des englischen spaCy-Modells für die Normalisierung
nlp_en = nlp_models['en']


def translate_text_if_needed(text: str, language: str):
    if language == 'en':
        return text  # Keine Übersetzung erforderlich

    translator = Translator()

    try:
        # Versuch, die Übersetzung durchzuführen
        translate = translator.translate(text, dest='en')
        translated_text = translate.text if translate else ''
    except Exception as e:
        print(f"Translation failed: {e}")
        # Fallback: Originaltext zurückgeben, falls die Übersetzung fehlschlägt
        translated_text = text

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
    if language not in nlp_models:
        raise ValueError(f"Unsupported language: {language}")

    # Verwende das vorab geladene Sprachmodell
    nlp = nlp_models[language]

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

def extract_keywords_with_synonyms(text: str, language: str, top_n_keywords: int = 10, num_topics: int = 3, num_words: int = 5):
    # Übersetze den Text, falls erforderlich
    translated_text = translate_text_if_needed(text, language)

    # Verwende das vorab geladene englische Sprachmodell
    nlp = nlp_en
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

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, iterations=500, random_state=42)
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
    nlp = nlp_models[language]
    normalized = []
    for keyword in keywords:
        doc = nlp(keyword.lower())
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and chunk.root.is_alpha:
                normalized.append(chunk.lemma_)

    # Entfernen von Duplikaten und leeren Strings
    return list(set(filter(None, normalized)))

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

if __name__ == '__main__':
    app.run(debug=False)