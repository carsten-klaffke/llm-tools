from transformers import MarianMTModel, MarianTokenizer

def download_and_save_model(model_name, local_dir):
    # Modell und Tokenizer aus Hugging Face herunterladen
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Speichere das Modell und den Tokenizer in einem lokalen Verzeichnis
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    print(f"Model {model_name} saved to {local_dir}")

marian_models_map = {
    'pt': 'Helsinki-NLP/opus-mt-mul-en',
}
for language, model_name in marian_models_map.items():
    local_dir = f"./models/{language}"  # Speichere das Modell in einem Verzeichnis für die Sprache
    download_and_save_model(model_name, local_dir)
