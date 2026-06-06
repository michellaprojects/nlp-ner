import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocessing import (
    normalize_sequence,
    word_level_bio_no_spans,
    build_features_for_inference,
    bio_to_entities,
    extract_text_from_file,
    Non_lemma_label,
)

app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "../../Modelling/model_result/crf_model.pkl"), "rb"))


# ═════════════════════════════════════════════════════════════════════════════
# RUN NER
# ═════════════════════════════════════════════════════════════════════════════

def run_ner(text: str) -> dict:
    raw_bio  = word_level_bio_no_spans(text)
    norm_bio = normalize_sequence(raw_bio)

    if not norm_bio:
        return {}

    features, norm_toks = build_features_for_inference(raw_bio, norm_bio)
    pred_tags = model.predict([features])[0]
    return bio_to_entities(norm_toks, pred_tags)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext != '.pdf':
        return jsonify({'error': f"Unsupported file type '{ext}'. Please upload PDF."}), 400

    text   = extract_text_from_file(file)
    result = run_ner(text)
    return jsonify({'status': 'success', 'data': result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
