import os
import re
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocessing import (
    is_tech_token,
    normalize_token,
    word_tokenize,
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
# PREPROCESSING HELPERS (CRF-specific — tidak ada di preprocessing.py)
# ═════════════════════════════════════════════════════════════════════════════

def normalize_sequence(bio_tokens, no_lemma_labels=Non_lemma_label):
    """Normalize a list of (token, BIO-tag) pairs, skipping noise tokens."""
    result = []
    for tok, tag in bio_tokens:
        entity = tag.split('-', 1)[1] if '-' in tag else None
        apply_lemma = entity not in no_lemma_labels if entity else True
        norm = normalize_token(tok, apply_lemma=apply_lemma)
        if norm is not None:
            result.append((norm, tag))
    return result

def word_level_bio_no_spans(text: str):
    """Tokenize text into (token, 'O') pairs — no spans needed for inference."""
    tokens = word_tokenize(text)
    return [(tok, 'O') for tok, _, _ in tokens]


# ═════════════════════════════════════════════════════════════════════════════
# CRF FEATURES
# ═════════════════════════════════════════════════════════════════════════════

_POS_RULES = [
    (re.compile(r'^\d{4}$'),               'CD_YEAR'),
    (re.compile(r'^\d+$'),                 'CD'),
    (re.compile(r'^[A-Z][a-z]+$'),         'NNP'),
    (re.compile(r'^[A-Z]{2,}$'),           'NNP_ABB'),
    (re.compile(r'.+ing$', re.I),          'VBG'),
    (re.compile(r'.+ed$', re.I),           'VBD'),
    (re.compile(r'.+er$', re.I),           'NN_ER'),
    (re.compile(r'.+ly$', re.I),           'RB'),
    (re.compile(r'.+tion$|.+sion$', re.I), 'NN_TION'),
    (re.compile(r'.+ment$', re.I),         'NN_MENT'),
    (re.compile(r'@'),                     'EMAIL'),
    (re.compile(r'[+#]|\d'),               'TECH'),
]

def regex_pos(token: str) -> str:
    for pattern, tag in _POS_RULES:
        if pattern.search(token):
            return tag
    return 'NN'

def token_features(raw_tok, norm_tok, idx, raw_seq, norm_seq):
    prev_raw = raw_seq[idx - 1] if idx > 0 else '<START>'
    next_raw = raw_seq[idx + 1] if idx < len(raw_seq) - 1 else '<END>'
    return {
        'token'            : norm_tok,
        'pos'              : regex_pos(raw_tok),
        'is_title_case'    : raw_tok.istitle(),
        'is_all_upper'     : raw_tok.isupper(),
        'is_all_lower'     : raw_tok.islower(),
        'is_digit'         : raw_tok.isdigit(),
        'has_digit'        : any(c.isdigit() for c in raw_tok),
        'is_alnum'         : raw_tok.isalnum(),
        'has_hyphen'       : '-' in raw_tok,
        'has_at'           : '@' in raw_tok,
        'has_dot'          : '.' in raw_tok,
        'has_plus'         : '+' in raw_tok,
        'has_slash'        : '/' in raw_tok,
        'token_length'     : len(raw_tok),
        'is_short'         : len(raw_tok) <= 2,
        'prefix2'          : norm_tok[:2],
        'prefix3'          : norm_tok[:3],
        'prefix4'          : norm_tok[:4],
        'suffix2'          : norm_tok[-2:],
        'suffix3'          : norm_tok[-3:],
        'suffix4'          : norm_tok[-4:],
        'prev_token'       : prev_raw.lower(),
        'prev_is_title'    : prev_raw.istitle(),
        'prev_is_upper'    : prev_raw.isupper(),
        'prev_pos'         : regex_pos(prev_raw),
        'next_token'       : next_raw.lower(),
        'next_is_title'    : next_raw.istitle(),
        'next_is_upper'    : next_raw.isupper(),
        'next_pos'         : regex_pos(next_raw),
        'is_sentence_start': idx == 0,
        'is_sentence_end'  : idx == len(raw_seq) - 1,
    }

def build_features_for_inference(raw_bio, norm_bio):
    """Build CRF feature dicts for inference (no gold labels needed)."""
    aligned = []
    raw_iter = iter(raw_bio)
    for norm_tok, norm_tag in norm_bio:
        for raw_tok, raw_tag in raw_iter:
            aligned.append((raw_tok, norm_tok))
            break

    raw_toks  = [r for r, _ in aligned]
    norm_toks = [n for _, n in aligned]
    return [
        token_features(raw_toks[i], norm_toks[i], i, raw_toks, norm_toks)
        for i in range(len(aligned))
    ], norm_toks


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
