"""
preprocessing.py
Fungsi preprocessing untuk inference NER berbasis CRF — dipakai app.py.
"""

import re
import unicodedata
from collections import defaultdict
import spacy

nlp = spacy.load('en_core_web_sm')

# ── Constants ─────────────────────────────────────────────────────────────────

Non_lemma_label = {
    'Name', 'Email Address', 'Graduation Year',
    'Location', 'College Name', 'Companies worked at'
}

_TECH_TOKEN    = re.compile(r'[+#]|\d|\.\w|^[A-Z]{1,5}$|^\.[a-z]+')
_NOISE_PATTERN = re.compile(r'^[^\w.#+]+$|^[\-–—]+$|^[•·▪●■◦○]+$|^[Ââ€¢§]+$')

LABEL_KEY_MAP = {
    'Name'                : 'name',
    'Designation'         : 'designation',
    'Companies worked at' : 'companies_worked_at',
    'Location'            : 'location',
    'Email Address'       : 'email',
    'College Name'        : 'college_name',
    'Degree'              : 'degree',
    'Graduation Year'     : 'graduation_year',
    'Skills'              : 'skills',
    'Years of Experience' : 'years_of_experience',
}

# ── Text normalization ────────────────────────────────────────────────────────

def is_tech_token(token: str) -> bool:
    return bool(_TECH_TOKEN.search(token))

def is_noise_token(token: str) -> bool:
    return bool(_NOISE_PATTERN.match(token)) or not token.strip()

def to_ascii(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def case_fold(token: str) -> str:
    return to_ascii(token).lower()

def remove_edge_noise(token: str) -> str:
    if token.startswith('##'):
        return '##' + re.sub(r'^[^\w#]+|[^\w]+$', '', token[2:])
    return re.sub(r'^[^\w.#+]+|[^\w.#+]+$', '', token)

def lemmatize(token: str) -> str:
    if is_tech_token(token):
        return token
    doc = nlp(token)
    return doc[0].lemma_ if doc else token

def normalize_token(token: str, apply_lemma: bool = True) -> str | None:
    is_tech = is_tech_token(token)
    token   = case_fold(token)
    token   = remove_edge_noise(token)
    if is_noise_token(token):
        return None
    if apply_lemma and not is_tech and len(token) > 2:
        token = lemmatize(token)
    return token or None

def normalize_sequence(
    bio_tokens     : list[tuple[str, str]],
    no_lemma_labels: set = Non_lemma_label
) -> list[tuple[str, str]]:
    """Normalize a list of (token, BIO-tag) pairs, skipping noise tokens."""
    result = []
    for tok, tag in bio_tokens:
        entity      = tag.split('-', 1)[1] if '-' in tag else None
        apply_lemma = entity not in no_lemma_labels if entity else True
        norm = normalize_token(tok, apply_lemma=apply_lemma)
        if norm is not None:
            result.append((norm, tag))
    return result

# ── Tokenization ──────────────────────────────────────────────────────────────

def word_tokenize(text: str) -> list[tuple]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]

def word_level_bio_no_spans(text: str) -> list[tuple[str, str]]:
    """Tokenize text into (token, 'O') pairs — no spans needed for inference."""
    return [(tok, 'O') for tok, _, _ in word_tokenize(text)]

# ── CRF Feature Extraction ────────────────────────────────────────────────────

def token_features(raw_tok: str, norm_tok: str, idx: int,
                   raw_seq: list[str], norm_seq: list[str]) -> dict:
    prev_raw = raw_seq[idx - 1] if idx > 0                else '<START>'
    next_raw = raw_seq[idx + 1] if idx < len(raw_seq) - 1 else '<END>'

    return {
        'token'            : norm_tok,
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
        'next_token'       : next_raw.lower(),
        'next_is_title'    : next_raw.istitle(),
        'next_is_upper'    : next_raw.isupper(),
        'is_sentence_start': idx == 0,
        'is_sentence_end'  : idx == len(raw_seq) - 1,
    }


def align_raw_norm(
    raw_bio : list[tuple[str, str]],
    norm_bio: list[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """
    Align raw and normalized BIO sequences.
    Normalization can DROP noise tokens — this pairs each normalized token
    back to its original raw form by skipping dropped tokens.
    Returns list of (raw_token, norm_token, bio_tag).
    """
    aligned  = []
    raw_iter = iter(raw_bio)
    for norm_tok, norm_tag in norm_bio:
        for raw_tok, raw_tag in raw_iter:
            if raw_tag == norm_tag:
                aligned.append((raw_tok, norm_tok, norm_tag))
                break
    return aligned


def build_features_for_inference(
    raw_bio : list[tuple[str, str]],
    norm_bio: list[tuple[str, str]]
) -> tuple[list[dict], list[str]]:
    """Build CRF feature dicts for inference. Returns (features, norm_tokens)."""
    aligned   = align_raw_norm(raw_bio, norm_bio)
    raw_toks  = [r for r, _, _ in aligned]
    norm_toks = [n for _, n, _ in aligned]

    features = [
        token_features(raw_toks[i], norm_toks[i], i, raw_toks, norm_toks)
        for i in range(len(aligned))
    ]
    return features, norm_toks

# ── BIO → structured dict ─────────────────────────────────────────────────────

def bio_to_entities(tokens: list, tags: list) -> dict:
    """Ubah list (token, BIO-tag) jadi dict entitas terstruktur."""
    entities = defaultdict(list)
    current_label, current_tokens = None, []

    for tok, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_label and current_tokens:
                entities[current_label].append(' '.join(current_tokens))
            current_label  = tag[2:]
            current_tokens = [tok]
        elif tag.startswith('I-') and current_label and tag[2:] == current_label:
            current_tokens.append(tok)
        else:
            if current_label and current_tokens:
                entities[current_label].append(' '.join(current_tokens))
            current_label, current_tokens = None, []

    if current_label and current_tokens:
        entities[current_label].append(' '.join(current_tokens))

    result = {}
    for label, values in entities.items():
        key = LABEL_KEY_MAP.get(label, label.lower().replace(' ', '_'))
        result[key] = list(dict.fromkeys(values))
    return result

# ── File text extraction ──────────────────────────────────────────────────────

def extract_text_from_file(file) -> str:
    filename = file.filename.lower()

    if filename.endswith('.pdf'):
        try:
            import pypdf
            reader = pypdf.PdfReader(file.stream)
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        except ImportError:
            return "[PDF parsing not available – install pypdf]"

    return "[Unsupported file format]"
