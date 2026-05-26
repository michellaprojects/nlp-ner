"""
preprocessing.py
Semua fungsi preprocessing untuk inference NER — dipisah dari app.py.
Berdasarkan Preprocessing.ipynb dan bilstm-crf-lstm.ipynb.
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

CHAR_PAD = '<CPAD>'
CHAR_UNK = '<CUNK>'
MAX_LEN  = 256
MAX_CHAR = 20

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

# ── Tokenization ──────────────────────────────────────────────────────────────

def word_tokenize(text: str) -> list[tuple]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]

# ── Encoding (BiLSTM) ─────────────────────────────────────────────────────────

def encode_token(tok: str, token2id: dict, unk_id: int) -> int:
    return token2id.get(tok, unk_id)

def encode_chars(tok: str, char2id: dict, max_char: int = MAX_CHAR) -> list[int]:
    pad = char2id.get(CHAR_PAD, 0)
    unk = char2id.get(CHAR_UNK, 1)
    ids = [char2id.get(c, unk) for c in tok[:max_char]]
    return ids + [pad] * (max_char - len(ids))

# ── Full inference preprocessing ──────────────────────────────────────────────

def preprocess_for_inference(text: str, token2id: dict, char2id: dict,
                              pad_id: int, unk_id: int,
                              max_len: int = MAX_LEN,
                              max_char: int = MAX_CHAR) -> tuple:
    """
    Tokenize, normalize, encode — siap masuk ke model BiLSTM-CRF.

    Returns
    -------
    t_tokens  : torch.Tensor (1, max_len)
    t_chars   : torch.Tensor (1, max_len, max_char)
    t_mask    : torch.Tensor (1, max_len) bool
    raw_kept  : list[str]  — original tokens setelah noise removal (untuk display)
    """
    import torch

    raw_tokens = word_tokenize(text)
    norm_tokens, raw_kept = [], []

    for raw_tok, _, _ in raw_tokens:
        norm = normalize_token(raw_tok)
        if norm is not None:
            norm_tokens.append(norm)
            raw_kept.append(raw_tok)

    tok_ids  = [encode_token(t, token2id, unk_id) for t in norm_tokens[:max_len]]
    char_ids = [encode_chars(t, char2id, max_char) for t in norm_tokens[:max_len]]
    real_len = len(tok_ids)

    pad_char = [char2id.get(CHAR_PAD, 0)] * max_char
    tok_ids  += [pad_id] * (max_len - real_len)
    char_ids += [pad_char] * (max_len - real_len)
    mask      = [True] * real_len + [False] * (max_len - real_len)

    t_tokens = torch.tensor([tok_ids],  dtype=torch.long)
    t_chars  = torch.tensor([char_ids], dtype=torch.long)
    t_mask   = torch.tensor([mask],     dtype=torch.bool)

    return t_tokens, t_chars, t_mask, raw_kept

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
        result[key] = list(dict.fromkeys(values))   # deduplicate, preserve order
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
