NLP Project: NER Resume extractions 
Models used: 
- CRF 
- HMM Generative 
- BiLSTM-CRF 
- BiLSTM-CRF + LSTM char embed
- DistilBERT 
- BERT 
- Llama  


## Setup Instructions

Follow these steps to run the project locally.

---

### 1. Clone the Repository

```bash
git clone <https://github.com/michellaprojects/nlp-ner>
cd nlp_ner
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

---

### 3. Activate Environment

**Windows (PowerShell):**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If `requirements.txt` is not provided:

```bash
pip install jupyterlab ipykernel pandas numpy scikit-learn spacy transformers datasets
```

---

### 5. Set Up Jupyter Kernel (Optional)

```bash
python -m ipykernel install --user --name nlp-ner --display-name "Python (nlp-ner)"
```

---

### 6. Run the Project

**Option A:  Jupyter Lab**

```bash
jupyter lab
```

**Option B: VSCode**

```bash
code .
```

Then select the `.venv` interpreter and kernel.

---

### 7. Verify Environment

Run in Python:

```python
import sys
print(sys.executable)
```

Expected: path should point to `.venv`.

---

### Notes

* Each project uses its own virtual environment (`.venv`)
* Do not move the `.venv` folder after creation
* If environment breaks, delete `.venv` and reinstall

---

### Quick Reset

```bash
rm -rf .venv   # Windows: rmdir /s /q .venv
python -m venv .venv
# activate again, then reinstall
```

