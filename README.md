# IR Assignment 1 — CISI Information Retrieval System

## Dataset

Download the CISI test collection from:
https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/

You need three files:
- `CISI.ALL` — 1,460 document abstracts
- `CISI.QRY` — 112 queries
- `CISI.REL` — relevance judgments

## Installation

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run.

## Running the app

```bash
streamlit run app.py
```

Then in the browser:
1. Upload `CISI.ALL`, `CISI.QRY`, and `CISI.REL` using the sidebar uploaders.
2. Click **Load & Index Dataset**.
3. Navigate between sections using the sidebar radio buttons.

## Sections

| Section | Content |
|---------|---------|
| A · Documents | Browse and view the document collection |
| B · Preprocessing | Step-by-step pipeline + stemming vs lemmatization evaluation |
| C · Phrase Query | Biword index vs positional index with false-positive analysis |
| D · Dictionary | BST vs B-Tree lookup benchmark |
| E · Tolerant Retrieval | Wildcard (k-gram), spelling correction, phonetic (Soundex) |
| G · Inferences | Full discussion and conclusions |

## File structure

```
ir_assignment/
├── app.py             # Main Streamlit application
├── cisi_parser.py     # CISI file format parsers
├── ir_core.py         # All IR data structures and algorithms
├── requirements.txt
└── README.md
```
