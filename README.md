# 📄 Adobe Hackathon – Round 1A Solution

## ✨ Challenge: Understanding PDF Structure (Connecting the Dots)

In this solution for Round 1A of the Adobe Hackathon, we built an intelligent PDF structure extractor that detects document outlines (Title, H1, H2, H3) from unstructured PDF files — without relying solely on font sizes. The extracted outline is saved in a hierarchical JSON format as required by the problem statement.

---

## 🧠 Problem Statement

Given a PDF (up to 50 pages), extract:

- **Title**
- **Headings**: H1, H2, H3
- Output format: JSON with hierarchy and page numbers

This enables smarter document experiences like semantic search, summarization, and outline-based navigation.

---

## 🏗️ Solution Overview

Our approach is divided into **two stages**:

### 1. Feature Extraction (via `create_csv.py`)

We use `PyMuPDF` to extract formatting features (e.g., font size, bold, italic, position, line spacing) from each line of the PDF. This raw data is then filtered to remove repetitive headers/footers and saved as a CSV file in the `csv/` directory.

Key Features Extracted:

- Font size, bold/italic flags
- Relative position on the page
- Character and word counts
- Line spacing, heading-like patterns (e.g., all caps, title case)

### 2. Heading Classification Model (via `model.py`)

We train a **Random Forest Classifier** on labeled PDF lines to predict heading levels (H1, H2, H3, or body text). It uses both primitive and engineered features like:

- `font_size_ratio`
- `is_bold`, `is_all_caps`
- `relative_y`, `formatting_score`
- `starts_with_number`, `font_emphasis`, etc.

The trained model classifies lines, and we post-process the results to generate a valid outline JSON.

---

## 🛠️ Libraries Used

| Library                | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| **PyMuPDF**            | PDF text + layout extraction                |
| **pandas**             | Data handling and CSV I/O                   |
| **numpy**              | Numerical operations                        |
| **scikit-learn**       | Random Forest classifier, feature selection |
| **joblib**             | Model serialization                         |
| **dateutil**, **pytz** | Timestamp generation (optional metadata)    |

Dependencies are pinned for consistency:

```

joblib==1.5.1
numpy==2.2.6
pandas==2.3.1
PyMuPDF==1.26.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.1
scipy==1.15.3
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2

```

---

## 🐳 Docker & Offline Requirements

- ✅ Compatible with `linux/amd64` (no GPU)
- ✅ No internet access required
- ✅ Model size ≤ 200MB
- ✅ Processes 50-page PDFs in <10 seconds
- ✅ All dependencies installed inside Docker

---

## 🚀 How to Build & Run (Docker)

You can build the image with:

```bash
docker build --platform linux/amd64 -t 1a .
```

Run the container using:

```bash
docker run --rm
  -v $(pwd)/input:/app/pdfs
  -v $(pwd)/output:/app/output
  --network none
   1a
```

It processes all `.pdf` files in the `/input` directory and writes corresponding `.json` outline files in `/output`.

---

## 📂 Directory Structure

```
.
├── create_csv.py            # PDF feature extractor
├── model.py                 # Heading classification model
├── csv/                     # Stores intermediate feature CSVs
├── pdfs/                    # Input PDFs
├── output/                  # Final outline JSONs
├── Dockerfile               # Offline-compatible Dockerfile
└── README.md
```

---
