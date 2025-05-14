import os
import uuid
import fitz  # PyMuPDF
from pathlib import Path
import re
import unidecode



def sanitize_filename(name: str) -> str:
    return os.path.basename(name)


def upload_pdf(uploaded_file, pdf_directory: str) -> Path:
    pdf_directory = Path(pdf_directory)
    pdf_directory.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4()}_{sanitize_filename(uploaded_file.name)}"
    save_path = pdf_directory / unique_name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def clean_hungarian_text(text: str) -> str:
    # a szöveg közbeni [1] cuccosok törlése
    text = re.sub(r"\[\d+\]", "", text)

    # Az 1.Bevezetés és a hasonlók kijelölése
    text = re.sub(r"(\d+\.\s*[A-Za-záéíóöőúüű]+)", r"[ÚJ SZEKCIÓ] \1", text)

    # extra sorkihagyások, tabok, spacek
    text = re.sub(r"\s+", " ", text)

    # ha egy paragrafus közepén lenen vége a mondatnak az ne történjen meg
    text = re.sub(r"(?<=[^\n])\n(?=\S)", " ", text)

    # ne legyen space irásjelek végén
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)

    # egyforma - legyenek
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    text = text.strip()
    return text

def remove_metadata_from_pdf(pdf_path: Path) -> None:
    with fitz.open(pdf_path) as doc:
        doc.metadata = {}
        doc.save(pdf_path)

def process_pdf(uploaded_file, pdf_directory: str = "uploaded_pdfs") -> str:
    pdf_path = upload_pdf(uploaded_file, pdf_directory)
    remove_metadata_from_pdf(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_hungarian_text(raw_text)
    return cleaned_text
