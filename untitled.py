# untitled.py
# PDF tabanlı yapay zeka asistanı + FAISS vektör veritabanı + 8B LLM + cache + gelişmiş ask() + terminal sohbet

import os
import re
import json
import textwrap
from dataclasses import dataclass
from time import perf_counter as timer

import requests
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# bitsandbytes opsiyonel (GPU + 4bit için)
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


# -----------------------------
# 1) CONFIG
# -----------------------------

@dataclass
class Config:
    # PDF
    pdf_path: str = "human-nutrition-text.pdf"
    pdf_url: str = "https://github.com/mrdbourke/simple-local-rag/raw/main/human-nutrition-text.pdf"
    page_number_offset: int = 41

    # Chunk
    num_sentence_chunk_size: int = 10
    min_token_length: int = 30  # çok kısa chunk'leri ele

    # Retrieval
    top_k: int = 5
    min_score_threshold: float = 0.20  # düşükse "bilmiyorum"

    # Cache
    cache_dir: str = "cache"
    corpus_json: str = "cache/corpus.json"
    faiss_index_path: str = "cache/faiss.index"
    embeddings_path: str = "cache/embeddings.npy"

    # Models
    embedding_model_name: str = "all-mpnet-base-v2"
    llm_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 8B


cfg = Config()

# Global runtime objects
embedding_model = None
faiss_index = None
corpus_chunks = None
tokenizer = None
llm_model = None


# -----------------------------
# 2) UTILS
# -----------------------------

def ensure_cache_dir():
    os.makedirs(cfg.cache_dir, exist_ok=True)

def print_wrapped(text: str, width: int = 90):
    print(textwrap.fill(text, width))

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def split_list(input_list, slice_size):
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]


# -----------------------------
# 3) PDF LOAD
# -----------------------------

def download_pdf_if_needed():
    if not os.path.exists(cfg.pdf_path):
        print(f"[INFO] {cfg.pdf_path} bulunamadı. İndiriliyor...")
        resp = requests.get(cfg.pdf_url)
        resp.raise_for_status()
        with open(cfg.pdf_path, "wb") as f:
            f.write(resp.content)
        print(f"[INFO] PDF indirildi: {cfg.pdf_path}")
    else:
        print(f"[INFO] '{cfg.pdf_path}' zaten var.")

def open_and_read_pdf():
    doc = fitz.open(cfg.pdf_path)
    pages_and_texts = []

    print("[INFO] PDF okunuyor (sayfa bazlı)...")
    for page_number, page in tqdm(list(enumerate(doc))):
        text = text_formatter(page.get_text())
        pages_and_texts.append(
            {
                "page_number": page_number - cfg.page_number_offset,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
            }
        )
    return pages_and_texts


# -----------------------------
# 4) CHUNKING
# -----------------------------

def build_sentencizer():
    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp

def build_pages_and_chunks(pages_and_texts):
    nlp = build_sentencizer()

    print("[INFO] Sayfa metinleri cümlelere ayrılıyor...")
    for item in tqdm(pages_and_texts):
        item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])

    print("[INFO] Cümleler chunk'lere ayrılıyor...")
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(item["sentences"], cfg.num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    print("[INFO] Chunk'ler tek liste haline getiriliyor...")
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)

            chunk_token_count = len(joined) / 4
            if chunk_token_count < cfg.min_token_length:
                continue

            pages_and_chunks.append(
                {
                    "page_number": item["page_number"],
                    "sentence_chunk": joined,
                    "chunk_char_count": len(joined),
                    "chunk_word_count": len(joined.split(" ")),
                    "chunk_token_count": chunk_token_count,
                }
            )
    return pages_and_chunks


# -----------------------------
# 5) EMBEDDINGS + FAISS (VECTOR DB)
# -----------------------------

def build_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Embedding modeli cihazı: {device}")
    model = SentenceTransformer(cfg.embedding_model_name, device=device)
    return model

def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def build_faiss_index(emb: np.ndarray):
    emb = normalize_rows(emb).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (normalize + inner product)
    index.add(emb)
    return index

def save_vector_db(corpus: list, emb: np.ndarray, index):
    ensure_cache_dir()
    with open(cfg.corpus_json, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    np.save(cfg.embeddings_path, emb)
    faiss.write_index(index, cfg.faiss_index_path)
    print("[INFO] Cache kaydedildi (corpus + embeddings + index).")

def load_vector_db_if_exists():
    if os.path.exists(cfg.corpus_json) and os.path.exists(cfg.embeddings_path) and os.path.exists(cfg.faiss_index_path):
        print("[INFO] Cache bulundu. Diskten yükleniyor...")
        with open(cfg.corpus_json, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        emb = np.load(cfg.embeddings_path)
        index = faiss.read_index(cfg.faiss_index_path)
        return corpus, emb, index
    return None, None, None

def build_or_load_vector_db():
    global embedding_model, corpus_chunks, faiss_index

    corpus, emb, index = load_vector_db_if_exists()
    if corpus is not None:
        corpus_chunks = corpus
        faiss_index = index
        # embedding_model retrieval için yine lazım
        if embedding_model is None:
            embedding_model = build_embedding_model()
        return

    # Cache yoksa oluştur
    download_pdf_if_needed()
    pages_and_texts = open_and_read_pdf()
    pages_and_chunks = build_pages_and_chunks(pages_and_texts)

    df = pd.DataFrame(pages_and_chunks)
    print("[INFO] Chunk istatistikleri:")
    print(df.describe().round(2))

    corpus = df.to_dict(orient="records")
    texts = [c["sentence_chunk"] for c in corpus]

    if embedding_model is None:
        embedding_model = build_embedding_model()

    print("[INFO] Chunk embedding'leri hesaplanıyor...")
    emb = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print("[INFO] FAISS index kuruluyor...")
    index = build_faiss_index(emb)

    save_vector_db(corpus, emb, index)

    corpus_chunks = corpus
    faiss_index = index


# -----------------------------
# 6) RETRIEVAL
# -----------------------------

def retrieve_relevant_resources(query: str, top_k: int = None, print_time: bool = True):
    global embedding_model, faiss_index, corpus_chunks

    if top_k is None:
        top_k = cfg.top_k

    if embedding_model is None or faiss_index is None or corpus_chunks is None:
        raise RuntimeError("[ERROR] Vector DB veya embedding modeli hazır değil!")

    q_emb = embedding_model.encode(query, convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    q_emb = np.expand_dims(q_emb.astype("float32"), axis=0)

    start = timer()
    scores, indices = faiss_index.search(q_emb, top_k)
    end = timer()

    if print_time:
        print(f"[INFO] Retrieval: {faiss_index.ntotal} vektörde arama {end - start:.4f} sn")

    scores = scores[0]
    indices = indices[0].astype(int)
    items = [corpus_chunks[i] for i in indices]
    return scores, items


# -----------------------------
# 7) 8B LLM LOAD (LLAMA 3.1 8B)
# -----------------------------

def build_llm_model(model_id: str = None):
    """
    8B model yükleme.
    - GPU varsa 4-bit quant önerilir (bitsandbytes gerekli).
    - HF gated model: HF login / token gerekli.
      (Token'ı koda yazma. Gerekirse environment:
        set HF_TOKEN=hf_xxx  (Windows CMD)
        $env:HF_TOKEN="hf_xxx" (PowerShell)
      ama en temizi: huggingface-cli login
    """
    global tokenizer, llm_model

    if model_id is None:
        model_id = cfg.llm_model_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] LLM cihazı: {device}")
    print(f"[INFO] LLM modeli: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cuda":
        if not BNB_AVAILABLE:
            raise RuntimeError(
                "[ERROR] GPU var ama bitsandbytes yok. Kur: pip install -U bitsandbytes"
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        # CPU: çalışır ama çok yavaş olabilir
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)


# -----------------------------
# 8) ASK (RAG)
# -----------------------------

def ask(
    query: str,
    temperature: float = 0.4,
    max_new_tokens: int = 256,
    top_k: int = None,
    return_answer_only: bool = True,
):
    global tokenizer, llm_model

    if tokenizer is None or llm_model is None:
        raise RuntimeError("[ERROR] LLM modeli hazır değil. Önce build_llm_model() çalıştır.")

    scores, context_items = retrieve_relevant_resources(query, top_k=top_k, print_time=True)

    best_score = float(scores[0]) if len(scores) else 0.0
    if best_score < cfg.min_score_threshold:
        answer_text = (
            "Bu soru için PDF içinde yeterli düzeyde bilgi bulamadım. "
            "Yanlış bilgi vermemek için cevap üretmiyorum."
        )
        return (answer_text if return_answer_only else (answer_text, context_items, scores))

    # Context metni
    context_text = ""
    for i, item in enumerate(context_items, start=1):
        context_text += f"[Paragraf {i} | Sayfa {item['page_number']} | Skor {scores[i-1]:.4f}]\n"
        context_text += item["sentence_chunk"] + "\n\n"

    prompt = f"""
You are a helpful assistant that answers questions ONLY using the PDF context below.
If the context is not enough, say you don't know. Do NOT make up facts.
Answer in clear and simple Turkish.

PDF Context:
{context_text}

Question: {query}

Answer (Turkish):
""".strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = llm_model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_text = decoded.replace(prompt, "").strip()

    if return_answer_only:
        return answer_text
    return answer_text, context_items, scores


# -----------------------------
# 9) CHAT LOOP
# -----------------------------

def chat_loop():
    print("\n📚 PDF Tabanlı Yapay Zeka Asistanı'na hoş geldin.")
    print("Çıkmak için 'q', 'quit', 'exit', 'çık' yazabilirsin.\n")

    while True:
        question = input("Soru: ").strip()
        if question.lower() in ["q", "quit", "exit", "çık", "çıkış"]:
            print("Görüşürüz 👋")
            break

        try:
            answer, context_items, scores = ask(
                query=question,
                return_answer_only=False,
            )
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        print("\n--- CEVAP ---\n")
        print(answer)

        print("\n--- KAYNAK PARAGRAFLAR (PDF'ten) ---")
        for i, item in enumerate(context_items, start=1):
            print(f"\n[Paragraf {i} | Sayfa {item['page_number']} | Skor {scores[i-1]:.4f}]")
            print_wrapped(item["sentence_chunk"], width=90)
        print("\n")


# -----------------------------
# 10) MAIN PIPELINE
# -----------------------------

def main():
    global embedding_model

    # 1) Vector DB hazırla (cache varsa hızlı)
    build_or_load_vector_db()

    # 2) LLM yükle (8B)
    build_llm_model(cfg.llm_model_id)

    # 3) Terminal sohbet
    chat_loop()


if __name__ == "__main__":
    main()
