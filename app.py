import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
import pickle
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

# --- AYARLAR ---
UPLOAD_FOLDER = "pdfs"
VECTORS_FOLDER = "vectors"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORS_FOLDER, exist_ok=True)

# MULTILINGUAL EMBEDDING MODELİ
# Türkçe ve İngilizceyi aynı uzayda anlar.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

app = Flask(__name__)
CORS(app)

# Modeli bellekte tutmak için global değişken
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] Embedding modeli yükleniyor ({device})...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return embedding_model

# --- PDF İŞLEME ---

def process_and_save_pdf(file_path, filename, offset=0):
    """PDF'i okur, chunklara böler, vektörleştirir ve kaydeder."""
    model = get_embedding_model()
    doc = fitz.open(file_path)
    chunks = []
    
    print(f"[PROCESS] {filename} işleniyor. Ofset: {offset}")
    
    for page_num, page in enumerate(doc):
        if page_num < offset: # Ofset kontrolü
            continue
            
        text = page.get_text().replace("\n", " ").strip()
        if len(text) < 50: continue # Boş sayfaları atla

        # Cümle bazlı chunking (10 cümlelik bloklar)
        sentences = text.split(". ")
        chunk_size = 10
        
        for i in range(0, len(sentences), chunk_size):
            chunk_text = ". ".join(sentences[i:i+chunk_size])
            if len(chunk_text) < 30: continue
            
            chunks.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": chunk_text
            })

    if not chunks:
        return False, "PDF'den metin çıkarılamadı."

    print(f"[EMBED] {len(chunks)} parça vektöre çevriliyor...")
    df = pd.DataFrame(chunks)
    
    # Embedding hesapla ve listeye çevir
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    df["embedding"] = list(embeddings)
    
    # Pickle olarak kaydet
    save_path = os.path.join(VECTORS_FOLDER, f"{filename}.pkl")
    df.to_pickle(save_path)
    
    return True, f"{len(chunks)} parça işlendi."

# --- VEKTÖR ARAMA (NUMPY DOT PRODUCT) ---


def search_in_vectors(query, selected_files, top_k=5):
    """Seçili dosyalarda matematiksel benzerlik araması yapar."""
    model = get_embedding_model()
    
    # 1. Seçili dosyaları yükle
    dataframes = []
    for fname in selected_files:
        pkl_path = os.path.join(VECTORS_FOLDER, f"{fname}.pkl")
        if os.path.exists(pkl_path):
            dataframes.append(pd.read_pickle(pkl_path))
    
    if not dataframes:
        return []

    # Verileri birleştir
    full_df = pd.concat(dataframes, ignore_index=True)
    
    # 2. Embeddingleri Matrise Çevir (Stacking)
    embeddings_matrix = np.stack(full_df["embedding"].values)
    
    # 3. Sorguyu Vektöre Çevir
    query_embedding = model.encode(query, convert_to_tensor=False)
    
    # 4. Dot Product (Benzerlik Hesaplama)
    # Vektörleri normalize et (Cosine Similarity için)
    embeddings_matrix = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    
    scores = np.dot(embeddings_matrix, query_embedding)
    
    # 5. En iyi sonuçları bul
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        row = full_df.iloc[idx]
        results.append({
            "score": float(scores[idx]),           # float() çevirimi önemli
            "text": row["text"],
            "filename": row["filename"],
            "page_number": int(row["page_number"]) # BURASI DÜZELDİ: int64 -> int
        })
        
    return results

# --- ENDPOINTLER ---

@app.route("/files", methods=["GET"])
def list_files():
    """İşlenmiş dosyaları listeler."""
    files = [f.replace(".pkl", "") for f in os.listdir(VECTORS_FOLDER) if f.endswith(".pkl")]
    return jsonify(files)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Dosya yok"}), 400
    
    file = request.files["file"]
    offset = int(request.form.get("offset", 0))
    
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    success, msg = process_and_save_pdf(file_path, filename, offset)
    
    if success:
        return jsonify({"message": msg, "filename": filename}), 200
    else:
        return jsonify({"error": msg}), 500

# app.py içindeki ask fonksiyonunu bununla değiştir:

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    selected_files = data.get("selected_files", [])
    model_name = data.get("model_name", "llama3.1")
    
    # Dil seçimi ve uzunluk ayarları KALDIRILDI.
    
    if not question or not selected_files:
        return jsonify({"error": "Soru ve dosya seçimi zorunlu."}), 400

    try:
        # 1. Arama Yap
        results = search_in_vectors(question, selected_files)
        
        if not results:
            return jsonify({"answer": "Seçilen belgelerde ilgili bilgi bulunamadı.", "sources": []})

        # 2. Context Oluştur
        context_text = ""
        for res in results:
            context_text += f"[Dosya: {res['filename']} | Sayfa: {res['page_number']}]\n{res['text']}\n\n"

        # 3. Prompt Hazırla (Sadeleştirilmiş)
        # Modele sadece bağlamı kullanmasını söylüyoruz, diline karışmıyoruz.
        prompt = f"""
        Aşağıdaki "BİLGİLER" kısmındaki metinleri kullanarak "SORU"yu cevapla. 
        Eğer bilgilerin içinde cevap yoksa "Bilmiyorum" de.
        
        BİLGİLER (CONTEXT):
        {context_text}
        
        SORU: {question}
        
        CEVAP:
        """
        
        print(f"[LLM] Model: {model_name} | Soru: {question}")
        
        # 4. Ollama'ya Gönder
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        answer = response['message']['content']
        
        # Kaynakları düzenle
        sources = [{
            "filename": r["filename"],
            "page_number": r["page_number"],
            "preview": r["text"][:100] + "..."
        } for r in results]

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        print(f"HATA: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)