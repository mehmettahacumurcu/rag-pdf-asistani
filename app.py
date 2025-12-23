import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- MODEL YÖNETİMİ ---
# Desteklenen Modeller
AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Hızlı, 128 Token
    "e5-base": "intfloat/multilingual-e5-base" # Akıllı, 512 Token (Önerilen)
}

# Global değişkenler (Hafıza Yönetimi İçin)
embedding_model = None
current_model_key = None

def get_embedding_model(model_key="e5-base"):
    global embedding_model, current_model_key
    
    # İstenen modelin HuggingFace ID'sini al
    target_model_name = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["e5-base"])
    
    # Eğer model zaten yüklüyse ve aynısıysa tekrar yükleme
    if embedding_model is not None and current_model_key == model_key:
        return embedding_model
    
    # Farklı bir model istendiyse, Eskisini Sil (VRAM Temizliği 🧹)
    if embedding_model is not None:
        print(f"[MEMORY] {current_model_key} hafızadan siliniyor...")
        del embedding_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Yeni model yükleniyor: {model_key} ({device})...")
    
    embedding_model = SentenceTransformer(target_model_name, device=device)
    current_model_key = model_key
    return embedding_model

# --- METİN PARÇALAMA (CHUNKING) ---
def chunk_text(text, chunk_size=500, overlap=100):
    """
    Metni karakter sayısına göre böler (Token sınırını aşmamak için).
    Örtüşme (overlap) sayesinde cümleler bölünse bile anlam kaybı olmaz.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # Kelime ortasından bölmemek için en yakın boşluğu bul
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end].strip()
        if len(chunk) > 30: # Çok kısa çöp parçaları alma
            chunks.append(chunk)
        
        # Bir sonraki parça için overlap kadar geri git
        start = end - overlap
        if start >= len(text): break
        
    return chunks

# --- PDF İŞLEME ---
def process_and_save_pdf(file_path, filename, model_key="e5-base", offset=0):
    """PDF'i okur, chunklara böler, SEÇİLEN MODELE göre vektörleştirir."""
    model = get_embedding_model(model_key)
    doc = fitz.open(file_path)
    chunks_data = []
    
    print(f"[PROCESS] {filename} işleniyor ({model_key})...")
    
    for page_num, page in enumerate(doc):
        if page_num < offset: continue
            
        text = page.get_text().replace("\n", " ").strip()
        if len(text) < 50: continue

        # Karakter bazlı chunking (Eski cümle bazlı sistem yerine)
        text_chunks = chunk_text(text, chunk_size=500, overlap=100)
        
        for chunk in text_chunks:
            chunks_data.append({
                "filename": filename,
                "page_number": page_num + 1,
                "text": chunk
            })

    if not chunks_data:
        return False, "PDF'den metin çıkarılamadı."

    print(f"[EMBED] {len(chunks_data)} parça vektöre çevriliyor...")
    df = pd.DataFrame(chunks_data)
    
    # --- E5 İÇİN ÖZEL PREFIX AYARI ---
    if "e5" in model_key:
        # E5 modelleri "passage: " etiketi ister
        texts_to_embed = ["passage: " + t for t in df["text"].tolist()]
        # E5 için normalize_embeddings=True önerilir
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, normalize_embeddings=True)
    else:
        # MiniLM için direkt metni veriyoruz
        embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
        
    df["embedding"] = list(embeddings)
    
    # --- KLASÖRLEME (MODEL BAZLI) ---
    # vectors/e5-base/dosya.pkl  veya  vectors/minilm/dosya.pkl
    model_folder = os.path.join(VECTORS_FOLDER, model_key)
    os.makedirs(model_folder, exist_ok=True)
    
    save_path = os.path.join(model_folder, f"{filename}.pkl")
    df.to_pickle(save_path)
    
    return True, f"{len(chunks_data)} parça işlendi ({model_key})."

# --- VEKTÖR ARAMA ---
def search_in_vectors(query, selected_files, model_key="e5-base", top_k=5):
    """Seçili modelin klasöründeki dosyalarda arama yapar."""
    model = get_embedding_model(model_key)
    
    dataframes = []
    # Sadece seçili modelin klasörüne bakıyoruz!
    model_folder = os.path.join(VECTORS_FOLDER, model_key)
    
    if not os.path.exists(model_folder):
        return []

    for fname in selected_files:
        pkl_path = os.path.join(model_folder, f"{fname}.pkl")
        if os.path.exists(pkl_path):
            dataframes.append(pd.read_pickle(pkl_path))
    
    if not dataframes:
        return []

    full_df = pd.concat(dataframes, ignore_index=True)
    embeddings_matrix = np.stack(full_df["embedding"].values)
    
    # --- E5 İÇİN SORGU PREFIX AYARI ---
    if "e5" in model_key:
        query_text = "query: " + query
        query_embedding = model.encode(query_text, convert_to_tensor=False, normalize_embeddings=True)
    else:
        query_embedding = model.encode(query, convert_to_tensor=False)
    
    # Benzerlik Hesaplama (Dot Product)
    # E5 normalize edildiği için Dot Product == Cosine Similarity
    if "e5" not in model_key:
        # MiniLM normalize edilmediyse burada edelim (garanti olsun)
        embeddings_matrix = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    
    scores = np.dot(embeddings_matrix, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        row = full_df.iloc[idx]
        results.append({
            "score": float(scores[idx]),
            "text": row["text"],
            "filename": row["filename"],
            "page_number": int(row["page_number"])
        })
        
    return results

# --- ENDPOINTLER ---

@app.route("/files", methods=["GET"])
def list_files():
    """Seçili modele ait dosyaları listeler."""
    # Frontend'den ?model_key=e5-base şeklinde parametre gelir
    model_key = request.args.get("model_key", "e5-base")
    
    folder = os.path.join(VECTORS_FOLDER, model_key)
    if not os.path.exists(folder):
        return jsonify([])
        
    files = [f.replace(".pkl", "") for f in os.listdir(folder) if f.endswith(".pkl")]
    return jsonify(files)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Dosya yok"}), 400
    
    file = request.files["file"]
    offset = int(request.form.get("offset", 0))
    # Frontend'den seçili embedding modelini alıyoruz
    embedding_model_key = request.form.get("embedding_model", "e5-base")
    
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Modeli parametre olarak gönderiyoruz
    success, msg = process_and_save_pdf(file_path, filename, model_key=embedding_model_key, offset=offset)
    
    if success:
        return jsonify({"message": msg, "filename": filename}), 200
    else:
        return jsonify({"error": msg}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    selected_files = data.get("selected_files", [])
    llm_model_name = data.get("model_name", "llama3.1") # Bu Qwen/Llama (LLM)
    embedding_model_key = data.get("embedding_model", "e5-base") # Bu E5/MiniLM (Embedding)
    
    if not question or not selected_files:
        return jsonify({"error": "Soru ve dosya seçimi zorunlu."}), 400

    try:
        # 1. Arama Yap (Model key ile)
        results = search_in_vectors(question, selected_files, model_key=embedding_model_key)
        
        if not results:
            return jsonify({"answer": "Seçilen belgelerde (veya bu embedding modelinde) ilgili bilgi bulunamadı.", "sources": []})

        # 2. Context Oluştur
        context_text = ""
        for res in results:
            context_text += f"[Dosya: {res['filename']} | Sayfa: {res['page_number']}]\n{res['text']}\n\n"

        # 3. Prompt Hazırla
        prompt = f"""
        Aşağıdaki "BİLGİLER" kısmındaki metinleri kullanarak "SORU"yu cevapla. 
        Eğer bilgilerin içinde cevap yoksa "Bilmiyorum" de.
        
        BİLGİLER (CONTEXT):
        {context_text}
        
        SORU: {question}
        
        CEVAP:
        """
        
        print(f"[LLM] Model: {llm_model_name} | Embed: {embedding_model_key} | Soru: {question}")
        
        # 4. Ollama'ya Gönder
        response = ollama.chat(model=llm_model_name, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        answer = response['message']['content']
        
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