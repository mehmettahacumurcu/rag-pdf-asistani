import { useState, useEffect } from "react";
import "./index.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  
  // --- MODEL AYARLARI ---
  // LLM (Cevap veren zeka)
  const [selectedModel, setSelectedModel] = useState("llama3.1");
  // Embedding (Metni anlayan zeka) - Varsayılan E5
  const [embeddingModel, setEmbeddingModel] = useState("e5-base");

  // Server Ayarları (Ngrok)
  const [serverUrl, setServerUrl] = useState(""); 

  // Dosya State'leri
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  
  // Upload State'leri
  const [uploadFile, setUploadFile] = useState(null);
  const [offset, setOffset] = useState(0);
  const [uploading, setUploading] = useState(false);

  // Embedding modeli değiştiğinde dosya listesini yenile
  useEffect(() => {
    fetchFiles();
  }, [embeddingModel]);

  const fetchFiles = async () => {
    try {
      // Backend'e hangi modelin klasörüne bakacağını söylüyoruz
      const res = await fetch(`http://localhost:5000/files?model_key=${embeddingModel}`);
      const data = await res.json();
      setAvailableFiles(data);
    } catch (err) {
      console.error("Yerel dosya listesi alınamadı.", err);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploading(true);
    
    const formData = new FormData();
    formData.append("file", uploadFile);
    formData.append("offset", offset);
    // Backend'e hangi model ile vektörleştireceğini söylüyoruz
    formData.append("embedding_model", embeddingModel);

    let logs = "";

    try {
      // 1. Önce Yerele (Laptop) Yükle
      const resLocal = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
      if (resLocal.ok) logs += `✅ Laptop: Dosya "${embeddingModel}" formatında işlendi.\n`;
      else logs += "❌ Laptop yüklemesi başarısız.\n";

      // 2. Server URL varsa Oraya da Yükle
      if (serverUrl.trim()) {
        try {
          const cleanUrl = serverUrl.replace(/\/$/, ""); 
          const resServer = await fetch(`${cleanUrl}/upload`, {
            method: "POST",
            body: formData,
          });
          if (resServer.ok) logs += `✅ EV PC: Dosya "${embeddingModel}" formatında işlendi.\n`;
          else logs += "❌ EV PC yüklemesi başarısız.\n";
        } catch (e) {
          logs += "⚠️ Evdeki PC'ye ulaşılamadı.\n";
        }
      }

      alert(logs);
      setUploadFile(null);
      setOffset(0);
      fetchFiles(); // Listeyi yenile

    } catch (err) {
      alert("Yükleme hatası.");
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async (target) => {
    if (!input.trim()) return;
    if (selectedFiles.length === 0) {
      alert("Lütfen dosya seçin!");
      return;
    }
    if (target === 'server' && !serverUrl.trim()) {
      alert("Lütfen önce Server (Ngrok) adresini gir!");
      return;
    }

    const userMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    const targetEndpoint = target === 'local' 
      ? "http://localhost:5000/ask" 
      : `${serverUrl.replace(/\/$/, "")}/ask`;

    const assistantLabel = target === 'local' ? "💻 Laptop" : "🏠 Ev PC";

    try {
      const res = await fetch(targetEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          selected_files: selectedFiles,
          model_name: selectedModel,     // LLM (Qwen/Llama)
          embedding_model: embeddingModel // Embedding (E5/MiniLM)
        }),
      });

      const data = await res.json();
      
      if (data.error) {
        setMessages((prev) => [...prev, { role: "assistant", text: `[${assistantLabel}] Hata: ` + data.error }]);
      } else {
        setMessages((prev) => [...prev, { 
          role: "assistant", 
          text: `[${assistantLabel}] ${data.answer}`, 
          sources: data.sources 
        }]);
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `[${assistantLabel}] Bağlantı hatası!` }]);
    } finally {
      setLoading(false);
    }
  };

  const toggleFile = (fname) => {
    setSelectedFiles(prev => 
      prev.includes(fname) ? prev.filter(f => f !== fname) : [...prev, fname]
    );
  };

  return (
    <div className="app-root">
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-icon">📚</span>
          <span className="logo-text">RAG Asistanı</span>
        </div>

        <div className="sidebar-section">
          <h3>⚙️ Ayarlar</h3>
          
          {/* EMBEDDING MODEL SEÇİMİ (YENİ) */}
          <div className="setting-group">
            <label className="setting-label">🧠 Embedding (Hafıza):</label>
            <select 
              value={embeddingModel} 
              onChange={(e) => {
                setEmbeddingModel(e.target.value);
                setSelectedFiles([]); // Model değişince seçimleri temizle
              }}
              className="model-select"
              
            >
              <option value="e5-base">E5-Base (Önerilen - Akıllı) 🌟</option>
              <option value="minilm">MiniLM (Hızlı - Eski)</option>
            </select>
            <p style={{fontSize:"9px", color:"#666", marginTop:"2px"}}>
              *Değişince dosya listesi yenilenir.
            </p>
          </div>

          <div className="setting-group">
            <label className="setting-label">🤖 Yapay Zeka (LLM):</label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-select"
            >
              <option value="llama3.1">Llama 3.1 (8B)</option>
              <option value="qwen2.5:3b">Qwen 2.5 (3B) - Hızlı</option>
              <option value="qwen2.5:14b">Qwen 2.5 (14B) - Türkçe (Ağır)</option>
              <option value="mistral-nemo">Mistral NeMo (12B)</option>
              <option value="solar">Solar (10.7B)</option>
            </select>
          </div>

          <div className="setting-group">
            <label className="setting-label">Ev PC Bağlantısı (Ngrok):</label>
            <input 
              type="text" 
              placeholder="https://xxxx.ngrok-free.app"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              className="model-select"
              style={{ fontSize: '11px' }}
            />
          </div>
        </div>

        <div className="sidebar-section upload-section">
          <h3>📤 PDF Yükle</h3>
          <p style={{fontSize:'10px', color:'#64748b', marginBottom:'5px'}}>
            (Seçili Embedding Modeli ile işlenir)
          </p>
          <input type="file" accept=".pdf" onChange={(e) => setUploadFile(e.target.files[0])} />
          <div className="offset-control">
            <label>Ofset:</label>
            <input type="number" min="0" value={offset} onChange={(e) => setOffset(e.target.value)} />
          </div>
          <button onClick={handleUpload} disabled={uploading || !uploadFile} className="upload-btn">
            {uploading ? "İşleniyor..." : "Yükle"}
          </button>
        </div>

        <div className="sidebar-section file-list-section">
          <h3>📂 Belgeler ({embeddingModel})</h3>
          {availableFiles.length === 0 ? <p className="no-files">Bu model için işlenmiş dosya yok.</p> : (
            <ul className="file-list">
              {availableFiles.map((f) => (
                <li key={f}>
                  <label>
                    <input type="checkbox" checked={selectedFiles.includes(f)} onChange={() => toggleFile(f)} />
                    {f}
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <main className="chat-wrapper">
        <header className="chat-header">
          <h1>Dokümanlarınla Konuş</h1>
          <p>
            {selectedFiles.length} belge seçili | 
            Hafıza: <strong>{embeddingModel}</strong> | 
            Zeka: <strong>{selectedModel}</strong>
          </p>
        </header>

        <div className="chat-box">
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <h2>Merhaba! 👋</h2>
                <p>Belgelerini seç ve sohbete başla.</p>
                <p style={{fontSize:"12px", color:"#888", marginTop:"10px"}}>
                   E5-Base modeli ile daha akıllı sonuçlar alabilirsiniz.
                </p>
              </div>
            )}
            {messages.map((m, idx) => (
              <div key={idx} className={`message-row ${m.role === "user" ? "message-user" : "message-assistant"}`}>
                <div className="avatar">{m.role === "user" ? "👤" : "🤖"}</div>
                <div className="bubble">
                  <div className="bubble-text">{m.text}</div>
                  {m.sources && m.sources.length > 0 && (
                    <div className="sources-block">
                      <div className="sources-title">Kaynaklar:</div>
                      <ul>
                        {m.sources.map((s, i) => (
                          <li key={i}>[{s.filename}] Syf {s.page_number} <span className="source-preview">"{s.preview}"</span></li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && <div className="loading">Yazıyor...</div>}
          </div>

          <div className="input-bar">
            <input 
              value={input} 
              onChange={(e) => setInput(e.target.value)} 
              placeholder="Soru sor..." 
              onKeyDown={(e) => e.key === 'Enter' && handleSend('local')}
            />
            <button 
              type="button" 
              onClick={() => handleSend('local')} 
              disabled={loading}
              style={{background: '#2563eb'}}
            >
              Gönder
            </button>
            
            <button 
              type="button" 
              onClick={() => handleSend('server')} 
              disabled={loading}
              title="Evdeki bilgisayarda çalıştırır"
              style={{background: '#7c3aed', marginLeft:'5px'}}
            >
              🚀 Server'a Gönder
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;