import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from googleapiclient.discovery import build

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Emosi YouTube IndoBERT", layout="wide")

# --- DEBUG INFO ---
with st.expander("🔍 Debug Info"):
    st.write("Working directory:", os.getcwd())
    if os.path.exists("model_save"):
        st.success("✅ model_save folder exists")
        for f in os.listdir("model_save"):
            size = os.path.getsize(os.path.join("model_save", f)) / (1024*1024)
            st.write(f"- {f}: {size:.2f} MB")
    else:
        st.error("❌ model_save not found!")

# --- KUNCI API YOUTUBE ---
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except:
    YOUTUBE_API_KEY = "AIzaSyBs9r3BO44zc4aROhhVWq2IXJWKsrByzkU"

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model_path = "./model_save"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# --- FUNGSI PREDIKSI ---
def predict_emotion(text, tokenizer, model, device):
    label_dict = {0: 'Senang', 1: 'Sedih', 2: 'Marah', 3: 'Takut', 4: 'Netral'}
    emoji_dict = {0: '😄', 1: '😢', 2: '😡', 3: '😱', 4: '😐'}
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    return label_dict[pred_idx], emoji_dict[pred_idx]

# --- FUNGSI SCRAPING ---
def get_comments(video_id, max_results=500):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        data = []
        next_page = None
        while len(data) < max_results:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100, pageToken=next_page, textFormat="plainText"
            ).execute()
            for item in res['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                data.append({
                    "Author": snippet['authorDisplayName'],
                    "Img": snippet['authorProfileImageUrl'],
                    "Text": snippet['textDisplay'],
                    "Likes": snippet.get('likeCount', 0),
                    "Date": snippet['publishedAt']
                })
            next_page = res.get('nextPageToken')
            if not next_page: break
        return data
    except Exception as e:
        st.error(f"Error getting comments: {e}")
        return []

# --- UI UTAMA ---
st.title("Analisis Emosi YouTube")

# Load model
try:
    tokenizer, model, device = load_model()
    st.success("✅ Model berhasil di-load!")
except:
    st.error("❌ Gagal load model. Cek debug info di atas.")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Filter & Ekspor")
search = st.sidebar.text_input("Cari kata kunci:")
emo_filter = st.sidebar.multiselect("Filter Emosi:", ["Senang", "Sedih", "Marah", "Takut", "Netral"])

url = st.text_input("Link Video YouTube:", placeholder="https://www.youtube.com/watch?v=...")
limit = st.select_slider("Limit Komentar:", options=[50, 100, 250, 500], value=100)

if st.button("Mulai Analisis", type="primary"):
    v_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if v_id:
        video_id = v_id.group(1)
        st.write(f"📹 Video ID: {video_id}")
        
        with st.spinner("Mengambil komentar..."):
            raw = get_comments(video_id, max_results=limit)
        
        if raw:
            st.success(f"✅ Berhasil ambil {len(raw)} komentar!")
            
            # Analisis emosi
            with st.spinner("Menganalisis emosi..."):
                counts = {'Senang': 0, 'Sedih': 0, 'Marah': 0, 'Takut': 0, 'Netral': 0}
                for item in raw:
                    emo, emj = predict_emotion(item['Text'], tokenizer, model, device)
                    item.update({"Emotion": emo, "Emoji": emj})
                    counts[emo] += 1
            
            # --- LAYOUT STATISTIK & VISUALISASI ---
            st.subheader("📊 Statistik & Visualisasi")
            col_stats, col_viz = st.columns([1, 2])
            
            with col_stats:
                st.write("**Jumlah Emosi**")
                for emo, num in counts.items():
                    st.metric(label=emo, value=num)
            
            with col_viz:
                tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
                with tab1:
                    st.bar_chart(pd.DataFrame.from_dict(counts, orient='index'))
                with tab2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    valid_labels = [k for k, v in counts.items() if v > 0]
                    valid_sizes = [v for v in counts.values() if v > 0]
                    colors = ['#4CAF50','#2196F3','#F44336','#FF9800','#9E9E9E']
                    ax.pie(valid_sizes, labels=valid_labels, autopct='%1.1f%%', startangle=140, colors=colors)
                    st.pyplot(fig)

            # --- DETAIL KOMENTAR ---
            st.subheader("💬 Detail Analisis Komentar")
            df = pd.DataFrame(raw)
            if search: 
                df = df[df['Text'].str.contains(search, case=False, na=False)]
            if emo_filter: 
                df = df[df['Emotion'].isin(emo_filter)]
            
            st.download_button("📥 Ekspor ke CSV", df.to_csv(index=False), "analisis_komentar.csv", "text/csv")

            # Scrollable box
            with st.container(height=500):
                for _, r in df.iterrows():
                    c1, c2 = st.columns([1, 12])
                    with c1:
                        st.markdown(f'<img src="{r["Img"]}" style="border-radius:50%; width:45px;">', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**{r['Author']}**")
                        st.write(r['Text'])
                        st.caption(f"Emosi: **{r['Emotion']} {r['Emoji']}** | 👍 {r['Likes']} | 📅 {r['Date']}")
                    st.markdown("---")
        else:
            st.error("❌ Tidak ada komentar yang ditemukan!")
    else:
        st.error("❌ Link YouTube tidak valid!")