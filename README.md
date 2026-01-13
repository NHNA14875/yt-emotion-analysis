# YouTube Emotion Analytics Dashboard (IndoBERT)
**Proyek UTS Sains Data & Komunikasi Interpersonal (STKI) - Kelompok 2**

Aplikasi analisis emosi berbasis Deep Learning yang dirancang untuk membantu *Content Creator* memahami respons emosional audiens melalui komentar YouTube secara otomatis dan real-time menggunakan model bahasa IndoBERT.

---

## 👥 Anggota Kelompok 2
1. **Naufal Hanif N.A** - A11.2023.14875
2. **Dhimas Bagus M** - A11.2023.14891
3. **Yobby Azriel Iqdhi V.** - A11.2023.14890
4. **Handis Pramudio** - A11.2023.14883
5. **Izzal Azmi R** - A11.2021.13703

---

## 🎯 Fitur Utama
- **Scraping Otomatis**: Mengambil hingga 500 komentar terbaru langsung dari YouTube via API.
- **Deep Learning Classifier**: Klasifikasi 5 emosi (Senang, Sedih, Marah, Takut, Netral) dengan model **IndoBERT**.
- **Visualisasi Interaktif**: Dashboard dengan statistik jumlah emosi, Bar Chart, dan Pie Chart (Persentase).
- **Text Finder & Filter**: Mencari kata kunci tertentu atau memfilter emosi spesifik dalam ribuan komentar.
- **Export CSV**: Mengunduh hasil klasifikasi lengkap untuk kebutuhan dokumentasi.
- **Scrollable Interface**: Antarmuka komentar yang rapi dengan fitur scrollbox.

---

## 🚀 Cara Penggunaan Aplikasi
1. **Akses Dashboard**: Buka URL aplikasi di Streamlit Cloud.
2. **Input URL**: Salin link video YouTube (misal: `https://www.youtube.com/watch?v=...`) ke kolom input.
3. **Atur Limit**: Tentukan jumlah komentar yang ingin dianalisis (50 hingga 500 komentar).
4. **Eksekusi**: Klik tombol **"Mulai Analisis"** dan tunggu model memproses data riil.
5. **Eksplorasi**: 
   - Lihat statistik distribusi emosi pada bagian grafik.
   - Gunakan **Sidebar** untuk mencari komentar dengan kata kunci tertentu.
   - Klik **"Unduh CSV"** jika ingin menyimpan hasil analisis ke laptop.

---

## 🔐 Panduan Konfigurasi API Key (PENTING)
Demi keamanan, API Key **tidak ditulis di dalam kode GitHub**. Kami menggunakan sistem **Streamlit Secrets**.

**Langkah Setting API:**
1. Masuk ke dashboard [Streamlit Cloud](https://share.streamlit.io/).
2. Klik ikon titik tiga `...` di samping nama aplikasi Anda, pilih **Settings**.
3. Pilih menu **Secrets**.
4. Masukkan kode berikut (ganti dengan YouTube API Key Anda):
   ```toml
   YOUTUBE_API_KEY = "MASUKKAN_API_KEY_ANDA_DISINI"