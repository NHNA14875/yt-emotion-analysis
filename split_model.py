# Script untuk memecah file besar agar bisa upload ke GitHub
import os

CHUNK_SIZE = 90 * 1024 * 1024 # 90MB per chunk (batas aman GitHub 100MB)
input_file = 'model_save/pytorch_model.bin'

if os.path.exists(input_file):
    with open(input_file, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            with open(f'{input_file}.part{chunk_num}', 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_num += 1
    print("Model berhasil dipecah! Sekarang hapus/pindahkan file pytorch_model.bin yang asli.")