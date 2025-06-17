# Deteksi Kepala, Kacamata, dan Topi dengan YOLO Kustom  
Program ini merupakan sistem deteksi objek untuk mendeteksi **kepala**, **kacamata**, dan **topi** pada gambar menggunakan algoritma YOLO yang telah dikustomisasi. Seluruh kode dan data berada di dalam folder `PROJECT PCD`.

## Fitur Utama
- Menggunakan arsitektur YOLO dengan backbone dan head yang dirancang sendiri.
- Dataset dan label diolah dalam format grid sesuai kebutuhan YOLO.
- Proses training, evaluasi, dan deteksi/inferensi terintegrasi.
- Visualisasi hasil deteksi lengkap dengan bounding box dan label kelas.
- 
## Cara Penggunaan
1. **Training Model:**
   - Jalankan `train.py` untuk melatih model dengan dataset yang sudah disiapkan.
2. **Deteksi Objek:**
   - Jalankan `detect.py` untuk melakukan deteksi pada gambar baru menggunakan model `.pt` hasil training.
3. **Visualisasi:**
   - Hasil deteksi dan visualisasi grid dapat dilihat di folder `output_yolo`.

## Catatan
- Pastikan semua dependensi (PyTorch, OpenCV, Ultralytics, dsb) sudah terinstall.
- Model YOLO yang digunakan telah dikustomisasi untuk mendeteksi kepala, kacamata, dan topi.

<img width="325" alt="Contoh Deteksi" src="https://github.com/user-attachments/assets/92bd6157-2732-42e2-be59-46b512d378ce" />
