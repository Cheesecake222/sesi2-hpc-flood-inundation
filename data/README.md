# Data

Folder ini sengaja hampir kosong (lihat `.gitignore`): data DEM nyata biasanya besar
dan sering punya syarat lisensi. Taruh data Anda di sini dan rujuk dari berkas konfigurasi.

`contoh/` berisi data kecil untuk mencoba jalur "data sendiri" (`configs/contoh_asc.toml`):

- `dem.asc`   : DEM sintetis 50 x 70 sel, 10 m (dibuat dengan `python -m floodsim make-dem`).
- `hujan.csv` : hyetograph contoh dengan kolom `time_min,intensity_mm_h`.

## Sumber data yang lazim dipakai

| Data | Sumber contoh | Catatan |
|---|---|---|
| DEM | DEMNAS (Badan Informasi Geospasial), SRTM, ALOS AW3D30, survei drone/LiDAR | periksa lisensi, resolusi, dan galat vertikal |
| Hujan | stasiun BMKG/BBWS, CHIRPS, GPM IMERG, radar cuaca | ubah ke `time_min,intensity_mm_h` |
| Debit sungai | pos duga air (BBWS/PUPR) atau hasil model hidrologi | ubah ke hidrograf `times_min` dan `q_m3s` |
| Tutupan lahan | peta tutupan lahan resmi, citra satelit terklasifikasi | petakan kelas ke nilai Manning dan infiltrasi |

Ketentuan format: grid beraturan dalam **meter** (proyeksi UTM), sel persegi, ESRI ASCII Grid
(`.asc`). Baris pertama berkas = baris paling utara.
