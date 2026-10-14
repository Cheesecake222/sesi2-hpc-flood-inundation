# Metode numerik, desain paralel, dan keterbatasan

Dokumen ini menjelaskan apa yang sebenarnya dihitung `floodsim`, bagaimana domain
dibagi ke banyak proses, bagaimana kebenarannya diuji, dan apa saja yang **belum**
dimodelkan.

## 1. Model fisik

Genangan permukaan dimodelkan dengan **formulasi inersia lokal** dari persamaan
air dangkal (*local inertial / "simplified shallow water"*). Pendekatan ini
dipakai pada model genangan 2D seperti LISFLOOD-FP karena jauh lebih murah dari
persamaan air dangkal penuh, tetapi tetap menangkap dinamika aliran pada dataran
banjir.

Pada grid reguler dengan ukuran sel $\Delta x$, kedalaman air $h$ dan elevasi
dasar $z$ disimpan di pusat sel, sedangkan debit per satuan lebar $q$ (m²/s)
disimpan di **sisi antar-sel** (grid terhuyung / *staggered*).

**Persamaan momentum** pada tiap sisi (arah-x; arah-y analog):

$$
q^{t+\Delta t}=\frac{q^{t}-g\,h_f\,\Delta t\,\dfrac{\eta_{j+1}-\eta_j}{\Delta x}}
{1+\dfrac{g\,\Delta t\,n^{2}\,|q^{t}|}{h_f^{7/3}}},
\qquad \eta=z+h
$$

dengan $h_f=\max(\eta_j,\eta_{j+1})-\max(z_j,z_{j+1})$ (kedalaman aliran pada sisi),
$n$ koefisien Manning (rata-rata kedua sel), dan $g=9{,}81$ m/s². Sisi dianggap
kering jika $h_f \le h_{tol}$ (default 1 mm). Bilangan Froude dibatasi
$Fr \le 1$ untuk kestabilan.

**Persamaan kontinuitas** pada tiap sel:

$$
h^{t+\Delta t}=h^{t}+\Delta t\left[\,r - f \;-\; \frac{\sum q_{\text{keluar}}-\sum q_{\text{masuk}}}{\Delta x}\,\right]
$$

dengan $r$ intensitas hujan (seragam), $f$ laju infiltrasi (dibatasi oleh air yang
ada di sel), ditambah debit sungai masuk pada sel-sel inflow.

**Langkah waktu** mengikuti syarat CFL berdasarkan kedalaman maksimum global:
$\Delta t = \alpha\,\Delta x/\sqrt{g\,h_{max}}$, dengan $\alpha=0{,}7$ dan batas
$[\Delta t_{min}, \Delta t_{max}]$.

**Batas domain.** Default: dinding tertutup. Sisi yang dipilih di
`model.open_boundaries` menjadi batas terbuka dengan aliran keluar dari rumus
Manning aliran seragam, $q=h^{5/3}\sqrt{S_0}/n$ (default $S_0=0{,}004$).

### Pembatas massa (flux limiter)

Setelah fluks dihitung, total aliran keluar tiap sel dibandingkan dengan air yang
tersedia. Jika berlebih, semua fluks keluar sel itu diskalakan dengan faktor
$\le 1$. Akibatnya:

- kedalaman tidak pernah negatif,
- **massa kekal sampai ketelitian mesin** (lihat neraca massa di `summary.json`),
- sisi antar-subdomain mendapat fluks yang identik di kedua sisi.

## 2. Desain paralel

Domain dipecah menjadi **slab baris** (1D *domain decomposition*). Tiap subdomain
menyimpan satu baris *ghost* di atas dan di bawah.

```
 rank 0   +--------------------+  baris interior
          |  ghost (dinding)   |
          |  ....              |
          +--------------------+  <- baris tepi dikirim ke rank 1
 rank 1   |  ghost <- dari 0   |
          |  ....              |
          |  ghost <- dari 2   |
          +--------------------+
 rank 2   ...
```

Satu langkah waktu terdiri atas:

1. `allreduce(max)` kedalaman maksimum untuk menentukan $\Delta t$ global,
2. **tukar halo** kedalaman $h$ (dua `Sendrecv` per rank),
3. `phase1`: hitung fluks kandidat, sumber, dan faktor pembatas massa,
4. **tukar halo** faktor pembatas pada baris tepi,
5. `phase2`: terapkan fluks yang sudah dibatasi dan perbarui $h$.

Kedua rank yang berbatasan menghitung fluks antar-subdomain dari data yang sama
persis, sehingga **hasil paralel identik bit-per-bit dengan serial**, berapa pun
jumlah prosesnya. Sifat ini diuji otomatis (`tests/test_floodsim.py`).

Tiga implementasi komunikasi (`floodsim/comm.py`) berbagi satu kode runner:

| Kelas | Fungsi |
|---|---|
| `MPIComm` | pembungkus `mpi4py` untuk klaster sungguhan |
| `ThreadComm` | replika semantik MPI dengan thread, untuk uji tanpa MPI |
| `SerialComm` | satu proses |

### Ekspektasi performa

- Volume komunikasi per rank per langkah ≈ 4 baris × `nx` nilai, **tidak mengecil**
  saat jumlah rank bertambah, sedangkan komputasi per rank mengecil. Karena itu
  efisiensi turun pada jumlah rank besar dan pada grid kecil (hukum Amdahl dan
  rasio komputasi:komunikasi).
- Jumlah rank maksimum = jumlah baris grid. Praktisnya, usahakan ≥ 20-50 baris per rank.
- Implementasi NumPy bersifat *memory-bound* dan membuat banyak array sementara.
  Untuk kinerja lebih tinggi, porting kernel `phase1`/`phase2` ke Numba, C/C++
  (OpenMP), atau GPU adalah langkah lanjutan yang wajar.
- Setiap rank saat ini membangun DEM penuh lalu mengambil potongannya. Untuk grid
  sangat besar, ganti dengan pembacaan terpartisi (mis. `mpi4py` + HDF5/GeoTIFF *windowed read*).

## 3. Validasi dan pengujian

| Uji | Yang dipastikan |
|---|---|
| Kekekalan massa (domain tertutup) | volume tersimpan = volume hujan, galat relatif < 1e-10 |
| Neraca massa lengkap | tersimpan = hujan + inflow − infiltrasi − keluar |
| Kedalaman tidak negatif | pembatas massa bekerja |
| *Lake-at-rest* | danau dengan muka air datar di atas topografi tak rata tetap diam (skema *well-balanced*) |
| Pengaliran batas terbuka | air hujan di lereng keluar lewat batas selatan |
| Slab = serial | 2, 3, 5 slab menghasilkan hasil identik dengan serial |
| Terdistribusi = serial | 1, 3, 4 rank (`ThreadComm`) identik dengan serial |
| I/O dan CLI | `.asc` bolak-balik, override konfigurasi, jalur DEM nyata |

> **Penting:** lulus uji di atas membuktikan *kebenaran numerik dan implementasi*,
> **bukan** keakuratan prakiraan untuk lokasi nyata. Untuk itu diperlukan
> kalibrasi dan validasi terhadap kejadian banjir yang terdokumentasi
> (jejak muka air, citra satelit, atau pengukuran lapangan).

## 4. Keterbatasan (yang perlu Anda ketahui sebelum memakai hasilnya)

- **Hujan seragam** di seluruh domain; belum ada medan hujan spasial (radar/satelit).
- **Infiltrasi konstan** per sel (mm/jam), tanpa kejenuhan tanah atau kondisi
  awal. Metode seperti SCS-CN atau Green-Ampt adalah pengembangan alami.
- **Alur sungai dimodelkan sebagai sel DEM** pada resolusi grid. Sungai yang jauh
  lebih sempit dari $\Delta x$, jembatan, gorong-gorong, pintu air, dan pompa
  **belum** dimodelkan. Bangunan hanya terwakili lewat kekasaran (Manning).
- **Kondisi awal kering**, termasuk alur sungai. Debit dasar sungai baru muncul
  sebagai inflow di hulu, sehingga beberapa menit awal tidak mewakili kondisi nyata.
- **Aliran keluar batas terbuka** memakai kemiringan tetap, bukan kurva debit-muka air.
- **Kelas bahaya** pada peta hanya memakai kedalaman maksimum. Bahaya sebenarnya
  juga bergantung pada kecepatan aliran dan durasi, yang belum dikeluarkan.
- DEM global (mis. SRTM) punya galat vertikal beberapa meter, dapat lebih besar
  daripada kedalaman banjir yang ingin dipelajari. Pakai DEM terbaik yang tersedia.
- **Bukan sistem peringatan dini operasional.**

## 5. Rujukan

- Bates, P.D., Horritt, M.S., Fewtrell, T.J. (2010). *A simple inertial formulation of
  the shallow water equations for efficient two-dimensional flood inundation
  modelling.* Journal of Hydrology 387, 33-45.
- de Almeida, G.A.M., Bates, P., Freer, J.E., Souvignet, M. (2012). *Improving the
  stability of a simple formulation of the shallow water equations for 2-D flood
  modeling.* Water Resources Research 48, W05528.
