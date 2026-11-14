# Hand Tracking — Gerçek Zamanlı El ve Parmak Takibi

Python, OpenCV ve MediaPipe kullanılarak geliştirilmiş, kameradan elin 21 referans noktasını tespit eden; parmakların açık/kapalı durumlarını ve açık parmak sayısını gösteren bir masaüstü uygulaması. Windows'ta el hareketleriyle slayt kontrolünü de destekler.

## Neler yapabiliyor?

- Web kameradan gerçek zamanlı görüntü alır.
- MediaPipe Hand Landmarker modeliyle en fazla **iki elin 21'er noktasını** bulur.
- Her parmağın açık/kapalı durumunu, eklem açıları ve mesafe oranları üzerinden hesaplar.
- Algılama titreşimini azaltmak için her parmak durumunu **4 ardışık karede** doğrular (varsayılan).
- Açık parmak sayısını, açık parmak adlarını, el iskeletini ve FPS'yi gösterir.
- `D` ile hata ayıklama ekranını açarak açıları, mesafeleri ve başarısız kalan eşikleri görüntüler.
- `S` ile Windows sunum/slayt kontrolünü açıp kapatır.

## Kullanılan teknolojiler

| Teknoloji | Görevi |
| --- | --- |
| Python | Akış, geometri ve kurallar |
| OpenCV (`cv2`) | Kamera, kare dönüşümleri, çizim ve pencere |
| MediaPipe Hand Landmarker | Önceden eğitilmiş modelle elin 21 referans noktasını belirleme |
| NumPy | Görüntü dizileri ve arayüz tuvali |
| Pillow | Türkçe metinler ve şeffaf paneller |
| Windows `ctypes` | Sağ/sol ok tuşuna basma olaylarını gönderme |

> Bu projede sıfırdan model eğitilmedi. Yapay zekâ ile el noktaları belirleniyor; parmak durumları Python'da yazılmış geometrik kurallarla hesaplanıyor.

## Kurulum (Windows, Python 3.13)

```powershell
# Proje klasöründeyken
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python hand_tracking.py --download-model
python hand_tracking.py
```

Kurulmuş Python 3.13 yoksa, `py -3.13` yerine yüklü uyumlu sürüm kullanılabilir; bu sürümdeki bağımlılıkların kullanılabilirliği doğrulanmalıdır.

**Not:** Model dosyası GitHub paketinde bulunmaz; `--download-model` komutu MediaPipe'ın yayımladığı modeli `models/hand_landmarker.task` yoluna indirir. Kurulum sırasında internet gerekir. **Atölye sürümü ZIP'inde** model önceden yer alır, indirme adımı atlanabilir.

Kamera açılmazsa Windows kamera izinlerini kontrol edin veya `python hand_tracking.py --camera 1` deneyin.

## Çalıştırma seçenekleri

```powershell
python hand_tracking.py                 # Normal kullanım
python hand_tracking.py --debug         # Açı/oranları ve başarısız eşikleri göster
python hand_tracking.py --slides        # Slayt kontrolünü baştan açık başlat
python hand_tracking.py --camera 1      # İkinci kamerayı kullan
python hand_tracking.py --stable-frames 6 # Daha yavaş ama daha kararlı durum güncellemesi
python -m unittest discover -s tests -v # Temel mantık testleri
```

## Klavye / jest kontrolü

- `Q` veya `Esc`: uygulamayı kapat
- `S`: slayt kontrolünü aç/kapat
- `D`: hata ayıklama bilgilerini aç/kapat
- **Önce 5 parmağın tümünü aç:** slayt komutunu etkinleştir
- **Yalnız işaret parmağı açık:** 0,7 saniye tutulunca sağ ok (sonraki slayt)
- **İşaret + orta parmak açık:** 0,7 saniye tutulunca sol ok (önceki slayt)
- Komutlar arasında en az 1,5 saniye bekleme vardır. Bir sonraki komut için tekrar açık avuç göstermek gerekir.

Slayt kontrolü Windows'a klavye tuşu gönderir; komutun sunuma ulaşması için sunum penceresinin **önde/odakta** olması gerekir. İki el aynı anda gösteriliyorsa slayt komutu devre dışıdır, ancak iki el de görüntülenebilir.

## Teknik işlem hattı

1. `cv2.VideoCapture` ile kameradan kare okunur.
2. BGR kare RGB'ye çevrilip `detect_for_video` ile MediaPipe'a verilir.
3. Model, her el için `hand_landmarks` (görüntü için) ve `hand_world_landmarks` (3B geometrik ölçüm için) üretir.
4. `analyze_fingers()` MCP/PIP/DIP eklem açılarını, parmak ucunun erişim oranını ve bileğe göre uzanmayı ölçer. Başparmak için ayrı eşikler uygulanır.
5. `StableState`, her parmağın yeni durumunu arka arkaya 4 kare gözlemlenmeden kesinleştirmez.
6. `draw_landmarks()` iskeleti; `draw_ui()` parmak adlarını, sayacı, FPS'yi ve panelleri görüntüler.
7. Slayt modu açıksa `GestureGate` hareketin 0,7 saniye korunmasını ve 1,5 saniyelik beklemeyi doğrular; `send_arrow` Windows ok tuşunu gönderir.

**Sınırlamalar:** Işık, elin kamera açısı, parmakların üst üste gelmesi ve model hataları sayımı etkileyebilir. Kodun eşikleri kalibre edilebilir. Slayt kontrolü yalnız Windows'tadır. Sistem görüntüleri diske kaydetmez veya ağ üzerinden göndermez; **modelin ilk indirilmesi** internet erişimi kullanır.

## Projenin dosyaları

```text
hand-tracking/
├── hand_tracking.py       # Ana uygulama
├── requirements.txt       # Python bağımlılıkları
├── .gitignore             # Sanal ortam ve model dosyalarını Git'ten hariç tutar
├── README.md              # Kurulum ve proje tanıtımı
└── tests/
    └── test_core.py       # Geometri, kararlılık ve jest mantığı testleri
```

## Model kaynağı

El referans noktaları Google MediaPipe Hand Landmarker modelinden elde edilir. İndirme adresi `hand_tracking.py` dosyasındaki `MODEL_URL` sabitinde tanımlanmıştır. Kaynak model uygulamanın kendi eğittiği model değildir.