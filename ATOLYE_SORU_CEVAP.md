# Takım Kaptanı Soru-Cevap Notları

**30 saniyelik sunum**

“Python, OpenCV ve MediaPipe ile gerçek zamanlı el ve parmak takibi yaptık. Kamera görüntüsünü MediaPipe Hand Landmarker modeline veriyoruz. Modelin belirlediği 21 üç boyutlu referans noktası arasındaki açıları ve uzunluk oranlarını hesaplayarak açık parmak sayısını buluyoruz. Tek karede yanlış karar vermemek için 4 ardışık kare doğrulaması ekledik. Aynı altyapıyla açık avuçtan sonra belirli jestleri sağ/sol klavye oklarına dönüştürüp slayt yönetebiliyoruz.”

## Olası sorular

**Yapay zekâyı siz mi eğittiniz?** Hayır. MediaPipe'ın önceden eğitilmiş Hand Landmarker modelini kullanıyoruz. Bizim algoritmamız el noktalarından parmak durumunu belirliyor.

**OpenCV ne yapıyor?** Kameradan kare alıyor, renk biçimini RGB'ye çeviriyor, ekranda iskeleti ve pencereyi çiziyor.

**21 nokta nedir?** Bilek ile parmaklardaki eklem ve uç noktalarının konumlarıdır; her el için toplam 21 nokta.

**Hangi parmağın açık olduğunu nasıl anlıyorsunuz?** Başparmak dışındaki parmaklarda MCP/PIP/DIP açılarını, parmak ucunun köke göre uzunluk oranını ve bileğe göre uzanmasını ölçüyoruz. Başparmak için ayrı açı ve açıklık koşulları var.

**Neden 4 kare bekliyorsunuz?** Kamera titremesi veya tek karelik yanlış algılamaların ekranda anlık değişiklik yaratmaması için. Varsayılan `--stable-frames 4`.

**Parmak sayısını MediaPipe mı söylüyor?** Hayır; MediaPipe el noktalarını veriyor, Python kodu açık/kapalı parmakları hesaplayıp sayıyor.

**İki el çalışıyor mu?** Evet, model `num_hands=2` ile başlatılıyor. İki eli de sayabiliyoruz; slayt komutları karışmaması için tek el varken etkin.

**Sunumu nasıl değiştiriyorsunuz?** Açık avuçla komut sistemini etkinleştirip yalnız işaret (sağ ok) veya işaret+orta (sol ok) gösteriyoruz; jest 0,7 saniye tutulmalı. Komutlar arasında 1,5 saniye bekleme var. Windows `ctypes` ile klavye olayı gönderiliyor.

**FPS nedir?** Saniyede işlenen kare sayısı; performansı izlemek için gösterilir.

**Görüntüler kaydediliyor mu?** Bu kodda hayır. İşlem kameradan yerel bellekte yapılıyor; model ilk kez indiriliyorsa internet kullanılıyor.

**Zayıf yönü nedir?** Kamera ışığı, kapalı/üst üste parmaklar ve el yönü algılamayı etkileyebilir. Eşikleri `--debug` ekranına bakarak ayarlayabiliriz. Slayt kontrolü Windows'a özel.

**Bugün neleri geliştirebilirsiniz?** Farklı eller/ışık koşulları için kalibrasyon, işaret dili sınıflandırması (bu sürümde yok), kontrol jestleri için ek doğrulama ve testler.

**Önemli:** `FINGERS` eklemleri ve `analyze_fingers` bizim geometri hesaplarımızdır; `GestureGate` jestlerden komut üretir; `StableState` ölçümleri filtreler; `draw_ui` Türkçe arayüzü çizer; `main` her şeyi birleştirir.