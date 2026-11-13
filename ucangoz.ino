"""
Proje: YOLOv8 ve OpenCV ile Video Üzerinden Gerçek Zamanlı Uçangöz Tespiti

Eğitilmiş bir YOLO nesne tespit modelini (drone_model.pt) ve OpenCV kütüphanesini kullanarak 
giriş videosundaki (Smarter Drone Swarms.mp4) uçangözleri kare kare tespit eden bir görüntü işleme yazılımıdır.
Program, güven skoru 0.20'nin üzerinde olan her bir quadcopter veya sabit kanatlı (fixed-wing) uçangözü yeşil 
sınırlayıcı kutu içine alır ve üzerine güven oranını yazdırır.
Ayrıca video görüntüsünün sağ tarafına 300 piksellik siyah bir bilgi paneli ekleyerek o karede uçangöz 
bulunup bulunmadığını ("Drone" / "Drone yok") anlık olarak gösterir, hem ekranda canlı oynatır hem de 
işlenmiş görüntüyü yeni bir video dosyası (drone_sonuc.mp4) olarak kaydeder.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


KLASOR = Path(__file__).resolve().parent
VIDEO = KLASOR / "Smarter Drone Swarms.mp4" # farkli videoda test icin video ismini degisin
MODEL = KLASOR / "drone_model.pt"   # farklı modeli test etmek icin model indirip ismini yazin
CIKTI = KLASOR / "drone_sonuc.mp4"
GUVEN = 0.20 # guven skoru = modelin tespit ettigi siniftan ne kadar emin olmasi durumu (0.20den fazla ise tespit edecek)


def main():
    model = YOLO(str(MODEL))
    video = cv2.VideoCapture(str(VIDEO))
    if not video.isOpened():
        raise RuntimeError(f"Video acilamadi: {VIDEO}")

    genislik = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25
    panel = 300
    kayit = cv2.VideoWriter(str(CIKTI), cv2.VideoWriter_fourcc(*"mp4v"), fps, (genislik + panel, yukseklik))
    if not kayit.isOpened():
        video.release()
        raise RuntimeError(f"Cikti videosu olusturulamadi: {CIKTI}")

    try:
        while True:
            ok, kare = video.read()
            if not ok:
                break
            # Modelin iki sinifi da drone turudur: quadcopter ve fixed-wing.
            sonuc = model.predict(kare, conf=GUVEN, agnostic_nms=True, verbose=False)[0]
            for kutu in sonuc.boxes:
                x1, y1, x2, y2 = map(int, kutu.xyxy[0].tolist())
                guven = float(kutu.conf[0])
                cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(kare, f"Drone {guven:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            goruntu = np.zeros ( (yukseklik, genislik + panel, 3), dtype=np.uint8)
            goruntu[:, :genislik] = kare
            yazi = "Drone" if len(sonuc.boxes) else "Drone yok"
            cv2.putText(goruntu, yazi, (genislik + 20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            kayit.write(goruntu)
            ekran = cv2.resize(goruntu, (1110, int(yukseklik * 1110 / (genislik + panel))))
            cv2.imshow("Drone tespiti - cikis: Q", ekran)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        kayit.release()
        cv2.destroyAllWindows()
    print(f"Sonuc kaydedildi: {CIKTI}")


if __name__ == "__main__":
    main()
