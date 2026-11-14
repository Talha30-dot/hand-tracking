"""YZT el takibi: python hand_tracking.py [--debug].

Tek uygulama dosyası: geometri, filtre, slayt, model, kamera ve Türkçe arayüz.
Kamera görüntüsü yerelde işlenir; kaydedilmez veya gönderilmez.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import acos, degrees, dist, isfinite, sqrt
from pathlib import Path
import argparse
import os
import time

ROOT = Path(__file__).resolve().parent
NAMES = ('Başparmak', 'İşaret', 'Orta', 'Yüzük', 'Serçe')
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
WINDOW = 'YZT - El Takibi'

@dataclass(frozen=True)
class Thresholds:
    pip: float = 135.0
    dip: float = 125.0
    mcp: float = 100.0
    reach: float = .72
    thumb_mcp: float = 125.0
    thumb_ip: float = 140.0
    thumb_spread: float = .45


def angle(a, b, c):
    """b noktasındaki iç açı. Düz parmakta 180 dereceye yaklaşır."""
    u = tuple(x-y for x, y in zip(a, b))
    v = tuple(x-y for x, y in zip(c, b))
    length = sqrt(sum(x*x for x in u)*sum(x*x for x in v))
    if length < 1e-12:
        return 0.0
    return degrees(acos(max(-1., min(1., sum(x*y for x, y in zip(u, v))/length))))


def analyze_fingers(points, thresholds=None):
    """Durumlar ve açıklanabilir ölçümler döndürür; nokta sırası MediaPipe 0–20.

    Dört parmak için MCP/PIP/DIP açıları, MCP–TIP mesafesinin toplam
    kemik uzunluğuna oranı ve TIP'in bilekten PIP'e göre uzaması incelenir.
    Böylece hafif eğrilik tolere edilirken yumruk ve avuca katlanmış düz
    parmaklar yalnızca düşük bir açı eşiği yüzünden açık sayılmaz.
    """
    if len(points) != 21 or any(len(p) != 3 or not all(isfinite(v) for v in p) for p in points):
        raise ValueError('21 adet sonlu, üç boyutlu landmark gerekli.')
    p, t = points, thresholds or Thresholds()
    width = dist(p[5], p[17])
    scale = max(dist(p[0], p[9]), width, 1e-9)

    tmcp, tip = angle(p[1], p[2], p[3]), angle(p[2], p[3], p[4])
    spread = dist(p[4], p[5])/max(width, 1e-9)
    away = (dist(p[4], p[17])-dist(p[3], p[17]))/scale
    thumb_open = width > 1e-8 and tmcp >= t.thumb_mcp and tip >= t.thumb_ip and spread >= t.thumb_spread and away > .02
    states = [thumb_open]
    metrics = [dict(name=NAMES[0], mcp=tmcp, pip=tip, dip=None, reach=spread, advance=away,
                    open=thumb_open, failed=[name for name, ok in (
                        ('MCP', tmcp >= t.thumb_mcp), ('IP', tip >= t.thumb_ip),
                        ('açıklık', spread >= t.thumb_spread), ('uzanma', away > .02)) if not ok])]
    for name, (mcp, pip, dip, tip) in zip(NAMES[1:], FINGERS):
        a_mcp = angle(p[0], p[mcp], p[pip])
        a_pip = angle(p[mcp], p[pip], p[dip])
        a_dip = angle(p[pip], p[dip], p[tip])
        length = dist(p[mcp], p[pip])+dist(p[pip], p[dip])+dist(p[dip], p[tip])
        reach = dist(p[mcp], p[tip])/max(length, 1e-9)
        advance = (dist(p[tip], p[0])-dist(p[pip], p[0]))/scale
        checks = [('MCP', a_mcp >= t.mcp), ('PIP', a_pip >= t.pip),
                  ('DIP', a_dip >= t.dip), ('oran', reach >= t.reach),
                  ('uzanma', advance > .04), ('kemik', length > 1e-8)]
        opened = all(ok for _, ok in checks)
        states.append(opened)
        metrics.append(dict(name=name, mcp=a_mcp, pip=a_pip, dip=a_dip, reach=reach,
                            advance=advance, open=opened, failed=[k for k, ok in checks if not ok]))
    return tuple(states), metrics


def finger_states(points, thresholds=None):
    return analyze_fingers(points, thresholds)[0]


class StableState:
    """Her parmağı bağımsız N ardışık kareyle doğrula; tek parmak diğerlerini bekletmez."""
    def __init__(self, frames=4):
        if frames < 1:
            raise ValueError('Kararlılık kare sayısı pozitif olmalı.')
        self.frames = frames
        self.reset()

    def reset(self):
        self.candidates = [None]*5
        self.counts = [0]*5
        self.values = [None]*5

    def update(self, value):
        if value is None:
            self.reset()
            return None
        for i, state in enumerate(value):
            self.counts[i] = self.counts[i]+1 if state == self.candidates[i] else 1
            self.candidates[i] = state
            if self.counts[i] >= self.frames:
                self.values[i] = state
        return None if None in self.values else tuple(self.values)


class GestureGate:
    def __init__(self, hold=.7, cooldown=1.5):
        self.hold, self.cooldown = hold, cooldown
        self.candidate = None
        self.since = 0.
        self.last = float('-inf')
        self.armed = False

    def update(self, states, now):
        if states is None:
            self.candidate = None
            self.armed = False
            return None
        gesture = {(False, True, False, False, False): 'right',
                   (False, True, True, False, False): 'left'}.get(states)
        if gesture != self.candidate:
            self.candidate, self.since = gesture, now
        if gesture is None:
            if all(states):  # Yeniden kurmak için gerçekten açık avuç gerekir.
                self.armed = True
            return None
        if self.armed and now-self.since >= self.hold and now-self.last >= self.cooldown:
            self.last, self.armed = now, False
            return gesture
        return None


def send_arrow(key):
    import ctypes
    import sys
    if sys.platform != 'win32':
        raise RuntimeError('Slayt kontrolü bu sürümde Windows gerektirir.')
    vk = {'left': 0x25, 'right': 0x27}[key]
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, 2, 0)


def download_model(path):
    from urllib.request import urlopen
    import shutil
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError('İndirilen model beklenenden küçük.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Model hazır: {path}')


@lru_cache(maxsize=24)
def font(size):
    from PIL import ImageFont
    candidates = [Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'segoeui.ttf',
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError('Türkçe font bulunamadı: Windows Segoe UI veya DejaVu Sans gerekli.')


def draw_ui(frame, observations, fps, slides, debug=False, thresholds=None):
    """1280x960 yerleşim, oranı korunmuş kamera ve çakışmayan paneller.

    OpenCV pencereyi ölçekler; panel geometrisi kamera 320x240 veya 1920x1080
    olduğunda değişmez. Uzun parmak adları ölçülerek satırlara bölünür.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    canvas = np.zeros((960, 1280, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    factor = min(1280/w, 960/h)
    resized = cv2.resize(frame, (max(1, round(w*factor)), max(1, round(h*factor))))
    rh, rw = resized.shape[:2]
    x, y = (1280-rw)//2, (960-rh)//2
    canvas[y:y+rh, x:x+rw] = resized
    base = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert('RGBA')
    overlay = Image.new('RGBA', base.size)
    draw = ImageDraw.Draw(overlay)

    def panel(box):
        draw.rounded_rectangle(box, radius=16, fill=(12, 20, 32, 210), outline=(75, 92, 112, 220))

    def text(x, y, value, size=24, color=(236, 242, 250)):
        draw.text((x, y), value, font=font(size), fill=color)

    def wrapped(value, max_width, size=23):
        lines, line = [], ''
        for word in value.split():
            trial = f'{line} {word}'.strip()
            if line and draw.textlength(trial, font=font(size)) > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines+[line]

    panel((20, 20, 440, 174))
    text(40, 32, 'YZT  •  EL TAKİBİ', 30)
    text(40, 80, f'FPS  {fps:.1f}   |   Slayt  {"Açık" if slides else "Kapalı"}')
    text(40, 124, 'Yeşil: açık   •   Turuncu: kapalı', 22)
    cards = observations or [dict(side=None, states=None)]
    for i, item in enumerate(cards):
        top = 20+i*156
        panel((654, top, 1260, top+142))
        states = item['states']
        label = 'El algılanmadı' if item['side'] is None else ('Sol el' if item['side'] == 'Left' else 'Sağ el')
        text(676, top+12, label, 23)
        text(676, top+42, str(sum(states)) if states is not None else '—', 62, (77, 231, 162))
        names = ' · '.join(name for name, opened in zip(NAMES, states) if opened) if states is not None else 'Elinizi gösterin' if item['side'] is None else 'Kararlılık bekleniyor'
        for row, line in enumerate(wrapped(names or 'Yumruk', 440)):
            text(786, top+48+row*30, line, 23)
    if slides:
        panel((20, 192, 610, 290))
        text(40, 204, 'Önce avucu aç; hareketi 0,7 sn tut.', 23)
        text(40, 246, 'İşaret: ileri   •   İşaret + orta: geri', 23)
    if debug:
        t = thresholds or Thresholds()
        panel((20, 542, 1260, 888))
        text(40, 550, f'DEBUG • Ham ölçümler | PIP ≥ {t.pip:g}°  DIP ≥ {t.dip:g}°  MCP ≥ {t.mcp:g}°  Oran ≥ {t.reach:g}', 21)
        for col, item in enumerate(observations):
            left = 40+col*610
            text(left, 591, ('Sol' if item['side'] == 'Left' else 'Sağ')+' el: MCP / PIP / DIP | oran | uzanma', 20)
            for row, metric in enumerate(item['metrics']):
                d = '—' if metric['dip'] is None else f"{metric['dip']:.0f}"
                values = f"{metric['name']}: {metric['mcp']:.0f} / {metric['pip']:.0f} / {d} | {metric['reach']:.2f} | {metric['advance']:.2f}"
                color = (77, 231, 162) if metric['open'] else (255, 174, 100)
                text(left, 630+row*47, values, 18, color)
                text(left, 652+row*47, 'AÇIK' if metric['open'] else 'KAPALI: '+', '.join(metric['failed']), 15, color)
    panel((20, 906, 1260, 950))
    text(40, 913, 'Q / Esc: Çıkış     S: Slayt aç/kapat     D: Debug     Slayt için sunumu öne alın (Alt+Tab)', 22)
    return cv2.cvtColor(np.asarray(Image.alpha_composite(base, overlay).convert('RGB')), cv2.COLOR_RGB2BGR)


def draw_landmarks(frame, landmarks, states, debug):
    import cv2
    h, w = frame.shape[:2]
    points = [(round((1-p.x)*(w-1)), round(p.y*(h-1))) for p in landmarks]
    for a, b in ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)):
        cv2.line(frame, points[a], points[b], (180, 185, 195), 2)
    for i, chain in enumerate(((1, 2, 3, 4),)+FINGERS):
        color = (150, 150, 150) if states is None else (110, 225, 65) if states[i] else (50, 155, 255)
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, points[a], points[b], color, 3)
    for i, point in enumerate(points):
        cv2.circle(frame, point, 4, (245, 245, 245), -1)
        if debug:
            cv2.putText(frame, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)


def parser():
    p = argparse.ArgumentParser(description='YZT: Türkçe arayüzlü canlı el takibi')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--model', type=Path, default=ROOT/'models'/'hand_landmarker.task')
    p.add_argument('--download-model', action='store_true')
    p.add_argument('--slides', action='store_true')
    p.add_argument('--stable-frames', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--debug', action='store_true', help='D ile de açılabilir; ham açılar ve başarısız kurallar')
    p.add_argument('--pip-angle', type=float, default=135.)
    p.add_argument('--dip-angle', type=float, default=125.)
    p.add_argument('--mcp-angle', type=float, default=100.)
    p.add_argument('--reach-ratio', type=float, default=.72)
    p.add_argument('--thumb-mcp-angle', type=float, default=125.)
    p.add_argument('--thumb-ip-angle', type=float, default=140.)
    p.add_argument('--thumb-spread', type=float, default=.45)
    return p


def main():
    args = parser().parse_args()
    if args.stable_frames < 1 or args.camera < 0 or args.max_frames < 0:
        print('Hata: Kamera/kare sınırı negatif olamaz; kararlılık en az 1 olmalı.')
        return 2
    angles = (args.pip_angle, args.dip_angle, args.mcp_angle, args.thumb_mcp_angle, args.thumb_ip_angle)
    if not all(0 < value <= 180 for value in angles) or not 0 < args.reach_ratio <= 1 or not 0 < args.thumb_spread <= 2:
        print('Hata: Açı 0–180, oran 0–1, başparmak açıklığı 0–2 aralığında olmalı (0 hariç).')
        return 2
    if args.download_model:
        try:
            download_model(args.model)
            return 0
        except (OSError, RuntimeError) as error:
            print(f'Model indirilemedi: {error}')
            return 1
    if args.headless and (args.slides or not args.max_frames):
        print('Hata: headless için max-frames gerekli; slayt kontrolü kapalı olmalı.')
        return 2
    if not args.model.is_file():
        print('Model bulunamadı. Çalıştırın: python hand_tracking.py --download-model')
        return 2
    try:
        import cv2
        import mediapipe as mp
        import PIL
    except ImportError as error:
        print(f'Eksik bağımlılık: {error}. python -m pip install -r requirements.txt')
        return 2
    thresholds = Thresholds(args.pip_angle, args.dip_angle, args.mcp_angle, args.reach_ratio,
                            args.thumb_mcp_angle, args.thumb_ip_angle, args.thumb_spread)
    cap = None
    frames = hands_seen = 0
    slides, debug = args.slides, args.debug
    gate = GestureGate()
    stable = {side: StableState(args.stable_frames) for side in ('Left', 'Right')}
    control_side = None
    try:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.6, min_hand_presence_confidence=.6, min_tracking_confidence=.6)
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise RuntimeError('Kamera açılamadı. Bağlantıyı, Windows kamera izinlerini ve diğer uygulamaları kontrol edin; --camera 1 deneyin.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(WINDOW, 1100, 825)
            previous = time.perf_counter()
            last_timestamp = -1
            fps = 0.
            while True:
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError('Kameradan kare alınamadı; bağlantı kesilmiş olabilir.')
                now = time.perf_counter()
                timestamp = max(last_timestamp+1, int(now*1000))
                last_timestamp = timestamp

                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                frame = cv2.flip(raw, 1)
                seen, observations = set(), []
                for landmarks, world, handedness in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
                    side = handedness[0].category_name
                    if side not in stable or side in seen:
                        continue
                    seen.add(side)
                    raw_states, metrics = analyze_fingers([(p.x, p.y, p.z) for p in world], thresholds)
                    states = stable[side].update(raw_states)
                    observations.append(dict(side=side, states=states, metrics=metrics))
                    hands_seen += 1
                    if not args.headless:
                        draw_landmarks(frame, landmarks, states, debug)
                for side in stable.keys()-seen:
                    stable[side].update(None)
                observations.sort(key=lambda item: item['side'])
                active_side = observations[0]['side'] if len(observations) == 1 else None
                if active_side != control_side:
                    gate = GestureGate()
                    control_side = active_side
                command = gate.update(observations[0]['states'] if slides and active_side else None, now)
                if command:
                    send_arrow(command)
                elapsed = max(time.perf_counter()-previous, 1e-6)
                fps = .9*fps+.1/elapsed if fps else 1/elapsed
                previous = time.perf_counter()
                frames += 1
                if not args.headless:
                    cv2.imshow(WINDOW, draw_ui(frame, observations, fps, slides, debug, thresholds))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord('s'):
                        slides = not slides
                        gate = GestureGate()
                    if key == ord('d'):
                        debug = not debug
                if args.max_frames and frames >= args.max_frames:
                    break
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f'Hata: {error}')
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    print(f'İşlenen kare: {frames}; algılanan el gözlemi: {hands_seen}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
