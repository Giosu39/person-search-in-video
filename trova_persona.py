"""
trova_persona.py — trova i frame in cui compare una persona in uno o più video.

Uso (esempio):
    python trova_persona.py --refs riferimenti --videos video --out risultati

Struttura attesa:
    riferimenti/   foto della persona (jpg/png), una persona per foto se possibile
    video/         i file .mp4 (o passa direttamente i file con --videos a.mp4 b.mp4)

Output in --out:
    rilevamenti.csv   ogni volto confrontato con somiglianza >= (soglia - 0.10),
                      con colonna "match" (utile per tarare la soglia)
    segmenti.csv      apparizioni raggruppate in intervalli inizio/fine per video
    frame/            i frame con match, con riquadro e punteggio disegnati
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm è facoltativo
    tqdm = None

from insightface.app import FaceAnalysis

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


# ---------- utilità I/O compatibili con percorsi Windows con accenti ----------

def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> bool:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------- modello ----------

def load_model(det_size: int, use_cpu: bool) -> FaceAnalysis:
    providers = ["CPUExecutionProvider"] if use_cpu else [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=-1 if use_cpu else 0, det_size=(det_size, det_size))

    # Segnala se la GPU è davvero in uso
    try:
        used = app.models["recognition"].session.get_providers()
        print(f"[info] Provider in uso: {used}")
        if not use_cpu and "CUDAExecutionProvider" not in used:
            print("[attenzione] CUDA non disponibile: si usa la CPU (più lento). "
                  "Vedi le note sull'installazione di onnxruntime-gpu.")
    except Exception:
        pass
    return app


def load_references(app: FaceAnalysis, refs_dir: Path) -> np.ndarray:
    files = sorted(p for p in refs_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"[errore] Nessuna immagine in {refs_dir}")

    embs = []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            print(f"[attenzione] Impossibile leggere {p.name}, salto.")
            continue
        faces = app.get(img)
        if not faces:
            print(f"[attenzione] Nessun volto trovato in {p.name}, salto.")
            continue
        if len(faces) > 1:
            print(f"[info] {p.name}: {len(faces)} volti, uso il più grande.")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embs.append(face.normed_embedding)
        print(f"[ok] Riferimento: {p.name}")

    if not embs:
        sys.exit("[errore] Nessun volto utilizzabile nelle foto di riferimento.")
    print(f"[info] {len(embs)} foto di riferimento valide.\n")
    return np.stack(embs)  # (N, 512), già normalizzati


# ---------- analisi video ----------

def process_video(app, refs, video: Path, args, writer, frames_dir: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[errore] Impossibile aprire {video.name}")
        return []

    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(vfps / args.fps))
    print(f"[video] {video.name}: {vfps:.2f} fps, {total} frame, "
          f"analizzo 1 frame ogni {step} (~{vfps / step:.2f} al secondo)")

    low = args.threshold - 0.10
    match_times = []
    idx = 0
    bar = tqdm(total=total, unit="frame", desc=video.stem[:30]) if tqdm and total else None
    t0 = time.time()

    while True:
        if idx % step != 0:
            ok = cap.grab()  # salta il frame senza decodificarlo del tutto
            if not ok:
                break
            idx += 1
            if bar:
                bar.update(1)
            continue

        ok, frame = cap.read()
        if not ok:
            break

        t = idx / vfps
        faces = app.get(frame)
        best_in_frame = -1.0
        annotated = None

        for f in faces:
            sim = float(np.max(refs @ f.normed_embedding))
            if sim < low:
                continue
            is_match = sim >= args.threshold
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            writer.writerow([video.name, fmt_ts(t), f"{t:.3f}", idx,
                             f"{sim:.4f}", int(is_match), x1, y1, x2, y2])
            if is_match:
                best_in_frame = max(best_in_frame, sim)
                if annotated is None:
                    annotated = frame.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{sim:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if best_in_frame >= 0:
            match_times.append((t, best_in_frame))
            if not args.no_save:
                name = f"{video.stem}_{fmt_ts(t).replace(':', '-')}_{best_in_frame:.2f}.jpg"
                imwrite_unicode(frames_dir / name, annotated)

        idx += 1
        if bar:
            bar.update(1)

    if bar:
        bar.close()
    cap.release()
    print(f"[video] {video.name}: {len(match_times)} frame con match "
          f"in {time.time() - t0:.0f} s\n")
    return match_times


def build_segments(match_times, max_gap):
    """Raggruppa i match vicini nel tempo in intervalli."""
    segs = []
    for t, s in match_times:
        if segs and t - segs[-1]["end"] <= max_gap:
            segs[-1]["end"] = t
            segs[-1]["n"] += 1
            segs[-1]["best"] = max(segs[-1]["best"], s)
        else:
            segs.append({"start": t, "end": t, "n": 1, "best": s})
    return segs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Trova una persona nei video tramite riconoscimento facciale.")
    ap.add_argument("--refs", required=True, type=Path, help="cartella con le foto di riferimento")
    ap.add_argument("--videos", required=True, nargs="+", type=Path, help="cartella o file video")
    ap.add_argument("--out", default=Path("risultati"), type=Path, help="cartella di output")
    ap.add_argument("--fps", default=2.0, type=float, help="frame analizzati al secondo (default 2)")
    ap.add_argument("--threshold", default=0.40, type=float,
                    help="soglia di somiglianza coseno (default 0.40; abbassa se perde apparizioni, alza se ci sono falsi positivi)")
    ap.add_argument("--det-size", default=640, type=int,
                    help="risoluzione del rilevatore (640 default; 960-1280 aiuta con volti piccoli, ma è più lento)")
    ap.add_argument("--cpu", action="store_true", help="forza l'uso della CPU")
    ap.add_argument("--no-save", action="store_true", help="non salvare le immagini dei frame")
    args = ap.parse_args()

    videos = []
    for v in args.videos:
        if v.is_dir():
            videos += sorted(p for p in v.iterdir() if p.suffix.lower() in VID_EXT)
        elif v.is_file():
            videos.append(v)
        else:
            print(f"[attenzione] Non trovato: {v}")
    if not videos:
        sys.exit("[errore] Nessun video da analizzare.")

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frame"
    frames_dir.mkdir(exist_ok=True)

    app = load_model(args.det_size, args.cpu)
    refs = load_references(app, args.refs)

    max_gap = max(2.0, 3.0 / args.fps)  # buchi brevi (volto girato, sfocato) non spezzano l'intervallo

    with open(args.out / "rilevamenti.csv", "w", newline="", encoding="utf-8-sig") as fd, \
         open(args.out / "segmenti.csv", "w", newline="", encoding="utf-8-sig") as fs:
        wd = csv.writer(fd, delimiter=";")
        ws = csv.writer(fs, delimiter=";")
        wd.writerow(["video", "timestamp", "secondi", "frame", "somiglianza", "match",
                     "x1", "y1", "x2", "y2"])
        ws.writerow(["video", "inizio", "fine", "durata_s", "frame_con_match", "somiglianza_max"])

        for v in videos:
            times = process_video(app, refs, v, args, wd, frames_dir)
            for s in build_segments(times, max_gap):
                ws.writerow([v.name, fmt_ts(s["start"]), fmt_ts(s["end"]),
                             f"{s['end'] - s['start']:.1f}", s["n"], f"{s['best']:.3f}"])
            fd.flush()
            fs.flush()

    print(f"Fatto. Risultati in: {args.out.resolve()}")


if __name__ == "__main__":
    main()