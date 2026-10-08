# person-search-in-video

Find the moments in your videos where a specific person appears.

You give the script a few photos of the person and one or more video files. It returns a list of time ranges where that person is on screen, plus the matching frames saved as images. Everything runs locally on your machine: no video or photo is uploaded anywhere.

> The script's console messages and output column names are in Italian. A short glossary is included in [Output](#output).

---

## Contents

1. [How it works](#how-it-works)
2. [What to expect](#what-to-expect)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Tutorial](#tutorial)
6. [Output](#output)
7. [Options](#options)
8. [Tuning the results](#tuning-the-results)
9. [Troubleshooting](#troubleshooting)
10. [Privacy and responsible use](#privacy-and-responsible-use)
11. [License](#license)
12. [Credits](#credits)

---

## How it works

1. **Reference faces.** Each photo in the reference folder is scanned for a face. If a photo contains several faces, the largest one is used. Each face is turned into an *embedding*, a 512-number vector that describes it.
2. **Frame sampling.** Videos are read with OpenCV. Only some frames are analysed (2 per second by default) rather than all 25–60, which keeps the run fast without missing much.
3. **Face matching.** Every face found in a sampled frame gets its own embedding. It is compared with all the reference embeddings using cosine similarity. If the best score is above the threshold, the frame counts as a match.
4. **Grouping.** Matches that are close in time are merged into segments (start → end). This way you get "00:03:12 – 00:04:05" instead of a hundred separate rows.

Detection and recognition use [InsightFace](https://github.com/deepinsight/insightface) with the `buffalo_l` model pack, running on ONNX Runtime, either on the GPU or on the CPU. Only the pack's detection and recognition models are loaded; its landmark and age/gender models are skipped because they aren't needed and would slow down every frame.

## What to expect

**It works well when:**
- the face is reasonably frontal (up to roughly a three-quarter view),
- the face is not too small in the frame,
- the lighting is decent,
- your reference photos are varied and sharp.

**It struggles with:**
- strict profile views, people seen from behind, or heavy motion blur,
- very small or distant faces (try a larger `--det-size`),
- sunglasses, masks, or hands covering the face,
- very old or very low-resolution footage,
- people who look alike, such as close relatives, which can cause false positives.

**Expect to review the results.** This is a search aid, not a verdict. Some true appearances will be missed, and occasionally someone else will match. The saved frames and the per-face scores in `rilevamenti.csv` make the review quick.

**Speed** depends heavily on your hardware, the video resolution, and the settings, so there are no reliable universal numbers. As a rough guide, a GPU is many times faster than a CPU. On a CPU, processing can take as long as the video itself, or longer. If a run on an NVIDIA machine feels that slow, the GPU is probably not being used (see [Troubleshooting](#troubleshooting)).

The progress bar counts **all** frames of the video, including the skipped ones. To get the number of frames actually analysed per second, divide the rate shown by the bar by the *N* in "analizzo 1 frame ogni N", which is printed when each video starts.

## Prerequisites

- **Python 3.11 or newer**, 64-bit. Recent `onnxruntime-gpu` releases only ship wheels for Python 3.11–3.14; check [PyPI](https://pypi.org/project/onnxruntime-gpu/#files) if you use a different version.
- **A few GB of free disk space** for the packages, the CUDA/cuDNN libraries installed through pip, and the model. The model pack downloads automatically on the first run.
- **Optional, but strongly recommended: an NVIDIA GPU with an up-to-date driver.** You do **not** need to install the CUDA Toolkit or cuDNN yourself: `requirements.txt` pulls them in through pip. Recent `onnxruntime-gpu` builds target CUDA 13, which requires a recent NVIDIA driver; see the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). Without a working GPU setup, the script automatically falls back to the CPU, which is much slower.
- **Windows only:** installing `insightface` may need the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), with the "Desktop development with C++" workload, because part of the package is compiled during installation.

**Tested on:** Windows, Python 3.14, NVIDIA RTX 4070 Super, `onnxruntime-gpu` 1.30. The script includes support for file paths with accented characters on Windows. It should also work on Linux and macOS, but it has not been tested there. On macOS there is no CUDA, so it runs on the CPU.

## Installation

```bash
git clone https://github.com/Giosu39/person-search-in-video.git
cd person-search-in-video

# Recommended: use a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**CPU-only machine?** Open `requirements.txt` and replace `onnxruntime-gpu[cuda,cudnn]` with `onnxruntime` before installing. **Never install both `onnxruntime` and `onnxruntime-gpu`:** they conflict, and the CPU-only one usually wins, silently disabling the GPU.

### Check that the GPU is detected

```bash
pip list | findstr onnxruntime        # Windows (use "grep" on Linux/macOS)
python -c "import onnxruntime as ort; ort.preload_dlls(); print(ort.get_available_providers())"
```

- The first command should list **only** `onnxruntime-gpu`.
- The second should include **`CUDAExecutionProvider`**. If you see only `AzureExecutionProvider` and `CPUExecutionProvider`, see [Troubleshooting](#troubleshooting).

## Tutorial

### 1. Prepare the reference photos

Create a folder called `riferimenti` and put **3–10 photos** of the person in it (`.jpg`, `.png`, `.bmp`, `.webp`).

Good reference photos:
- are sharp, with the face clearly visible,
- show different angles, lighting conditions, and, if possible, different ages or hairstyles,
- ideally show only that person. If there are other people, the largest face is used.

The quality and variety of these photos matter more than any setting.

### 2. Put the videos in a folder

Create a folder called `video` and copy your files into it (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`).

```
person-search-in-video/
├── trova_persona.py
├── riferimenti/
│   ├── photo1.jpg
│   └── photo2.jpg
└── video/
    ├── holiday.mp4
    └── party.mp4
```

### 3. Run it

```bash
python trova_persona.py --refs riferimenti --videos video --out risultati
```

You can also pass individual files instead of a folder:

```bash
python trova_persona.py --refs riferimenti --videos video/holiday.mp4 video/party.mp4
```

On the first run, the model pack is downloaded to `~/.insightface/models/`, which is a few hundred MB. Then the script prints which ONNX Runtime providers it is using. **If you see `CUDAExecutionProvider`, the GPU is active.** If it isn't listed, the script is running on the CPU. See [Troubleshooting](#troubleshooting).

### 4. Look at the results

Open `risultati/segmenti.csv` first. It lists every time range where the person was found. Then look through `risultati/frame/` to check the matches visually.

### 5. Adjust and re-run if needed

If appearances are missing, or the wrong people match, see [Tuning the results](#tuning-the-results).

## Output

Everything is written to the `--out` folder (default: `risultati`).

| File / folder | Content |
|---|---|
| `segmenti.csv` | One row per appearance, as a time range. **Start here.** |
| `rilevamenti.csv` | One row per detected face with similarity ≥ threshold − 0.10, including near-misses. Useful for tuning. |
| `frame/` | Matching frames as JPEG images, with a green box and the score drawn on each matched face. Filenames contain the video name, timestamp, and score. |

The CSV files use `;` as the separator and UTF-8 with BOM encoding, so they open correctly in Excel with European regional settings.

**Column glossary:**

| Column | Meaning |
|---|---|
| `video` | Video file name |
| `inizio` / `fine` | Segment start / end (`HH:MM:SS.mmm`) |
| `durata_s` | Segment duration in seconds |
| `frame_con_match` | Number of matching sampled frames in the segment |
| `somiglianza_max` / `somiglianza` | Highest / per-face similarity score (−1 to 1, higher means more similar) |
| `timestamp` / `secondi` / `frame` | Position in the video as a timestamp, in seconds, and as a frame index |
| `match` | `1` if the score is at or above the threshold, `0` if it's a near-miss |
| `x1, y1, x2, y2` | Face bounding box in pixels |

## Options

| Option | Default | Description |
|---|---|---|
| `--refs` | *(required)* | Folder with the reference photos |
| `--videos` | *(required)* | One or more video files or folders |
| `--out` | `risultati` | Output folder |
| `--fps` | `2` | Frames analysed per second of video |
| `--threshold` | `0.40` | Minimum cosine similarity for a match |
| `--det-size` | `640` | Face detector input size. Larger values find smaller faces but run slower. |
| `--cpu` | off | Force CPU even if a GPU is available |
| `--no-save` | off | Don't save frame images, only the CSV files |

## Tuning the results

The default threshold of **0.40** is a starting point, not a value guaranteed for your footage. Use `rilevamenti.csv` to calibrate it:

- Filter the rows with `match = 0` and look at their scores. If rows around 0.33–0.39 are actually the right person, lower the threshold.
- If frames in `frame/` show someone else, check their scores and raise the threshold just above them.

| Problem | Try |
|---|---|
| Missed appearances | `--threshold 0.33` and/or more varied reference photos |
| Wrong people matched | `--threshold 0.45`–`0.50` |
| Person is small or far away | `--det-size 960` or `1280` |
| Person appears only briefly | `--fps 4` or higher |
| Too slow | lower `--fps`, keep `--det-size 640`, use `--no-save` |

Example:

```bash
python trova_persona.py --refs riferimenti --videos video --threshold 0.35 --det-size 960 --fps 3
```

## Troubleshooting

**`pip install insightface` fails with a compiler error (Windows)**
Install the Microsoft C++ Build Tools with the "Desktop development with C++" workload, open a new terminal, and retry.

**The script says CUDA is not available / `CUDAExecutionProvider` is missing**
A typical symptom is a warning like *"Specified provider 'CUDAExecutionProvider' is not in available provider names. Available providers: 'AzureExecutionProvider, CPUExecutionProvider'"*. That list means the **CPU-only `onnxruntime` package** is the one in use. The progress bar will crawl at just a few frames per second. Fix it with a clean reinstall:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]"
```

Then update your NVIDIA driver and run the checks in [Check that the GPU is detected](#check-that-the-gpu-is-detected). If `onnxruntime` reappears in `pip list` after installing other packages, uninstall just that one again. If CUDA is still missing, compare your setup with the [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). The script still works on the CPU in the meantime.

**"Nessun volto trovato" (no face found) for a reference photo**
That photo is skipped. Use a sharper or more frontal photo, or crop it closer to the face.

**"Impossibile aprire" (cannot open) a video**
OpenCV couldn't decode the file. Try re-encoding it to a standard H.264 MP4, for example with `ffmpeg -i input.ext -c:v libx264 -c:a copy output.mp4`. Also try moving it to a path without special characters.

**The model download fails**
Check your internet connection and any firewall or proxy. The model pack is fetched once and cached in `~/.insightface/models/`.

## Privacy and responsible use

Face recognition is processing of biometric data. Under laws such as the EU GDPR, it is subject to strict rules when it concerns other people.

- Use this tool on footage you have the right to process, such as your own videos or family videos, and preferably with the consent of the people involved.
- Don't use it to track, identify, or profile people without their knowledge.
- **Don't commit your videos, reference photos, or results to a public repository.** The included `.gitignore` excludes the default folders and common media files, but double-check before pushing.

## License

The code in this repository is released under the license in the `LICENSE` file.

**Important:** the license of this repository does **not** cover the InsightFace models. The InsightFace *code* is MIT-licensed. However, its README states that the pretrained models, including the `buffalo_l` pack downloaded by this script, are provided for **non-commercial research purposes only**, and asks commercial users to contact InsightFace for licensing. See the [InsightFace README](https://github.com/deepinsight/insightface) for the current terms. This summary is not legal advice.

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and recognition
- [ONNX Runtime](https://onnxruntime.ai/) for model inference
- [OpenCV](https://opencv.org/) for video decoding and image I/O