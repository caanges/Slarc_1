#KRÄVER FFMPEG och Cv2

import os
import subprocess

# =========================
# INSTÄLLNINGAR
# =========================

#sätt in path till videon
VIDEO_FILE = r"C:\Users\Emil\Downloads\IMG_5073.MOV"



#anpassa FPS med hur lång videon är
NUM_FRAMES = 30
VIDEO_DURATION_SECONDS = 11





# MAPP VID SCRIPTETS PLATS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER = os.path.join(BASE_DIR, "Drone_fotage")

FRAME_FILENAME_PATTERN = "frame_%03d.jpg"

# =========================
# KONTROLL VIDEO
# =========================

print("Kontrollerar video...")

if not os.path.exists(VIDEO_FILE):
    raise FileNotFoundError(f"Can't find the video: {VIDEO_FILE}")

print("Video hittad")

# =========================
# SKAPA MAPP VID SCRIPT
# =========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Output-mapp:", OUTPUT_FOLDER)

# =========================
# BERÄKNA FPS
# =========================

fps = NUM_FRAMES / VIDEO_DURATION_SECONDS

output_pattern = os.path.join(OUTPUT_FOLDER, FRAME_FILENAME_PATTERN)

# =========================
# KÖR FFMPEG
# =========================

result = subprocess.run([
    "ffmpeg",
    "-i", VIDEO_FILE,
    "-vf", f"fps={fps}",
    "-vsync", "vfr",
    output_pattern
], capture_output=True, text=True)

# =========================
# FELKOLL
# =========================

if result.returncode != 0:
    print("FFMPEG FEL:")
    print(result.stderr)
    raise Exception("FFmpeg Failed")

print("Klart!")

# =========================
# RESULTAT
# =========================

files = os.listdir(OUTPUT_FOLDER)

print(f"{len(files)} Pictures created in:")
print(OUTPUT_FOLDER)