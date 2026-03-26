import cv2
import os

FOLDER = "../data/person_a/"
IMAGE_W, IMAGE_H = 320, 240
BLUR_THRESHOLD = 100      # 低于这个分数认为模糊
FACE_MIN_RATIO = 0.10     # 脸占画面面积低于10%认为太小

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

total = 0
blurry = []
too_small = []
ok = []

for f in sorted(os.listdir(FOLDER)):
    if not f.endswith(".png"):
        continue
    total += 1
    path = os.path.join(FOLDER, f)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 模糊检测
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 人脸大小检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_ratio = (w * h) / (IMAGE_W * IMAGE_H)
    else:
        face_ratio = 0.0

    if blur_score < BLUR_THRESHOLD:
        blurry.append((f, blur_score))
    elif face_ratio < FACE_MIN_RATIO:
        too_small.append((f, face_ratio))
    else:
        ok.append(f)

print(f"总计: {total} 张")
print(f"清晰可用: {len(ok)} 张")
print(f"模糊 (blur < {BLUR_THRESHOLD}): {len(blurry)} 张")
for f, s in blurry:
    print(f"  {f}  blur={s:.1f}")
print(f"脸太小 (face < {FACE_MIN_RATIO*100:.0f}%): {len(too_small)} 张")
for f, r in too_small:
    print(f"  {f}  face={r*100:.1f}%")
