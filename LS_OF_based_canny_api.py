import os
import cv2
import numpy as np
import shutil
import zipfile

def Outlier_filtering(zip_path: str, save_dir: str, sharpness_threshold: float = 0.03):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len([name for name in zip_ref.namelist() if name.endswith((".jpg", ".jpeg", ".png"))])
        for idx, filename in enumerate(zip_ref.namelist()):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing {idx + 1}/{total_files}: {filename}")
                image_bytes = zip_ref.read(filename)
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                h, w = image.shape[:2]
                save_path = os.path.join(save_dir, os.path.basename(filename))

                if h != w:
                    cv2.imwrite(save_path, image)
                    continue

                center = (w // 2, h // 2)
                mask = np.zeros((h, w), np.uint8)
                cv2.circle(mask, center, radius=100, color=(255), thickness=-1)
                selected_region = cv2.bitwise_and(image, image, mask=mask)
                gray = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_sharpness = np.sum(edges) / (h * w)

                if edge_sharpness > sharpness_threshold:
                    cv2.imwrite(save_path, image)

                print(f"Processed {filename}: sharpness = {edge_sharpness}")
