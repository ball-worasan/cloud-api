import os
from pathlib import Path
import cv2
import numpy as np

# รับที่อยู่ปัจจุบัน (Path object รองรับ Unicode ได้ดี)
current_dir = Path.cwd()
# join กับโฟลเดอร์ images และชื่อไฟล์ images_0.png
image_path = current_dir / "images" / "images_110.png"
print("Image path:", image_path)

# อ่านไฟล์ด้วย open() ในโหมดไบนารี
try:
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    # แปลงข้อมูลไบนารีเป็น numpy array
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    # Decode ภาพโดยใช้ OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("ไม่สามารถโหลดภาพได้ กรุณาตรวจสอบ path:", image_path)
    else:
        print("----- รายละเอียดของภาพ -----")
        print("Shape:", image.shape)
        print("Data type:", image.dtype)
        print("Min value:", np.min(image))
        print("Max value:", np.max(image))
except Exception as e:
    print("Error reading image file:", e)
