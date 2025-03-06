import numpy as np
import cv2
import os


def save_npz_as_images(npz_filename, output_folder="images"):
    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # โหลดข้อมูลจากไฟล์ .npz
    data = np.load(npz_filename)

    # วน loop ตาม key ที่มีในไฟล์ .npz
    for key in data.files:
        arr = data[key]
        print(f"Key: {key}, shape: {arr.shape}")

        # กรณีที่ข้อมูลเป็นหลายภาพ (4 มิติ) เช่น (n, H, W, 1)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            num_images = arr.shape[0]
            for i in range(num_images):
                # ดึงภาพที่ index i
                image = arr[i]  # shape: (H, W, 1)
                # ลบ channel dimension ถ้ามี (จะได้ shape: (H, W))
                image = np.squeeze(image, axis=-1)

                # ถ้า dtype ไม่ใช่ uint8 ให้ normalize เป็นช่วง 0-255
                if image.dtype != np.uint8:
                    image = cv2.normalize(
                        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                    ).astype(np.uint8)

                # ตั้งชื่อไฟล์ เช่น images_0.png, images_1.png, ...
                output_path = os.path.join(output_folder, f"{key}_{i}.png")
                cv2.imwrite(output_path, image)
                print(f"Saved {output_path}")

        else:
            # กรณีที่ข้อมูลเป็นภาพเดียว (2 หรือ 3 มิติ)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)

            if arr.dtype != np.uint8:
                arr = cv2.normalize(
                    arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                ).astype(np.uint8)

            output_path = os.path.join(output_folder, f"{key}.png")
            cv2.imwrite(output_path, arr)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    # เรียกใช้งานฟังก์ชัน โดยระบุชื่อไฟล์ npz และโฟลเดอร์ที่จะบันทึกภาพ
    save_npz_as_images("2023_04_29_grayscale.npz", output_folder="images")
