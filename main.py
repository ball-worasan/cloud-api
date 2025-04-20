import re
import cv2  # type: ignore
import json
import base64
import logging
import uvicorn  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from io import BytesIO
from sklearn.svm import SVR  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.multioutput import MultiOutputRegressor  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException  # type: ignore
from fastapi.responses import Response, HTMLResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
from load_weights import load_cloud_model  # type: ignore

# import aiohttp  # type: ignore
# import asyncio

# เพิ่มเติม: ใช้ PyTesseract ในการทำ OCR
import pytesseract  # type: ignore

# pytesseract.image_to_string
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ------------------------------------------------------------------------------
# การตั้งค่าเริ่มต้น
# ------------------------------------------------------------------------------
DEBUG = False  # ตั้งค่าเริ่มต้นเป็น False (ปิด DEBUG) สามารถเปลี่ยนได้ใน environment หรือ config
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ค่าคงที่
CROP_PIXEL_DATA_PATH = Path("pixel_data.json")
BACKGROUND_IMAGE_PATH = Path("background_image.npy")
CLOUD_MODEL_PATH = Path("cloud_model.keras")
DEFAULT_DBZ_MIN, DEFAULT_DBZ_MAX = 8, 72
REPLACE_COLORS = np.array(
    [
        [79, 125, 0],
        [51, 204, 0],
        [0, 255, 71],
        [79, 102, 0],
        [0, 255, 102],
        [33, 87, 0],
        [71, 176, 0],
        [18, 222, 51],
    ],
    dtype=np.uint8,
)

# โหลดไฟล์ที่จำเป็นตอนเริ่มต้น
try:
    BACKGROUND_IMAGE = np.load(BACKGROUND_IMAGE_PATH)
    if BACKGROUND_IMAGE.shape != (800, 800, 3):
        raise ValueError(
            f"Background image must be 800x800x3, got {BACKGROUND_IMAGE.shape}"
        )
    logger.info("Background image loaded successfully")
except Exception as e:
    logger.error(f"Failed to load {BACKGROUND_IMAGE_PATH}: {e}")
    raise RuntimeError("Failed to initialize server")

try:
    cloud_model = load_cloud_model(CLOUD_MODEL_PATH)
    logger.info("Cloud model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load {CLOUD_MODEL_PATH}: {e}")
    raise RuntimeError("Failed to initialize server")

# ------------------------------------------------------------------------------
# ตั้งค่า FastAPI และ CORS
# ------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Next-Time"],
)


# ------------------------------------------------------------------------------
# โมเดลแปลงค่า dbz <-> RGB
# ------------------------------------------------------------------------------
class ColorModel:
    def __init__(self) -> None:
        self.color_values_reversed = [
            [255, 255, 255],
            [9, 9, 255],
            [0, 0, 255],
            [0, 44, 255],
            [0, 95, 255],
            [0, 145, 255],
            [0, 193, 255],
            [252, 2, 254],
            [254, 51, 254],
            [254, 144, 254],
            [254, 176, 254],
            [254, 199, 254],
            [254, 27, 1],
            [254, 47, 1],
            [254, 39, 48],
            [254, 116, 7],
            [254, 132, 7],
            [254, 151, 7],
            [254, 182, 7],
            [254, 203, 7],
            [254, 221, 7],
            [254, 253, 7],
            [253, 245, 13],
            [245, 247, 60],
            [231, 254, 7],
            [95, 255, 0],
            [68, 250, 2],
            [44, 219, 22],
            [0, 200, 53],
            [0, 173, 71],
            [0, 124, 79],
            [0, 101, 77],
            [0, 87, 33],
        ]
        self.dbz_values = np.linspace(72, 8, len(self.color_values_reversed))
        self.X_rgb = np.array(self.color_values_reversed)
        self.y_dbz = np.array(self.dbz_values)
        self._train_rgb_to_dbz_model()
        self._train_dbz_to_rgb_model()

    def _train_rgb_to_dbz_model(self):
        try:
            self.scaler_X_rgb = StandardScaler()
            self.scaler_y_dbz = StandardScaler()
            X_rgb_scaled = self.scaler_X_rgb.fit_transform(self.X_rgb)
            y_dbz_scaled = self.scaler_y_dbz.fit_transform(
                self.y_dbz.reshape(-1, 1)
            ).ravel()
            param_grid_svr = {
                "C": [0.1, 1, 10, 100],
                "epsilon": [0.01, 0.1, 1],
                "gamma": ["scale", "auto", 0.01, 0.1, 1],
            }
            grid_search_dbz = GridSearchCV(
                SVR(kernel="rbf"),
                param_grid_svr,
                cv=3,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            grid_search_dbz.fit(X_rgb_scaled, y_dbz_scaled)
            self.best_svr_dbz_model = grid_search_dbz.best_estimator_
            if DEBUG:
                logger.debug(
                    f"RGB-to-dBZ model trained with best params: {grid_search_dbz.best_params_}"
                )
        except Exception as e:
            logger.error(f"Error training RGB-to-dBZ model: {e}")
            raise RuntimeError("Failed to initialize ColorModel")

    def _train_dbz_to_rgb_model(self):
        try:
            X_dbz = self.y_dbz.reshape(-1, 1)
            Y_rgb = self.X_rgb
            self.scaler_X_dbz = StandardScaler()
            self.scaler_Y_rgb = StandardScaler()
            X_dbz_scaled = self.scaler_X_dbz.fit_transform(X_dbz)
            Y_rgb_scaled = self.scaler_Y_rgb.fit_transform(Y_rgb)
            svr_for_rgb = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
            self.multi_svr_rgb = MultiOutputRegressor(svr_for_rgb, n_jobs=-1)
            self.multi_svr_rgb.fit(X_dbz_scaled, Y_rgb_scaled)
            if DEBUG:
                logger.debug("dBZ-to-RGB model trained successfully")
        except Exception as e:
            logger.error(f"Error training dBZ-to-RGB model: {e}")
            raise RuntimeError("Failed to initialize ColorModel")


color_model = ColorModel()


# ------------------------------------------------------------------------------
# ฟังก์ชันช่วยเหลือ
# ------------------------------------------------------------------------------
def image_to_base64(image: np.ndarray) -> str:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array")
    success, img_encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(img_encoded).decode("utf-8")


def crop_image(image: np.ndarray, crop_area: Tuple[int, int, int, int]) -> np.ndarray:
    if image.shape[:2] < (crop_area[3] - crop_area[1], crop_area[2] - crop_area[0]):
        raise ValueError(
            f"Image size {image.shape[:2]} too small for crop area {crop_area}"
        )
    pil_image = Image.fromarray(image)
    return np.array(pil_image.crop(crop_area))


def replace_pixels(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    if image.shape != background.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match background shape {background.shape}"
        )
    same_as_background = np.all(image == background, axis=-1)
    color_mask = np.any(
        [np.all(image == color, axis=-1) for color in REPLACE_COLORS], axis=0
    )
    image[same_as_background | color_mask] = [0, 0, 0]  # RGB black
    return image


def predict_dbz_from_image_batch(
    images: np.ndarray, svr_model, scaler_X, scaler_y, dbz_min=8, dbz_max=72
) -> np.ndarray:
    n, h, w, c = images.shape
    if c != 3:
        raise ValueError(f"Expected 3 color channels, got {c}")
    images_bgr = images[..., ::-1]  # RGB -> BGR
    reshaped_images = images_bgr.reshape(-1, 3)
    scaled_rgb = scaler_X.transform(reshaped_images)
    predicted_dbz_scaled = svr_model.predict(scaled_rgb)
    predicted_dbz = scaler_y.inverse_transform(
        predicted_dbz_scaled.reshape(-1, 1)
    ).ravel()
    normalized_dbz = (predicted_dbz - dbz_min) / (dbz_max - dbz_min) * 255
    return np.clip(normalized_dbz, 0, 255).astype(np.uint8).reshape(n, h, w, 1)


def dbz_to_rgb_batch(dbz_array: np.ndarray) -> np.ndarray:
    dbz_scaled = color_model.scaler_X_dbz.transform(dbz_array.reshape(-1, 1))
    rgb_pred_scaled = color_model.multi_svr_rgb.predict(dbz_scaled)
    return np.clip(
        color_model.scaler_Y_rgb.inverse_transform(rgb_pred_scaled), 0, 255
    ).astype(np.uint8)


def convert_grayscale_to_rgb(
    image: np.ndarray,
    dbz_min: float = DEFAULT_DBZ_MIN,
    dbz_max: float = DEFAULT_DBZ_MAX,
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("Empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        )
    h, w = image.shape
    dbz_vals = (image.astype(np.float32) / 255.0) * (dbz_max - dbz_min) + dbz_min
    rgb_image = dbz_to_rgb_batch(dbz_vals.flatten()).reshape(h, w, 3)
    # สามารถแก้เงื่อนไขการ mask สีดำตามต้องการได้
    black_mask = (dbz_vals >= 8.0) & (dbz_vals <= 22.0)
    rgb_image[black_mask] = [0, 0, 0]
    return rgb_image


def load_crop_pixel_data(path: Path) -> List[Tuple[int, int, List[int]]]:
    if not path.exists():
        logger.warning(f"{path} not found, returning empty list")
        return []
    with path.open("r") as f:
        data_dict: Dict[str, List[int]] = json.load(f)
    pixel_list = []
    for pixel_str, color_value in data_dict.items():
        try:
            x, y = map(int, pixel_str.split(","))
            if len(color_value) != 3:
                raise ValueError(f"Invalid color value: {color_value}")
            pixel_list.append((x, y, color_value))
        except Exception as e:
            logger.warning(f"Invalid pixel data '{pixel_str}': {e}")
    if DEBUG:
        logger.debug(f"Loaded {len(pixel_list)} valid pixels from {path}")
    return pixel_list


try:
    crop_pixel_data_list = load_crop_pixel_data(CROP_PIXEL_DATA_PATH)
    logger.info("Pixel Data loaded successfully")
except Exception as e:
    logger.error(f"Failed to load {CROP_PIXEL_DATA_PATH}: {e}")
    raise RuntimeError("Failed to initialize server")


def replace_black_pixels(
    image: np.ndarray, pixel_data_list: List[Tuple[int, int, List[int]]]
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("Empty image")
    h, w = image.shape[:2]
    valid_pixels = [
        (x, y, color) for x, y, color in pixel_data_list if 0 <= x < w and 0 <= y < h
    ]
    if not valid_pixels:
        return image
    xs, ys, colors = zip(*valid_pixels)
    xs, ys = np.array(xs, dtype=int), np.array(ys, dtype=int)
    colors = np.array(colors, dtype=np.uint8)
    black_pixels = np.all(image[ys, xs] == 0, axis=1)
    if np.any(black_pixels):
        image[ys[black_pixels], xs[black_pixels]] = colors[black_pixels]
    return image


def prepare_input_for_model(grayscale_images: List[np.ndarray]) -> np.ndarray:
    if len(grayscale_images) != 8:
        raise ValueError(f"Expected 8 images, got {len(grayscale_images)}")
    stacked_images = np.stack(grayscale_images, axis=0)  # Shape: (8, 333, 333)
    input_data = stacked_images[
        np.newaxis, ..., np.newaxis
    ]  # Shape: (1, 8, 333, 333, 1)
    input_data = input_data.astype(np.float32) / 255.0  # Normalize เป็น [0, 1]
    return input_data


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        if DEBUG:
            logger.debug(f"Received file: {file.filename}, size: {len(contents)} bytes")

        image_loaded = cv2.imdecode(
            np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image_loaded is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        restored_rgb_image = convert_grayscale_to_rgb(image_loaded)
        restored_rgb_image = cv2.cvtColor(restored_rgb_image, cv2.COLOR_BGR2RGB)
        restored_rgb_image = replace_black_pixels(
            restored_rgb_image, crop_pixel_data_list
        )

        success, img_encoded = cv2.imencode(".png", restored_rgb_image)
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed")

        return Response(
            content=img_encoded.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=output_image.png"},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /convert: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/prepare-images")
async def prepare_images(images: List[UploadFile] = File(...)):
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        # 1) อ่านข้อมูลไฟล์ทั้งหมดครั้งเดียว เก็บใน list
        file_contents_list = []
        for upload_file in images:
            contents = await upload_file.read()
            if not contents:
                raise HTTPException(
                    status_code=400, detail=f"Empty file: {upload_file.filename}"
                )
            file_contents_list.append(contents)

        # ============ เพิ่มตัวเก็บผล OCR =============
        # เก็บเป็น list ของ string เช่น "10:54 / 21-Mar-2024"
        ocr_results = []

        # 2) OCR: ตรวจจับข้อความในพื้นที่ (x: 801-967, y: 0-100)
        for idx, contents in enumerate(file_contents_list):
            decoded_img = cv2.imdecode(
                np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
            )
            if decoded_img is None:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image: {images[idx].filename}"
                )

            # ตัวอย่าง: เช็กขนาดภาพก่อน OCR
            if decoded_img.shape[1] >= 967 and decoded_img.shape[0] >= 100:
                # เลือก Crop บริเวณ [y0:y1, x0:x1]
                ocr_region = decoded_img[30:60, 801:967]
                text_found = pytesseract.image_to_string(
                    Image.fromarray(ocr_region)
                ).strip()
                logger.info(f"OCR (Image {idx+1}): {text_found}")
                ocr_results.append(text_found)
            else:
                logger.info(
                    f"OCR Skipped (Image {idx+1}): Image size too small for 801-969x0-100 region."
                )
                ocr_results.append("")  # เก็บเป็นค่าว่างถ้าไม่เจอ

        # ==================== ส่วน Parsing OCR =========================
        # สมมติรูปแบบเป็น "HH:MM / DD-MMM-YYYY" เช่น "10:54 / 21-Mar-2024"
        # เราจะ parse เป็น datetime
        def parse_ocr_datetime(ocr_text: str) -> datetime:
            """
            พยายามแยกเวลา (HH:MM) และวันที่ (DD-Mon-YYYY) จาก string
            Return เป็น datetime object (ถ้า parse สำเร็จ)
            """
            # ตัวอย่าง regex: (\d{1,2}:\d{2}) => เวลา , (\d{1,2}-[A-Za-z]{3}-\d{4}) => วันที่
            pattern = r"(\d{1,2}:\d{2})\s*/\s*(\d{1,2}-[A-Za-z]{3}-\d{4})"
            match = re.search(pattern, ocr_text)
            if not match:
                return None  # หรือ raise error

            time_str = match.group(1)  # เช่น "10:54"
            date_str = match.group(2)  # เช่น "21-Mar-2024"

            # แปลง date_str เป็น datetime => "%d-%b-%Y" (day-month_abbrev-year)
            # "%b" จะ parse "Mar" "Jan" "Feb" etc.
            try:
                date_part = datetime.strptime(
                    date_str, "%d-%b-%Y"
                )  # ได้ 2024-03-21 00:00:00
                # แปลง time_str เป็น เวลา
                hh, mm = time_str.split(":")
                date_part = date_part.replace(hour=int(hh), minute=int(mm))
                return date_part
            except ValueError:
                return None

        # หากมี 8 รูป เราจะ parse วันที่ + เวลาทั้ง 8
        times_ocr = []
        if len(images) == 8:
            for text in ocr_results:
                dt = parse_ocr_datetime(text)
                if dt is None:
                    # ถ้า parse ไม่ได้ จะเก็บ None
                    times_ocr.append(None)
                else:
                    times_ocr.append(dt)
            # ตัวอย่าง: times_ocr = [datetime(2024,3,21,10,54), datetime(2024,3,21,11,0), ...]

        # 3) ประมวลผลภาพตามขั้นตอน (Crop, Replace, Predict dBZ)
        grayscale_images = []
        for idx, contents in enumerate(file_contents_list):
            logger.info(f"Processing image {idx+1}: {images[idx].filename}")
            decoded_img = cv2.imdecode(
                np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
            )
            if decoded_img is None:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image: {images[idx].filename}"
                )

            # Step 2: Crop 1
            crop_area = (0, 0, 800, 800)
            cropped_img = crop_image(decoded_img, crop_area)

            # Step 3: Replace Pixels
            processed_img = replace_pixels(cropped_img, BACKGROUND_IMAGE)

            # Step 4: Crop 2
            crop_area_2 = (400, 67, 733, 400)
            cropped_img_2 = crop_image(processed_img, crop_area_2)

            # Step 5: Predict dBZ (ได้ grayscale uint8)
            grayscale_batch = predict_dbz_from_image_batch(
                cropped_img_2[np.newaxis, ...],
                color_model.best_svr_dbz_model,
                color_model.scaler_X_rgb,
                color_model.scaler_y_dbz,
                DEFAULT_DBZ_MIN,
                DEFAULT_DBZ_MAX,
            )
            grayscale_image = grayscale_batch[0, ..., 0]  # (333, 333)
            grayscale_images.append(grayscale_image)

        # ==================== เช็ควันที่และเวลา (เมื่อมี 8 ภาพ) ====================
        next_time_str = None  # เอาไว้เก็บเวลาถัดไปในรูป string
        if len(images) == 8:
            try:
                # กรอง None ออก
                valid_datetimes = [dt for dt in times_ocr if dt is not None]

                # ตรวจว่าทั้ง 8 รูป parse ได้ครบหรือไม่
                if len(valid_datetimes) != 8:
                    logger.warning(
                        "Some OCR time cannot be parsed or missing. Skip checking times."
                    )
                else:
                    # 1) เช็คว่า วัน/เดือน/ปี ตรงกันหมดไหม
                    #    เอาวันที่ภาพแรกเป็นเกณฑ์
                    base_date = valid_datetimes[
                        0
                    ].date()  # ex: datetime.date(2024,3,21)
                    all_same_date = all(
                        dt.date() == base_date for dt in valid_datetimes
                    )
                    if not all_same_date:
                        logger.warning("Date is not consistent across all 8 images.")
                    else:
                        # 2) เช็ค interval ระหว่างภาพ
                        intervals = []
                        for i in range(7):
                            diff = valid_datetimes[i + 1] - valid_datetimes[i]
                            intervals.append(diff)

                        # สมมติว่าเราคาดว่า interval เท่ากันหมด
                        # ก็เช็คว่า intervals[i] == intervals[0] สำหรับ i in range(1..len)
                        # (เผื่อมีเคสต่างกันเล็กน้อย, อาจต้องยืดหยุ่น)
                        if all(iv == intervals[0] for iv in intervals):
                            # ได้ interval หลัก
                            main_interval = intervals[0]  # type: timedelta
                            # แปลงเป็นนาที
                            interval_minutes = int(main_interval.total_seconds() // 60)
                            logger.info(
                                f"Images have consistent interval: {interval_minutes} minutes"
                            )

                            # 3) คำนวณเวลาถัดไป
                            last_dt = valid_datetimes[-1]
                            next_dt = last_dt + main_interval
                            next_time_str = next_dt.strftime("%H:%M")  # เช่น "11:42"
                            logger.info(f"Next time = {next_time_str}")
                        else:
                            logger.warning(
                                "Time intervals among images are not consistent."
                            )

            except Exception as e:
                logger.error(f"Error checking date/time: {e}")

        # 4) หากมี 8 ภาพ ให้ทำ Cloud Prediction (ทำไปแล้วข้างบน)
        if len(images) == 8:
            try:
                input_data = prepare_input_for_model(
                    grayscale_images
                )  # (1, 8, 333, 333, 1)
                predicted_cloud = cloud_model.predict(input_data, verbose=0)
                if predicted_cloud.shape != (1, 333, 333, 1):
                    raise ValueError(
                        f"Expected output shape (1, 333, 333, 1), got {predicted_cloud.shape}"
                    )
                predicted_cloud_image = (predicted_cloud[0, ..., 0] * 255).astype(
                    np.uint8
                )
            except Exception as e:
                logger.error(f"Error in Cloud Prediction: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Cloud Prediction failed: {str(e)}"
                )

        # 5) Convert ส่งออก (ใช้ภาพทำนายถ้า 8 ภาพ, ไม่งั้นใช้ grayscale สุดท้าย)
        try:
            image_for_convert = (
                predicted_cloud_image if len(images) == 8 else grayscale_images[-1]
            )
            # ---------- ส่ง Response ----------
            success, img_encoded = cv2.imencode(".png", image_for_convert)
            if not success:
                raise HTTPException(status_code=500, detail="Grayscale encoding failed")

            return Response(
                content=img_encoded.tobytes(),
                media_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=output_image.png",
                    "X-Next-Time": next_time_str,
                    "Access-Control-Expose-Headers": "X-Next-Time",
                },
            )

            logger.info("Image processed and converted successfully")
            return response

        except Exception as e:
            logger.error(f"Error in Step Convert: {e}")
            raise HTTPException(status_code=500, detail=f"Convert failed: {str(e)}")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /prepare-images: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
