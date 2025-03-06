import cv2
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import json
import logging
import uvicorn
from typing import Optional, Dict, Tuple
from PIL import Image as PILImage
from PIL.ExifTags import TAGS

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ค่าคงที่
DEBUG = False
CROP_PIXEL_DATA_PATH = "crop_pixel_data.json"
DEFAULT_DBZ_MIN, DEFAULT_DBZ_MAX = 8, 72


# โหลดโมเดลและข้อมูลฝึกสอน
class ColorModel:
    def __init__(self):
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
        self._train_model()

    def _train_model(self):
        try:
            X_rgb = np.array(self.color_values_reversed)
            y_dbz = np.array(self.dbz_values)
            X_dbz = y_dbz.reshape(-1, 1)

            self.scaler_X_dbz = StandardScaler()
            self.scaler_Y_rgb = StandardScaler()
            X_dbz_scaled = self.scaler_X_dbz.fit_transform(X_dbz)
            Y_rgb_scaled = self.scaler_Y_rgb.fit_transform(X_rgb)

            svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
            self.model = MultiOutputRegressor(svr)
            self.model.fit(X_dbz_scaled, Y_rgb_scaled)

            if DEBUG:
                logger.debug("Model trained successfully")
                logger.debug(f"X_dbz shape: {X_dbz.shape}, Y_rgb shape: {X_rgb.shape}")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise RuntimeError("Failed to initialize color model")


color_model = ColorModel()


# โหลดข้อมูล crop pixel
def load_crop_pixel_data(path: str) -> Dict[str, list]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} pixel mappings from {path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load crop pixel data: {e}")
        return {}


crop_pixel_data = load_crop_pixel_data(CROP_PIXEL_DATA_PATH)


# ฟังก์ชันยูทิลิตี้
def dbz_to_rgb_batch(dbz_array: np.ndarray) -> np.ndarray:
    try:
        dbz_array = np.array(dbz_array, dtype=np.float32).reshape(-1, 1)
        dbz_scaled = color_model.scaler_X_dbz.transform(dbz_array)
        rgb_pred_scaled = color_model.model.predict(dbz_scaled)
        rgb_pred = color_model.scaler_Y_rgb.inverse_transform(rgb_pred_scaled)
        return np.clip(rgb_pred, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Error in dbz_to_rgb_batch: {e}")
        raise


def convert_grayscale_to_rgb(
    grayscale_image: np.ndarray,
    dbz_min: float = DEFAULT_DBZ_MIN,
    dbz_max: float = DEFAULT_DBZ_MAX,
) -> np.ndarray:
    try:
        if grayscale_image is None or grayscale_image.size == 0:
            raise ValueError("Empty or invalid grayscale image")

        if len(grayscale_image.shape) == 3:
            grayscale_image = grayscale_image.squeeze()

        if len(grayscale_image.shape) != 2:
            raise ValueError(f"Invalid image shape: {grayscale_image.shape}")

        h, w = grayscale_image.shape
        dbz_vals = (grayscale_image.astype(np.float32) / 255.0) * (
            dbz_max - dbz_min
        ) + dbz_min
        black_mask = (dbz_vals >= 8.0) & (dbz_vals <= 28.0)
        not_black_mask = ~black_mask
        dbz_not_black = dbz_vals[not_black_mask]

        unique_dbz = (
            np.unique(dbz_not_black) if dbz_not_black.size > 0 else np.array([])
        )
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        if unique_dbz.size > 0:
            unique_rgb = dbz_to_rgb_batch(unique_dbz)
            dbz_to_color_map = dict(zip(unique_dbz, unique_rgb))
            rgb_image[not_black_mask] = [dbz_to_color_map[v] for v in dbz_not_black]

        return rgb_image
    except Exception as e:
        logger.error(f"Error in convert_grayscale_to_rgb: {e}")
        raise


def replace_black_pixels(image: np.ndarray, pixel_data: Dict[str, list]) -> np.ndarray:
    try:
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image")

        h, w, _ = image.shape
        for pixel_str, color_value in pixel_data.items():
            try:
                x, y = map(int, pixel_str.split(","))
                if 0 <= x < w and 0 <= y < h and np.array_equal(image[y, x], [0, 0, 0]):
                    image[y, x] = np.array(color_value, dtype=np.uint8)
            except (ValueError, IndexError):
                logger.warning(f"Invalid pixel data: {pixel_str}")
                continue
        return image
    except Exception as e:
        logger.error(f"Error in replace_black_pixels: {e}")
        raise


def analyze_colors(image: np.ndarray) -> Dict[str, any]:
    try:
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image")

        colors = image.reshape(-1, 3)
        non_black_mask = ~np.all(colors == [0, 0, 0], axis=1)
        non_black_colors = colors[non_black_mask]

        unique_colors, counts = np.unique(non_black_colors, axis=0, return_counts=True)
        color_analysis = {
            "total_non_black_pixels": int(non_black_colors.shape[0]),
            "unique_colors": int(unique_colors.shape[0]),
            "color_counts": {
                f"RGB{tuple(color)}": int(count)
                for color, count in zip(unique_colors, counts)
                if count > 1
            },
        }
        return color_analysis
    except Exception as e:
        logger.error(f"Error in analyze_colors: {e}")
        raise


def read_date_taken_from_exif(image_bytes: bytes) -> Optional[str]:
    try:
        with BytesIO(image_bytes) as bf:
            img_pil = PILImage.open(bf)
            exif_data = img_pil.getexif()
            if exif_data:
                for tag_id, val in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ("DateTimeOriginal", "DateTime"):
                        return str(val)
        return None
    except Exception as e:
        logger.debug(f"EXIF read error: {e}")
        return None


# FastAPI endpoint
@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Received file size: {len(contents)} bytes")

        # อ่าน EXIF
        date_taken = read_date_taken_from_exif(contents)

        # โหลดและประมวลผลภาพ
        nparr = np.frombuffer(contents, np.uint8)
        image_loaded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image_loaded is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        grayscale_image = (
            cv2.cvtColor(image_loaded, cv2.COLOR_BGR2GRAY)
            if len(image_loaded.shape) == 3 and image_loaded.shape[2] == 3
            else image_loaded
        )

        # แปลงภาพ
        restored_rgb_image = convert_grayscale_to_rgb(grayscale_image)
        restored_rgb_image = cv2.cvtColor(restored_rgb_image, cv2.COLOR_BGR2RGB)
        restored_rgb_image = replace_black_pixels(restored_rgb_image, crop_pixel_data)

        # วิเคราะห์สี
        color_info = analyze_colors(restored_rgb_image)

        # เข้ารหัสภาพ
        success, img_encoded = cv2.imencode(".png", restored_rgb_image)
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed")

        # สร้าง response
        response_data = {
            "color_analysis": color_info,
            "image": img_encoded.tobytes().hex(),
        }
        headers = {"X-Date-Taken": date_taken or ""}
        return JSONResponse(content=response_data, headers=headers)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /convert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # uvicorn main:app --reload
