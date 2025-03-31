from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import json
import os

app = FastAPI()

class PixelUpdate(BaseModel):
    pixels: Dict[str, List[int]]

@app.post("/update_pixel_data")
async def update_pixel_data(data: PixelUpdate):
    file_path = "pixel_data.json"

    # โหลดข้อมูลเดิมจากไฟล์ (ถ้ามี)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                old_data = json.load(f)
        except json.JSONDecodeError:
            old_data = {}
    else:
        old_data = {}

    # Merge ข้อมูลใหม่
    for key, value in data.pixels.items():
        old_data[key] = value

    # เขียนลงไฟล์
    with open(file_path, "w") as f:
        json.dump(old_data, f, indent=4)

    return {"status": "ok", "message": "Pixel data updated.", "data": old_data}

