import albumentations as A
import cv2
import numpy as np
import torch
from augraphy import *
import cv2
import random
import requests
from PIL import Image
from io import BytesIO
import random
import numpy as np
import cv2
from PIL import Image as PILImage
import os


def get_random_pexels_image(
    query="texture", orientation="landscape", size=(720, 960), api_key=None
):
    headers = {"Authorization": api_key}

    # Get a random page of results (max 80 images per query)
    page = random.randint(1, 5)
    per_page = 15
    url = f"https://api.pexels.com/v1/search?query={query}&orientation={orientation}&per_page={per_page}&page={page}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    photos = data.get("photos", [])
    if not photos:
        raise ValueError("No photos returned for this query")

    # Pick a random photo from the results
    photo = random.choice(photos)
    image_url = photo["src"]["small"]

    # Download and return as PIL image
    img_response = requests.get(image_url)
    img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize to match your expected size if needed
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def generate_stamps(stamp_amount=40, output_dir="/content/stamps", api_key=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {stamp_amount} random Pexels images in {output_dir}...")
    for i in range(stamp_amount):
        img = get_random_pexels_image(
            query="grunge texture", size=(100, 100), api_key=api_key
        )
        cv2.imwrite(f"{output_dir}/pexels_texture_{i}.jpg", img)


def get_foreground():
    int_foreground = random.randint(0, 39)
    return f"/content/stamps/pexels_texture_{int_foreground}.jpg"


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))  # Full RGB range


def get_aug():
    augraphy_pipeline = AugraphyPipeline(
        [
            # PatternGenerator(p=0.3),
            Scribbles(p=0.4),
            # BrightnessTexturize(p=.7),
            #PageBorder(p=0.2),
            # BindingsAndFasteners(
            #     foreground=get_foreground(),
            #     edge_offset=(0, 960 / 2),
            #     nscales=(0.5, 3),
            #     p=0.4,
            # ),
            LightingGradient(p=0.6),
        ]
    )

    return augraphy_pipeline


def get_albu_pipeline():
    albumentations_pipeline = A.Compose(
        [
            # A.PadIfNeeded(
            #     min_height="1200",
            #     min_width="900",
            #     position="random",
            #     border_mode=cv2.BORDER_CONSTANT,
            #     fill=random_color(),
            #     fill_mask=0,
            #     p=0.75,
            # ),
            # A.Perspective(
            #     scale=[0.05, 0.1],
            #     keep_size=False,
            #     fit_output=False,
            #     interpolation=cv2.INTER_LINEAR,
            #     mask_interpolation=cv2.INTER_NEAREST,
            #     border_mode=(
            #         cv2.BORDER_REFLECT_101
            #         if random.random() > 0.5
            #         else cv2.BORDER_CONSTANT
            #     ),
            #     value=0,
            #     fill=random_color(),
            #     fill_mask=0,
            #     p=0.6,
            # ),
            A.RGBShift(
                r_shift_limit=[-20, 20],
                g_shift_limit=[-20, 20],
                b_shift_limit=[-20, 20],
                p=0.8,
            ),
            # A.RandomCropFromBorders(
            #     crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, p=0.3
            # ),
        ]
    )
    return albumentations_pipeline

    # === Combined augmentor ===


def augment_example(example):
    img: PILImage.Image = example["image"]

    # Convert to NumPy for Albumentations
    image_np = np.array(img)

    # Apply Albumentations
    alt = get_albu_pipeline()
    image_np = alt(image=image_np)["image"]

    # Apply Augraphy
    augraphy_pipeline = get_aug()
    img_aug = augraphy_pipeline(image_np)

    # Convert back to PIL
    img_aug = PILImage.fromarray(img_aug)
    example["image"] = img_aug

    return example
