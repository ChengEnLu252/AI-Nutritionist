# test_sam_minimal.py (擴展版)
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import cv2
import numpy as np
from PIL import Image
import sys # for python version

# 【重要】修改為你下載的 SAM ViT-Base 模型的檢查點路徑
SAM_CHECKPOINT_PATH_VIT_B = "models/sam_checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE_VIT_B = "vit_b"
DEVICE = "cpu"

# 【重要】修改為你的測試圖片路徑
TEST_IMAGE_PATH = "test_image/影像.jpg" 

# 之前提供的 draw_masks_on_image 函式 (你需要把它複製到這裡)
def draw_masks_on_image(image_pil, masks_data):
    if not masks_data or image_pil is None:
        return image_pil if image_pil else Image.new("RGB", (100,100), "lightgray")
    image_cv = np.array(image_pil.convert("RGB"))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    overlay = image_cv.copy()
    output_image = image_cv.copy()
    sorted_masks = sorted(masks_data, key=lambda x: x['area'], reverse=True)
    for mask_info in sorted_masks:
        mask = mask_info['segmentation']
        color = np.random.randint(50, 200, size=(3,), dtype=np.uint8).tolist()
        for c_channel in range(3):
            overlay[mask, c_channel] = color[c_channel]
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_image_rgb)

print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"嘗試載入 SAM 模型 ({MODEL_TYPE_VIT_B}) 從: {SAM_CHECKPOINT_PATH_VIT_B} 到 {DEVICE}")

if not os.path.exists(SAM_CHECKPOINT_PATH_VIT_B):
    print(f"錯誤: 檢查點檔案未找到於: {SAM_CHECKPOINT_PATH_VIT_B}")
    exit()
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"錯誤: 測試圖片未找到於: {TEST_IMAGE_PATH}")
    exit()

sam_model = None
mask_generator = None
try:
    sam_model = sam_model_registry[MODEL_TYPE_VIT_B](checkpoint=SAM_CHECKPOINT_PATH_VIT_B)
    print("sam_model_registry[MODEL_TYPE_VIT_B](...) 成功。")
    sam_model.to(device=DEVICE)
    print(f"sam_model.to(device='{DEVICE}') 成功。模型已移至 CPU。")

    mask_generator = SamAutomaticMaskGenerator(sam_model)
    print("SamAutomaticMaskGenerator 初始化成功。")

    print(f"\n讀取測試圖片: {TEST_IMAGE_PATH}")
    input_image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    input_image_numpy = np.array(input_image_pil)

    print("使用 SAM mask_generator.generate() 進行分割...")
    masks = mask_generator.generate(input_image_numpy)
    print(f"SAM 生成了 {len(masks)} 個遮罩。")

    if masks:
        print("繪製遮罩到圖片上...")
        annotated_image = draw_masks_on_image(input_image_pil, masks)
        annotated_image.save("sam_output_test.jpg") # 儲存結果圖片
        print("已儲存分割結果到 sam_output_test.jpg")

    print("最小化 SAM 模型載入與分割測試成功！")

except Exception as e:
    print(f"在最小化 SAM 模型載入與分割測試中發生錯誤: {e}")
    import traceback
    traceback.print_exc()