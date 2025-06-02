# src/util/multimodal_module.py
import os
import traceback # 用於印出詳細的錯誤追蹤
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2 # OpenCV，用於圖像處理和繪製
import numpy as np # NumPy，用於高效的數值計算，尤其是陣列操作
from PIL import Image, ImageOps # Pillow (PIL Fork)，用於圖像的載入、轉換和處理
import torch # PyTorch 核心函式庫
import torchvision # 根據你的 PyTorch 用法，有時也需要 (例如，如果用到 torchvision 的轉換或模型)
import pandas as pd
# 從 Meta AI 的 segment-anything 套件匯入必要的元件
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print("成功從 segment_anything 匯入 SamAutomaticMaskGenerator。 (來自 multimodal_module)")
except ImportError:
    print("錯誤：未能匯入 segment_anything。請確保已正確安裝。(來自 multimodal_module)")
    SamAutomaticMaskGenerator = None # 設為 None 以便後續檢查，避免直接出錯
    
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    print("成功從 transformers 匯入 AutoImageProcessor 和 AutoModelForImageClassification。")
except ImportError:
    print("錯誤：未能匯入 transformers。請確保已安裝 transformers 套件。")
    AutoImageProcessor = None
    AutoModelForImageClassification = None


SAM_CHECKPOINT_PATH = "models/sam_checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu"

sam_model_global = None # 用於儲存已載入的基礎 SAM 模型
sam_mask_generator = None # 初始化為 None

VIT_MODEL_NAME = "SeyedAli/Food-Image-Classification-ViT" # << 【重要】更新為你找到的模型名稱
vit_processor = None
vit_model = None

def load_sam_model():
    global sam_model_global
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"multimodal_module.load_sam_model: SAM 檢查點檔案未找到於: {SAM_CHECKPOINT_PATH}")
        return False
    try:
        print(f"multimodal_module.load_sam_model: 正在載入 SAM 基礎模型 ({MODEL_TYPE}) 到 {DEVICE}...")
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam_model_global = sam # 將載入的模型賦值給全域變數
        print("multimodal_module.load_sam_model: SAM 基礎模型載入成功。")
        return True
    except Exception as e:
        print(f"multimodal_module.load_sam_model: 載入 SAM 基礎模型過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def _ensure_mask_generator_initialized():
    """確保 SamAutomaticMaskGenerator 已被初始化"""
    global sam_mask_generator, sam_model_global
    if sam_mask_generator is None:
        if sam_model_global is None:
            print("_ensure_mask_generator_initialized: 基礎 SAM 模型尚未載入，無法初始化產生器。")
            if not load_sam_model(): # 嘗試再次載入基礎模型
                return False # 如果基礎模型載入失敗

        if sam_model_global is not None: # 再次檢查基礎模型是否已載入
            try:
                print("_ensure_mask_generator_initialized: 正在初始化 SamAutomaticMaskGenerator...")
                sam_mask_generator = SamAutomaticMaskGenerator(sam_model_global)
                print("_ensure_mask_generator_initialized: SamAutomaticMaskGenerator 初始化成功。")
                return True
            except Exception as e:
                print(f"_ensure_mask_generator_initialized: 初始化 SamAutomaticMaskGenerator 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            return False # 基礎模型仍然未載入
    return True # 已初始化

def load_vit_model():
    global vit_processor, vit_model
    if AutoImageProcessor is None or AutoModelForImageClassification is None: # 檢查 transformers 是否成功匯入
        print("multimodal_module.load_vit_model: Transformers 元件未匯入，無法載入 ViT 模型。")
        return False
    try:
        print(f"multimodal_module.load_vit_model: 正在載入 ViT 圖像處理器從: {VIT_MODEL_NAME}...")
        vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
        print(f"multimodal_module.load_vit_model: 正在載入 ViT 分類模型從: {VIT_MODEL_NAME} 到 {DEVICE}...")
        vit_model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_NAME)
        vit_model.to(DEVICE)
        vit_model.eval() # 設定為評估模式，關閉 dropout 等
        print(f"ViT 食物分類模型 ({VIT_MODEL_NAME}) 載入並設定成功。")
        # 你可以印出模型的標籤來看看它能識別哪些食物類別
        # print("模型可識別的標籤 (部分):", {k: vit_model.config.id2label[k] for k in list(vit_model.config.id2label)[:10]})
        return True
    except Exception as e:
        print(f"multimodal_module.load_vit_model: 載入 ViT 模型過程中發生錯誤: {e}")
        traceback.print_exc()
        return False

def classify_image_batch(image_list_pil):
    if vit_processor is None or vit_model is None:
        print("classify_image_batch: ViT 模型或處理器未載入。")
        # 為每個圖像返回一個統一的錯誤訊息或標籤
        return ["ViT模型未載入"] * len(image_list_pil) if image_list_pil else []

    if not image_list_pil: # 如果傳入空列表
        return []

    try:
        # 預處理圖像: images 可以是單個 PIL Image 或 PIL Image 的列表
        inputs = vit_processor(images=image_list_pil, return_tensors="pt").to(DEVICE)

        # 進行預測
        with torch.no_grad(): # 在推理時不需要計算梯度
            logits = vit_model(**inputs).logits

        # 獲取預測類別的索引
        predicted_class_idx = logits.argmax(-1).tolist() # .tolist() 將 tensor 轉為 python list

        # 將索引轉換為人類可讀的標籤名稱
        # vit_model.config.id2label 是一個字典，將索引映射到標籤名稱
        predicted_labels = [vit_model.config.id2label[idx] for idx in predicted_class_idx]
        return predicted_labels

    except Exception as e:
        print(f"classify_image_batch: 圖像分類時發生錯誤: {e}")
        traceback.print_exc()
        return ["分類錯誤"] * len(image_list_pil) # 為每個圖像返回錯誤訊息
    

ENGLISH_TO_CHINESE_FOOD_MAP = {
    "apple": "蘋果",
    "banana": "香蕉",
    "orange": "柳橙", # 或 "橙子"
    "strawberry": "草莓",
    "grape": "葡萄",
    "watermelon": "西瓜",
    "pineapple": "鳳梨", # 或 "菠蘿"
    "mango": "芒果",
    "kiwi": "奇異果", # 或 "獼猴桃"
    "avocado": "酪梨", # 或 "牛油果"
    "tomato": "番茄", # 或 "西紅柿"
    "potato": "馬鈴薯", # 或 "土豆"
    "carrot": "胡蘿蔔",
    "broccoli": "西蘭花", # 或 "青花菜"
    "spinach": "菠菜",
    "lettuce": "生菜", # 或特定品種的萵苣
    "cucumber": "黃瓜",
    "onion": "洋蔥",
    "garlic": "大蒜",
    "bell pepper": "甜椒", # 可能需要根據顏色細分，例如 "red bell pepper" -> "紅甜椒"
    "chicken breast": "雞胸肉",
    "beef steak": "牛排",
    "salmon": "鮭魚",
    "shrimp": "蝦",
    "egg": "雞蛋",
    "rice": "米飯",
    "bread": "麵包",
    "cheese": "起司", # 或 "芝士"
    "milk": "牛奶",
    "yogurt": "優格", # 或 "酸奶"
    "coffee": "咖啡",
    "tea": "茶",
    "almond": "杏仁",
    "walnut": "核桃",
    "chickpea": "鷹嘴豆", # 對應你截圖中的鷹嘴豆
    "sweet potato": "地瓜", # 或 "甘薯", 對應你截圖中的烤地瓜塊
    # ... 你需要根據模型實際輸出的英文標籤和你的需求不斷擴充這個列表 ...
    # 模型的標籤可能是小寫，也可能包含空格，你需要觀察實際輸出來調整 key
}

def translate_food_labels_to_chinese(english_labels):
    chinese_labels = []
    for label_en in english_labels:
        # 將英文標籤轉為小寫並去除多餘空格，以提高匹配成功率
        processed_label_en = label_en.lower().strip()
        # 優先查找完全匹配，如果沒有，可以考慮部分匹配或更複雜的邏輯
        chinese_name = ENGLISH_TO_CHINESE_FOOD_MAP.get(processed_label_en, label_en) # 如果找不到翻譯，則返回原始英文標籤
        chinese_labels.append(chinese_name)
    return chinese_labels


SIMPLE_PORTION_ESTIMATES_GRAMS = {
    "蘋果": 150,  # 一個中等蘋果大約克數
    "香蕉": 100,  # 一根中等香蕉
    "雞胸肉": 120, # 一份常見的雞胸肉
    "番茄": 80,   # 一個中等番茄
    "生菜": 50,   # 一些生菜葉子
    "酪梨": 70,   # 半顆中等酪梨
    "米飯": 200,  # 一碗米飯
    # ... 你需要為你的 ViT 模型可能識別出的主要食材添加條目和預估克數 ...
    # 預設值，如果食材不在字典中
    "default": 100 
}

# 新增一個函式來查詢你的營養資料庫 (df_map)
# 這個函式會根據食材中文名稱，從你的 RAG 資料對照表中查找營養資訊
# df_map 應該是你在 load_rag_components() 中載入的那個 DataFrame
def get_nutrition_data_for_food(food_name_ch): # 接收一個參數
    # 在函式內部動態獲取 df_map
    current_df_map_from_rag = None # 先初始化
    try:
        from .rag_pipeline import df_map as imported_df_map # 相對匯入
        if imported_df_map is not None and not imported_df_map.empty:
            current_df_map_from_rag = imported_df_map
            print(f"get_nutrition_data_for_food: 成功從 rag_pipeline 獲取 df_map，資料筆數: {len(current_df_map_from_rag)}")
        else:
            print(f"get_nutrition_data_for_food: 從 rag_pipeline 獲取的 df_map 為空或 None。")
            return None
    except ImportError:
        print("get_nutrition_data_for_food: 無法從 rag_pipeline 匯入 df_map。")
        return None
    except AttributeError: # 如果 rag_pipeline 中還沒有 df_map 這個變數
        print("get_nutrition_data_for_food: rag_pipeline 模組中可能尚未定義 df_map。")
        return None
    
    if not food_name_ch:
        return None

    # 使用 current_df_map_from_rag 進行查詢
    # 確保 'food_name' 是 df_map 中的正確欄位名
    match = current_df_map_from_rag[current_df_map_from_rag['food_name'].astype(str).str.contains(str(food_name_ch), case=False, na=False)]
    
    if not match.empty:
        nutrition_data_per_100g = match.iloc[0] 
        # 確保 'calories_kcal' 等是你 df_map 中的實際欄位名
        print(f"為「{food_name_ch}」找到營養資料: 每100g 熱量 {nutrition_data_per_100g.get('calories_kcal', '未知')}")
        return nutrition_data_per_100g
    else:
        print(f"未能為「{food_name_ch}」在資料庫中找到匹配的營養資料。")
        return None

def handle_image_analysis(uploaded_image_pil):
    # 1. 初始檢查
    if uploaded_image_pil is None:
        print("handle_image_analysis: 錯誤 - 未收到任何圖片。")
        return None, "請先上傳圖片。", None

    print(f"handle_image_analysis: 接收到圖片，類型: {type(uploaded_image_pil)}, 尺寸: {uploaded_image_pil.size}")

    # 確保 SAM 分割器已初始化 (這會呼叫 _ensure_mask_generator_initialized)
    if not _ensure_mask_generator_initialized() or sam_mask_generator is None:
        print("handle_image_analysis: 錯誤 - SAM 影像分割元件未能成功初始化。")
        return uploaded_image_pil, "錯誤：影像分割元件未能成功初始化。", None
    
    # 確保 ViT 模型已載入 (這會呼叫 load_vit_model)
    if vit_processor is None or vit_model is None:
        print("handle_image_analysis: ViT 模型未載入，嘗試載入...")
        if not load_vit_model(): # 嘗試載入 ViT
            return uploaded_image_pil, "錯誤：食材分類 ViT 模型未能載入。", None
            
    # 在函式執行時動態獲取 RAG 的 df_map
    try:
        from .rag_pipeline import df_map as current_df_map_from_rag # 相對匯入
        if current_df_map_from_rag is None or current_df_map_from_rag.empty:
            print("handle_image_analysis: 從 rag_pipeline 獲取的 df_map 為空或 None。")
            return uploaded_image_pil, "錯誤：營養資料庫 (df_map) 未能載入或為空。", None
    except ImportError:
        print("handle_image_analysis: 無法從 rag_pipeline 匯入 df_map。")
        return uploaded_image_pil, "錯誤：無法連接到營養資料庫。", None

    try:
        print("handle_image_analysis: 開始 SAM 影像分割流程...")
        image_rgb_numpy = np.array(uploaded_image_pil.convert("RGB"))
        print(f"handle_image_analysis: 圖片已轉換為 NumPy array (RGB), shape: {image_rgb_numpy.shape}")

        print("handle_image_analysis: 呼叫 SAM mask_generator.generate()...")
        masks_data = sam_mask_generator.generate(image_rgb_numpy)
        
        if masks_data is None:
             print("handle_image_analysis: SAM mask_generator.generate() 返回 None。")
             return uploaded_image_pil, "影像分割時 SAM 模型未返回有效結果。", None

        print(f"handle_image_analysis: SAM 模型成功生成了 {len(masks_data)} 個原始遮罩。")

        if not masks_data:
            print("handle_image_analysis: SAM 未能在圖片中偵測到任何物件/遮罩 (masks_data 列表為空)。")
            return uploaded_image_pil, "未能在此圖片中偵測到任何物件/遮罩。", None
        
        # --- 遮罩過濾邏輯 ---
        print("multimodal_module.handle_image_analysis: 開始過濾遮罩...")
        image_total_area = image_rgb_numpy.shape[0] * image_rgb_numpy.shape[1]
        filtered_masks_data = []
        for mask_info in masks_data:
            min_area_threshold = 100  # 可調整
            max_area_ratio_threshold = 0.85 # 可調整

            if mask_info['area'] < min_area_threshold:
                print(f"  - 過濾掉小面積遮罩: area={mask_info['area']}")
                continue
            if mask_info['area'] > image_total_area * max_area_ratio_threshold:
                print(f"  - 過濾掉大面積背景遮罩: area={mask_info['area']}")
                continue
            filtered_masks_data.append(mask_info)
        
        print(f"multimodal_module.handle_image_analysis: 過濾後剩下 {len(filtered_masks_data)} 個遮罩。")

        if not filtered_masks_data:
            print("multimodal_module.handle_image_analysis: 過濾後沒有剩下任何有效遮罩。")
            # 仍然顯示原始圖片的分割（基於未過濾的 masks_data），但提示沒有有效食材區域
            annotated_image_pil_for_display = draw_masks_on_image(uploaded_image_pil.copy(), masks_data)
            return annotated_image_pil_for_display, "未能從圖片中過濾出有效的食材區域。", None # cropped_item_images_pil 為 None

        # --- 提取裁剪圖像區域 (基於過濾後的遮罩) ---
        cropped_item_images_pil = [] # 初始化為空列表
        print("multimodal_module.handle_image_analysis: 開始提取物件圖像區域 (基於過濾後的遮罩)...")
        
        for i, mask_info in enumerate(filtered_masks_data): # 使用 filtered_masks_data
            bbox = mask_info['bbox'] 
            x, y, w, h = map(int, bbox)
            
            # print(f"\n  正在處理過濾後的遮罩 {i}:") # 可以取消註解以進行詳細調試
            # print(f"    原始 bbox={mask_info['bbox']}, 轉換後 x={x}, y={y}, w={w}, h={h}, 面積(area)={mask_info['area']}")

            if w <= 0 or h <= 0:
                print(f"    - 警告: 物件 {i} 的 bbox 寬高無效 (w={w}, h={h})，跳過。")
                continue
                
            left, upper, right, lower = x, y, x + w, y + h
            # print(f"    - 準備裁剪區域: left={left}, upper={upper}, right={right}, lower={lower}")
            
            try:
                cropped_item_pil = uploaded_image_pil.crop((left, upper, right, lower))
                if cropped_item_pil.size[0] == 0 or cropped_item_pil.size[1] == 0:
                    print(f"    - 警告: 物件 {i} 的邊界框導致了空的裁剪區域，跳過。")
                    continue
                cropped_item_images_pil.append(cropped_item_pil)
                # print(f"    - 成功提取物件 {i}，區域大小: {cropped_item_pil.size}")
            except Exception as e:
                print(f"    - 錯誤: 物件 {i} 裁剪時發生錯誤: {e}，跳過。")
                continue
        
        print(f"\n總共提取了 {len(cropped_item_images_pil)} 個有效物件圖像。")
        
        # --- 食材分類、翻譯、份量估算、營養映射 ---
        predicted_labels_en = []
        if cropped_item_images_pil: # 只有在成功提取到裁剪圖像後才進行分類
            print(f"準備對 {len(cropped_item_images_pil)} 個裁剪圖像進行分類...")
            predicted_labels_en = classify_image_batch(cropped_item_images_pil) # 假設此函式已定義
        else:
            print("沒有裁剪的圖像可供分類。")
        
        predicted_labels_ch = []
        if predicted_labels_en:
            print(f"準備翻譯英文標籤: {predicted_labels_en}")
            predicted_labels_ch = translate_food_labels_to_chinese(predicted_labels_en) # 假設此函式已定義

        processed_food_items = []
        total_calories = 0.0
        # ... (可以加入總蛋白質、脂肪、碳水化合物的累加變數)

        if predicted_labels_ch: # 確保有中文標籤才進行後續處理
            for i, food_name_ch in enumerate(predicted_labels_ch):
                if i >= len(cropped_item_images_pil): continue # 避免索引超出裁剪圖像列表的範圍

                # 份量估算 (使用簡化版)
                estimated_grams = SIMPLE_PORTION_ESTIMATES_GRAMS.get(food_name_ch, SIMPLE_PORTION_ESTIMATES_GRAMS["default"])
                
                # 熱量與營養映射
                nutrition_data = get_nutrition_data_for_food(food_name_ch)
                
                item_calories = 0.0
                # ... (其他營養素 item_protein = 0.0 等)
                
                if nutrition_data is not None:
                    # 確保欄位名與你的 df_map 中的實際欄位名一致
                    calories_val = nutrition_data.get('calories_kcal', 0)
                    # protein_val = nutrition_data.get('protein_g', 0) 
                    
                    # 處理可能的非數值情況
                    try:
                        calories_per_100g = float(calories_val if pd.notna(calories_val) else 0)
                        # protein_per_100g = float(protein_val if pd.notna(protein_val) else 0)
                    except ValueError:
                        calories_per_100g = 0
                        # protein_per_100g = 0
                        print(f"警告：食材 '{food_name_ch}' 的營養數值轉換失敗。calories_val='{calories_val}'")


                    item_calories = (calories_per_100g / 100.0) * estimated_grams
                    total_calories += item_calories
                    # ... (計算並累加其他營養素)
                
                processed_food_items.append({
                    "name": food_name_ch,
                    "estimated_grams": round(estimated_grams, 1),
                    "estimated_calories": round(item_calories, 1),
                })
        
        # --- 構建最終的狀態訊息 ---
        status_message_parts = [f"影像分割完成，過濾後找到 {len(filtered_masks_data)} 個主要物件區域。"]
        if processed_food_items:
            status_message_parts.append("\n初步營養估算結果：")
            for item in processed_food_items:
                status_message_parts.append(f"  - {item['name']}: 約 {item['estimated_grams']}克, 熱量約 {item['estimated_calories']}大卡")
            status_message_parts.append(f"\n整餐預估總熱量：約 {round(total_calories, 1)} 大卡")
        elif cropped_item_images_pil: # 有裁剪圖像但可能分類或營養查詢失敗
            status_message_parts.append("已提取食材區域，但未能完成所有食材的營養分析。")
        else: # 連裁剪圖像都沒有
             status_message_parts.append("未能從圖片中提取有效的食材區域進行分析。")
        
        final_status_message = "\n".join(status_message_parts)
        
        # --- 準備回傳 ---
        # 繪製過濾後的遮罩到圖片上進行顯示
        annotated_image_pil_for_display = draw_masks_on_image(uploaded_image_pil.copy(), filtered_masks_data)

        print(f"\n--- handle_image_analysis 即將回傳 ---")
        print(f"回傳的分割圖片類型: {type(annotated_image_pil_for_display)}")
        if annotated_image_pil_for_display: print(f"回傳的分割圖片尺寸: {annotated_image_pil_for_display.size}")
        print(f"回傳的狀態訊息 (final_status_message):\n---\n{final_status_message}\n---")
        print(f"回傳的裁剪圖片列表 (cropped_item_images_pil) 長度: {len(cropped_item_images_pil)}")

        return annotated_image_pil_for_display, final_status_message, cropped_item_images_pil

    except Exception as e:
        print(f"handle_image_analysis: 處理圖片時發生嚴重錯誤: {e}")
        traceback.print_exc()
        return uploaded_image_pil, f"處理圖片時發生錯誤: {str(e)}", None

# draw_masks_on_image 函式定義 (與之前相同)
def draw_masks_on_image(image_pil, masks_data):
    # ... (省略，與之前提供的版本相同) ...
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