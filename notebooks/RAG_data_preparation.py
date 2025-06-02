# 1. 導入必要的函式庫
import pandas as pd
import os

# 2. 設定檔案路徑
# 原始數據路徑 (相對於專案根目錄)
raw_data_path = "data/raw/"
# 處理後數據儲存路徑 (相對於專案根目錄)
processed_data_path = "data/processed/"

# 衛福部資料庫檔案名稱 (根據你的截圖)
taiwan_food_db_filename = "衛福部台灣食品營養成分資料庫.csv"

# USDA 資料庫檔案名稱 (先嘗試使用單一的 CSV 檔案)
usda_food_db_filename = "USDA_FDC_data.csv"

os.makedirs(processed_data_path, exist_ok=True)

# --- 處理衛福部台灣食品營養成分資料庫 ---
print(f"開始處理 {taiwan_food_db_filename}...")
tw_file_path = os.path.join(raw_data_path, taiwan_food_db_filename)

if os.path.exists(tw_file_path):
    try:
        # 嘗試不同的編碼，直到中文正確顯示
        # 常見中文編碼: 'utf-8', 'big5', 'cp950'
        df_tw = pd.read_csv(tw_file_path, encoding='utf-8',header = 1) # 先嘗試 utf-8
    except UnicodeDecodeError:
        try:
            df_tw = pd.read_csv(tw_file_path, encoding='cp950',header = 1) # 嘗試 cp950 (常用於台灣繁體系統的 CSV)
        except UnicodeDecodeError:
            try:
                df_tw = pd.read_csv(tw_file_path, encoding='big5',header = 1)
            except Exception as e:
                print(f"載入 {taiwan_food_db_filename} 時發生錯誤，嘗試了 utf-8, cp950, big5 編碼: {e}")
                df_tw = pd.DataFrame() # 建立空 DataFrame
    except Exception as e:
        print(f"載入 {taiwan_food_db_filename} 時發生非預期錯誤: {e}")
        df_tw = pd.DataFrame()

    if not df_tw.empty:
        print("\n--- 衛福部資料庫 初步探索 ---")
        print("欄位名稱 (Columns):")
        print(df_tw.columns.tolist()) # 印出所有欄位名稱列表
        print("\n前 5 筆資料 (Head):")
        print(df_tw.head())
        print("\n資料資訊 (Info):")
        df_tw.info()
        print("\n缺失值統計 (Null Values per Column):")
        print(df_tw.isnull().sum())

        # ***** 你需要根據上面印出的欄位名稱來修改以下部分 *****
        # 5. 選擇相關欄位 (範例，請替換成實際的欄位名)
        # 例如，假設你發現欄位名是 '樣品名稱', '修正熱量(kcal)', '蛋白質(g)' 等
        relevant_cols_tw = [
            '樣品名稱', # 假設這是食品名稱的欄位
            # '食品分類', # 如果有的話
            '熱量(kcal)', # 假設這是熱量的欄位 (名稱可能不同，例如 '修正熱量(kcal)')
            '總碳水化合物(g)', # 假設
            '粗蛋白(g)',     # 假設
            '粗脂肪(g)',     # 假設
            '膳食纖維(g)'    # 假設
            # ... 添加其他你需要的營養素欄位 ...
        ]
        
        # 篩選出實際存在的欄位
        existing_relevant_cols_tw = [col for col in relevant_cols_tw if col in df_tw.columns]
        print(f"\n選擇的衛福部相關欄位 (存在的): {existing_relevant_cols_tw}")
        df_tw_selected = df_tw[existing_relevant_cols_tw].copy()

        # 6. 資料清洗
        print("\n--- 衛福部資料庫 資料清洗 (請根據實際情況調整) ---")
        # 重新命名欄位 (範例，請根據實際欄位名和你的偏好修改)
        rename_map_tw = {
            '樣品名稱': 'food_name',
            '熱量(kcal)': 'calories_kcal', # 確認熱量欄位名稱
            '總碳水化合物(g)': 'carbohydrates_g',
            '粗蛋白(g)': 'protein_g',
            '粗脂肪(g)': 'fat_g',
            '膳食纖維(g)': 'dietary_fiber_g'
            # ... 其他需要重新命名的欄位 ...
        }
        actual_rename_map_tw = {k: v for k, v in rename_map_tw.items() if k in df_tw_selected.columns}
        df_tw_selected.rename(columns=actual_rename_map_tw, inplace=True)

        # 處理數值欄位的缺失值和類型轉換 (使用 rename 後的欄位名)
        numeric_cols = ['calories_kcal', 'carbohydrates_g', 'protein_g', 'fat_g', 'dietary_fiber_g']
        for col in numeric_cols:
            if col in df_tw_selected.columns:
                df_tw_selected[col] = pd.to_numeric(df_tw_selected[col], errors='coerce').fillna(0)
        
        # 確保 food_name 沒有缺失值並為字串
        if 'food_name' in df_tw_selected.columns:
            df_tw_selected.dropna(subset=['food_name'], inplace=True)
            df_tw_selected['food_name'] = df_tw_selected['food_name'].astype(str)
        else:
            print("警告: 衛福部資料中未找到 'food_name' (或其重命名後) 的欄位，無法繼續處理。")


        # 7. 創建用於 Embedding 的文本描述 (使用 rename 後的欄位名)
        if 'food_name' in df_tw_selected.columns:
            print("\n--- 衛福部資料庫 創建 Embedding 文本 ---")
            def create_embedding_text_tw(row):
                texts = []
                texts.append(f"食品名稱: {row.get('food_name', '未知食品')}.")
                # if '食品分類_renamed' in row and pd.notna(row['食品分類_renamed']): # 假設 '食品分類' 被 rename 成 '食品分類_renamed'
                #    texts.append(f"分類: {row['食品分類_renamed']}.")
                if pd.notna(row.get('calories_kcal')): texts.append(f"熱量: {row['calories_kcal']:.1f} 大卡.") # 保留一位小數
                if pd.notna(row.get('protein_g')): texts.append(f"蛋白質: {row['protein_g']:.1f} 克.")
                if pd.notna(row.get('carbohydrates_g')): texts.append(f"碳水化合物: {row['carbohydrates_g']:.1f} 克.")
                if pd.notna(row.get('fat_g')): texts.append(f"脂肪: {row['fat_g']:.1f} 克.")
                if pd.notna(row.get('dietary_fiber_g')): texts.append(f"膳食纖維: {row['dietary_fiber_g']:.1f} 克.")
                # ... 添加更多你認為重要的營養資訊 ...
                return " ".join(texts)

            df_tw_selected['embedding_text'] = df_tw_selected.apply(create_embedding_text_tw, axis=1)
            print("Embedding 文本範例 (衛福部):")
            print(df_tw_selected[['food_name', 'embedding_text']].head())

            # 8. 儲存處理後的資料
            processed_tw_filename = "processed_taiwan_food_data.csv"
            df_tw_selected.to_csv(os.path.join(processed_data_path, processed_tw_filename), index=False, encoding='utf-8-sig')
            print(f"\n衛福部處理完成的資料已儲存到: {os.path.join(processed_data_path, processed_tw_filename)}")
        else:
            print("因缺少 'food_name' 欄位，無法為衛福部資料創建 embedding text 或儲存。")
    else:
        print(f"衛福部資料庫 {taiwan_food_db_filename} 為空或載入失敗。")
else:
    print(f"找不到衛福部資料庫檔案: {tw_file_path}")



# --- 處理 USDA FDC 資料庫 (從多個 CSV 檔案) ---
print("\n--- 開始處理 USDA FDC 資料庫 (從多個 CSV) ---")
# USDA 解壓縮後 CSV 檔案存放的路徑
usda_extracted_path = os.path.join(raw_data_path, "USDA_FDC_extracted_data/")

if os.path.exists(usda_extracted_path):
    try:
        print(f"從 {usda_extracted_path} 載入 USDA CSV 檔案...")
        # 選擇性載入欄位以節省記憶體，你需要根據實際需求調整 'usecols'
        df_food = pd.read_csv(os.path.join(usda_extracted_path, "food.csv"),
                              usecols=['fdc_id', 'description', 'data_type'], low_memory=False)
        df_nutrient = pd.read_csv(os.path.join(usda_extracted_path, "nutrient.csv"),
                                  usecols=['id', 'name', 'unit_name'], low_memory=False)
        df_food_nutrient = pd.read_csv(os.path.join(usda_extracted_path, "food_nutrient.csv"),
                                       usecols=['fdc_id', 'nutrient_id', 'amount'], low_memory=False)

        print("\n--- USDA food.csv 初步探索 ---")
        print("food.csv Columns:", df_food.columns.tolist())
        print(df_food.head())
        # 你可能需要根據 'data_type' 欄位篩選想使用的食品種類
        # 例如: df_food = df_food[df_food['data_type'] == 'sr_legacy'] # 只選擇 SR Legacy 食品

        print("\n--- USDA nutrient.csv 初步探索 ---")
        print("nutrient.csv Columns:", df_nutrient.columns.tolist())
        print(df_nutrient[['id', 'name', 'unit_name']].head())
        
        # *** 非常重要：你需要查看 nutrient.csv 來確定你關心的營養素的確切英文名稱和單位 ***
        # 例如，你可能關心的營養素（這些名稱需要你在 nutrient.csv 中核對！）：
        target_nutrient_english_names = [
            'Protein',                      # 蛋白質
            'Total lipid (fat)',            # 總脂肪
            'Carbohydrate, by difference',  # 碳水化合物 (差量法)
            'Energy',                       # 能量/熱量
            'Fiber, total dietary'          # 總膳食纖維
            # 添加其他你需要的營養素的英文名稱
        ]
        
        # 將 nutrient.csv 中的 'id' 欄位重命名為 'nutrient_id' 以便與 food_nutrient.csv 合併
        df_nutrient.rename(columns={'id': 'nutrient_id'}, inplace=True)

        # 1. 將 food_nutrient 與 nutrient 合併，獲取營養素名稱和單位
        print("\n合併 food_nutrient 和 nutrient 表...")
        df_merged_nutrients = pd.merge(df_food_nutrient, df_nutrient, on='nutrient_id')
        print("合併後 df_merged_nutrients 的前幾行:")
        print(df_merged_nutrients.head())

        # 2. 篩選出我們關心的營養素
        print(f"\n篩選目標營養素: {target_nutrient_english_names}")
        df_filtered_target_nutrients = df_merged_nutrients[df_merged_nutrients['name'].isin(target_nutrient_english_names)]
        
        # *** 特別注意 'Energy' 的單位 ***
        # 'Energy' 可能有 'KCAL' 和 'KJ' 兩種單位。你需要選擇一種，通常是 'KCAL'。
        # 例如，只保留 'Energy' 且單位是 'KCAL' 的數據：
        is_energy = df_filtered_target_nutrients['name'] == 'Energy'
        is_kcal = df_filtered_target_nutrients['unit_name'] == 'KCAL'
        df_energy_kcal = df_filtered_target_nutrients[is_energy & is_kcal]
        # 其他非 Energy 的營養素
        df_other_nutrients = df_filtered_target_nutrients[~is_energy]
        # 合併回篩選後的營養素表
        df_final_filtered_nutrients = pd.concat([df_other_nutrients, df_energy_kcal])
        print("篩選並處理單位後的營養素數據前幾行:")
        print(df_final_filtered_nutrients.head())


        # 3. 進行數據透視，將長表轉換為寬表 (每種食物一行，各種營養素為列)
        print("\n進行數據透視...")
        df_pivoted_usda = df_final_filtered_nutrients.pivot_table(
            index='fdc_id',         # 以食物 FDC ID 為索引
            columns='name',         # 以營養素名稱為欄位
            values='amount'         # 填充的值為營養素含量
        ).reset_index()
        print("透視表 df_pivoted_usda 的前幾行:")
        print(df_pivoted_usda.head())
        print("透視表 df_pivoted_usda 的欄位:", df_pivoted_usda.columns.tolist())


        # 4. 將透視後的營養素數據與食物描述資訊合併
        print("\n合併食物描述與透視後的營養素數據...")
        df_usda_processed = pd.merge(df_food[['fdc_id', 'description']], df_pivoted_usda, on='fdc_id', how='inner')
        print("最終合併的 USDA 數據前幾行:")
        print(df_usda_processed.head())

        # 5. 清理與重命名欄位 (欄位名會是 target_nutrient_english_names 中的值)
        # 根據你的 embedding_text 函數和後續使用，統一欄位名
        # 這些 key (左邊) 必須是 df_usda_processed 中實際存在的欄位名 (來自 target_nutrient_english_names)
        usda_rename_map = {
            'description': 'food_name',
            'Energy': 'calories_kcal', # 假設上面已篩選為 KCAL
            'Protein': 'protein_g',
            'Total lipid (fat)': 'fat_g',
            'Carbohydrate, by difference': 'carbohydrates_g',
            'Fiber, total dietary': 'dietary_fiber_g'
            # ... 其他根據 target_nutrient_english_names 中的實際名稱進行映射 ...
        }
        # 只重命名實際存在的欄位
        actual_usda_rename_map = {k: v for k, v in usda_rename_map.items() if k in df_usda_processed.columns}
        df_usda_processed.rename(columns=actual_usda_rename_map, inplace=True)
        print("重命名欄位後的 USDA 數據前幾行:")
        print(df_usda_processed.head())
        print("重命名欄位後的 USDA 數據欄位:", df_usda_processed.columns.tolist())
        
        # 處理數值欄位的缺失值 (透視後可能產生 NaN) 和類型轉換
        # 使用 rename 後的欄位名
        usda_numeric_cols = ['calories_kcal', 'protein_g', 'fat_g', 'carbohydrates_g', 'dietary_fiber_g']
        for col in usda_numeric_cols:
            if col in df_usda_processed.columns:
                df_usda_processed[col] = pd.to_numeric(df_usda_processed[col], errors='coerce').fillna(0)

        # 6. 創建用於 Embedding 的文本描述 (使用 rename 後的欄位名)
        if 'food_name' in df_usda_processed.columns:
            print("\n--- USDA 資料庫 創建 Embedding 文本 ---")
            def create_embedding_text_usda(row):
                texts = []
                texts.append(f"食品名稱: {row.get('food_name', '未知食品')}.")
                if pd.notna(row.get('calories_kcal')): texts.append(f"熱量: {row['calories_kcal']:.1f} 大卡.")
                if pd.notna(row.get('protein_g')): texts.append(f"蛋白質: {row['protein_g']:.1f} 克.")
                if pd.notna(row.get('carbohydrates_g')): texts.append(f"碳水化合物: {row['carbohydrates_g']:.1f} 克.")
                if pd.notna(row.get('fat_g')): texts.append(f"脂肪: {row['fat_g']:.1f} 克.")
                if pd.notna(row.get('dietary_fiber_g')): texts.append(f"膳食纖維: {row['dietary_fiber_g']:.1f} 克.")
                # ... 添加更多你認為重要的營養資訊 ...
                return " ".join(texts)

            df_usda_processed['embedding_text'] = df_usda_processed.apply(create_embedding_text_usda, axis=1)
            print("Embedding 文本範例 (USDA):")
            print(df_usda_processed[['food_name', 'embedding_text']].head())

            # 7. 儲存處理後的 USDA 資料
            processed_usda_multicsv_filename = "processed_usda_fdc_combined_data.csv"
            df_usda_processed.to_csv(os.path.join(processed_data_path, processed_usda_multicsv_filename), index=False, encoding='utf-8-sig')
            print(f"\nUSDA (來自多CSV合併) 處理完成的資料已儲存到: {os.path.join(processed_data_path, processed_usda_multicsv_filename)}")
        else:
            print("因缺少 'food_name' 欄位 (或其重命名後)，無法為 USDA 資料創建 embedding text 或儲存。")

    except FileNotFoundError as e:
        print(f"處理 USDA 資料時發生 FileNotFoundError: {e}. 請確保 USDA_FDC_extracted_data 資料夾以及其中的 food.csv, nutrient.csv, food_nutrient.csv 檔案存在。")
    except Exception as e:
        print(f"處理 USDA 資料時發生非預期錯誤: {e}")
else:
    print(f"找不到 USDA 資料夾: {usda_extracted_path}. 請創建此資料夾並將 USDA 的 CSV 檔案放入其中。")


# --- (可選) 合併處理後的衛福部和 USDA 數據集 ---
# ... (這部分邏輯與之前類似，但你需要確保兩個 DataFrame df_tw_selected 和 df_usda_processed 有相似且可合併的欄位結構) ...

print("\n--- 資料準備腳本執行完畢 ---")