# rag_pipeline.py (或 app.py 中的相關部分，如果 RAG 邏輯在那裡)

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI # 或者你使用的 LLM 的對應函式庫
import os
import traceback # 用於更詳細的錯誤輸出

# --- 1. 從 synonyms.py 匯入 ---
# 假設 synonyms.py 和 rag_pipeline.py 都在 src/util/ 目錄下
from .synonyms import chinese_food_synonym_dictionary, expand_query_with_synonyms
# 如果 synonyms.py 在 src/ 而 rag_pipeline.py 在 src/util/，則使用:
# from ..synonyms import chinese_food_synonym_dictionary, expand_query_with_synonyms


# --- 2. 全域 RAG 元件載入 (保持不變) ---
FAISS_INDEX_PATH = "models/faiss_index/nutrition_faiss.index" # 確保這是相對於專案根目錄的路徑
MAPPING_DATA_PATH = "models/faiss_index/indexed_food_data.csv" # 確保這是相對於專案根目錄的路徑
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-small'
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL_NAME = "gemma3:4b"

faiss_index = None
df_map = None
embedding_model_global = None
llm_client_global = None

def load_rag_components():
    global faiss_index, df_map, embedding_model_global, llm_client_global,LLM_MODEL_NAME
    # ... (載入 RAG 元件的程式碼，與你之前的版本相同) ...
    # 確保路徑 FAISS_INDEX_PATH 和 MAPPING_DATA_PATH 是正確的
    # 例如，如果 rag_pipeline.py 在 src/util/ 中，而執行是從專案根目錄，
    # 則 "models/..." 這樣的路徑是正確的。
    print("正在載入 RAG 核心組件...")
    try:
        # 構造相對於當前腳本的絕對路徑 (如果需要)
        # current_script_path = os.path.dirname(os.path.abspath(__file__))
        # project_root_path = os.path.abspath(os.path.join(current_script_path, "..", "..")) # 從 src/util 回到專案根目錄

        # faiss_abs_path = os.path.join(project_root_path, FAISS_INDEX_PATH)
        # mapping_abs_path = os.path.join(project_root_path, MAPPING_DATA_PATH)
        
        # 或者，如果你的執行腳本 (如 app.py) 總是在專案根目錄，
        # 並且 rag_pipeline.py 是作為模組被匯入，那麼直接使用相對於根目錄的路徑通常可行。
        # 我們這裡假設 FAISS_INDEX_PATH 和 MAPPING_DATA_PATH 是相對於執行 app.py 時的根目錄。
        OLLAMA_BASE_URL_FOR_RAG = "http://localhost:11434/v1"
        RAG_LLM_MODEL_NAME_CONFIG = "gemma3:4b"
        if os.path.exists(FAISS_INDEX_PATH):
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"FAISS 索引載入完畢，包含 {faiss_index.ntotal} 個向量。")
        else:
            print(f"錯誤: FAISS 索引檔案未找到於 {FAISS_INDEX_PATH}")
            faiss_index = None

        if os.path.exists(MAPPING_DATA_PATH):
            df_map = pd.read_csv(MAPPING_DATA_PATH)
            print(f"資料對照表載入完畢，共 {len(df_map)} 筆資料。")
        else:
            print(f"錯誤: 資料對照表檔案未找到於 {MAPPING_DATA_PATH}")
            df_map = pd.DataFrame()
        
        embedding_model_global = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Embedding 模型 ({EMBEDDING_MODEL_NAME}) 載入完畢。")
        
        llm_client_global = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        LLM_MODEL_NAME = RAG_LLM_MODEL_NAME_CONFIG # 將載入的模型名稱賦值給全域變數
        print(f"rag_pipeline.py: LLM 客戶端已設定，目標 URL: {OLLAMA_BASE_URL_FOR_RAG}，模型: {LLM_MODEL_NAME}")
        print(f"LLM 客戶端已設定，目標 URL: {OLLAMA_BASE_URL}，模型: {LLM_MODEL_NAME}")
        print("RAG 核心組件載入完畢。")

    except Exception as e:
        print(f"載入 RAG 組件時發生嚴重錯誤: {e}")
        print(f"rag_pipeline.py: 初始化 LLM 客戶端失敗: {e}")
        llm_client_global = None
        LLM_MODEL_NAME = None
        traceback.print_exc()

# --- 3. RAG 查詢處理核心函式 ---
def get_rag_response_from_pipeline(query_text, chat_history_list_ignored, k=3): # k 是最終返回給LLM的文檔數
    print(f"RAG Pipeline 接收到原始查詢: {query_text}")

    # 檢查 RAG 元件是否都已正確載入
    components_loaded = (
        faiss_index is not None and \
        df_map is not None and not df_map.empty and \
        embedding_model_global is not None and \
        llm_client_global is not None
    )

    if not components_loaded:
        print("警告: RAG 系統元件未完全載入或 df_map 為空，嘗試重新載入...")
        load_rag_components() # 假設 load_rag_components 會正確初始化這些全域變數
        
        # 再次檢查
        components_loaded_after_retry = (
            faiss_index is not None and \
            df_map is not None and not df_map.empty and \
            embedding_model_global is not None and \
            llm_client_global is not None
        )
        if not components_loaded_after_retry:
            return "抱歉，RAG 系統元件未能載入，無法處理您的請求。", [], []


    try:
        # === 3.1 查詢擴展/重寫 ===
        # 現在 expand_query_with_synonyms 和 chinese_food_synonym_dictionary 是從 synonyms.py 匯入的
        rewritten_or_expanded_queries = expand_query_with_synonyms(query_text, chinese_food_synonym_dictionary)
        print(f"擴展/重寫後的查詢列表: {rewritten_or_expanded_queries}")

        all_retrieved_indices_scores = {} 
        k_per_variant = 3 # 每個查詢變體檢索的文檔數 (可以調整)

        # === 3.2 對每個擴展/重寫後的查詢進行 Embedding 和 FAISS 搜索，並合併結果 ===
        for q_variant in rewritten_or_expanded_queries:
            print(f"  處理查詢變體: {q_variant}")
            query_embedding_variant = embedding_model_global.encode(q_variant)
            query_vector_variant = query_embedding_variant.reshape(1, -1).astype('float32')
            distances_variant, indices_variant = faiss_index.search(query_vector_variant, k_per_variant)
            
            for i in range(len(indices_variant[0])):
                idx = indices_variant[0][i]
                dist = distances_variant[0][i]
                if idx != -1: # FAISS 可能返回 -1
                    if idx not in all_retrieved_indices_scores or dist < all_retrieved_indices_scores[idx]:
                        all_retrieved_indices_scores[idx] = dist
        
        if not all_retrieved_indices_scores:
            return "根據您的查詢（及同義詞擴展），目前找不到相關的營養資訊。", [], []

        sorted_retrieved_items = sorted(all_retrieved_indices_scores.items(), key=lambda item: item[1])
        final_retrieved_indices = [item[0] for item in sorted_retrieved_items[:k]] # 取最終的 k 個
        print(f"合併篩選後，最終檢索到的索引 IDs: {final_retrieved_indices}")

        # === 3.3 檢索原始文本/資料 ===
        retrieved_documents_details = []
        context_for_llm_parts = []
        if final_retrieved_indices:
            for rank, idx_in_map in enumerate(final_retrieved_indices):
                if 0 <= idx_in_map < len(df_map):
                    doc_info = {
                        "id": f"來源 {rank+1}",
                        "food_name": df_map.iloc[idx_in_map].get('food_name', 'N/A'),
                        "text_content": df_map.iloc[idx_in_map].get('embedding_text', ''),
                        "faiss_id": int(idx_in_map)
                    }
                    retrieved_documents_details.append(doc_info)
                    context_for_llm_parts.append(f"{doc_info['id']} ({doc_info['food_name']}):\n{doc_info['text_content']}")
                else:
                     print(f"  - 警告: 檢索到的索引 ID {idx_in_map} 超出 df_map 的範圍。")
        
        if not retrieved_documents_details:
            return "根據您的查詢（及同義詞擴展），目前找不到相關的營養資訊。", [], []

        # === 3.4 建構 Prompt ===
        context_for_llm = "\n\n".join(context_for_llm_parts)
        system_prompt = '''
                        # 角色設定
                        你是一位具備專業營養與膳食規畫知識的「AI 營養師」。  
                        你的首要任務是：  
                        1. 依據可信來源（內建資料庫 + 實時檢索）如實回答使用者詢問的**食物健康資訊**。  
                        2. 當使用者提出「菜單／飲食計畫」需求時，先**引導使用者輸入自身身體健康狀況**（例：年齡、性別、體重、活動量、慢性疾病、過敏、目標體重等），再以手上資料為基礎**客製化一份對使用者最有利的菜單**。  
                        3. 所有飲食健康相關問題皆由你負責；若問題超出能力或缺乏資料，須**誠實告知限制，或引導使用者提出更精確問題**。  
                        4. **回覆語言必須與使用者輸入語言一致**；如使用者以中文提問，回答用中文；以英文提問，回答用英文。  

                        # 回答風格與格式
                        - 內容務求「正確、可追溯、易讀」。  
                        - 如引用外部或內部資料，於回答末尾以括號或分段方式列出簡短來源（例：`[衛福部食品營養成分資料庫]` 或 `(USDA FDC, 2024)`），避免冗長 URL。  
                        - 對健康建議使用**簡潔分點**；菜單請以**表格**或**日曆式排版**呈現（如 Breakfast / Lunch / Dinner）。  
                        - 若使用者未提供足夠健康資訊即要求菜單，先禮貌提示：「為了更精準建議，請補充以下資訊：⋯」。  

                        # 邊界與安全守則
                        - **勿提供醫療診斷或處方藥劑量**；必要時建議諮詢專科醫師或註冊營養師。  
                        - 若使用者請你評估療效不足、危險飲食（極端斷食、單一食物減肥…），需指出風險並提供科學依據。  
                        - 若使用者問題超出資料庫範圍，回答「目前無足夠可信資料可供參考」並嘗試提供下一步建議（如尋找專家）。  

                        # 結束提醒  
                        始終保持專業、客觀與同理心。若任何回應可能影響使用者健康，務必鼓勵其於實際採行前諮詢合格醫療或營養專業人員。
                        '''
        user_prompt_content = f"來源資料：\n{context_for_llm}\n\n問題：\n{query_text}\n\n回答："

        # === 3.5 呼叫 LLM ===
        print(f"呼叫 LLM ({LLM_MODEL_NAME})...")
        chat_completion = llm_client_global.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt_content}],
            model=LLM_MODEL_NAME,
        )
        llm_answer = chat_completion.choices[0].message.content
        
        return llm_answer, retrieved_documents_details, [] # 最後一個空列表是為了匹配 Gradio 的輸出格式

    except Exception as e:
        print(f"RAG pipeline 執行時發生錯誤: {e}")
        traceback.print_exc()
        return "處理您的請求時發生內部錯誤，請稍後再試。", [], []


# --- 主測試區塊 (如果直接執行此檔案) ---
# 這個區塊只有在 `python src/util/rag_pipeline.py` 這樣執行時才會運行
# 如果你是從 app.py 匯入並呼叫 get_rag_response_from_pipeline，則這個區塊不會在 app.py 運行時執行
if __name__ == '__main__':
    print("直接執行 rag_pipeline.py 進行測試...")
    load_rag_components() # 載入 RAG 元件

    if not all([faiss_index, df_map, embedding_model_global, llm_client_global]):
        print("RAG 元件未能完全載入，測試終止。")
    else:
        test_query_rag1 = "香蕉的熱量有多少？"
        answer1, sources1, _ = get_rag_response_from_pipeline(test_query_rag1, [])
        print(f"\n查詢 1: {test_query_rag1}")
        print(f"回答 1: {answer1}")
        if sources1:
            print("參考資料 1:")
            for src in sources1:
                print(f"- {src['id']} ({src['food_name']}) (FAISS ID: {src['faiss_id']})")

        test_query_rag2 = "士多啤梨有什麼營養" # 使用同義詞
        answer2, sources2, _ = get_rag_response_from_pipeline(test_query_rag2, [])
        print(f"\n查詢 2: {test_query_rag2}")
        print(f"回答 2: {answer2}")
        if sources2:
            print("參考資料 2:")
            for src in sources2:
                print(f"- {src['id']} ({src['food_name']}) (FAISS ID: {src['faiss_id']})")