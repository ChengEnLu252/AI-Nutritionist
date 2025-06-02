import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI # 假設你使用 OpenAI 相容的 API (如Ollama)
from langchain.chains import SQLDatabaseChain

# --- 全域變數 ---
# 【重要】強烈建議使用環境變數來設定這些敏感資訊
# 例如：os.environ.get("SUPABASE_DB_URI")
# 為了範例，我們先在這裡定義，但請務必替換成你自己的，並且不要直接提交密碼到版本控制
SUPABASE_DB_URI = "postgresql://postgres:Andy25206@db.ucbufcoeqqzalafkiznq.supabase.co:5432/postgres" # << 【請替換】
# 從 rag_pipeline 匯入 LLM 設定，或在此處重新初始化
# 假設 rag_pipeline.py 中有 OLLAMA_BASE_URL 和 LLM_MODEL_NAME
try:
    from .rag_pipeline import OLLAMA_BASE_URL, LLM_MODEL_NAME
    print("langchain_utils.py: 成功從 .rag_pipeline 匯入 LLM 設定。")
except ImportError:
    print("langchain_utils.py: 警告 - 未能從 .rag_pipeline 匯入 LLM 設定。將使用預設值。")
    OLLAMA_BASE_URL = "http://localhost:11434/v1" # Fallback
    LLM_MODEL_NAME = "gemma3:4b" # Fallback


langchain_db = None
sql_db_chain = None
langchain_llm = None

def init_langchain_components():
    """初始化 LangChain 需要的 SQLDatabase 和 SQLDatabaseChain"""
    global langchain_db, sql_db_chain, langchain_llm

    if "YOUR-PASSWORD" in SUPABASE_DB_URI or "YOUR-PROJECT-REF" in SUPABASE_DB_URI:
        print("錯誤 (langchain_utils.py): Supabase 資料庫連接 URI 未正確設定。請更新 SUPABASE_DB_URI。")
        return False

    try:
        print("langchain_utils.py: 正在初始化 LangChain SQLDatabase...")
        # 可以指定包含哪些表，以限制 LLM 的操作範圍和提高相關性
        # include_tables = ['users', 'weight_logs', 'diet_logs', 'user_goals']
        # db = SQLDatabase.from_uri(SUPABASE_DB_URI, include_tables=include_tables)
        langchain_db = SQLDatabase.from_uri(SUPABASE_DB_URI) # 不指定則包含所有可見表
        print("langchain_utils.py: LangChain SQLDatabase 初始化成功。")

        print(f"langchain_utils.py: 正在初始化 LangChain LLM ({LLM_MODEL_NAME})...")
        langchain_llm = ChatOpenAI(
            openai_api_base=OLLAMA_BASE_URL,
            openai_api_key="ollama", # Ollama 通常不需要 API Key，但 LangChain 可能需要一個佔位符
            model_name=LLM_MODEL_NAME,
            temperature=0 # 為了讓 SQL 生成和答案更具確定性，設為 0
        )
        print("langchain_utils.py: LangChain LLM 初始化成功。")

        print("langchain_utils.py: 正在創建 SQLDatabaseChain...")
        # verbose=True 可以看到 LLM 生成的 SQL 查詢和中間步驟
        sql_db_chain = SQLDatabaseChain.from_llm(llm=langchain_llm, db=langchain_db, verbose=True, return_intermediate_steps=False)
        # return_intermediate_steps=True 可以讓你看到 SQL 查詢和結果，方便調試
        print("langchain_utils.py: SQLDatabaseChain 創建成功。")
        return True

    except Exception as e:
        print(f"langchain_utils.py: 初始化 LangChain 元件時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_user_database_with_langchain(user_question: str, user_id: str = None):
    """
    使用 LangChain SQLDatabaseChain 根據自然語言問題查詢使用者資料庫。
    """
    global sql_db_chain
    if sql_db_chain is None:
        print("query_user_database_with_langchain: SQLDatabaseChain 未初始化。")
        if not init_langchain_components(): # 嘗試初始化
             return "抱歉，資料庫查詢功能目前無法使用（初始化失敗）。"

    # 為了讓 LLM 知道查詢是針對特定使用者的，可以在問題中加入 user_id
    # 或者，更好的方式是透過 Prompt Engineering 或 LangChain 的其他機制來限定查詢範圍
    # 這裡我們先假設 LLM 需要從問題中推斷或被告知 user_id
    
    # 基礎的 Prompt 增強，告知 LLM 當前使用者上下文 (如果 user_id 有提供)
    # 你可能需要更複雜的 Prompt 工程來確保 SQL 查詢正確地使用了 user_id 進行過濾
    contextual_question = user_question
    if user_id:
        # 嘗試讓 LLM 在生成 SQL 時考慮 user_id
        # 這一步的有效性取決於 LLM 的能力和 SQLDatabaseChain 的內部 Prompt
        contextual_question = f"針對 user_id 為 '{user_id}' 的使用者，回答以下問題：{user_question}"
        # 或者，更明確地指示資料庫中 user_id 欄位的用途
        # contextual_question = f"資料庫中的 'users' 表有 'user_id' 和 'username'。'weight_logs', 'diet_logs', 'user_goals' 表都有一個 'user_id' 外鍵關聯到 'users' 表的 'user_id'。現在，針對 user_id 為 '{user_id}' 的使用者，回答問題：{user_question}"
    
    print(f"LangChain 準備查詢的問題: {contextual_question}")

    try:
        # .run() 方法通常返回最終的自然語言答案
        # 如果 SQLDatabaseChain 設定了 return_intermediate_steps=True，則 .invoke() 或 .call() 返回的會是字典，包含中間步驟
        # result = sql_db_chain.run(contextual_question) # .run() 只返回最終答案
        
        # 使用 .invoke() 來獲取更完整的輸出（如果 return_intermediate_steps=True）
        # result_dict = sql_db_chain.invoke({"query": contextual_question})
        # answer = result_dict.get("result", "無法從查詢結果中提取答案。")
        # intermediate_steps = result_dict.get("intermediate_steps", [])
        # print(f"中間步驟 (SQL查詢等): {intermediate_steps}")

        # 為了簡單起見，先用 .run()
        answer = sql_db_chain.run(contextual_question)
        print(f"LangChain SQLDatabaseChain 回答: {answer}")
        return answer
    except Exception as e:
        print(f"使用 LangChain 查詢資料庫時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return f"抱歉，查詢您的歷史數據時發生錯誤: {e}"

if __name__ == '__main__':
    # 這個區塊用於直接執行此檔案時進行測試
    print("直接執行 langchain_utils.py 進行測試...")
    if init_langchain_components():
        # 假設你的資料庫中已經有一個 user_id='2dd009b1-96b9-4665-b291-af9c0f052eef' (從你之前的日誌看)
        # 並且該用戶有一些體重記錄
        test_user_id = "YOUR_EXISTING_USER_ID_FOR_TESTING" # << 【請替換成你資料庫中已存在的 user_id】
        if "YOUR_EXISTING_USER_ID_FOR_TESTING" == test_user_id:
            print("請在 langchain_utils.py 的測試區塊中替換 YOUR_EXISTING_USER_ID_FOR_TESTING 為一個實際的 user_id。")
        else:
            print(f"\n測試查詢1 (針對 user_id: {test_user_id}): 我最近一次記錄的體重是多少？")
            response1 = query_user_database_with_langchain("我最近一次記錄的體重是多少？", user_id=test_user_id)
            print(f"回應1: {response1}")

            print(f"\n測試查詢2 (針對 user_id: {test_user_id}): 我上週記錄了哪些飲食？")
            response2 = query_user_database_with_langchain("我上週記錄了哪些飲食？", user_id=test_user_id)
            print(f"回應2: {response2}")
            
            print(f"\n測試查詢3 (不指定 user_id，看 LLM 如何處理): username 為 '承恩' 的使用者有多少條體重記錄？")
            response3 = query_user_database_with_langchain("username 為 '承恩' 的使用者有多少條體重記錄？")
            print(f"回應3: {response3}")
    else:
        print("LangChain 元件初始化失敗，無法執行測試。")