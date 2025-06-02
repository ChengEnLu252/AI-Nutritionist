from supabase import create_client, Client
import os
from datetime import datetime, timedelta

# 從 Supabase 專案設定中獲取這些值
# 建議使用環境變數來儲存這些敏感資訊，而不是直接寫在程式碼中
# 但為了範例簡單，我們先直接定義 (之後你可以改成讀取環境變數)
SUPABASE_URL = "https://ucbufcoeqqzalafkiznq.supabase.co"  # << 【重要】替換成你的 Supabase Project URL
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjYnVmY29lcXF6YWxhZmtpem5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg3ODkyOTYsImV4cCI6MjA2NDM2NTI5Nn0.oRVDWKdGAIU6nJIdqbwh-fhQ4tqTm4PrsY956QsEhXk" # << 【重要】替換成你的 Service Role Key (非常重要，妥善保管)

supabase: Client = None

def init_supabase_client():
    """初始化 Supabase 客戶端"""
    global supabase
    if SUPABASE_URL == "YOUR_SUPABASE_PROJECT_URL" or SUPABASE_SERVICE_KEY == "YOUR_SUPABASE_SERVICE_ROLE_KEY":
        print("錯誤：Supabase URL 或 Service Key 未設定。請在 database_utils.py 中更新它們。")
        return False
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase 客戶端初始化成功。")
        return True
    except Exception as e:
        print(f"Supabase 客戶端初始化失敗: {e}")
        return False

# --- User 相關操作 ---
def get_or_create_user(username: str):
    """根據用戶名獲取用戶，如果不存在則創建"""
    if supabase is None: return None, "Supabase client not initialized."
    
    try:
        # 檢查用戶是否存在
        response = supabase.table('users').select("user_id, username").eq('username', username).execute()
        if response.data:
            print(f"找到用戶: {username}, ID: {response.data[0]['user_id']}")
            return response.data[0], None # 返回用戶數據和 None (表示無錯誤)
        else:
            # 創建用戶
            print(f"用戶 {username} 不存在，正在創建...")
            response = supabase.table('users').insert({"username": username}).execute()
            if response.data:
                print(f"用戶 {username} 創建成功: {response.data[0]}")
                return response.data[0], None
            else:
                # Supabase v2 insert 後可能不直接返回 data，而是 count
                if response.count is not None and response.count > 0:
                     # 需要再次查詢以獲取新用戶的 user_id 和其他資訊
                     new_user_response = supabase.table('users').select("user_id, username").eq('username', username).execute()
                     if new_user_response.data:
                         print(f"用戶 {username} 創建成功 (再次查詢): {new_user_response.data[0]}")
                         return new_user_response.data[0], None
                error_message = f"創建用戶 {username} 失敗: {response.error.message if response.error else '未知錯誤'}"
                print(error_message)
                return None, error_message
    except Exception as e:
        error_message = f"操作用戶 {username} 時發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message

# --- Weight Log 相關操作 ---
def add_weight_log(user_id: str, weight_kg: float):
    """為指定用戶添加體重記錄"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        data, error = supabase.table('weight_logs').insert({
            "user_id": user_id,
            "weight_kg": weight_kg
        }).execute()
        
        # Supabase Python client v1+ insert 操作的返回格式可能改變
        # 通常 data 是一個列表，包含插入的記錄；error 是 None 或一個錯誤物件
        # execute() 返回的物件包含 data 和 error 屬性
        
        # 處理 Supabase v2 insert 後 data 可能為 None 但 count > 0 的情況
        actual_data = getattr(data, 'data', data) # 嘗試獲取 data 屬性中的列表
        if actual_data:
            print(f"成功為用戶 {user_id} 添加體重記錄: {actual_data[0]}")
            return actual_data[0], None
        elif hasattr(data, 'count') and data.count is not None and data.count > 0:
            print(f"成功為用戶 {user_id} 添加體重記錄 (count: {data.count})。")
            # 如果需要返回剛插入的數據，可能需要再次查詢或依賴觸發器
            return {"user_id": user_id, "weight_kg": weight_kg, "status": "inserted"}, None
        elif getattr(data, 'error', error):
             err_obj = getattr(data, 'error', error)
             error_message = f"添加體重記錄失敗: {err_obj.message if hasattr(err_obj, 'message') else err_obj}"
             print(error_message)
             return None, error_message
        else:
            print(f"添加體重記錄操作完成，但未返回預期數據。Response: {data}")
            return None, "添加體重記錄操作完成，但未返回預期數據。"

    except Exception as e:
        error_message = f"添加體重記錄時發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message

def get_weight_logs(user_id: str, days_limit: int = 7): # 改為 days_limit
    """獲取指定用戶最近指定天數內的體重記錄"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        start_date = (datetime.now() - timedelta(days=days_limit)).strftime('%Y-%m-%d %H:%M:%S')
        response = supabase.table('weight_logs').select("*").eq('user_id', user_id)\
                           .gte('logged_at', start_date)\
                           .order('logged_at', desc=True).execute()
        # ... (後續錯誤處理和返回邏輯不變) ...
        if hasattr(response, 'data') and response.data:
            return response.data, None
        elif hasattr(response, 'error') and response.error:
            return None, f"獲取體重記錄失敗: {response.error.message}"
        return [], None
    except Exception as e:
        return None, f"獲取體重記錄時發生錯誤: {str(e)}"

# --- Diet Log 相關操作 ---
def add_diet_log(user_id: str, description: str, meal_type: str = None, estimated_calories: float = None):
    """為指定用戶添加飲食日誌"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        log_entry = {
            "user_id": user_id,
            "description": description,
        }
        if meal_type:
            log_entry["meal_type"] = meal_type
        if estimated_calories is not None: # 允許 0 卡路里
            log_entry["estimated_calories"] = estimated_calories
            
        response = supabase.table('diet_logs').insert(log_entry).execute()
        
        # 檢查 response.data (Supabase v1) 或 response.count (Supabase v2)
        if hasattr(response, 'data') and response.data:
            print(f"成功為用戶 {user_id} 添加飲食日誌: {response.data[0]}")
            return response.data[0], None
        elif hasattr(response, 'count') and response.count is not None and response.count > 0:
            # Supabase v2 insert 後可能不直接返回 data，但可以確認插入成功
            # 如果需要返回剛插入的數據，可能需要再次查詢或依賴資料庫觸發器返回
            print(f"成功為用戶 {user_id} 添加飲食日誌 (count: {response.count})。")
            return {"user_id": user_id, "description": description, "status": "inserted"}, None
        elif hasattr(response, 'error') and response.error:
             error_message = f"添加飲食日誌失敗: {response.error.message}"
             print(error_message)
             return None, error_message
        else:
            print(f"添加飲食日誌操作完成，但未返回預期數據。Response: {response}")
            return None, "添加飲食日誌操作完成，但未返回預期數據。"
            
    except Exception as e:
        error_message = f"添加飲食日誌時發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message

def get_diet_logs(user_id: str, days_limit: int = 7): # 改為 days_limit
    """獲取指定用戶最近指定天數內的飲食日誌"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        start_date = (datetime.now() - timedelta(days=days_limit)).strftime('%Y-%m-%d %H:%M:%S')
        response = supabase.table('diet_logs').select("*").eq('user_id', user_id)\
                           .gte('logged_at', start_date)\
                           .order('logged_at', desc=True).execute()
        # ... (後續錯誤處理和返回邏輯不變) ...
        if hasattr(response, 'data') and response.data:
            return response.data, None
        elif hasattr(response, 'error') and response.error:
            return None, f"獲取飲食日誌失敗: {response.error.message}"
        return [], None
    except Exception as e:
        return None, f"獲取飲食日誌時發生錯誤: {str(e)}"

# --- User Goal 相關操作 ---
def add_user_goal(user_id: str, description: str, goal_type: str = None, target_value: str = None, target_date_str: str = None):
    """為指定用戶添加目標"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        goal_entry = {
            "user_id": user_id,
            "description": description,
        }
        if goal_type:
            goal_entry["goal_type"] = goal_type
        if target_value:
            goal_entry["target_value"] = target_value
        if target_date_str: # 假設傳入的是 'YYYY-MM-DD' 格式的字串
            goal_entry["target_date"] = target_date_str 
            
        response = supabase.table('user_goals').insert(goal_entry).execute()

        if hasattr(response, 'data') and response.data:
            print(f"成功為用戶 {user_id} 添加目標: {response.data[0]}")
            return response.data[0], None
        elif hasattr(response, 'count') and response.count is not None and response.count > 0:
            print(f"成功為用戶 {user_id} 添加目標 (count: {response.count})。")
            return {"user_id": user_id, "description": description, "status": "inserted"}, None
        elif hasattr(response, 'error') and response.error:
             error_message = f"添加目標失敗: {response.error.message}"
             print(error_message)
             return None, error_message
        else:
            print(f"添加目標操作完成，但未返回預期數據。Response: {response}")
            return None, "添加目標操作完成，但未返回預期數據。"

    except Exception as e:
        error_message = f"添加目標時發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message

def get_user_goals(user_id: str, status: str = 'active'):
    """獲取指定用戶特定狀態的目標"""
    if supabase is None: return None, "Supabase client not initialized."
    try:
        query = supabase.table('user_goals').select("*").eq('user_id', user_id)
        if status:
            query = query.eq('status', status)
        response = query.order('created_at', desc=True).execute()
        
        if hasattr(response, 'data') and response.data:
            return response.data, None
        elif hasattr(response, 'error') and response.error:
            error_message = f"獲取用戶目標失敗: {response.error.message}"
            print(error_message)
            return None, error_message
        else:
            return [], None
    except Exception as e:
        error_message = f"獲取用戶目標時發生錯誤: {str(e)}"
        print(error_message)
        return None, error_message