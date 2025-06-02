# app.py

import torch
torch.set_num_threads(1) # 或者一個較小的值，如 2 或 4
import os
os.environ["OMP_NUM_THREADS"] = "1" # 通常與 torch.set_num_threads(1) 效果類似或互補
import gradio as gr
import pandas as pd # 如果 app.py 本身不用 pandas，可以考慮移除 (但保留也無妨)
import os
from src.util.rag_pipeline import load_rag_components, get_rag_response_from_pipeline, df_map as rag_df_map_global_var
import src.util.rag_pipeline as rag_module # 使用模組名來存取其全域變數

# --- 1. 從你的 RAG pipeline 和 multimodal 模組匯入函式 ---
try:
    # 從 rag_pipeline 匯入 df_map，並在載入後使用
    from src.util.rag_pipeline import load_rag_components, get_rag_response_from_pipeline, df_map as rag_df_map_global
    print("成功從 src.util.rag_pipeline 匯入函式和 df_map。")
except ImportError as e:
    print(f"從 src.util.rag_pipeline 匯入函式失敗: {e}")
    print("請確保 src/util/rag_pipeline.py 檔案存在，並且 src 和 src/util 資料夾下有 __init__.py 檔案。")
    print(f"從 src.util.rag_pipeline 匯入函式、df_map 或 LLM 設定失敗: {e}")
    rag_df_map = None # Fallback
    def load_rag_components(): print("錯誤：load_rag_components 未能載入。")
    def get_rag_response_from_pipeline(query_text, chat_history_list_ignored, k=3):
        return "錯誤：RAG pipeline 未能載入。", [], []
    
try:
    from src.util.multimodal_module import load_sam_model, handle_image_analysis, load_vit_model
    print("成功從 src.util.multimodal_module 匯入函式。")
except ImportError as e:
    print(f"從 src.util.multimodal_module 匯入函式失敗: {e}")
    def load_sam_model():
        print("錯誤：load_sam_model (來自 multimodal_module) 未能載入。")
        return False
    def handle_image_analysis(uploaded_image_pil):
        if uploaded_image_pil:
            return uploaded_image_pil, "錯誤：handle_image_analysis (來自 multimodal_module) 未能載入，返回原始圖片。"
        return None, "錯誤：handle_image_analysis (來自 multimodal_module) 未能載入。"

try:
    from src.util.database_utils import (
        init_supabase_client, get_or_create_user, 
        add_weight_log, get_weight_logs,
        add_diet_log, get_diet_logs,
        add_user_goal, get_user_goals
    )
    print("成功從 src.util.database_utils 匯入函式。")
except ImportError as e:
    # ... (database_utils fallback) ...
    pass

try:
    from src.util.langchain_utils import init_langchain_components, query_user_database_with_langchain
    print("成功從 src.util.langchain_utils 匯入函式。")
except ImportError as e:
    print(f"從 src.util.langchain_utils 匯入函式失敗: {e}")
    def init_langchain_components(): print("錯誤: init_langchain_components 未能載入。"); return False
    def query_user_database_with_langchain(query, user_id=None): return "錯誤: LangChain 查詢功能未能載入。"


# --- 2. RAG 營養師的 Gradio 聊天介面處理函式 ---
def handle_chat_interaction(user_message, chat_history_list_of_dicts):
    current_chat_session = chat_history_list_of_dicts + [{"role": "user", "content": user_message}]
    answer_text, sources, _ = get_rag_response_from_pipeline(user_message, [], k=3)
    current_chat_session.append({"role": "assistant", "content": answer_text})
    
    formatted_sources_json = []
    final_markdown_string_for_sources = "### 詳細來源預覽\n\n"
    if sources:
        for src in sources:
            source_item_json = {
                "來源編號": src.get("id", "N/A"), "食品名稱": src.get("food_name", "N/A"),
                "FAISS_ID": src.get("faiss_id", "N/A"), "內容預覽": src.get("text_content", "")[:150] + "..."}
            formatted_sources_json.append(source_item_json)
            final_markdown_string_for_sources += f"--- ({src.get('id', 'N/A')}) ---\n"
            final_markdown_string_for_sources += f"**食品名稱:** {src.get('food_name', 'N/A')}\n\n"
            final_markdown_string_for_sources += f"**FAISS ID:** {src.get('faiss_id', 'N/A')}\n\n"
            final_markdown_string_for_sources += f"**原始文本 (預覽):**\n```text\n{src.get('text_content', '')}\n```\n\n"
    else:
        formatted_sources_json = [{"message": "本次查詢無特定參考資料。"}]
        final_markdown_string_for_sources = "本次查詢無特定參考資料。"
    return current_chat_session, formatted_sources_json, final_markdown_string_for_sources, ""

# --- 3. 個人化長期減脂教練的處理函式 (佔位) ---
def handle_coach_interaction(username_input, weight_input_str, diet_log_input):
    if not username_input:
        return "請輸入用戶名。", "無歷史體重記錄。", "無飲食日誌記錄。"

    user_data, error = get_or_create_user(username_input)
    if error:
        return f"用戶操作失敗: {error}", "", ""
    
    user_id = user_data['user_id']
    
    # 添加體重記錄 (如果提供了)
    if weight_input_str:
        try:
            weight_kg = float(weight_input_str)
            _, error_weight = add_weight_log(user_id, weight_kg)
            if error_weight:
                print(f"添加體重記錄警告: {error_weight}")
        except ValueError:
            print(f"無效的體重輸入: {weight_input_str}")
            
    # TODO: 添加飲食日誌的邏輯 (add_diet_log)
    if diet_log_input:
        print(f"用戶 {username_input} (ID: {user_id}) 記錄飲食: {diet_log_input}")
        # _, error_diet = add_diet_log(user_id, diet_log_input) # 假設有這個函式

    # 獲取並顯示最近的體重記錄
    recent_weights, error_get_weights = get_weight_logs(user_id, limit=5)
    weight_history_display = "最近體重記錄:\n"
    if error_get_weights:
        weight_history_display += f"  獲取失敗: {error_get_weights}"
    elif recent_weights:
        for log in recent_weights:
            weight_history_display += f"  - {log['logged_at'][:10]}: {log['weight_kg']} kg\n"
    else:
        weight_history_display += "  暫無記錄。"
        
    # TODO: 獲取並顯示飲食日誌和目標，並生成 LLM 建議

    coach_advice = f"你好 {username_input}！你的資料已更新。(示意)"
    diet_history_display = f"飲食日誌記錄 (示意):\n - {diet_log_input if diet_log_input else '暫無本次記錄'}"

    return coach_advice, weight_history_display, diet_history_display

def handle_coach_submit_all_data(username_input, weight_input_str, meal_type_input, diet_description_input, calorie_input_str,
                                 goal_description_input, goal_type_input, goal_value_input, goal_target_date_input):
    # ... (你之前獲取或創建 user_data, user_id 的邏輯，以及添加體重、飲食、目標到資料庫的邏輯保持不變) ...
    if not username_input: # 基本的用戶名檢查
        return "請輸入用戶名。", "", "", "", "用戶名為空"
    
    user_data, error = get_or_create_user(username_input)
    if error:
        return f"用戶操作失敗: {error}", "", "", "", f"用戶錯誤 - {error}"
    user_id = user_data['user_id']
    submission_status_parts = []

    # 添加體重記錄
    if weight_input_str:
        try:
            weight_kg = float(weight_input_str)
            _, error_weight = add_weight_log(user_id, weight_kg)
            if error_weight: submission_status_parts.append(f"體重記錄警告: {error_weight}")
            else: submission_status_parts.append("體重已記錄")
        except ValueError:
            submission_status_parts.append(f"無效體重: {weight_input_str}")
            
    # 添加飲食日誌
    if diet_description_input:
        calories = None
        if calorie_input_str:
            try: calories = float(calorie_input_str)
            except ValueError: submission_status_parts.append(f"飲食熱量無效: {calorie_input_str}")
        _, error_diet = add_diet_log(user_id, diet_description_input, meal_type_input, calories)
        if error_diet: submission_status_parts.append(f"飲食記錄警告: {error_diet}")
        else: submission_status_parts.append("飲食已記錄")

    # 添加目標
    if goal_description_input:
        _, error_goal = add_user_goal(user_id, goal_description_input, goal_type_input, goal_value_input, goal_target_date_input)
        if error_goal: submission_status_parts.append(f"目標設定警告: {error_goal}")
        else: submission_status_parts.append("新目標已設定")

    # --- 生成 LLM 教練建議 ---
    print(f"\n準備為用戶 {username_input} (ID: {user_id}) 生成教練建議...")
    recent_weights_for_llm, _ = get_weight_logs(user_id, days_limit=7)
    recent_diets_for_llm, _ = get_diet_logs(user_id, days_limit=7)
    active_goals_for_llm, _ = get_user_goals(user_id, status='active')

    user_data_summary = format_data_for_llm(username_input, recent_weights_for_llm, recent_diets_for_llm, active_goals_for_llm)

    # 從 rag_module 中獲取已初始化的 LLM client 和 model name
    current_llm_client = rag_module.llm_client_global
    current_llm_model_name = rag_module.LLM_MODEL_NAME

    coach_advice_from_llm = generate_llm_coach_advice(user_data_summary, current_llm_client, current_llm_model_name) # << 傳遞參數
    # ------------------------

    # --- 更新顯示的歷史記錄 (這部分邏輯保持不變或略作調整) ---
    weight_history_display = "最近體重記錄:\n"
    if recent_weights_for_llm: # 使用已獲取的數據
        for log in recent_weights_for_llm: weight_history_display += f"  - {log.get('logged_at','未知')[:16]}: {log.get('weight_kg','N/A')} kg\n"
    else: weight_history_display += "  暫無最近7天記錄。"
        
    diet_history_display = "最近飲食日誌:\n"
    if recent_diets_for_llm: # 使用已獲取的數據
        for log in recent_diets_for_llm:
            cal_info = f" (約 {log.get('estimated_calories', 'N/A')} 大卡)" if log.get('estimated_calories') is not None else ""
            meal_info = f"{log.get('meal_type', '')}: " if log.get('meal_type') else ""
            diet_history_display += f"  - {log.get('logged_at','未知')[:16]}: {meal_info}{log.get('description','')}{cal_info}\n"
    else: diet_history_display += "  暫無最近7天記錄。"

    goal_display = "進行中目標:\n"
    if active_goals_for_llm: # 使用已獲取的數據
        for goal in active_goals_for_llm:
            target_info = f" (目標值: {goal.get('target_value', 'N/A')})" if goal.get('target_value') else ""
            goal_display += f"  - [{goal.get('goal_type','通用')}] {goal.get('description','')}{target_info}\n"
    else: goal_display += "  暫無進行中目標。"
    
    final_submission_status = "日誌/目標提交狀態：" + ("; ".join(submission_status_parts) if submission_status_parts else "無新操作執行")

    # 將 LLM 生成的建議顯示在 coach_advice_output
    return coach_advice_from_llm, weight_history_display, diet_history_display, goal_display, final_submission_status

def format_data_for_llm(username, weight_logs, diet_logs, active_goals):
    """將從資料庫獲取的數據格式化為 LLM prompt 的一部分"""
    prompt_parts = [f"這是使用者「{username}」的近期健康記錄：\n"]

    prompt_parts.append("# 過去7天飲食摘要：")
    if diet_logs:
        for log in diet_logs:
            cal_info = f" (約 {log.get('estimated_calories', '未記錄')} 大卡)" if log.get('estimated_calories') is not None else ""
            meal_info = f"{log.get('meal_type', '')}: " if log.get('meal_type') else ""
            log_date = log.get('logged_at', '未知日期')[:10] # 只取日期部分
            prompt_parts.append(f"- {log_date} {meal_info}{log.get('description', '無描述')}{cal_info}")
    else:
        prompt_parts.append("- 過去7天飲食記錄較少或未提供。")
    prompt_parts.append("\n")

    prompt_parts.append("# 過去7天體重變化：")
    if weight_logs:
        for log in weight_logs:
            log_date = log.get('logged_at', '未知日期')[:10]
            prompt_parts.append(f"- {log_date}: {log.get('weight_kg', '未記錄')} kg")
    else:
        prompt_parts.append("- 過去7天體重記錄較少或未提供。")
    prompt_parts.append("\n")

    prompt_parts.append("# 目前設定的目標：")
    if active_goals:
        for goal in active_goals:
            target_info = f" (目標值: {goal.get('target_value', '未設定')})" if goal.get('target_value') else ""
            type_info = f"[{goal.get('goal_type', '通用')}] " if goal.get('goal_type') else ""
            prompt_parts.append(f"- {type_info}{goal.get('description', '無描述')}{target_info}")
    else:
        prompt_parts.append("- 目前未設定明確目標。")
    
    return "\n".join(prompt_parts)

def generate_llm_coach_advice(user_data_summary_text, client, model_to_use):
    """呼叫 LLM 生成教練建議"""
    if not client or not model_to_use: # << 修改判斷條件
        print("LLM 客戶端或模型名稱未正確傳遞，無法生成建議。")
        return "抱歉，AI 教練暫時無法提供建議（模型配置錯誤）。"

    system_prompt = "你是一位專業且富有同理心的減重與健康教練。"
    user_task_prompt = f"""{user_data_summary_text}

請根據以上資訊，完成以下任務：
1.  對使用者過去7天的飲食和體重變化（如果有的話）做一個簡短的回顧與總結（約100-150字）。
2.  根據他的記錄和現有目標（如果有的話），為他設定接下來一週的3個具體的、可衡量的、可達成的、相關的、有時限的 (SMART) 健康目標。目標應具體可行，並具鼓勵性。
請以專業且友善的語氣提供回饋。
"""
    try:
        print(f"向 LLM ({model_to_use}) 發送請求以生成教練建議...")
        chat_completion = client.chat.completions.create( # << 使用傳入的 client
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_task_prompt}
            ],
            model=model_to_use, # << 使用傳入的 model_to_use
        )
        advice = chat_completion.choices[0].message.content
        print("LLM 教練建議生成成功。")
        return advice
    except Exception as e:
        print(f"呼叫 LLM 生成教練建議時發生錯誤: {e}")
        # ... (traceback)
        return f"抱歉，生成 AI 教練建議時發生錯誤: {e}"

# --- 新增：處理 LangChain 歷史數據查詢的函式 ---
def handle_langchain_history_query(username_input, history_query_input):
    if not username_input:
        return "請先在上方輸入您的用戶名。"
    if not history_query_input:
        return "請輸入您想查詢的歷史數據問題。"

    user_data, error = get_or_create_user(username_input) # 複用之前的函式獲取 user_id
    if error or not user_data:
        return f"無法識別用戶「{username_input}」或獲取用戶ID失敗。"
    
    user_id = user_data['user_id']
    
    # 呼叫 LangChain 的查詢函式
    answer = query_user_database_with_langchain(history_query_input, user_id)
    return answer

# --- 4. Gradio UI with Blocks and Tabs ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange"), title="專業減重輔助系統") as demo:
    gr.Markdown("# 🥗 專業减重AI營養師 🧑‍⚕️") # 應用程式總標題

    df_map_state = gr.State(None) # 初始化一個 State 來儲存 df_map

    with gr.Tabs():
        # --- 分頁一：RAG 營養師 ---
        with gr.TabItem("RAG營養師"):
            gr.Markdown("### 🤖 AI 營養師")
            gr.Markdown("輸入您關於食品營養的問題，系統將根據資料庫提供回答與參考來源。")
            with gr.Row():
                with gr.Column(scale=2):
                    rag_chatbot_display = gr.Chatbot(
                        label="聊天視窗", type='messages', height=600,
                        avatar_images=(None, "https://img.icons8.com/fluency/96/robot.png")
                    )
                    rag_user_input_textbox = gr.Textbox(
                        label="輸入您的問題：", placeholder="例如：香蕉的熱量有多少？", lines=1
                    )
                    rag_submit_button = gr.Button("💬 發送訊息 (營養師)", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 📖 參考資料")
                    rag_sources_json_output = gr.JSON(label="JSON 格式來源")
                    gr.Markdown("---")
                    gr.Markdown("#### 詳細來源預覽 (Markdown)")
                    rag_sources_md_output_area = gr.Markdown(label="來源內容摘要")
            
            rag_chat_state = gr.State([]) 
            df_map_state = gr.State(None)

            rag_submit_button.click(
                fn=handle_chat_interaction, 
                inputs=[rag_user_input_textbox, rag_chat_state], 
                outputs=[rag_chat_state, rag_sources_json_output, rag_sources_md_output_area, rag_user_input_textbox]
            ).then(
                lambda history: history, inputs=[rag_chat_state], outputs=[rag_chatbot_display]
            )
            rag_user_input_textbox.submit(
                fn=handle_chat_interaction, 
                inputs=[rag_user_input_textbox, rag_chat_state], 
                outputs=[rag_chat_state, rag_sources_json_output, rag_sources_md_output_area, rag_user_input_textbox]
            ).then(
                lambda history: history, inputs=[rag_chat_state], outputs=[rag_chatbot_display]
            )

        # --- 分頁二：拍照估算熱量 ---
        with gr.TabItem("拍照估算熱量"):
            gr.Markdown("### 📸 拍照估算熱量與菜單建議")
            gr.Markdown("上傳您的餐點照片，系統將嘗試分割食材並為後續熱量估算做準備。")
            with gr.Row():
                with gr.Column(scale=1):
                    mm_input_image = gr.Image(type="pil", label="上傳餐點照片", sources=["upload", "webcam", "clipboard"], height=400)
                    mm_process_button = gr.Button("🔍 開始分析圖片", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("#### 分析結果")
                    mm_output_annotated_image = gr.Image(label="影像分割結果", height=400)
                    mm_status_textbox = gr.Textbox(label="處理狀態", interactive=False)
            
            # 新增一個 Gallery 元件來展示裁剪出來的食材圖片 (可選，作為初步展示)
            gr.Markdown("---")
            gr.Markdown("#### 提取的食材區域 (示意)")
            mm_cropped_gallery = gr.Gallery(label="提取的食材圖像", height=200, columns=5, object_fit="contain")
            
            mm_process_button.click(
            fn=handle_image_analysis,
            inputs=[mm_input_image], # << 將 df_map_state 作為輸入
            outputs=[mm_output_annotated_image, mm_status_textbox, mm_cropped_gallery] # 假設 handle_image_analysis 現在回傳3個值
            )

        # --- 分頁三：個人減脂教練 ---
        with gr.TabItem("個人減脂教練"):
            gr.Markdown("### 💪 個人化長期減脂教練")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 身份與記錄")
                    coach_username_input = gr.Textbox(label="您的用戶名 (例如：王小明)")
                    coach_weight_input = gr.Textbox(label="今日體重 (公斤，例如：65.5)")
                    coach_meal_type_input = gr.Dropdown(label="餐點類型", choices=["早餐", "午餐", "晚餐", "點心", "其他"], value="其他")
                    coach_diet_description_input = gr.Textbox(label="記錄您的餐點描述 (例如：燕麥牛奶，一顆蘋果)", lines=2)
                    coach_calorie_input = gr.Textbox(label="該餐點估計熱量 (可選，例如：350)")
                    
                    gr.Markdown("#### 設定新目標")
                    coach_goal_description_input = gr.Textbox(label="目標描述 (例如：本週運動三次)", lines=2)
                    coach_goal_type_input = gr.Dropdown(label="目標類型", choices=["體重目標", "運動目標", "飲食習慣目標", "其他"], value="其他")
                    coach_goal_value_input = gr.Textbox(label="目標值 (例如：60kg, 3次/週, 每日喝水2L)")
                    coach_goal_target_date_input = gr.Textbox(label="目標日期 (YYYY-MM-DD，可選)") # 或者使用 gr. डेटpicker 如果 Gradio 更新了
                    
                    coach_submit_button = gr.Button("📝 提交記錄 / 設定目標 / 更新資訊", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 教練反饋與日誌")
                    coach_advice_output = gr.Textbox(label="教練建議/狀態", lines=3, interactive=False)
                    coach_submission_status_output = gr.Textbox(label="日誌/目標提交狀態", lines=2, interactive=False)
                    coach_weight_history_output = gr.Textbox(label="體重歷史", lines=7, interactive=False)
                    coach_diet_history_output = gr.Textbox(label="飲食日誌摘要", lines=7, interactive=False)
                    coach_goals_output = gr.Textbox(label="進行中目標", lines=7, interactive=False)


            # --- 新增：LangChain 歷史數據查詢區塊 ---
            gr.Markdown("---")
            gr.Markdown("#### 🗣️ 與 AI 教練對話 (查詢您的歷史數據)")
            coach_history_query_input = gr.Textbox(label="向 AI 教練提問您的歷史記錄 (例如：我上週的平均體重是多少？)：", placeholder="輸入問題...")
            coach_history_query_button = gr.Button("🔍 查詢歷史")
            coach_history_query_output = gr.Markdown(label="AI 教練的回覆 (基於歷史數據)") # 使用 Markdown 以便更好地顯示格式

            # 原有的提交按鈕事件綁定
            coach_submit_button.click(
                fn=handle_coach_submit_all_data, # 你之前的函式，現在它會生成 LLM 建議
                inputs=[coach_username_input, coach_weight_input, coach_meal_type_input, coach_diet_description_input, coach_calorie_input,
                        coach_goal_description_input, coach_goal_type_input, coach_goal_value_input, coach_goal_target_date_input],
                outputs=[coach_advice_output, coach_weight_history_output, coach_diet_history_output, coach_goals_output, coach_submission_status_output]
            )

            # 新增的 LangChain 查詢按鈕事件綁定
            coach_history_query_button.click(
                fn=handle_langchain_history_query,
                inputs=[coach_username_input, coach_history_query_input], # 需要用戶名來確定 user_id
                outputs=[coach_history_query_output]
            )


# --- 5. 主程式入口 ---
if __name__ == "__main__":
    print("準備啟動 Gradio 應用程式...")
    
    # 1. 初始化 Supabase Client (來自 database_utils)
    if not init_supabase_client():
        print("警告：Supabase 客戶端未能初始化！")

    # 2. 載入 RAG 元件 (這也會初始化 RAG 使用的 LLM client)
    print("將呼叫 load_rag_components() 來載入 RAG 核心組件...")
    load_rag_components() 
    
    # 3. 初始化 LangChain 元件 (包括 SQLDatabase 和 SQLDatabaseChain)
    print("將呼叫 init_langchain_components() 來設定 LangChain...")
    if not init_langchain_components():
        print("警告：LangChain 元件未能初始化！歷史數據查詢功能可能受限。")
    
    # 4. 載入 SAM 和 ViT 模型 (如果還沒在其他地方的初始化函式中包含)
    from src.util.multimodal_module import load_sam_model, load_vit_model # 假設你匯入了
    print("將呼叫 load_sam_model() 和 load_vit_model()...")
    load_sam_model() 
    load_vit_model()

    print("啟動 Gradio 服務...")
    demo.launch(debug=True)