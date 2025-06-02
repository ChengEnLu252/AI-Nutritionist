# synonyms.py

# --- 中文食物及營養相關同義詞辭典 ---
# 用於 RAG 查詢重寫或擴展，以處理同義詞問題。
# 你可以持續擴充這個詞典。
# 鍵：標準化/常用詞彙
# 值：同義詞列表

chinese_food_synonym_dictionary = {
    # 蔬菜類
    "番茄": ["西紅柿", "狼桃", "柑仔蜜"],
    "馬鈴薯": ["土豆", "洋芋", "薯仔", "洋山芋"], # 注意："土豆" 在某些地區也指花生
    "玉米": ["玉蜀黍", "苞谷", "苞米", "番麥"],
    "青椒": ["燈籠椒", "柿子椒", "甜椒", "大椒"],
    "黃瓜": ["小黃瓜", "青瓜", "胡瓜"],
    "苦瓜": ["涼瓜"],
    "茄子": ["矮瓜"],
    "花椰菜": ["花菜", "菜花", "白花椰"],
    "西蘭花": ["青花菜", "綠花椰", "綠花菜"],
    "高麗菜": ["包心菜", "甘藍", "捲心菜", "包菜", "結球甘藍", "洋白菜"],
    "大白菜": ["紹菜", "黃芽白"], # Napa Cabbage
    "菠菜": ["菠薐菜", "赤根菜"],
    "白蘿蔔": ["蘿蔔", "菜頭"],
    "胡蘿蔔": ["紅蘿蔔", "甘筍"],
    "洋蔥": ["蔥頭"],
    "生薑": ["薑"],
    "大蒜": ["蒜", "蒜頭"],
    "香菇": ["冬菇", "椎茸"],
    "金針菇": ["金菇"],
    "木耳": ["黑木耳"],
    "空心菜": ["蕹菜", "通菜", "藤菜"],
    "韭菜": ["起陽草"],
    "芹菜": ["旱芹"],
    "萵苣": ["生菜", "A菜"],

    # 水果類
    "草莓": ["士多啤梨"],
    "鳳梨": ["菠蘿", "黃梨", "旺來"],
    "西瓜": [],
    "香蕉": [],
    "蘋果": [],
    "橘子": ["桔子"],
    "柳橙": ["橙子", "柳丁"],
    "葡萄柚": ["西柚"],
    "奇異果": ["獼猴桃"],
    "木瓜": [],
    "芒果": ["杧果"],
    "酪梨": ["牛油果", "鱷梨"],
    "番石榴": ["芭樂"],
    "荔枝": [],
    "龍眼": ["桂圓"], # 桂圓通常指龍眼乾
    "檸檬": [],
    "百香果": ["雞蛋果", "熱情果"],
    "釋迦": ["番荔枝", "佛頭果"],
    "火龍果": ["紅龍果", "仙蜜果"],

    # 肉類/蛋白質
    "豬肉": [],
    "雞肉": [],
    "牛肉": [],
    "羊肉": [],
    "魚肉": ["魚"],
    "蝦": ["蝦子", "海米"],
    "雞蛋": ["蛋", "雞卵"],
    "豆腐": [],
    "豆漿": [],

    # 穀物/堅果/其他
    "米飯": ["飯", "白飯"],
    "麵條": ["麵"],
    "花生": ["土豆", "落花生", "長生果"], # 注意："土豆" 在北方某些方言指花生
    "芝麻": ["胡麻"],
    "杏仁": ["杏仁果"],
    "核桃": ["胡桃"],
    "腰果": ["雞腰果"],

    # 營養/飲食術語
    "熱量": ["卡路里", "大卡"],
    "蛋白質": ["蛋白"],
    "脂肪": ["油脂", "油份"],
    "碳水化合物": ["醣類", "碳水"],
    "維生素": ["維他命"],
    "膳食纖維": ["纖維"],
    "礦物質": [],
    "膽固醇": [],
    "鈉": [],
    "鈣": [],
    "鐵": [],
    "減肥": ["瘦身", "減重", "塑形"],
    "食譜": ["菜譜"],
    "菜單": ["餐單"],
    "零食": ["點心", "小吃"],
    "飲料": ["飲品"],
}

def expand_query_with_synonyms(query, synonym_dict):
    """
    根據提供的同義詞字典擴展或標準化查詢。
    這是一個基礎版本，可以進一步優化。

    Args:
        query (str): 使用者的原始查詢。
        synonym_dict (dict): 同義詞字典，
                             鍵為標準詞，值為同義詞列表。

    Returns:
        list: 包含原始查詢和可能的同義詞變體的查詢列表。
    """
    expanded_terms = set()  # 使用集合避免重複的查詢變體
    
    # 為了更精確的匹配，實際應用中建議使用中文分詞工具 (如 jieba)
    # 以下為簡化版，直接進行子字串匹配
    words_in_query = query 

    # 策略1: 如果查詢中包含標準詞，則嘗試用其同義詞替換，生成新的查詢
    for standard_term, synonyms_list in synonym_dict.items():
        if standard_term in words_in_query:
            for synonym in synonyms_list:
                expanded_terms.add(words_in_query.replace(standard_term, synonym))
    
    # 策略2: 如果查詢中包含某個同義詞，則嘗試用其標準詞替換，生成新的查詢
    # (這個策略可以幫助標準化查詢，你也可以選擇只用這個策略，或者與策略1結合)
    for standard_term, synonyms_list in synonym_dict.items():
        for synonym in synonyms_list:
            if synonym in words_in_query:
                expanded_terms.add(words_in_query.replace(synonym, standard_term))

    # 確保原始查詢始終被包含
    expanded_terms.add(query)
    
    return list(expanded_terms)

# --- 你可以在這裡加入一些測試程式碼 (可選) ---
if __name__ == '__main__':
    # 這個區塊的程式碼只有在直接執行 `python synonyms.py` 時才會運行
    # 用於測試你的字典和函式是否按預期工作
    
    test_query1 = "我想知道番茄的卡路里"
    expanded1 = expand_query_with_synonyms(test_query1, chinese_food_synonym_dictionary)
    print(f"原始查詢 1: {test_query1}")
    print(f"擴展後查詢 1: {expanded1}")
    # 預期輸出可能包含: "我想知道西紅柿的卡路里" 等

    test_query2 = "士多啤梨有哪些營養？"
    expanded2 = expand_query_with_synonyms(test_query2, chinese_food_synonym_dictionary)
    print(f"\n原始查詢 2: {test_query2}")
    print(f"擴展後查詢 2: {expanded2}")
    # 預期輸出可能包含: "草莓有哪些營養？"

    test_query3 = "薯仔的熱量"
    expanded3 = expand_query_with_synonyms(test_query3, chinese_food_synonym_dictionary)
    print(f"\n原始查詢 3: {test_query3}")
    print(f"擴展後查詢 3: {expanded3}")
    # 預期輸出可能包含: "馬鈴薯的熱量", "土豆的熱量" 等