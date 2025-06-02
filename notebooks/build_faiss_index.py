import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle # For saving the mapping list

# --- 1. Configuration: Set up your file paths ---
# 假設你的 Notebook/腳本是從專案根目錄 '/Users/lvchengen/生成式AI/' 執行的
processed_data_path = "data/processed"  # 直接指向專案根目錄下的 data/processed
faiss_index_output_path = "models/faiss_index" # 直接指向專案根目錄下的 models/faiss_index

# Names of your processed data files
taiwan_processed_file = "processed_taiwan_food_data.csv"
usda_processed_file = "processed_usda_fdc_combined_data.csv" # Or your processed USDA file name

# Output filenames
faiss_index_filename = "nutrition_faiss.index"
faiss_mapping_filename = "nutrition_faiss_mapping.pkl" # We'll save a list of original identifiers

# Ensure output directory exists
os.makedirs(faiss_index_output_path, exist_ok=True)

print(f"Processed data will be loaded from: {processed_data_path}")
print(f"FAISS index will be saved to: {faiss_index_output_path}")

# --- 2. Load Processed Data ---
df_list = []
df_taiwan = None
df_usda = None

# Load Taiwan data
try:
    path_tw = os.path.join(processed_data_path, taiwan_processed_file)
    if os.path.exists(path_tw):
        df_taiwan = pd.read_csv(path_tw)
        if 'embedding_text' in df_taiwan.columns:
            df_list.append(df_taiwan[['food_name', 'embedding_text']].copy()) # Keep relevant columns
            print(f"Successfully loaded and selected columns from: {taiwan_processed_file}, {len(df_taiwan)} entries.")
        else:
            print(f"Warning: '{taiwan_processed_file}' is missing 'embedding_text' column.")
    else:
        print(f"Warning: File not found - {path_tw}")
except Exception as e:
    print(f"Error loading {taiwan_processed_file}: {e}")

# Load USDA data
try:
    path_usda = os.path.join(processed_data_path, usda_processed_file)
    if os.path.exists(path_usda):
        df_usda = pd.read_csv(path_usda)
        if 'embedding_text' in df_usda.columns:
            df_list.append(df_usda[['food_name', 'embedding_text']].copy()) # Keep relevant columns
            print(f"Successfully loaded and selected columns from: {usda_processed_file}, {len(df_usda)} entries.")
        else:
            print(f"Warning: '{usda_processed_file}' is missing 'embedding_text' column.")
    else:
        print(f"Warning: File not found - {path_usda}")
except Exception as e:
    print(f"Error loading {usda_processed_file}: {e}")

# Combine data if both DataFrames are loaded and not empty
if not df_list:
    print("Error: No data loaded. Exiting.")
    exit()

combined_df = pd.concat(df_list, ignore_index=True)
combined_df.dropna(subset=['embedding_text'], inplace=True) # Drop rows where embedding_text might be null
combined_df['embedding_text'] = combined_df['embedding_text'].astype(str) # Ensure it's string

if combined_df.empty:
    print("Error: No text data available for embedding after loading and cleaning. Exiting.")
    exit()

all_texts_to_embed = combined_df['embedding_text'].tolist()
# This list will be our mapping from FAISS index ID back to original food info
# Each item could be a dictionary with more details if needed, e.g., combined_df.to_dict('records')
# For simplicity, we'll save the combined_df's 'food_name' or the whole DataFrame.
# Let's save a list of food names corresponding to the order of embeddings.
# We'll actually save the entire combined_df (or relevant parts) as the mapping.
# This 'combined_df' will now serve as our reference table for the FAISS indices.
print(f"\nTotal of {len(all_texts_to_embed)} unique text entries prepared for embedding.")
print("Example texts for embedding:")
for i, text in enumerate(all_texts_to_embed[:3]):
    print(f"  {i}: {text[:100]}...") # Print first 100 chars

# --- 3. Load Embedding Model ---
# Using the model specified in your project plan [cite: 15]
model_name = 'intfloat/multilingual-e5-small'
print(f"\nLoading sentence transformer model: {model_name}...")
try:
    embedding_model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure 'sentence-transformers' is installed and you have an internet connection.")
    exit()

# --- 4. Generate Embeddings ---
print("\nGenerating embeddings for all texts... This might take a while. ⏳")
# The model 'intfloat/multilingual-e5-small' expects a prefix for query vs passage.
# For passages (the documents you are indexing), the suggested prefix is 'passage: '.
# However, the original e5 models also work without prefixes, but using them can improve performance.
# Let's assume for now we are embedding them as passages.
# If your RAG query side also uses a 'query: ' prefix, this is consistent.
# For simplicity in this step, we can embed without prefix, or add 'passage: '
# texts_with_prefix = ["passage: " + text for text in all_texts_to_embed]
# embeddings = embedding_model.encode(texts_with_prefix, show_progress_bar=True)

# Simpler approach if not using query/passage prefixes, or if model handles it:
embeddings = embedding_model.encode(all_texts_to_embed, show_progress_bar=True)

print(f"Embeddings generated. Shape: {embeddings.shape}") # Should be (number_of_texts, dimension_of_embeddings)

if embeddings.shape[0] == 0:
    print("Error: No embeddings were generated. Exiting.")
    exit()

# --- 5. Build FAISS Index ---
dimension = embeddings.shape[1]  # Get the dimension of the embeddings
print(f"\nBuilding FAISS index with dimension {dimension}...")

# Using IndexFlatL2 - a basic but effective index for dense vectors with L2 distance search
index = faiss.IndexFlatL2(dimension)
print(f"FAISS index created. Is trained: {index.is_trained}")

# Add the generated embeddings to the index
# FAISS expects float32 type for embeddings
index.add(embeddings.astype('float32'))
print(f"Embeddings added to FAISS index. Total vectors in index: {index.ntotal}")

# --- 6. Save FAISS Index and Mapping Data ---
# Save the FAISS index
faiss_index_full_path = os.path.join(faiss_index_output_path, faiss_index_filename)
faiss.write_index(index, faiss_index_full_path)
print(f"FAISS index saved to: {faiss_index_full_path}")

# Save the combined_df which maps the FAISS index ID (its row number) to food_name and embedding_text
# This way, when you retrieve an ID from FAISS, you can look it up in this DataFrame
mapping_full_path = os.path.join(faiss_index_output_path, "indexed_food_data.csv") # Saving as CSV for readability
combined_df.to_csv(mapping_full_path, index_label='faiss_id', encoding='utf-8-sig')
print(f"Mapping data (food details corresponding to FAISS IDs) saved to: {mapping_full_path}")

# Example of saving with pickle (alternative, good for lists/dicts)
# mapping_list_full_path = os.path.join(faiss_index_output_path, faiss_mapping_filename)
# with open(mapping_list_full_path, 'wb') as f:
#     pickle.dump(combined_df[['food_name', 'embedding_text']].to_dict('records'), f) # Saving list of dicts
# print(f"Mapping list (pickle) saved to: {mapping_list_full_path}")


print("\n--- FAISS index building process complete! 🎉 ---")