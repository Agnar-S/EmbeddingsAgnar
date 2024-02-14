import os
import json
from sentence_transformers import SentenceTransformer

# Step 1: Install Sentence Transformers library if not already installed
# Command: pip install sentence-transformers
# Step 2: Load the Norwegian language model directly
model = SentenceTransformer("NbAiLab/nb-bert-base")

# Step 3: Read and Encode Text Files
text_files_directory = "data"  # Directory where your text files are stored

# Ensure the embeddings directory exists
embeddings_directory = "embeddings"
if not os.path.exists(embeddings_directory):
    os.makedirs(embeddings_directory)

for filename in os.listdir(text_files_directory):
    if filename.endswith(".txt"):  # Ensure you're reading only text files
        file_path = os.path.join(text_files_directory, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
            embedding = model.encode(text)

            # Store the embedding in a separate JSON file
            embedding_file_path = os.path.join(embeddings_directory, f"{filename}.json")
            with open(embedding_file_path, "w") as embedding_file:
                json.dump({"filename": filename, "embedding": embedding.tolist()}, embedding_file)

            print(f"Embedding for {filename} saved to {embedding_file_path}")
