
from huggingface_hub import HfApi
import os
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_SPACE_REPO", "sriharimudakavi/engine-condition-predictor") # Use env var or default
api = HfApi()
api.upload_folder(folder_path=".", repo_id=REPO_ID, repo_type="space", token=HF_TOKEN)
print("✅ Uploaded to Hugging Face Space successfully.")
