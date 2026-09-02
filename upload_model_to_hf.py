import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()  # reads .env into environment variables
token = os.getenv("HF_TOKEN")

api = HfApi(token=token)

repo_id = "UItraviolet/yolo_multicart"  # change this

# Create the repo if it doesn't exist yet (safe to call even if it does)
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

# Upload the weights file
api.upload_file(
    path_or_fileobj="runs/segment/train-2/weights/best.pt",
    path_in_repo="runs/segment/train-2/weights/best.pt",
    repo_id=repo_id,
    repo_type="model",
    token=token,
)

print(f"Uploaded! View it at: https://huggingface.co/{repo_id}")