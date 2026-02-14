# HG Camera Counter

This repository contains the HG Camera Counter application (runtime + controller).

Quick setup

1. Install Python dependencies:
```powershell
python -m pip install -r requirements.txt
```

2. (Recommended) Install Git LFS and track model files:
```powershell
# Install Git LFS: https://git-lfs.github.com/
git lfs install
git lfs track "models/*.pt"
```

3. Do NOT commit secrets. Copy `data/config/config.template.yaml` -> `data/config/config.yaml` and fill keys.

GitHub push (example):
```powershell
cd "C:\Users\Huagr\Downloads\project_count (1)\project_count"
git init
git add .gitattributes .gitignore
git add -A
git commit -m "Initial import"
# Create remote on GitHub via web, then add remote (HTTPS example):
git remote add origin https://github.com/YOUR_USERNAME/HGCameraCounter.git
git branch -M main
git push -u origin main
```

If you want, I can create the remote and push for you if you provide the remote URL and confirm you want me to run git commands from this environment.