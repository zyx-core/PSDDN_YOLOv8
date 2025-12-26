# Git Upload Guide for PSDDN Project

## Step 1: Configure Git (One-time setup)

Set your Git username and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or set it only for this project:

```bash
cd C:\Users\arsha\PSDDN_YOLOv8
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 2: Initialize Repository

The repository is already initialized. Now commit your changes:

```bash
cd C:\Users\arsha\PSDDN_YOLOv8

# Remove the ultralytics submodule issue
git rm -rf --cached ultralytics_repo

# Add only the custom PSDDN file from ultralytics
git add ultralytics_repo/ultralytics/utils/psddn_loss.py

# Add all other files
git add .

# Create initial commit
git commit -m "Initial commit: PSDDN on YOLOv8n implementation"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com
2. Click "New repository" (+ icon in top right)
3. Name it: `PSDDN-YOLOv8`
4. Description: "Point-Supervised Deep Detection Network on YOLOv8n for crowd counting"
5. Choose Public or Private
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

## Step 4: Connect and Push

GitHub will show you commands. Use these:

```bash
cd C:\Users\arsha\PSDDN_YOLOv8

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/PSDDN-YOLOv8.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 5: Verify Upload

Go to your GitHub repository URL:
`https://github.com/YOUR_USERNAME/PSDDN-YOLOv8`

You should see all files uploaded!

## What Gets Uploaded

✅ **Included:**
- All scripts (8 core + 3 test files)
- Documentation (README, guides)
- Custom PSDDN loss file
- Example annotations
- .gitignore

❌ **Excluded (by .gitignore):**
- Training outputs (runs/, results/)
- Model weights (*.pt, *.pth)
- Dataset images
- Python cache files
- Full ultralytics repo (only custom loss file included)

## Alternative: Using GitHub Desktop

If you prefer a GUI:

1. Download GitHub Desktop: https://desktop.github.com/
2. Install and sign in
3. File → Add Local Repository → Select `C:\Users\arsha\PSDDN_YOLOv8`
4. Commit changes
5. Publish repository to GitHub

## Updating Later

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

## Important Notes

- The `.gitignore` file prevents uploading large files (datasets, model weights)
- Only upload code and documentation
- Share datasets separately (e.g., via Google Drive)
- Model weights can be uploaded to releases if needed

## Need Help?

If you get errors, common solutions:

**"Permission denied"**: Set up SSH key or use personal access token
**"Repository not found"**: Check the remote URL with `git remote -v`
**"Merge conflict"**: Pull first with `git pull origin main`
