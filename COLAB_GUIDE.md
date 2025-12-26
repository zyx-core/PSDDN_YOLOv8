# Google Colab Training Guide

## ✅ Yes! You can train PSDDN on Google Colab (FREE GPU)

I've created a complete Colab notebook: **`PSDDN_Training_Colab.ipynb`**

---

## 🚀 How to Use

### Step 1: Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload `PSDDN_Training_Colab.ipynb`

**Option B: From GitHub**
1. Push the notebook to your GitHub repo
2. Go to https://colab.research.google.com/
3. Click "GitHub" tab
4. Enter: `zyx-core/PSDDN_YOLOv8`
5. Select `PSDDN_Training_Colab.ipynb`

### Step 2: Enable GPU

1. In Colab, click "Runtime" → "Change runtime type"
2. Select "GPU" (T4 GPU - FREE!)
3. Click "Save"

### Step 3: Run All Cells

1. Click "Runtime" → "Run all"
2. Or run cells one-by-one with Shift+Enter

**That's it!** The notebook will:
- ✅ Install dependencies
- ✅ Clone your repo
- ✅ Download ShanghaiTech dataset
- ✅ Convert annotations
- ✅ Generate pseudo GT
- ✅ Create curriculum folds
- ✅ Train for 100 epochs (~2-3 hours)
- ✅ Run inference
- ✅ Generate results

---

## 📊 What the Notebook Does

### 11 Automated Steps:

1. **Setup Environment** - Install PyTorch, Ultralytics, etc.
2. **Clone Repository** - Get your PSDDN code
3. **Download Dataset** - ShanghaiTech Part B (400 train, 316 test)
4. **Convert Annotations** - MAT → JSON
5. **Generate Pseudo GT** - Point → Bounding boxes
6. **Create Curriculum** - Sort by difficulty
7. **Configure Data** - Create YAML config
8. **Train Model** - 100 epochs with curriculum learning
9. **Run Inference** - Test on 316 images
10. **View Results** - MAE, MSE, AP metrics
11. **Download Results** - Zip and download

---

## ⏱️ Timeline

| Step | Time |
|------|------|
| Setup (Steps 1-7) | ~10 minutes |
| Training (Step 8) | ~2-3 hours |
| Inference (Step 9) | ~5 minutes |
| **Total** | **~2.5-3.5 hours** |

---

## 💡 Advantages of Colab

✅ **Free GPU** (Tesla T4)
✅ **No local setup** required
✅ **Pre-installed** PyTorch, CUDA
✅ **12 hours** continuous runtime
✅ **Save to Google Drive** option
✅ **Share results** easily

---

## 📝 Important Notes

### Runtime Limits
- **Free tier**: 12 hours max per session
- **Solution**: Training takes ~2-3 hours, so you're safe!
- **Tip**: If disconnected, results are in `/content/` (lost on disconnect)

### Save Results
Run this cell to save to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r runs /content/drive/MyDrive/PSDDN_Results
```

### Download Results
The notebook includes a download cell:
```python
!zip -r results.zip runs results
from google.colab import files
files.download('results.zip')
```

---

## 🎯 Expected Results

For ShanghaiTech Part B:
- **MAE**: ~7-10 (Mean Absolute Error)
- **MSE**: ~12-15 (Mean Squared Error)
- **AP**: ~0.6-0.8 (Average Precision)

---

## 🔧 Troubleshooting

### Issue: "No GPU available"
**Solution**: Runtime → Change runtime type → GPU

### Issue: "Session disconnected"
**Solution**: 
1. Reconnect
2. Re-run from last checkpoint
3. Or save to Google Drive regularly

### Issue: "Out of memory"
**Solution**: Reduce batch size in Step 8:
```python
batch=8,  # instead of 16
```

### Issue: "Runtime timeout"
**Solution**: Training should complete in 2-3 hours. If not:
- Reduce epochs: `epochs=50`
- Use smaller image size: `imgsz=512`

---

## 🚀 Quick Start (3 Steps)

1. **Upload** `PSDDN_Training_Colab.ipynb` to Colab
2. **Enable GPU** (Runtime → Change runtime type)
3. **Run All** (Runtime → Run all)

**Wait 2-3 hours → Get results!** ✨

---

## 📦 What You Get

After training completes:
- `best.pt` - Best model weights
- `predictions.json` - Test set predictions
- `metrics.json` - Evaluation metrics (MAE, MSE, AP)
- `visualizations/` - Prediction images with boxes
- `report.html` - Full evaluation report

---

## 💾 Alternative: Use Your Own Dataset

To use a different dataset:

1. Upload your data to Colab:
```python
from google.colab import files
uploaded = files.upload()
```

2. Modify Step 3 to use your data
3. Adjust image sizes in Step 5

---

## ✅ Recommended Workflow

1. **First run**: Use the notebook as-is (ShanghaiTech Part B)
2. **Analyze**: Check results, identify issues
3. **Fine-tune**: Adjust hyperparameters
4. **Re-train**: Run again with new settings
5. **Deploy**: Download best model

---

## 🎓 Learning Tips

- **Watch the training**: Monitor loss curves in real-time
- **Check visualizations**: See predictions on test images
- **Compare metrics**: MAE, MSE, AP tell different stories
- **Experiment**: Try different batch sizes, learning rates

---

## 🌟 Next Steps After Training

1. **Analyze errors**: Which images have high MAE?
2. **Try Part A**: Denser crowds, harder challenge
3. **Fine-tune**: Adjust hyperparameters
4. **Deploy**: Use model for real applications
5. **Share**: Upload results to GitHub

---

**Ready to train?** Just upload the notebook and click "Run all"! 🚀
