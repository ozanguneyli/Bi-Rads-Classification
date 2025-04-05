# BI-RADS Classification

This project contains **Mammography Images Classification Model** for **BI-RADS categories** using deep learning with attention mechanisms.

## 📌 Features
- **DICOM Preprocessing**: Cropping with YOLOv5-based breast region detection, Mask, Clahe, Dicom Windowing
- **Advanced Architecture**: RegNet backbone + Multi-head Attention Mechanism
- **Multi-class Classification**: BI-RADS 0 to 5 support

## 🛠️ Installation
```bash
git clone https://github.com/ozanguneyli/Bi-Rads-Classification.git
cd Bi-Rads-Classification
pip install -r requirements.txt
```

## Project Structure
```
📂 bi_rads_classification/
├── Breast_Cropper.pt       # YOLOv5-based cropper weights
├── model.py                # Main model architecture
├── preprocessing_pipeline.ipynb  # Full preprocessing workflow
├── requirements.txt
└── README.md
```


## 📊 Results (Our Implementation)

- **Model Evaluation Metrics**:  Bar chart comparing key performance metrics (Best val, test, F1 score)

![Evaluation Metrics](images/evaluation_metrics_plot.png)

- **Confusion Matrix**: Visual representation of the model's true positives, false positives, true negatives, and false negatives

![Confussion Matrix](images/confussion_matrix.png)

## License

[MIT License](LICENSE)
