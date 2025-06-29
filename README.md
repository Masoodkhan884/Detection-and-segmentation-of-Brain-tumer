
# 🧠 Brain Tumor Detection & Segmentation

An AI-powered web application for detecting and segmenting brain tumors from MRI scans using YOLOv11 (object detection) and SAM2 (segmentation). Built with Python and Streamlit.

---

## 🚀 Features

- 🎯 Accurate tumor **detection** using YOLOv11
- 🏷️ Automatic tumor **classification** (e.g., Glioma, Meningioma, etc.)
- 🧠 Tumor **segmentation** using SAM2
- 🖼️ Upload and process any MRI image
- ⚡ Fast and user-friendly **Streamlit** interface

---

## 📸 Sample Output

| Original MRI | Detected Tumor | Segmented Tumor |
|--------------|----------------|------------------|
| ![Original](images/original.jpg) | ![Detection](images/detection.jpg) | ![Segmented](images/segmentation.jpg) |

---

## 📂 Folder Structure

```
brain_tumor_detection_app/
│
├── app.py                 # Main Streamlit app
├── best.pt                # YOLOv11 trained model
├── sam2_b.pt              # SAM2 segmentation model
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── images/                # Sample images for README (optional)
```

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/brain_tumor_detection_app.git
cd brain_tumor_detection_app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## 📌 Requirements

- Python 3.9+
- `streamlit`
- `ultralytics`
- `opencv-python`
- `pillow`
- `numpy`

All dependencies are listed in [`requirements.txt`](./requirements.txt).

---

## 📊 Models Used

- **YOLOv11**: For object detection and classification
- **SAM2 (Segment Anything Model)**: For precise tumor region segmentation

> Models are trained on a labeled brain tumor MRI dataset (Roboflow/YAML-based).

---

## 🧪 Example Tumor Types Detected

- Glioma
- Meningioma
- Pituitary
- No Tumor (if no detection)

---

## 💡 Future Improvements

- Add **download** button for segmented output
- Deploy on **Streamlit Cloud**
- Add **patient report generation** (PDF)

---

## 🙋‍♂️ Author

**Masood Khan**  
[LinkedIn](https://www.linkedin.com/) • [GitHub](https://github.com/your-username)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
