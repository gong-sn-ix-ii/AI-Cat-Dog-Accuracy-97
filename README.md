<div align="center">

# 🐱 vs 🐶 — AI Cat & Dog Classifier

### โมเดล CNN จำแนกภาพหมา–แมว ด้วยความแม่นยำ **97%**

*Binary Image Classification ด้วย Dataset 25,000 รูป — เริ่มจาก Clean Data จนถึง Fine-tune โมเดล*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)](https://github.com/gong-sn-ix-ii/AI-Cat-Dog-Accuracy-97)
[![Dataset](https://img.shields.io/badge/Dataset-25K%20Images-blue?style=flat-square)](https://github.com/gong-sn-ix-ii/AI-Cat-Dog-Accuracy-97)
[![Model](https://img.shields.io/badge/Model-CNN%20200x200x128-A855F7?style=flat-square)](https://github.com/gong-sn-ix-ii/AI-Cat-Dog-Accuracy-97)
[![Developer](https://img.shields.io/badge/developer-Kitsada%20Khamnuan-06B6D4?style=flat-square)](https://gong-ix-ii-dev.com)

</div>

---

## 🎯 ที่มาของโปรเจกต์

ตอนเริ่มสนใจสาย Computer Vision ผมอยากลองทำโปรเจกต์ Classic ที่นักเรียน Deep Learning ทุกคนเคยทำ — **Cats vs Dogs Classification** เพื่อใช้เป็น sandbox ในการเรียนรู้ทั้งกระบวนการ ตั้งแต่ Data Cleaning, Model Design, Training, Tuning ไปจนถึง Evaluation

จุดที่อยากท้าทายตัวเองคือ — *เริ่มจาก Vanilla CNN ที่เขียนเองทุก Layer ไม่ใช่ Transfer Learning จาก Pre-trained Model* แล้วดูว่าด้วยขนาด Network ที่จำกัด เราจะดัน Accuracy ไปได้ถึงไหน

ผลที่ได้ — **97% บน Test Set** จาก dataset 25,000 รูป ที่ผ่านการ Clean และ Fine-tune หลายรอบ

---

## 📊 ผลลัพธ์โดยรวม

<div align="center">

| Metric | Value |
|---|---|
| 🎯 **Test Accuracy** | **97%** |
| 📦 **Dataset Size** | 25,000 รูป |
| 🧠 **Model** | `9712_Model_200x200x128` |
| 🖼️ **Input Size** | 200 × 200 px (RGB) |
| 🏗️ **Architecture** | CNN (Convolutional Neural Network) |
| 🔢 **Classes** | 2 (Binary: Cat / Dog) |

</div>

> 💡 ชื่อโมเดล `9712_Model_200x200x128` มาจาก — Validation Accuracy 97.12% + Input 200×200 + 128 Filters ใน Conv Layer หลัก

---

## 🏗️ สถาปัตยกรรมโมเดล

โมเดลออกแบบเป็น **CNN ที่ Stack ขึ้นมาเอง** ไม่ใช้ Pre-trained — เพื่อทดสอบความเข้าใจในการออกแบบ Architecture จริง ๆ

```
Input (200, 200, 3) RGB Image
        │
        ▼
┌────────────────────────────┐
│ Conv2D + ReLU + MaxPool    │ × หลาย Block
│ (32 → 64 → 128 filters)    │ ค่อย ๆ เพิ่มความลึก
└────────────┬───────────────┘
             │
        Flatten
             │
             ▼
┌────────────────────────────┐
│ Dense + Dropout            │ ป้องกัน Overfit
│ Dense (Sigmoid)            │ Output: Cat or Dog
└────────────────────────────┘
        │
        ▼
Binary Classification (0 = Cat, 1 = Dog)
```

| Layer Type | บทบาท |
|---|---|
| **Conv2D + MaxPooling** | สกัด Spatial Features จากภาพ (Edge → Shape → Pattern) |
| **128 Filters** | ใน Layer สุดท้ายของ Conv Block ก่อน Flatten |
| **Dropout** | ลด Overfitting — สำคัญมากกับ Dataset ขนาดนี้ |
| **Sigmoid Output** | Binary Classification → ให้ค่า Probability เดียว |

---

## 🛠️ เทคโนโลยีที่ใช้

| Stack | บทบาท |
|---|---|
| 🐍 **Python** | ภาษาหลักทั้ง Data Pipeline และ Model |
| 🔶 **TensorFlow / Keras** | สร้างและเทรน CNN |
| 📓 **Jupyter Notebook** | Workspace สำหรับ EDA, Training, และ Visualization |
| 📊 **NumPy / Pandas** | จัดการ Tensor และ Metadata |
| 🎨 **Matplotlib / Seaborn** | Plot Training History และ Confusion Matrix |
| 🖼️ **OpenCV / PIL** | Image Preprocessing |

---

## 🔬 กระบวนการพัฒนา

### 1️⃣ Data Cleaning

ก่อนเทรนต้องลุยทำความสะอาดข้อมูลก่อน — Dataset 25K รูปไม่ได้สะอาดเสมอไป:

- กรองรูปเสีย (Corrupted Files) ที่ Load ไม่ได้
- ตรวจ Label ผิด (รูปแมวที่ติด Label หมา หรือกลับกัน) ด้วยการ Sample ตรวจ
- Resize ทุกรูปเป็น `200×200 px` ให้เข้ากับ Input ของโมเดล
- Normalize ค่า Pixel เป็น `[0, 1]` (หาร 255)

### 2️⃣ Data Augmentation

เพื่อลด Overfitting และสร้าง Variation ให้โมเดล:

- Horizontal Flip
- Rotation (±20°)
- Zoom (±10%)
- Brightness Shift

### 3️⃣ Training

- **Train / Validation / Test Split** — แยก data ก่อนทำอะไรก็ตาม (ไม่ peek ดู test set)
- เทรนจน Validation Loss เริ่มไม่ลด → ใช้ **Early Stopping** ตัดให้
- ใช้ **ModelCheckpoint** เก็บ Best Weights ไว้ใช้ Inference

### 4️⃣ Fine-tuning

หลังเทรนรอบแรก ผมปรับ Hyperparameter หลายตัวเพื่อดัน Accuracy:

- ลด **Learning Rate** ตอน Plateau (ReduceLROnPlateau)
- ปรับ **Dropout Rate** หาจุดที่ Train/Val ไม่แยกออกจากกันมาก
- ลอง **Batch Size** ต่าง ๆ เพื่อหา Sweet Spot ระหว่าง Speed vs Stability
- ผลสุดท้าย → **Test Accuracy 97.12%**

---

## 📂 โครงสร้างโปรเจกต์

```
AI-Cat-Dog-Accuracy-97/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA + Visualize ข้อมูล
│   ├── 02_data_cleaning.ipynb         # ล้างและเตรียม Dataset
│   ├── 03_model_training.ipynb        # เทรน CNN
│   └── 04_evaluation.ipynb            # Confusion Matrix + ผลลัพธ์
├── models/
│   └── 9712_Model_200x200x128.h5      # โมเดลตัวสุดท้าย (97% accuracy)
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── docs/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── sample_predictions.png
├── requirements.txt
└── README.md
```

---

## 🚀 วิธีรันโปรเจกต์

### สิ่งที่ต้องเตรียม

- Python `>= 3.9`
- TensorFlow `>= 2.10`
- Jupyter Notebook / JupyterLab
- GPU (แนะนำ — เทรนเร็วกว่ามาก)

### ขั้นตอนการติดตั้ง

```bash
# 1. Clone repository
git clone https://github.com/gong-sn-ix-ii/AI-Cat-Dog-Accuracy-97.git
cd AI-Cat-Dog-Accuracy-97

# 2. สร้าง virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. เตรียม Dataset
# Download Dogs vs Cats จาก Kaggle:
# https://www.kaggle.com/c/dogs-vs-cats
# แตกไฟล์ไว้ที่ data/ ตามโครงสร้างด้านบน

# 5. เปิด Jupyter และรัน notebook ตามลำดับ
jupyter notebook
```

### Inference (ใช้โมเดลที่เทรนแล้ว)

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# โหลดโมเดล
model = load_model('models/9712_Model_200x200x128.h5')

# เตรียมรูป
img = image.load_img('your_image.jpg', target_size=(200, 200))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# ทำนาย
pred = model.predict(x)[0][0]
label = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
confidence = pred if pred > 0.5 else 1 - pred
print(f"{label} — Confidence: {confidence:.2%}")
```

---

## 💡 บทเรียนที่ได้

### 🧠 Data quality สำคัญกว่า Model complexity

ตอนแรกผมพยายามเพิ่ม Layer ให้ Network ลึกขึ้นเพื่อดัน Accuracy — แต่กลับ Overfit เร็วขึ้น สุดท้ายสิ่งที่ทำให้ Accuracy ขึ้นจริง ๆ คือ **Data Cleaning + Augmentation** ที่ดีกว่า ไม่ใช่ Model ที่ใหญ่กว่า

### 📉 Overfitting คือศัตรูตัวจริง

ระหว่างเทรนเห็น Training Accuracy พุ่ง 99% แต่ Validation อยู่แค่ 85% — เป็นสัญญาณคลาสสิกของ Overfit ผมแก้ด้วย Dropout + Augmentation + Early Stopping จนช่องว่าง Train vs Val แคบลงเหลือ 1-2%

### ⚙️ Reproducibility

ทุก Notebook มี `random_seed` set ไว้ + Save weights แต่ละ Checkpoint — ทำให้กลับมารัน Inference ได้ผลเดิมเสมอ เป็นนิสัยที่ใช้กับโปรเจกต์ ML ทุกตัวหลังจากนั้น

---

## 👨‍💻 ผู้พัฒนา

<table>
<tr>
<td>

### Kitsada Khamnuan (กฤษฎา คำนวน)

*AI Application Developer · Junior Software Engineer · Cybersecurity Enthusiast*

📍 ชลบุรี / กรุงเทพฯ, ประเทศไทย

🌐 **Portfolio:** [gong-ix-ii-dev.com](https://gong-ix-ii-dev.com)
💼 **GitHub:** [@gong-sn-ix-ii](https://github.com/gong-sn-ix-ii)
💬 **LinkedIn:** [Kitsada Khamnuan](https://www.linkedin.com/in/kitsada-khamnuan-2a6729407/)

</td>
</tr>
</table>

---

## 📄 License

โปรเจกต์นี้พัฒนาเพื่อการศึกษาและเป็นส่วนหนึ่งของ Portfolio
Dataset อ้างอิงจาก [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) (ลิขสิทธิ์ของ Kaggle/Microsoft Research)

---

<div align="center">

### 🧠 *"Models ไม่ได้ฉลาดเพราะใหญ่ — แต่ฉลาดเพราะข้อมูลสะอาด"*

**⭐ ถ้าคุณชอบโปรเจกต์นี้ ฝากกด Star เป็นกำลังใจให้ผู้พัฒนาด้วยนะครับ ⭐**

Made with 💜 by [Kitsada Khamnuan](https://gong-ix-ii-dev.com)

</div>
