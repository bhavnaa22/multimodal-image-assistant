# 🖼️ Multimodal Image Assistant

A dark-themed web application for **image captioning** and **visual question answering (VQA)** using BLIP vision-language models and Gradio.

Users upload an image, generate a natural language caption, and ask free-form questions about the image—all through a simple, interactive interface.

---

## ✨ Features

- **Image Captioning**  
  Automatically generates descriptive captions using `Salesforce/blip-image-captioning-base`.

- **Visual Question Answering**  
  Answers natural-language questions about images (e.g., "How many people?" or "What color is the car?") using `Salesforce/blip-vqa-base`.

- **Modern Dark UI**  
  Professional two-column Gradio interface optimized for demos and usability.

- **Easy Deployment**  
  Runs locally or on Hugging Face Spaces with just `python app.py`.

---

## 🧠 Tech Stack

| Component | Tool |
|-----------|------|
| **Models** | Salesforce BLIP (caption + VQA) |
| **Deep Learning** | PyTorch |
| **Model Hub** | Hugging Face Transformers |
| **Web Framework** | Gradio |
| **Deployment** | Hugging Face Spaces (or local) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

### Locally

```bash
python app.py
```

The app will start at `http://127.0.0.1:7860` by default.

### On Hugging Face Spaces

Push `app.py`, `requirements.txt`, and this `README.md` to a Hugging Face Space. The Space will auto-build and deploy.

---

## 📖 How to Use

1. **Upload an image**  
   Click the upload area or drag-and-drop any image.

2. **Generate a caption**  
   Click **"✨ Generate Caption"** to get a natural language description.

3. **Ask questions**  
   Type any question about the image in the "Your Question" field and click **"🔍 Answer Question"**.

### Example Questions

- "What is happening in this image?"
- "How many people are there?"
- "What color is the main object?"
- "Is this indoors or outdoors?"

---

## 📁 Project Structure

```
.
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── system_paper.pdf    # Technical system description (optional)
```

### `app.py` Contents

- Model loading (BLIP caption + VQA at startup)
- Inference functions: `generate_caption()` and `answer_question()`
- Gradio UI layout with dark theme
- Event callbacks for button clicks

### `requirements.txt`

```
torch
torchvision
transformers
Pillow
gradio
```

---

## 🎓 About This Project

This project demonstrates:

- Modern **multimodal models** combining vision and language understanding.
- A complete **end-to-end ML pipeline** from model loading to web deployment.
- **Lightweight inference** using pre-trained models without custom training.
- Professional UI/UX in a data science application.

---

## 📝 System Architecture

```
User Input (Image + Question)
              ↓
        Gradio Interface
              ↓
    ┌────────┴────────┐
    ↓                 ↓
Caption Function  VQA Function
    ↓                 ↓
BLIP Caption      BLIP VQA
   Model           Model
    ↓                 ↓
  Text Output    Text Answer
    ↓                 ↓
    └────────┬────────┘
             ↓
      Display in UI
```

---

## ⚡ Performance Notes

- **First run**: Models download (~2–3 GB) on first inference. This takes 1–2 minutes.
- **Subsequent runs**: Much faster (a few seconds per inference).
- **Hardware**: Works on CPU (slower) or GPU (faster). Hugging Face free tier provides CPU.

---

## 🔧 Customization

### Change the theme

In `app.py`, replace `gr.themes.Monochrome()` with another theme:

```python
theme=gr.themes.Soft()  # or gr.themes.Default(), etc.
```

### Adjust model inference parameters

Modify the `num_beams` or `max_length` in the inference functions:

```python
out = cap_model.generate(
    **inputs,
    max_length=60,      # Longer captions
    num_beams=5,        # Better quality (slower)
)
```

### Use larger BLIP models

Replace the model checkpoints:

```python
# For larger models:
BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"  # Instead of -base
)
```

---



## 📧 Support

For questions or issues, please open a GitHub Issue or email me: bhavnaa2277@gmail.com
---

