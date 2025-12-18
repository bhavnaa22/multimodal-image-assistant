
import gradio as gr
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,
)

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------- Models ----------------
print("Loading BLIP Caption model...")
cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
cap_model.eval()
print("✅ Caption model ready")

print("Loading BLIP VQA model...")
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)
vqa_model.eval()
print("✅ VQA model ready")


# ---------------- Inference ----------------
def generate_caption(image: Image.Image):
    if image is None:
        return "📤 Please upload an image first."
    try:
        inputs = cap_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = cap_model.generate(
                **inputs,
                max_length=50,
                num_beams=3,
            )
        caption = cap_processor.decode(out[0], skip_special_tokens=True)
        return f"✨ {caption}"
    except Exception as e:
        return f"Error while generating caption: {e}"


def answer_question(image: Image.Image, question: str):
    if image is None:
        return "📤 Please upload an image first."
    if not question.strip():
        return "❓ Please type a question about the image."
    try:
        inputs = vqa_processor(
            images=image,
            text=question,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = vqa_model.generate(
                **inputs,
                max_length=20,
                num_beams=3,
            )
        answer = vqa_processor.batch_decode(out, skip_special_tokens=True)[0]
        return f"💡 {answer}"
    except Exception as e:
        return f"Error while answering question: {e}"


# ---------------- Dark CSS ----------------
custom_css = """
:root {
  --primary: #38bdf8;
  --primary-dark: #0ea5e9;
  --accent: #22c55e;
  --bg: #020617;
  --bg-elevated: #020617;
  --card: #020617;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --border: #1f2937;
  --shadow-soft: 0 18px 40px rgba(0, 0, 0, 0.75);
}
body {
  background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #000000 100%);
  color: var(--text);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.gradio-container {
  max-width: 1100px;
  margin: 0 auto !important;
  padding: 24px 18px !important;
}
/* Main card */
main {
  border-radius: 18px !important;
  background: radial-gradient(circle at top left, #020617 0, #020617 40%, #000000 100%) !important;
  border: 1px solid rgba(148, 163, 184, 0.35) !important;
  box-shadow: var(--shadow-soft) !important;
}
/* Title area */
.gradio-container h1 {
  font-size: 2.3rem !important;
  font-weight: 800 !important;
  margin-bottom: 0.4rem !important;
  letter-spacing: 0.02em;
  background: linear-gradient(135deg, #38bdf8, #22c55e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.gradio-container h2 {
  font-size: 1.15rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
}
/* Generic text */
.gradio-container p {
  color: var(--muted);
  font-size: 0.98rem;
}
/* "Cards" */
.block, .gr-panel, .tabs {
  border-radius: 16px !important;
  background: rgba(15, 23, 42, 0.82) !important;
  border: 1px solid rgba(148, 163, 184, 0.35) !important;
}
/* Left column panel */
#left-panel {
  padding: 14px 16px !important;
}
/* Right column panel */
#right-panel {
  padding: 14px 16px !important;
}
/* Buttons */
button {
  border-radius: 9999px !important;
  font-weight: 600 !important;
  padding: 9px 20px !important;
  border: none !important;
  background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
  color: #e5e7eb !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease !important;
}
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(56, 189, 248, 0.35) !important;
  filter: brightness(1.1);
}
/* Inputs */
textarea, input, .gr-box {
  border-radius: 12px !important;
  border-color: var(--border) !important;
  background: #020617 !important;
  color: var(--text) !important;
}
label {
  color: var(--muted) !important;
}
/* Image component */
.gradio-container img {
  border-radius: 14px !important;
}
/* Example / hint list */
ul.usage-hints {
  list-style: disc;
  margin-left: 1.25rem;
  color: var(--muted);
  font-size: 0.95rem;
}
ul.usage-hints li {
  margin-bottom: 0.25rem;
}
/* Mobile */
@media (max-width: 768px) {
  .gradio-container {
    padding: 18px 12px !important;
  }
}
"""


# ---------------- UI ----------------
with gr.Blocks(title="Multimodal Image Assistant") as demo:
    # Header
    gr.Markdown(
        """
# 🖼️ Multimodal Image Assistant
A dark-mode BLIP demo for **image captioning** and **visual question answering**.
"""
    )

    with gr.Row():
        # ---------- Left: Image & caption ----------
        with gr.Column(elem_id="left-panel"):
            gr.Markdown("### 📤 Upload image")
            image_input = gr.Image(
                type="pil",
                label="Upload or drag an image here",
            )

            gr.Markdown("### 📝 Caption the scene")
            caption_btn = gr.Button("✨ Generate Caption")
            caption_output = gr.Textbox(
                label="Generated Caption",
                interactive=False,
                lines=3,
                placeholder="Caption will appear here...",
            )

            caption_btn.click(
                fn=generate_caption,
                inputs=image_input,
                outputs=caption_output,
            )

        # ---------- Right: VQA & tips ----------
        with gr.Column(elem_id="right-panel"):
            gr.Markdown("### ❓ Ask about the image")

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the person doing? How many animals are there?",
                lines=2,
            )
            vqa_btn = gr.Button("🔍 Answer Question")
            vqa_output = gr.Textbox(
                label="AI Answer",
                interactive=False,
                lines=2,
                placeholder="Answer will appear here...",
            )

            vqa_btn.click(
                fn=answer_question,
                inputs=[image_input, question_input],
                outputs=vqa_output,
            )

            gr.Markdown("### 💡 How to use")
            gr.Markdown(
                """
<ul class="usage-hints">
  <li>Upload any photo (people, objects, street scenes, etc.).</li>
  <li>Click <strong>“Generate Caption”</strong> to get a quick description.</li>
  <li>Then ask follow‑up questions like:
    <br>• <em>“What color is the car?”</em>
    <br>• <em>“How many people are there?”</em>
    <br>• <em>“Is this indoors or outdoors?”</em>
  </li>
</ul>
""",
                elem_classes=[],
            )

    gr.Markdown(
        """
---
**Tech stack:** BLIP (Salesforce), Hugging Face Transformers, Gradio  
**Tasks:** Image captioning ▪ visual question answering ▪ multimodal reasoning
"""
    )

# In Gradio 6, theme/css go in launch()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    theme=gr.themes.Monochrome(),
    css=custom_css,
)
