
import gradio as gr
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    BlipForQuestionAnswering,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BLIP Caption model...")
cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
cap_model.eval()

print("Loading BLIP VQA model...")
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)
vqa_model.eval()

def generate_caption(image):
    if image is None:
        return "📤 Please upload an image first."
    inputs = cap_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = cap_model.generate(**inputs, max_length=50, num_beams=3)
    caption = cap_processor.decode(out[0], skip_special_tokens=True)
    return f"✨ {caption}"

def answer_question(image, question):
    if image is None:
        return "📤 Please upload an image first."
    if not question.strip():
        return "❓ Please type a question about the image."
    inputs = vqa_processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vqa_model.generate(**inputs, max_length=20, num_beams=3)
    answer = vqa_processor.batch_decode(out, skip_special_tokens=True)[0]
    return f"💡 {answer}"

with gr.Blocks(title="Multimodal Image Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🖼️ Multimodal Image Assistant\n"
        "**Image Captioning + Visual Question Answering with BLIP**"
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="📤 Upload Image", scale=2)

    gr.Markdown("### 📝 Image Captioning")
    with gr.Row():
        caption_output = gr.Textbox(label="Generated Caption", interactive=False, lines=2, scale=3)
        caption_btn = gr.Button("🎯 Caption", variant="primary", scale=1)

    caption_btn.click(generate_caption, inputs=image_input, outputs=caption_output)

    gr.Markdown("### ❓ Visual Question Answering")
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., How many people are there? What color is the car?",
            lines=2,
            scale=2,
        )
        vqa_btn = gr.Button("💡 Answer", variant="primary", scale=1)

    with gr.Row():
        vqa_output = gr.Textbox(label="AI Answer", interactive=False, lines=2)

    vqa_btn.click(answer_question, inputs=[image_input, question_input], outputs=vqa_output)

demo.launch(share=True)
