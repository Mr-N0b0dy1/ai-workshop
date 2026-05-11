# app.py — Vision + Language Pipeline
# Model: ibm-granite/granite-4.1-3b-instruct
# Detector: YOLOv8n

import gradio as gr
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline

# ── Config ────────────────────────────────────────────────────
MODEL_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
YOLO_CKPT = "yolov8n.pt"
DEVICE    = 0 if torch.cuda.is_available() else -1

# ── Load models (once at startup) ─────────────────────────────
print(f"[INFO] Loading YOLO from {YOLO_CKPT} ...")
yolo_model = YOLO(YOLO_CKPT)

print(f"[INFO] Loading LLM: {MODEL_ID} ...")
llm_pipe = hf_pipeline(
    "text-generation",
    model=MODEL_ID,
    device=DEVICE,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# ── Prompt modes ──────────────────────────────────────────────
# Each entry: (label, system_prompt)
MODES: dict[str, str] = {

    "🔍 Scene Description": (
        "You are a precise visual analyst. "
        "Given a list of detected objects, describe the overall scene in 2-3 clear, "
        "vivid sentences. Mention the likely setting, activities, and atmosphere. "
        "Do not list objects — synthesise them into a coherent narrative."
    ),

    "⚠️ Safety Advisor": (
        "You are a certified safety inspector. "
        "Given a list of detected objects, identify all potential safety hazards "
        "in bullet-point form. For each hazard, state the risk and one mitigation step. "
        "Be specific and practical. If no hazards are evident, say so clearly."
    ),

    "🌿 Environmental Impact": (
        "You are an environmental scientist. "
        "Given the detected objects, assess the likely environmental footprint of this scene. "
        "Comment on waste, energy, or ecological concerns you can infer. "
        "Suggest one concrete sustainability improvement. Keep it under 80 words."
    ),

    "📸 Instagram Caption": (
        "You are a creative social media copywriter. "
        "Write ONE punchy Instagram caption (under 30 words) inspired by the detected objects, "
        "followed by 5 relevant hashtags on a new line. "
        "Be witty, relatable, and platform-appropriate."
    ),

    "🎓 Educational Explainer": (
        "You are an engaging science educator writing for curious teenagers. "
        "Pick the single most interesting object from the list and explain one surprising, "
        "scientifically accurate fact about it in 2-3 sentences. "
        "Use accessible language and end with a thought-provoking question."
    ),

    "🛒 Shopping Assistant": (
        "You are a helpful personal shopper. "
        "Based on the detected objects, suggest 3 complementary products a person might need "
        "or want to buy. For each suggestion, give the product name and a one-sentence reason. "
        "Format as a numbered list."
    ),

    "🎨 Art Critique": (
        "You are a witty contemporary art critic. "
        "Treat the scene described by the detected objects as if it were an art installation. "
        "Give it a pretentious title and write a 2-sentence museum-style critique that "
        "explores its themes and deeper meaning, however absurd."
    ),

    "🧩 Technical Analysis": (
        "You are a technical documentation writer. "
        "Given the detected objects, produce a structured analysis: "
        "(1) Object inventory with likely material/function, "
        "(2) Dominant category (e.g. electronics, furniture, vehicles), "
        "(3) Any notable object interactions or dependencies. "
        "Be concise and factual."
    ),
}

MODE_LABELS = list(MODES.keys())

# ── LLM call ─────────────────────────────────────────────────
def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    prompt_str = llm_pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = llm_pipe(
        prompt_str,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=llm_pipe.tokenizer.eos_token_id,
    )

    text = output[0]["generated_text"].strip()

    # Strip any residual role tokens Granite may emit
    for tag in ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"]:
        text = text.replace(tag, "")

    return text.strip()

# ── Main pipeline ─────────────────────────────────────────────
def run_pipeline(image, confidence: float, mode_label: str, max_tokens: int):
    if image is None:
        return None, "⚠️ Please upload an image first.", "", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # ── YOLO inference ────────────────────────────────────────
    results     = yolo_model.predict(image, conf=confidence, verbose=False)
    result      = results[0]
    annotated   = Image.fromarray(result.plot()[:, :, ::-1])

    boxes = result.boxes
    names = result.names

    if len(boxes) == 0:
        detection_text = "No objects detected."
        labels_str     = "nothing identifiable"
        summary_str    = "0 objects detected"
    else:
        detections     = [(names[int(b.cls[0])], float(b.conf[0])) for b in boxes]
        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        detection_text = "\n".join([f"• {n}  ({c:.0%})" for n, c in detections])
        labels_str     = ", ".join([n for n, _ in detections])
        unique_labels  = list(dict.fromkeys([n for n, _ in detections]))
        summary_str    = (
            f"{len(detections)} detection(s) · "
            f"{len(unique_labels)} unique class(es): {', '.join(unique_labels)}"
        )

    # ── Image metadata ────────────────────────────────────────
    w, h = image.size
    meta_text = f"Resolution: {w}×{h} px  |  {summary_str}"

    # ── LLM response ──────────────────────────────────────────
    system_prompt = MODES[mode_label]
    user_prompt   = (
        f"Objects detected in the image: {labels_str}.\n"
        f"Total detections: {len(boxes)}."
    )

    response = call_llm(system_prompt, user_prompt, max_tokens=int(max_tokens))

    return annotated, response, detection_text, meta_text

# ── Gradio UI ─────────────────────────────────────────────────
CSS = """
.gr-button-primary {
    background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
    border: 1px solid #0f3460 !important;
    color: #e94560 !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #0f3460, #1a1a2e) !important;
}
.label-text { font-size: 0.8em; color: #888; }
footer { display: none !important; }
"""

with gr.Blocks(title="Vision + Language Pipeline", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔗 Vision + Language Pipeline
        **YOLOv8n** object detection  ·  **IBM Granite 4.1-3B** language understanding
        Upload an image, pick an analysis mode, and run the pipeline.
        """
    )

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────
        with gr.Column(scale=1):
            img_in = gr.Image(label="📤 Upload Image", type="pil", height=300)

            mode_in = gr.Dropdown(
                choices=MODE_LABELS,
                value=MODE_LABELS[0],
                label="🎯 Analysis Mode",
            )

            with gr.Row():
                conf_in = gr.Slider(
                    0.10, 0.90, value=0.25, step=0.05,
                    label="Confidence Threshold",
                )
                tokens_in = gr.Slider(
                    50, 400, value=200, step=25,
                    label="Max Output Tokens",
                )

            btn = gr.Button("▶  Run Pipeline", variant="primary", size="lg")

            meta_out = gr.Textbox(
                label="📊 Image Info",
                interactive=False,
                lines=1,
                placeholder="Metadata will appear here after running.",
            )

        # ── Right column: outputs ─────────────────────────────
        with gr.Column(scale=1):
            img_out = gr.Image(label="🖼️ Annotated Output", height=300)

            llm_out = gr.Textbox(
                label="LLM Response",
                lines=7,
                interactive=False,
                placeholder="AI response will appear here.",
            )

            det_out = gr.Textbox(
                label="📋 Detections (sorted by confidence)",
                lines=6,
                interactive=False,
                placeholder="Detected objects will appear here.",
            )

    gr.Examples(
        examples=[],   # add example image paths here if desired
        inputs=[img_in],
    )

    btn.click(
        fn=run_pipeline,
        inputs=[img_in, conf_in, mode_in, tokens_in],
        outputs=[img_out, llm_out, det_out, meta_out],
    )

    gr.Markdown(
        "<p style='text-align:center; color:#888; font-size:0.75em;'>"
        "Powered by IBM Granite 4.1 · Ultralytics YOLOv8 · Gradio"
        "</p>"
    )

demo.launch()