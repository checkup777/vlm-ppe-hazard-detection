import os
# Must be set BEFORE torch initializes CUDA. Without this, PyTorch's
# use_deterministic_algorithms(True) below does not fully cover cuBLAS
# matrix multiplications - cuBLAS can still pick from workspace-size-
# dependent algorithm variants non-deterministically, which is one
# concrete source of run-to-run output drift on identical inputs.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re
import base64
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import gc
import torch
import gradio as gr

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/qwen2-vl-2b")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256*28*28,
    max_pixels=768*28*28
)

def query_model(image, question, max_tokens=150, stop_strings=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    # stop_strings, when provided, halts generation the moment the given
    # string appears in the output - used here to cut generation off before
    # the model starts a second, hallucinated PPE/Hazard/Verdict block after
    # already answering. Requires transformers >= 4.39.
    extra_kwargs = {}
    if stop_strings:
        extra_kwargs["stop_strings"] = stop_strings
        extra_kwargs["tokenizer"] = processor.tokenizer

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            **extra_kwargs,
        )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs, text
    return output_text[0]

description_question = (
    "Describe this image in a few sentences, covering: "
    "1) what activity or task is taking place, "
    "2) what personal protective equipment (PPE) is visibly present on the people shown, "
    "3) what PPE items are missing or not visible, and "
    "4) whether any worker is physically touching, operating, or within arm's reach of machinery "
    "or equipment that is running, moving, or energized right now. If no machinery is visible, or "
    "it is present but idle/unattended, say so explicitly rather than describing it as hazardous. "
    "Write naturally, in your own words."
)

def build_conclusion_question(description):
    return (
        f"Here is a description of a worksite image:\n\"{description}\"\n\n"
        "Base your judgment ONLY on the facts stated in the description above. Do not introduce any "
        "detail, count, or fact that is not explicitly mentioned in it.\n\n"
        "First, answer two questions in one sentence each:\n"
        "1) PPE: Is EVERY visible worker wearing a hardhat AND industrial-appropriate "
        "clothing (not jeans, flannel, t-shirts, or business attire)?\n"
        "2) Hazard: Is a worker actually touching, operating, or within arm's reach of equipment "
        "that is running, moving, or energized - as opposed to equipment that is merely present, "
        "idle, or in the background? Equipment that is idle or unattended is NOT a hazard.\n\n"
        "Safe only if PPE is fully compliant AND no active hazard is present.\n\n"
        "Respond in exactly this format, once, and stop immediately after the Hazard line:\n"
        "1) PPE: [Yes/No] - [brief reason]\n"
        "2) Hazard: [Yes/No] - [brief reason]"
    )

def extract_ppe_hazard_verdict(judgment_text):
    """
    Ignores whatever the model itself wrote after 'Verdict:' - that line has
    repeatedly been shown to contradict the model's own PPE/Hazard answers
    directly above it. Instead, this pulls out just the two Yes/No sub-answers
    (which have been more reliably reasoned) and applies the stated safety
    rule in plain code: Unsafe if PPE is No, OR Hazard is Yes. Safe only if
    PPE is Yes AND Hazard is No.
    """
    text = judgment_text.lower()

    ppe_match = re.search(r"1\)\s*ppe:\s*(yes|no)", text)
    hazard_match = re.search(r"2\)\s*hazard:\s*(yes|no)", text)

    if not ppe_match or not hazard_match:
        return "unclear"

    ppe_ok = ppe_match.group(1) == "yes"
    hazard_present = hazard_match.group(1) == "yes"

    if not ppe_ok or hazard_present:
        return "unsafe"
    return "safe"

def analyze_image(image):
    """
    Runs the full two-call pipeline on a single PIL image. Returns
    (description, raw_judgment, verdict).
    """
    description = query_model(image, description_question, max_tokens=200)
    conclusion_question = build_conclusion_question(description)
    model_conclusion = query_model(
        image, conclusion_question, max_tokens=120,
        stop_strings=["\n\n1)"]
    )
    verdict = extract_ppe_hazard_verdict(model_conclusion)

    del conclusion_question
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return description, model_conclusion, verdict


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def run_batch_inspection(files, progress=gr.Progress()):
    if not files:
        return [], "No files uploaded.", "", ""

    results = []
    table_rows = []

    for idx, file in enumerate(files):
        progress((idx, len(files)), desc=f"Analyzing {os.path.basename(file.name)}")
        filename = os.path.basename(file.name)
        image = Image.open(file.name).convert("RGB")

        description, model_conclusion, verdict = analyze_image(image)

        thumb = image.copy()
        thumb.thumbnail((300, 300))
        buffered = BytesIO()
        thumb.save(buffered, format="JPEG", quality=85)
        thumbnail_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        thumbnail_data_uri = f"data:image/jpeg;base64,{thumbnail_b64}"

        results.append({
        "filename": filename,
        "description": description,
        "judgment": model_conclusion,
        "verdict": verdict,
        "thumbnail": thumbnail_data_uri,
        })
        table_rows.append([filename, verdict.upper()])

        del image, thumb, buffered
        gc.collect()
        torch.cuda.empty_cache()

    safe_count = sum(1 for r in results if r["verdict"] == "safe")
    unsafe_count = sum(1 for r in results if r["verdict"] == "unsafe")
    unclear_count = sum(1 for r in results if r["verdict"] == "unclear")
    summary = (
        f"Total analyzed: {len(results)}  |  "
        f"Safe: {safe_count}  |  Unsafe: {unsafe_count}  |  Unclear: {unclear_count}"
    )

    report_blocks = []
    for r in results:
        verdict_color = (
            "#c0392b" if r["verdict"] == "unsafe"
            else "#27ae60" if r["verdict"] == "safe"
            else "#7f8c8d"
        )
        report_blocks.append(f"""
        <div style="display:flex; gap:16px; border:1px solid #ddd; border-radius:8px;
                    padding:14px; margin-bottom:14px; align-items:flex-start;">
            <img src="{r['thumbnail']}" style="width:220px; border-radius:6px; flex-shrink:0;" />
            <div>
                <h4 style="margin:0 0 6px 0;">
                    {r['filename']}
                    <span style="color:{verdict_color}; margin-left:8px;">{r['verdict'].upper()}</span>
                </h4>
                <p style="margin:4px 0;"><b>Description:</b> {r['description']}</p>
                <p style="margin:4px 0;"><b>Model judgment:</b> {r['judgment']}</p>
            </div>
        </div>
        """)
    full_report = "".join(report_blocks)

    # Final model-generated summary and recommendation, mirroring the
    # original script's closing step.
    progress((len(files), len(files)), desc="Generating overall recommendation")
    all_judgments_text = "\n\n".join(
        f"{r['filename']}: {r['judgment']}" for r in results
    )
    summary_prompt = (
        f"Here are the safety judgments for {len(results)} worksite images:\n\n"
        f"{all_judgments_text}\n\n"
        f"Out of these, {safe_count} were judged safe and {unsafe_count} were judged unsafe. "
        "Write a brief overall summary of the findings, and provide practical recommendations for "
        "what site management should do to address the unsafe conditions found."
    )
    last_image = Image.open(files[-1].name).convert("RGB")
    recommendation = query_model(last_image, summary_prompt, max_tokens=450)
    del last_image
    gc.collect()
    torch.cuda.empty_cache()

    return table_rows, summary, full_report, recommendation


with gr.Blocks(title="VLM PPE Safety Inspection") as demo:
    gr.Markdown("## Vision-LLM PPE & Hazard Safety Inspection")
    gr.Markdown(
        "Upload one or more worksite images. Each image is checked for PPE compliance "
        "and active equipment hazards; the final verdict is computed in Python from the "
        "model's PPE/Hazard sub-answers, not from its own self-reported Verdict line."
    )

    file_input = gr.File(
        file_count="multiple",
        file_types=["image"],
        label="Upload worksite images"
    )
    run_btn = gr.Button("Run Inspection", variant="primary")

    summary_box = gr.Textbox(label="Summary", interactive=False)
    results_table = gr.Dataframe(
        headers=["Filename", "Verdict"],
        label="Per-image results"
    )
    recommendation_box = gr.Textbox(
        label="Overall Summary & Recommendation",
        interactive=False,
        lines=6
    )
    report_box = gr.HTML(label="Full report")

    run_btn.click(
        fn=run_batch_inspection,
        inputs=file_input,
        outputs=[results_table, summary_box, report_box, recommendation_box]
    )

if __name__ == "__main__":
    demo.launch()