# Text-to-Image (Streamlit + Hugging Face Inference)
# ---------------------------------------------------
# What this is:
#   A minimal Streamlit app that generates images from text using
#   Hugging Face Inference API via huggingface_hub.InferenceClient.
#
# How to use:
#   1) Add your Hugging Face token in Streamlit Secrets as HF_TOKEN
#      (or set an environment variable HF_TOKEN). Then click "Generate".
#   2) Pick a model in the sidebar and tweak width/height, steps, etc.
#   3) Download the image or keep a small session gallery on-page.
#
# Notes:
#   - If no token is provided, the app runs in "demo mode" and creates
#     a lightweight placeholder image so the UI remains interactive.
#   - Keep sizes modest on Streamlit Cloud (e.g., 768x768) to avoid timeouts.

import io
import os
from typing import Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Text → Image (Hugging Face)", page_icon="🎨", layout="centered")

def get_hf_token() -> Optional[str]:
    return st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

def make_demo_image(text: str, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            r = int(180 + 60 * (x / w))
            g = int(120 + 80 * (y / h))
            b = int(200 - 80 * ((x + y) / (w + h)))
            img.putpixel((x, y), (r, g, b))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, w // 28))
    except Exception:
        font = ImageFont.load_default()
    box = (40, 40, w - 40, h - 40)
    draw.rectangle(box, outline=(255, 255, 255), width=2)
    wrapped = []
    line = ""
    for word in text.split():
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) < (w - 120):
            line = test
        else:
            wrapped.append(line)
            line = word
    if line:
        wrapped.append(line)
    y = h // 2 - len(wrapped) * (font.size + 6) // 2
    for ln in wrapped:
        tw = draw.textlength(ln, font=font)
        draw.text(((w - tw) / 2, y), ln, fill=(255, 255, 255), font=font)
        y += font.size + 6
    return img

@st.cache_resource(show_spinner=False)
def get_client(token: str):
    return InferenceClient(token=token)

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

st.sidebar.header("Settings")
st.sidebar.caption("Add your Hugging Face token in **Settings → Secrets → HF_TOKEN**.")

token = get_hf_token()
token_status = "✅ Found" if token else "⚠️ Not set (demo mode)"
st.sidebar.write(f"**Token:** {token_status}")

model = st.sidebar.selectbox(
    "Model",
    [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-2-1",
    ],
    index=0
)

w = st.sidebar.slider("Width", 384, 1024, 768, step=64)
h = st.sidebar.slider("Height", 384, 1024, 768, step=64)
steps = st.sidebar.slider("Inference steps", 4, 50, 20)
guidance = st.sidebar.slider("Guidance scale", 0.0, 12.0, 7.5, step=0.1)
seed = st.sidebar.number_input("Seed (optional, -1 = random)", value=-1, step=1)
neg = st.sidebar.text_input("Negative prompt (optional)", "")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: keep sizes modest on Streamlit Cloud to avoid timeouts.")

st.title("🎨 Text → Image")
prompt = st.text_area("Describe the image you want:", placeholder="e.g., Astronaut riding a horse, photorealistic, golden hour", height=100)

col_gen, col_clear = st.columns([1, 1])
generate = col_gen.button("Generate", type="primary", use_container_width=True)
clear = col_clear.button("Clear Gallery", use_container_width=True)

if "gallery" not in st.session_state:
    st.session_state.gallery = []

if clear:
    st.session_state.gallery = []
    st.experimental_rerun()

if generate and prompt.strip():
    with st.spinner("Creating image..."):
        if not token or not HF_AVAILABLE:
            img = make_demo_image(prompt.strip(), w, h)
            st.info("Demo mode: set HF_TOKEN to generate with a real model.")
        else:
            try:
                client = get_client(token)
                img = client.text_to_image(
                    prompt=prompt.strip(),
                    negative_prompt=neg or None,
                    model=model,
                    guidance_scale=float(guidance),
                    num_inference_steps=int(steps),
                    width=int(w),
                    height=int(h),
                    seed=None if int(seed) < 0 else int(seed),
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                img = make_demo_image("Generation error — showing demo image", w, h)

        st.image(img, caption=prompt.strip(), use_container_width=False)
        st.session_state.gallery.insert(0, img)

        png_bytes = pil_to_bytes(img, "PNG")
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name="generation.png",
            mime="image/png",
            use_container_width=True
        )

if st.session_state.gallery:
    st.subheader("Session gallery")
    for i, im in enumerate(st.session_state.gallery[:8], start=1):
        st.image(im, caption=f"#{i}", use_container_width=False)

st.caption("Powered by Hugging Face Inference • Streamlit • Made by Christopher + Animaeus")
