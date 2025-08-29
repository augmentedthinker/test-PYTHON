# Text-to-Image (Streamlit + Hugging Face Inference) — v1.1
# Fix: clamp num_inference_steps for models like FLUX.1-schnell (max 16).

import io
import os
from typing import Optional, Dict, Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Text → Image (Hugging Face)", page_icon="🎨", layout="centered")

# --- Model presets: sane bounds for Streamlit Cloud + router limits ---
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # Fast FLUX: router caps steps to 16 on many providers (e.g., Nebius)
    "black-forest-labs/FLUX.1-schnell": {"max_steps": 16, "max_size": 1024, "default_steps": 12},
    # SD 2.1 typically tolerates higher steps; keep modest
    "stabilityai/stable-diffusion-2-1": {"max_steps": 50, "max_size": 1024, "default_steps": 30},
}

def get_hf_token() -> Optional[str]:
    return st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

def get_hf_provider() -> Optional[str]:
    # Optional override, e.g., "fal-ai"
    return st.secrets.get("HF_PROVIDER", os.getenv("HF_PROVIDER"))

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
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) < (w - 120):
            line = test
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = h // 2 - len(lines) * (font.size + 6) // 2
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        draw.text(((w - tw) / 2, y), ln, fill=(255, 255, 255), font=font)
        y += font.size + 6
    return img

@st.cache_resource(show_spinner=False)
def get_client(token: str, provider: Optional[str] = None):
    if provider:
        return InferenceClient(provider=provider, api_key=token)
    return InferenceClient(token=token)

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ---- Sidebar ----
st.sidebar.header("Settings")
st.sidebar.caption("Add your Hugging Face token in **Settings → Secrets → HF_TOKEN**. Optional: `HF_PROVIDER` (e.g., `fal-ai`).")

token = get_hf_token()
provider = get_hf_provider()
token_status = "✅ Found" if token else "⚠️ Not set (demo mode)"
prov_status = f" (provider: {provider})" if provider else ""
st.sidebar.write(f"**Token:** {token_status}{prov_status}")

model = st.sidebar.selectbox("Model", list(MODEL_PRESETS.keys()), index=0)
preset = MODEL_PRESETS[model]
max_steps = int(preset["max_steps"])
default_steps = int(preset["default_steps"])

w = st.sidebar.slider("Width", 384, preset["max_size"], 768, step=64)
h = st.sidebar.slider("Height", 384, preset["max_size"], 768, step=64)
steps = st.sidebar.slider(f"Inference steps (≤ {max_steps})", 4, max_steps, default_steps)
guidance = st.sidebar.slider("Guidance scale", 0.0, 12.0, 7.5, step=0.1)
seed = st.sidebar.number_input("Seed (optional, -1 = random)", value=-1, step=1)
neg = st.sidebar.text_input("Negative prompt (optional)", "")

st.sidebar.markdown("---")
st.sidebar.caption("Router errors about steps? This app auto-clamps to the model’s limits.")

# ---- Main ----
st.title("🎨 Text → Image")
prompt = st.text_area("Describe the image you want:", placeholder="e.g., Astronaut riding a horse, photorealistic, golden hour", height=100)

c1, c2 = st.columns([1, 1])
generate = c1.button("Generate", type="primary", use_container_width=True)
clear = c2.button("Clear Gallery", use_container_width=True)

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
                client = get_client(token, provider)
                # Clamp steps defensively
                use_steps = max(1, min(int(steps), max_steps))
                img = client.text_to_image(
                    prompt=prompt.strip(),
                    negative_prompt=neg or None,
                    model=model,
                    guidance_scale=float(guidance),
                    num_inference_steps=use_steps,
                    width=int(w),
                    height=int(h),
                    seed=None if int(seed) < 0 else int(seed),
                )
            except Exception as e:
                # Surface helpful hint if it's a 4xx with JSON detail
                st.error(f"Generation failed: {e}")
                st.info("Hint: If the error mentions `num_inference_steps`, try reducing steps "
                        f"(this model allows up to {max_steps}).")
                img = make_demo_image("Generation error — showing demo image", w, h)

        st.image(img, caption=prompt.strip(), use_container_width=False)
        st.session_state.gallery.insert(0, img)

        st.download_button(
            "Download PNG",
            data=pil_to_bytes(img, "PNG"),
            file_name="generation.png",
            mime="image/png",
            use_container_width=True
        )

if st.session_state.gallery:
    st.subheader("Session gallery")
    for i, im in enumerate(st.session_state.gallery[:8], start=1):
        st.image(im, caption=f"#{i}", use_container_width=False)

st.caption("Powered by Hugging Face Inference • Streamlit • Made by Christopher + Animaeus")
