# Text-to-Video (Streamlit + Hugging Face Inference) — v1.0
# Minimal Streamlit app that generates videos from text prompts using
# Hugging Face Inference API (huggingface_hub.InferenceClient).
#
# - Put your HF token in Streamlit Secrets as HF_TOKEN (or env var).
# - Optional: HF_PROVIDER (e.g., "novita").
# - If token is missing, the app shows a tiny placeholder video instead.

import os
from typing import Optional

import streamlit as st

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Text → Video", page_icon="🎬", layout="centered")

# -------- Utils --------
def get_hf_token() -> Optional[str]:
    return st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

def get_hf_provider() -> Optional[str]:
    return st.secrets.get("HF_PROVIDER", os.getenv("HF_PROVIDER"))

@st.cache_resource(show_spinner=False)
def get_client(token: str, provider: Optional[str] = None):
    if provider:
        return InferenceClient(provider=provider, api_key=token)
    return InferenceClient(token=token)

def make_demo_video_bytes(duration: float = 3.0, fps: int = 24) -> bytes:
    """Create a tiny placeholder MP4 so the UI works without a token.
    Uses moviepy + imageio-ffmpeg; writes to /tmp then returns bytes.
    """
    try:
        from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
        import tempfile, os

        # simple background
        bg = ColorClip(size=(360, 240), color=(120, 30, 200), duration=duration).set_fps(fps)

        # attempt to overlay 'Demo Mode' text; if TextClip fails (no fonts),
        # fall back to just the color clip.
        try:
            txt = TextClip("Demo Mode", fontsize=40, color="white").set_duration(duration).set_pos("center")
            clip = CompositeVideoClip([bg, txt])
        except Exception:
            clip = bg

        tmpfile = os.path.join(tempfile.gettempdir(), "demo_placeholder.mp4")
        clip.write_videofile(tmpfile, codec="libx264", audio=False, verbose=False, logger=None)
        with open(tmpfile, "rb") as f:
            return f.read()
    except Exception:
        # absolute fallback: return empty bytes (Streamlit will show nothing)
        return b""

# -------- Sidebar --------
st.sidebar.header("Settings")
st.sidebar.caption("Add your HF token in Secrets → HF_TOKEN. Optional: HF_PROVIDER (e.g., novita).")

token = get_hf_token()
provider = get_hf_provider()
status = "✅ Found" if token else "⚠️ Not set (demo mode)"
prov_status = f" (provider: {provider})" if provider else ""
st.sidebar.write(f"**Token:** {status}{prov_status}")

models = [
    "Wan-AI/Wan2.1-T2V-14B",
]
model = st.sidebar.selectbox("Model", models, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: keep prompts concise; T2V can be slow on free tiers.")

# -------- Main --------
st.title("🎬 Text → Video")
prompt = st.text_area("Describe the video you want:", "A young man walking on the street", height=80)

c1, c2 = st.columns([1,1])
generate = c1.button("Generate", type="primary", use_container_width=True)
clear = c2.button("Clear Gallery", use_container_width=True)

if "gallery" not in st.session_state:
    st.session_state.gallery = []

if clear:
    st.session_state.gallery = []
    st.experimental_rerun()

if generate and prompt.strip():
    with st.spinner("Creating video... (this may take up to a minute)"):
        if not token or not HF_AVAILABLE:
            st.info("Demo mode: add HF_TOKEN to run with a real model.")
            vid_bytes = make_demo_video_bytes()
        else:
            try:
                client = get_client(token, provider)
                # text_to_video returns raw bytes for the video file (mp4/webm)
                vid = client.text_to_video(prompt.strip(), model=model)
                vid_bytes = vid if isinstance(vid, (bytes, bytearray)) else vid.read()
            except Exception as e:
                st.error(f"Generation failed: {e}")
                vid_bytes = make_demo_video_bytes()

        if vid_bytes:
            st.video(vid_bytes)
            st.session_state.gallery.insert(0, vid_bytes)
            st.download_button(
                "Download video",
                data=vid_bytes,
                file_name="generation.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        else:
            st.warning("Could not render demo video; please set HF_TOKEN and try again.")

if st.session_state.gallery:
    st.subheader("Session gallery")
    for i, vb in enumerate(st.session_state.gallery[:4], start=1):
        st.video(vb)

st.caption("Powered by Hugging Face Inference • Streamlit • Made by Christopher + Animaeus")
