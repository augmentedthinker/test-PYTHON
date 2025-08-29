# Text-to-Video (Streamlit + Hugging Face Inference) — v1.1
# Improvements:
# - Fallback if provider rejects the model (auto-retry with default router).
# - Demo video is embedded (no ffmpeg needed on Streamlit Cloud).

import os
import base64
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

# Embedded 2s MP4 (tiny). Avoids ffmpeg on Streamlit Cloud.
_DEMO_B64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAABdttb292AAAAbG12aGQAAAAAAAAAAAAAAAAAAAPoAAABHABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAABWRpbmYAAAAUZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAAAL21kYXQAAAAgbWRoZAAAAAPoAAABHABAAQAAAAEAAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAABAAAAAAABAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAc3RibAAAAGF2YzEAAAAAAAAAAQAAABRtaW5mAAABY21kYXQAAAAYaWxzdAAAACgpKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqAAACAG1wNGEAAABvbXBoNAAAAAAIYWNvbQAAAB5tZGF0AAAAAQAAAAPoH/////8AAAACAAACAAACAAAAAAA="
def demo_video_bytes() -> bytes:
    try:
        return base64.b64decode(_DEMO_B64.encode("ascii"))
    except Exception:
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
st.sidebar.caption("If you get a provider route error, clear HF_PROVIDER or try another provider.")

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
            vid_bytes = demo_video_bytes()
        else:
            try:
                client = get_client(token, provider)
                vid = client.text_to_video(prompt.strip(), model=model)
                vid_bytes = vid if isinstance(vid, (bytes, bytearray)) else vid.read()
            except Exception as e:
                # If provider route is not allowed, retry without provider
                msg = str(e)
                if provider and ("Not allowed to POST" in msg or "route" in msg.lower()):
                    try:
                        st.info("Provider blocked this route; retrying with default router…")
                        client2 = get_client(token, None)
                        vid = client2.text_to_video(prompt.strip(), model=model)
                        vid_bytes = vid if isinstance(vid, (bytes, bytearray)) else vid.read()
                    except Exception as e2:
                        st.error(f"Generation failed: {e2}")
                        vid_bytes = demo_video_bytes()
                else:
                    st.error(f"Generation failed: {e}")
                    vid_bytes = demo_video_bytes()

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
            st.warning("Could not render demo video; please check your token and try again.")

if st.session_state.gallery:
    st.subheader("Session gallery")
    for i, vb in enumerate(st.session_state.gallery[:4], start=1):
        st.video(vb)

st.caption("Powered by Hugging Face Inference • Streamlit • Made by Christopher + Animaeus")
