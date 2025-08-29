# Text-to-Video (Streamlit + Hugging Face Inference) — v1.2
# - Bundled demo.mp4 (no ffmpeg needed)
# - Auto-retry without provider if provider blocks the route
# - Clear instructions in UI

import os
from typing import Optional

import streamlit as st

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

st.set_page_config(page_title="Text → Video", page_icon="🎬", layout="centered")

def get_hf_token() -> Optional[str]:
    return st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

def get_hf_provider() -> Optional[str]:
    return st.secrets.get("HF_PROVIDER", os.getenv("HF_PROVIDER"))

@st.cache_resource(show_spinner=False)
def get_client(token: str, provider: Optional[str] = None):
    if provider:
        return InferenceClient(provider=provider, api_key=token)
    return InferenceClient(token=token)

def demo_video_bytes() -> bytes:
    # Read the bundled demo asset
    path = os.path.join(os.path.dirname(__file__), "assets", "demo.mp4")
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""

# ---- Sidebar
st.sidebar.header("Settings")
st.sidebar.caption("Set **HF_TOKEN** in Secrets. Optional: **HF_PROVIDER** (try removing if you see route errors).")

token = get_hf_token()
provider = get_hf_provider()
status = "✅ Found" if token else "⚠️ Not set (demo mode)"
st.sidebar.write(f"**Token:** {status}" + (f" (provider: {provider})" if provider else ""))

model = st.sidebar.selectbox("Model", ["Wan-AI/Wan2.1-T2V-14B"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("If you get a 'Not allowed to POST' error, clear HF_PROVIDER and rerun.")

# ---- Main
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
    with st.spinner("Creating video... (may take up to a minute)"):
        if not token or not HF_AVAILABLE:
            st.info("Demo mode: add HF_TOKEN to run with a real model.")
            vb = demo_video_bytes()
        else:
            try:
                client = get_client(token, provider)
                out = client.text_to_video(prompt.strip(), model=model)
                vb = out if isinstance(out, (bytes, bytearray)) else out.read()
            except Exception as e:
                msg = str(e)
                # Retry without provider if the route is blocked
                if provider and ("Not allowed to POST" in msg or "route" in msg.lower()):
                    try:
                        st.info("Provider blocked this route; retrying with default router…")
                        client2 = get_client(token, None)
                        out = client2.text_to_video(prompt.strip(), model=model)
                        vb = out if isinstance(out, (bytes, bytearray)) else out.read()
                    except Exception as e2:
                        st.error(f"Generation failed: {e2}")
                        vb = demo_video_bytes()
                else:
                    st.error(f"Generation failed: {e}")
                    vb = demo_video_bytes()

        if vb:
            st.video(vb)
            st.session_state.gallery.insert(0, vb)
            st.download_button("Download video", data=vb, file_name="generation.mp4", mime="video/mp4", use_container_width=True)
        else:
            st.warning("Could not render demo video; please check your token and try again.")

if st.session_state.gallery:
    st.subheader("Session gallery")
    for i, vb in enumerate(st.session_state.gallery[:4], 1):
        st.video(vb)

st.caption("Powered by Hugging Face Inference • Streamlit • Made by Christopher + Animaeus")
