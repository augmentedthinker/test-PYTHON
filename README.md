# Text → Video (Streamlit + Hugging Face) — v1.1

Fixes
- Embedded demo MP4 (no ffmpeg needed on Streamlit Cloud).
- Auto-retry without provider if a provider blocks the route (e.g., `Not allowed to POST ... for provider novita`).

## Secrets
```
HF_TOKEN = "hf_xxx..."     # required
# optional, try removing if you see route errors
HF_PROVIDER = "novita"
```

## Deploy
- Push to GitHub, set entry to `streamlit_app.py`, add Secrets, run.
