# Text → Video (Streamlit + Hugging Face)

Minimal Streamlit app to generate short videos from text prompts using the Hugging Face Inference API.

## Files
- `streamlit_app.py`
- `requirements.txt`

## Streamlit Secrets
```
HF_TOKEN = "hf_xxx..."     # required
HF_PROVIDER = "novita"     # optional
```

## Deploy
- Push to a public GitHub repo.
- On Streamlit Cloud, set the entry file to `streamlit_app.py`.
- Add Secrets as above, then click **Generate**.

### Notes
- Default model: `Wan-AI/Wan2.1-T2V-14B`.
- If the token is missing, the app shows a tiny placeholder demo clip.
- Video generation can take 30–60 seconds (or more) depending on model/provider.
