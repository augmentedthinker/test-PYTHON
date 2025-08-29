# Text → Image (Streamlit + Hugging Face) — v1.1

Update: clamps `num_inference_steps` for models like `FLUX.1-schnell` (max 16). Optional `HF_PROVIDER` (e.g., `fal-ai`).

## Setup
- Add `streamlit_app.py` and `requirements.txt` to a GitHub repo.
- On Streamlit Cloud add Secrets:
```
HF_TOKEN = "hf_xxx_your_token_here"
# optional
HF_PROVIDER = "fal-ai"
```
- Deploy with entrypoint `streamlit_app.py`.

If you see a router error about steps, reduce steps; this app auto-clamps based on the selected model.
