# Text → Image (Streamlit + Hugging Face)

A tiny Streamlit app that turns prompts into images using the Hugging Face Inference API. Minimal dependencies, deploys on Streamlit Community Cloud.

## 1) Create a public GitHub repo
Add these files to the repo:
- `streamlit_app.py`
- `requirements.txt`

*(Optional)* Add a LICENSE and a cool preview gif later.

## 2) Add your Hugging Face token
On Streamlit Cloud: **App → Settings → Secrets** and add

```
HF_TOKEN = "hf_xxx_your_token_here"
```

Alternatively, you can set an environment variable `HF_TOKEN`, but Secrets is easiest.

## 3) Deploy on Streamlit Community Cloud
- Click **New app**, select your repo, set `streamlit_app.py` as the entry file.
- Wait for build. Open the URL.
- In the app, pick a model (e.g., `black-forest-labs/FLUX.1-schnell`), write a prompt, and click **Generate**.

### Notes
- If the token is missing, the app runs in a playful **demo mode** so the UI is never blank.
- Keep image sizes modest (e.g., 768×768) to avoid timeouts on free tiers.
- You can add more models that support `text_to_image` on Hugging Face Inference.

### Troubleshooting
- **Model not allowed / 403**: Your token might not have access; try another model or check your HF permissions.
- **Timeout**: Reduce width/height or steps.
- **No images**: Confirm `HF_TOKEN` is set in Secrets and the repo has `requirements.txt`.
