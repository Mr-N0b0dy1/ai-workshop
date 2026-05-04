# Day 1 Cheat Sheet — AI Landscape & First Model
### Keep this open during the session

---

## The Core Pipeline (memorise this)

```
DATA  ──→  MODEL  ──→  OUTPUT  ──→  INTEGRATION
```

Everything in AI is a version of this. Ask yourself for any AI system:
- What is the **data** (input)?
- What is the **model** (who trained it, on what)?
- What is the **output** (format, structure)?
- What is the **integration** (what uses this output)?

---

## AI Types — Quick Reference

| Type | What it does | Example |
|---|---|---|
| Classical ML | Learns patterns from structured data | Spam filter, price prediction |
| Deep Learning | Neural networks on raw data | Everything below |
| Computer Vision | Understands images & video | Face ID, object detection |
| NLP | Understands & generates text | ChatGPT, translation |
| Generative AI | Creates new content | Stable Diffusion, GPT-4 |

---

## Hugging Face Pipeline — Syntax

```python
from transformers import pipeline

# Load any model
model = pipeline("TASK_TYPE", model="MODEL_ID")

# Run it
result = model(YOUR_INPUT)
```

**Common task types:**
- `"image-classification"` — what object is in this image?
- `"text-classification"` — what category/sentiment is this text?
- `"text-generation"` — continue this text
- `"zero-shot-classification"` — classify into categories you define
- `"image-to-text"` — describe an image in words
- `"automatic-speech-recognition"` — speech to text

---

## Colab Quick Reference

| Action | Shortcut |
|---|---|
| Run current cell | Shift + Enter |
| Run and insert new cell | Alt + Enter |
| Add cell above | Ctrl + M + A |
| Add cell below | Ctrl + M + B |
| Delete cell | Ctrl + M + D |
| Comment/uncomment | Ctrl + / |

**If something breaks:**
1. Read the **last line** of the error — that's the actual problem
2. Make sure cells ran **in order** (top to bottom)
3. If all else fails: Runtime → Restart and run all

---

## Models Used Today

| Model | Task | Size | Notes |
|---|---|---|---|
| `google/vit-base-patch16-224` | Image classification | ~330MB | Trained on 14M images, 1000 categories |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment | ~260MB | Fast, accurate for English sentiment |
| `gpt2` | Text generation | ~500MB | Old (2019) but fast — good for demos |
| `facebook/bart-large-mnli` | Zero-shot classification | ~1.6GB | Flexible — define your own categories |
| `nateraw/food` | Food classification | ~350MB | 101 food categories |

---

## Find More Models

Go to: **https://huggingface.co/models**

Filter by:
- **Task** (top left) — pick your use case
- **Language** (if doing NLP)
- **Sort by Downloads** — most popular = most tested

A model card tells you:
- What it was trained on
- What it's good at
- Code to run it
- Known limitations

---

## Key Vocabulary

| Word | Meaning |
|---|---|
| **Model** | A trained neural network stored as a file of weights |
| **Inference** | Running a trained model on new data (what we did today) |
| **Training** | The process of adjusting a model's weights using data |
| **Fine-tuning** | Taking a pretrained model and training it more on specific data |
| **Pipeline** | A wrapper that handles preprocessing + model + postprocessing |
| **Weights** | The numbers inside a model that encode what it learned |
| **Confidence score** | How certain the model is (0–1, where 1 = 100%) |
| **Zero-shot** | Running a model on tasks it was never explicitly trained for |

---

## Tomorrow — Day 2

You'll need:
- Today's Colab notebook (already have it)
- A local install if the instructor sends it: **Ollama** → https://ollama.ai

Optional prep: First 10 minutes of Andrej Karpathy "Intro to Large Language Models" on YouTube.
