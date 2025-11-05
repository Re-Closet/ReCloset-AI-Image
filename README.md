# ♻️ ReCloset-AI

**2025 GDGoC Solution Challenge**  
**Team ReCloset — AI Image Classification Repository**

---

## Purpose

ReCloset-AI uses machine learning to automatically classify the condition of secondhand clothes based on uploaded images. Our AI model detects signs of wear, tear, stains, and other damage to sort clothes into one of the following seven categories:

### Classification Categories

1. Large tear  
2. Wear / Small tear  
3. Shrinkage / Stretching / Wrinkling  
4. Buckle / Button / Zipper damage  
5. Oil / Food / Chemical stain  
6. Ink  
7. Mold  

---

## Installation

### environment

We recommend using `conda`:

```bash
conda create -n ReClosetAI python=3.9
conda activate ReClosetAI
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

If you want to launch the AI classification server (FastAPI):

Make sure the FastAPI app is defined inside ai_server.py as app.

```bash
uvicorn ai_server:app --reload
```
## Training the Model
```bash
python train.py
```
## Authors

Team ReCloset – AI Division
Contact: github.com/wis-hyun

## License
This project is licensed under the MIT License.

---

