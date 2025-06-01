# CLIP Zero-Shot Image Classification

A straightforward implementation demonstrating **text-based image retrieval** and **zero-shot prediction** using **OpenAI's CLIP model**.

---

## Overview

This project showcases two powerful capabilities of CLIP:

* **Zero-Shot Classification** (`zero-shot-prediction.py`): Predicts the class of an image without being explicitly trained on that specific class. It does this by comparing the image's "visual fingerprint" to the "text fingerprints" of various class labels.
* **Text-Based Image Understanding** (`txt-based-image-retrieval.ipynb`): Utilizes natural language to understand and categorize images.

The example uses the **CIFAR-100 dataset** (a collection of 10,000 test images across 100 object categories like "cat," "dog," "airplane," etc.) to demonstrate how CLIP can accurately identify objects in images simply by being provided with their class names as text descriptions.
