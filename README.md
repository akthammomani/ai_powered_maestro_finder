# **AI-Powered Maestro Finder**
<p align="center">
  <img src="https://github.com/user-attachments/assets/fd53a6fb-6c1b-4f75-915d-2e727dc322c4"
       alt="Image"
       width="80%">
</p>

This project is a part of Neural Networks and Deep Learning (AAI-511-02) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- Project Status: Ongoing

## **Introduction**

**AI‑Powered Maestro Finder** is a lightweight web app that can name the composer behind a piece of classical music in seconds.  
Upload a MIDI file or record a short audio snippet and the app returns the most likely maestro (Bach, Beethoven, Chopin, or Mozart) along with confidence scores and a quick piano‑roll visual. Think of it as "Shazam for classical composers", built entirely on symbolic‑music AI.

## **Objective**

* Make composer recognition accessible to students, musicians, and curious listeners.  
* Demonstrate an end‑to‑end ML pipeline: data wrangling &rarr; augmentation &rarr; feature extraction &rarr; deep‑learning Models &rarr; friendly UI.  
* Provide a clean template that others can extend with more composers or new classification tasks (style, period, instrument).

## **Methods**

| Stage | Key steps |
|-------|-----------|
| **Data wrangling** | Deduplicated and cleaned ~1.6 k raw MIDI files; fixed note overlaps, trimmed out‑of‑range pitches, removed empty tracks. |
| **Class balancing** | Pitch‑shift, time‑stretch, and velocity‑jitter on minority classes to reach &asymp; 1 k files per composer. |
| **Feature extraction** | *Two parallel views*  <br> 1. **Piano‑rolls** (88 pitches × 512 time frames) for a CNN.  <br> 2. **Event tokens** (note‑on/off + time‑shift) for an LSTM. |
| **Modeling** | Trained a CNN on piano‑roll "images" and an LSTM on token sequences; final prediction is the average of both softmax outputs. |
| **Inference pipeline** | MIDI upload &rarr; features on‑the‑fly &rarr; ensemble &rarr; probabilities.  <br> Live audio passes through BasicPitch to extract notes before the same pipeline. |
| **Front‑end** | Streamlit app shows top‑3 composers, confidence bar chart, and a mini piano‑roll preview for context. |

