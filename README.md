<p align="left">
  <!-- Release / Repo -->
  <a href="https://github.com/YOUR_USER/YOUR_REPO/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/YOUR_USER/YOUR_REPO?display_name=tag&sort=semver">
  </a>
  <a href="https://github.com/YOUR_USER/YOUR_REPO/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/YOUR_USER/YOUR_REPO">
  </a>

  <!-- Python / Frameworks -->
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Keras Tuner" src="https://img.shields.io/badge/Keras%20Tuner-enabled-E00000?logo=keras&logoColor=white">
  <img alt="Basic Pitch" src="https://img.shields.io/badge/Basic%20Pitch-ONNX-4A90E2">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white">

  <!-- Quality / Style (optional) -->
  <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  <!-- If you add CI later, switch these on:
  <img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/YOUR_USER/YOUR_REPO/tests.yml?label=tests">
  <img alt="Coverage" src="https://img.shields.io/codecov/c/github/YOUR_USER/YOUR_REPO?logo=codecov">
  -->

  <!-- Model / Demo -->
  <img alt="Model accuracy" src="https://img.shields.io/badge/CNN%20accuracy-98.4%25-brightgreen">
  <!-- If you host the app, add your URL here -->
  <!-- <a href="https://YOUR-STREAMLIT-APP-URL">
    <img alt="Demo" src="https://img.shields.io/badge/demo-streamlit.cloud-0A0A0A?logo=streamlit">
  </a> -->

  <!-- Notebooks -->
  <a href="https://colab.research.google.com/drive/REPLACE_WITH_DATA_WRANGLING_COLAB_ID">
    <img alt="Colab: Data wrangling" src="https://img.shields.io/badge/Colab-Data%20wrangling-F9AB00?logo=googlecolab&logoColor=white">
  </a>
  <a href="https://colab.research.google.com/drive/REPLACE_WITH_CNN_COLAB_ID">
    <img alt="Colab: CNN modeling" src="https://img.shields.io/badge/Colab-CNN%20modeling-F9AB00?logo=googlecolab&logoColor=white">
  </a>
</p>



# **AI-Powered Maestro Finder** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://ai-powered-maestro-finder.streamlit.app//)
<p align="center">
  <img src="https://github.com/user-attachments/assets/fd53a6fb-6c1b-4f75-915d-2e727dc322c4"
       alt="Image"
       width="80%">
</p>

This project is a part of Neural Networks and Deep Learning (AAI-511-02) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- Project Status: Completed

Lightweight web app that can name the composer behind a classical piano piece in seconds.  
Upload a MIDI file or record a short audio snippet; the app predicts **Bach, Beethoven, Chopin, or Mozart** and shows clear confidence bars, a piano-roll, and optional sheet-music preview.

## **Introduction**

**AI-Powered Maestro Finder** turns symbolic music into quick, explainable predictions. It’s a complete pipeline—from raw MIDIs to trained models to an interactive Streamlit app—built to be readable, reproducible, and easy to extend.

## **Objective**

- Make composer recognition accessible to students, musicians, and curious listeners.
- Demonstrate an end-to-end ML workflow:
  data wrangling → augmentation → feature extraction → modeling → friendly UI.
- Provide a clean template you can extend with new composers or tasks (style, period, instrument).

## Features

- **Upload MIDI** or **record audio** (auto-transcribed to MIDI with Basic Pitch).
- **Confidence bars** and **piano-roll** visualization.
- **Optional sheet-music view** (MusicXML rendered via OpenSheetMusicDisplay).
- **On-device inference** using a compact CNN.
- Clear, modular code: `utils/` for I/O, features, inference, and visualization.

## **Methods**

| Stage | Key steps |
|------|-----------|
| **Data wrangling** | De-duplicate and clean raw MIDIs; trim to piano range (A0–C8); remove empty or corrupted tracks. |
| **Class balancing** | Pitch-shift, time-stretch, and velocity-jitter minority classes to balance counts. |
| **Feature extraction** | Two parallel views: (1) **Piano-roll** 88×512 @ 8 fps for CNN, (2) **Event tokens** for LSTM (used for research and comparison). |
| **Modeling** | Train **CNN** on piano-rolls and **LSTM** on sequences; evaluate both. |
| **Final selection** | **CNN** chosen for the app based on accuracy, stability, and speed. |
| **Inference** | MIDI → piano-roll → CNN → probabilities. For audio, run Basic Pitch → MIDI → same pipeline. |
| **Front end** | Streamlit UI with confidence bars, piano-roll, and optional sheet music. |

## **Results (test split)**

| Model | Input window | Accuracy | Macro F1 |
|------|--------------|----------|----------|
| **CNN** | Piano-roll 88 × 512 | **0.984** | **0.984** |
| LSTM | Piano-roll 88 × 512 | 0.830 | 0.832 |

**Takeaway:** with piano-roll inputs, the CNN's inductive bias (2D convs over time×pitch + residuals + SE) aligns better with musical texture and converges faster.  
We use the **CNN** as the production model in the app.

