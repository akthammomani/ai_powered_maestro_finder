# **AI-Powered Maestro Finder — App** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://ai-powered-maestro-finder.streamlit.app//)

A fast, lightweight **Streamlit** app that identifies the composer of a classical piano piece.  
Upload a **MIDI** file or **record audio**; the app predicts **Bach, Beethoven, Chopin, or Mozart**, and shows confidence bars, a piano-roll, and optional sheet-music preview.

<p align="center">
  <img src="assets/hero_screenshot.png" alt="App screenshot" width="800"/>
</p>

---

## **Key features**

- **Two inputs**: upload `.mid/.midi` or record a short audio snippet.
- **Audio → MIDI** with **Basic Pitch (ONNX)** for fast, reliable transcription.
- **Compact CNN** classifier on 88×512 piano-roll windows (8 fps).
- **Visuals**: confidence **horizontal bars**, clean **piano-roll**, optional **MusicXML** score.
- **Friendly UX**: one-page Streamlit UI with clear messages and examples.

---

## pipeline

app.py
utils/
  inference.py          # load model + predict_composer()
  audio_utils.py        # audio -> MIDI (Basic Pitch, sanity checks)
  vis_utils.py          # charts (confidence bars, piano-roll)
  score_utils.py        # MusicXML export + OSMD render (optional)
  label_map.json
assets/images/          # UI images
style.css               # optional custom styles
model/best_cnn.keras    # trained CNN



