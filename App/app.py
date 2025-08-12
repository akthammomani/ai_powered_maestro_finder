import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image
import pretty_midi
import base64
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["BASIC_PITCH_BACKEND"] = "onnx"

from utils.inference import predict_composer
from utils.audio_utils import convert_audio_to_midi
from utils.vis_utils import plot_pianoroll_plotly_clean, plot_confidence_bars
from utils.inference import _prep_roll, MODEL, COMPOSERS
from utils.score_utils import midi_to_musicxml_str, render_musicxml_osmd

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def is_valid_piano_midi(midi_path, min_notes=10, min_duration=2.0):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        total_notes = sum(len(inst.notes) for inst in pm.instruments)
        duration = pm.get_end_time()
        return total_notes >= min_notes and duration >= min_duration
    except Exception:
        return False

def extract_best_512(pm: pretty_midi.PrettyMIDI, fs: int = 8, window: int = 512) -> np.ndarray:
    """
    Training-exact roll: (88, 512), binary {0,1}, FS=8, left crop/pad.
    Uses all instruments; rows 21..108 (A0..C8).
    """
    roll = pm.get_piano_roll(fs=fs)[21:109, :]      # (88, T), velocities 0..127
    roll = (roll > 0).astype(np.uint8)              # binary like training

    T = roll.shape[1]
    out = np.zeros((88, window), dtype=np.uint8)
    out[:, :min(T, window)] = roll[:, :window]      # left crop/pad
    return out

def piano_likeness_flags(pm_obj, fs=8):
    roll = (pm_obj.get_piano_roll(fs=fs)[21:109, :] > 0).astype(np.uint8)  # (88,T)
    if roll.size == 0 or roll.shape[1] == 0:
        return False, ["empty_roll"]
    notes_per_frame = roll.sum(axis=0)
    mean_polyphony  = float(notes_per_frame.mean())          # ~1 for singing
    chord_frames    = float((notes_per_frame >= 3).mean())   # fraction of frames with chords
    unique_pitches  = int((roll.sum(axis=1) > 0).sum())
    density         = float(roll.sum() / (roll.shape[0]*roll.shape[1]))

    reasons = []
    if mean_polyphony < 1.2: reasons.append("mostly_monophonic")
    if chord_frames   < 0.05: reasons.append("few_chords")
    if unique_pitches < 8:    reasons.append("very_few_pitches")
    if density        < 0.01: reasons.append("very_sparse")

    ok = len(reasons) == 0
    return ok, reasons


# ----- PAGE CONFIG + CUSTOM STYLES ----- 
logo = Image.open("assets/images/logo.png")
st.set_page_config(
    page_title="AI-Powered Maestro Finder", 
    page_icon=logo,
    layout="wide")

# ----- Custom CSS ----- 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.markdown("""
<style>
  :root{
    /* card image size */
    --imgW: 450px;
    --imgH: 300px;

    /* inner padding around the image (match the cards) */
    --padX: 18px;     /* left/right */
    --padY: 18px;     /* top/bottom */

    /* full white frame (image + padding), exact same as cards */
    --frameW: calc(var(--imgW) + 2*var(--padX));
  }
  /* container that forces the frame to the right side */
  .hero-right{
    display: flex;
    justify-content: flex-end;   /* push to right */
    width: 100%;
  }

  body { font-family: 'Segoe UI', sans-serif; }
  .section-wrap { max-width: 1100px; margin: 0 auto; }

  .subheading {
      text-transform: uppercase; color: #1CB65D; font-size: 14px;
      font-weight: bold; letter-spacing: 1px; margin-bottom: 8px;
  }
  .headline {
      font-size: 30px; font-weight: 800; line-height: 1.2;
      margin-bottom: 0.5rem; color: #1c1c1c;
  }
  .description {
      font-size: 20px; line-height: 1.6; color: #555;
      max-width: 900px; margin-top: 10px;
  }
  a { text-decoration: underline; font-size: 16px; }

  /* Header image frame – identical look/width to cards */
  .img-frame {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
    padding: 18px;
    width: 475px !important;  /* image + padding */
    height: 336px !important; /* image + padding */
    box-sizing: border-box;
}
  .img-frame img {
    width: 450px !important;
    height: 300px !important;
    /*object-fit: cover;  /* crop instead of stretch */
    border-radius: 6px;
    display: block;
}

  /* Card container style (unchanged) */
  .card{
  background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.07);
  padding: var(--padY) var(--padX); border:2px solid transparent;
  height:100%; display:flex; flex-direction:column;
}
.card h4{
  font-size: clamp(16px, 1.6vw, 20px);
  line-height:1.25; margin:12px 0 8px; font-weight:700;
  /* reserve ~2 lines so all boxes align */
  min-height: 2.6em;
  display:-webkit-box; -webkit-box-orient:vertical; overflow:hidden; -webkit-line-clamp:2;
}
.card p{
  font-size: clamp(14px, 1.3vw, 16px);
  line-height:1.5; color:#444; margin:0;
  /* reserve ~3 lines */
  min-height: 4.5em;
  display:-webkit-box; -webkit-box-orient:vertical; overflow:hidden; -webkit-line-clamp:3;
}
  /* Ensure card images are exactly the same size as header image */
  .card img{
    width: var(--imgW);
    height: var(--imgH);
    object-fit: cover;
    border-radius: 6px;
    display: block;
    margin: 0 auto;
  }
</style>
""", unsafe_allow_html=True)


# ----- Plotly confidence pie ----- 
def plot_confidence_pie(pred_probs: dict):
    """
    pred_probs: dict like {"Bach": 0.62, "Mozart": 0.28, "Beethoven": 0.10}
    Renders a donut chart with nice formatting.
    """
    if not pred_probs:
        st.info("No probabilities to chart.")
        return

    labels = list(pred_probs.keys())
    values = [float(pred_probs[k]) for k in labels]

    # Normalize in case they don't sum to 1.0
    s = sum(values)
    if s > 0:
        values = [v / s for v in values]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,                         # donut
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent:.1%} (%{value:.3f})<extra></extra>",
                marker=dict(
                    colors=['#7E3FF2', '#38BDF8', '#22C55E', '#F59E0B', '#EF4444'],
                    line=dict(color='white', width=2)
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- WRAPPED LAYOUT TO CENTER THE CONTENT ----- 
left_pad, main_col, right_pad = st.columns([1, 6, 1])

with main_col:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{get_base64_image('assets/images/logo.png')}" 
                 alt="Logo" style="width: 60px; height: 60px; border-radius: 8px;">
            <h1 style="font-size: 50px; font-weight: 800; margin: 0;">
                AI-Powered Maestro Finder
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ----- HEADER SECTION  — tightened spacing ----- 
    col1, col2 = st.columns([7, 3], gap="small")  

    with col1:
        st.markdown('<div class="subheading">Shazam—but for classical composers</div>', unsafe_allow_html=True)
        st.markdown('<div class="headline">Identify composers with AI precision</div>', unsafe_allow_html=True)
        st.markdown("""
                <div class="description">
                  <p>
                    AI-Powered Maestro Finder instantly reveals the composer behind your MIDI files or piano recordings. Upload or record a snippet, 
                    and our advanced AI analyzes the music to predict whether Bach, Beethoven, Chopin, or Mozart composed it. Perfect for students, 
                    musicians, and enthusiasts seeking quick, accurate insights into classical masterpieces:
                  </p>
                  <ul>
                    <li>Works with <b>MIDI uploads</b> or <b>live mic recordings</b></li>
                    <li><b>Confidence bars</b> plus a piano-roll (and optional sheet music)</li>
                    <li>Fast, lightweight, and easy to extend with more composers</li>
                  </ul>
                </div>
        """, unsafe_allow_html=True)
        st.markdown("[![](https://img.shields.io/badge/GitHub%20-AI--Powered%20Maestro%20Finder-informational)](https://github.com/akthammomani/ai_powered_maestro_finder)")

    with col2:
        st.markdown("""<br>""", unsafe_allow_html=True)
        st.markdown("""<br>""", unsafe_allow_html=True)

        header_img_data = get_base64_image("assets/images/image_1.jpg")
        st.markdown(
            f"""
            <div class="hero-right">
            <div class="img-frame">
                <img src="data:image/jpeg;base64,{header_img_data}" alt="Piano" />
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("""---""")

    # ----- TOOL OVERVIEW SECTION ----- 
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown('<div class="subheading">AI Music Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline">Identify composers from MIDI or piano recordings</div>', unsafe_allow_html=True)

    tool_col1, tool_col2, tool_col3 = st.columns(3)

    card_style = """
                  <div class="card">
                    <div style="text-align:center;">
                      <img src="data:image/jpeg;base64,{img_data}" width="{img_width}" height="{img_height}" />
                    </div>
                    <h4 class="card-title">{title}</h4>
                    <p class="card-desc">{description}</p>
                  </div>
                """

    with tool_col1:
        st.markdown(card_style.format(
            img_data=get_base64_image("assets/images/image_2.jpg"),
            img_width="450", img_height="300",
            title="Composer identifier for MIDI files",
            description="Identify classical composers from your MIDI files with precision and ease."
        ), unsafe_allow_html=True)
    with tool_col2:
        st.markdown(card_style.format(
            img_data=get_base64_image("assets/images/image_3.jpg"),
            img_width="450", img_height="300",
            title="Real-time piano composer detection",
            description="Discover the composer behind live piano recordings quickly and accurately."
        ), unsafe_allow_html=True)
    with tool_col3:
        st.markdown(card_style.format(
            img_data=get_base64_image("assets/images/image_4.jpg"),
            img_width="450", img_height="300",
            title="Classical composer confidence scoring",
            description="Evaluate composer predictions with confidence scores and visual insights for deeper musical analysis."
        ), unsafe_allow_html=True)

    # ----- FILE UPLOAD / RECORD SECTION ----- 
    st.markdown("""---""")
    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    st.header("Upload your MIDI file or record live piano")

    st.info(
        "Please provide a clean solo piano recording (no background instruments or noise). "
        "Poor-quality or non-piano audio may result in failed or inaccurate transcription."
    )

    col_up, col_rec = st.columns(2, gap="large")

    uploaded_midi = None
    wav_path = None

    with col_up:
        st.markdown("<h4>Upload MIDI</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a .mid file", type=["mid", "midi"])

    with col_rec:
        st.markdown("<h4>Record Audio</h4>", unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record your audio snippet")

    # ----- Priority: uploaded MIDI > recorded audio ----- 
    if uploaded_file is not None:
        uploaded_midi = uploaded_file
    elif recorded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(recorded_audio.getbuffer())
            wav_path = tmp_audio.name
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_midi:
                midi_out = tmp_midi.name
            with st.spinner("Transcribing audio → MIDI..."):
                convert_audio_to_midi(wav_path, midi_out)
            uploaded_midi = midi_out  # path string
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            uploaded_midi = None
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    st.markdown('</div>', unsafe_allow_html=True) 

    # ----- INFERENCE ----- 
    if uploaded_midi:
        with st.spinner("Analyzing composition..."):
            if isinstance(uploaded_midi, str):
                midi_path = uploaded_midi
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_midi:
                    tmp_midi.write(uploaded_midi.read())
                    midi_path = tmp_midi.name
    
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)

                # does this look like solo piano?
                ok, reasons = piano_likeness_flags(pm, fs=8)
                if not ok:
                    st.warning(
                        "This clip doesn’t look like solo piano (likely voice/whistle). "
                        f"Reasons: {', '.join(reasons)}. Prediction may be unreliable."
                    )
                # *stop* here:
                    st.stop()
    
                # densest 512 frames, (88,512) in 0..127
                pr = extract_best_512(pm, fs=10, window=512)
                #st.write({
                    #"roll_shape": pr.shape,
                    #"nonzero_frac": float(np.count_nonzero(pr) / pr.size),
                  #  "max": int(pr.max()),
                #})
                # DEBUG: raw probs and predicted label (prettiest check)
                #raw = MODEL.predict(_prep_roll(pr), verbose=0)[0]
                #st.write({"probs": np.round(raw, 4).tolist(), "sum": float(raw.sum())})
                #pred_idx = int(np.argmax(raw, axis=-1))
               # st.write({"predicted_label": COMPOSERS[pred_idx]})
    
                if not is_valid_piano_midi(midi_path):
                    st.warning(
                        "The MIDI appears too short or sparse. "
                        "Please try a clearer solo piano clip."
                    )
                else:
                    pred_probs, viz_roll = predict_composer(pr)  # viz_roll: (88,512)
                    #st.write("Softmax:", list(pred_probs.items()))
    
                    pie_col, viz_col = st.columns([1, 1], gap="large")

                    with pie_col:
                        st.subheader("Confidence")
                        st.markdown("""<br>""", unsafe_allow_html=True)
                        plot_confidence_bars(pred_probs)
                    
                    with viz_col:
                        st.subheader("Visualization")
                        tab_roll, tab_sheet = st.tabs(["Piano-roll", "Sheet music"])
                    
                        with tab_roll:
                            plot_pianoroll_plotly_clean(viz_roll)
                    
                        with tab_sheet:
                            with st.spinner("Rendering Sheet Music…"):
                                try:
                                    # keep it light; adjust height as you like
                                    xml = midi_to_musicxml_str(midi_path)
                                    render_musicxml_osmd(xml, height=320, compact=True)
                                except Exception as e:
                                    st.warning(f"Couldn’t render sheet music: {e}")
    
            except Exception as e:
                st.error(f"Failed to analyze MIDI: {e}")
            finally:
                try:
                    os.remove(midi_path)
                except Exception:
                    pass

# ----- Footer: Contact form + links ----- 

with st.container():
    # keep everything visually centered a bit
    _padL, mid, _padR = st.columns([2, 12, 2])
    with mid:
        st.divider()
        with st.expander("Leave Us a Comment or Question"):
            contact_form = """
                <form action=https://formsubmit.co/aktham.momani81@gmail.com method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder="Your name" required>
                    <input type="email" name="email" placeholder="Your email" required>
                    <textarea name="message" placeholder="Your message here"></textarea>
                    <button type="submit">Send</button>
                </form>
            """
            st.markdown(contact_form, unsafe_allow_html=True)

            # Use Local CSS File
            local_css("style.css")

        # ----- Contacts / badges row ----- 
        with mid:
            st.markdown(
                """
                ### Contacts
                [![](https://img.shields.io/badge/GitHub-Follow-informational)](https://github.com/akthammomani)
                [![](https://img.shields.io/badge/LinkedIn-Connect-informational)](https://www.linkedin.com/in/akthammomani/)
                [![](https://img.shields.io/badge/Open%20an-Issue-informational)](https://github.com/akthammomani/ai_powered_maestro_finder/issues)
                [![MAIL Badge](https://img.shields.io/badge/-aktham.momani81@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:aktham.momani81@gmail.com)](mailto:aktham.momani81@gmail.com)
                
                ###### © Aktham Momani, 2025. All rights reserved.
                """,
                unsafe_allow_html=True,
            )
