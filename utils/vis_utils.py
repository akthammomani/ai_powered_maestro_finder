import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# a consistent color palette for your app
_PIE_COLORS = ['#7E3FF2', '#38BDF8', '#22C55E', '#F59E0B', '#EF4444']

# tiny helper so we don't depend on pretty_midi just to label C-notes
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_name(n: int) -> str:
    octave = (n // 12) - 1
    return f"{_NOTE_NAMES[n % 12]}{octave}"

def plot_confidence_pie(pred_probs: dict, height: int = 340):
    """
    Plot a donut (pie) of composer probabilities using Plotly.
    pred_probs: dict, e.g. {"Bach":0.62,"Mozart":0.2,"Beethoven":0.1,"Chopin":0.08}
    """
    if not pred_probs:
        st.info("No probabilities to chart.")
        return

    labels = list(pred_probs.keys())
    values = [float(pred_probs[k]) for k in labels]
    s = sum(values)
    if s > 0:
        values = [v / s for v in values]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{percent:.1%} (%{value:.3f})<extra></extra>",
            marker=dict(colors=_PIE_COLORS[:len(labels)], line=dict(color='white', width=2)),
            sort=False
        )]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=height, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_pianoroll_plotly_clean(pr: np.ndarray):
    """
    pr: numpy array (88, T) with velocities 0..127.
        Row 0 corresponds to MIDI 21 (A0), row 87 to MIDI 108 (C8).
    """
    rows, T = pr.shape  # rows=88
    assert rows == 88, f"Expected 88 pitch rows, got {rows}"

    # Build the image: origin='lower' means row 0 appears at the bottom.
    fig = px.imshow(
        pr,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=0, zmax=127
    )

    # We only show ticks for C notes (MIDI 24 = C1 ... MIDI 108 = C8)
    midi_low = 21  # row 0 -> MIDI 21
    c_midis = list(range(24, 109, 12))  # 24, 36, 48, 60, 72, 84, 96, 108
    tick_vals = [m - midi_low for m in c_midis]              # row indices for those C's
    tick_text = [midi_to_name(m) for m in c_midis]           # 'C1'..'C8'

    fig.update_yaxes(
        title="Pitch",
        tickmode="array",    # <-- only use these ticks
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=False,
        zeroline=False
    )
    fig.update_xaxes(
        title="Time frames",
        showgrid=False,
        zeroline=False
    )

    # Clean layout: single title only, no inner subplot title
    fig.update_layout(
        #title="Piano-roll Visualization (88 × 512)",
        coloraxis_colorbar=dict(title="Velocity"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
