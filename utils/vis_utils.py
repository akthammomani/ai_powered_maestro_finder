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

def plot_confidence_bars(pred_probs: dict, inside_threshold: float = 0.98):
    """
    Horizontal confidence bars with smart labels:
    - If prob >= inside_threshold → put the % label INSIDE the bar (right end)
    - else → put the % label OUTSIDE to the right
    """
    if not pred_probs:
        st.info("No probabilities to chart.")
        return

    # sort by prob desc
    labels, vals = zip(*sorted(pred_probs.items(), key=lambda kv: kv[1], reverse=False))
    vals = np.array(vals, dtype=float)
    s = vals.sum()
    if s > 0:
        vals = vals / s

    texts = [f"{v*100:.1f}%" for v in vals]
    # per-bar text position
    pos = ["inside" if v >= inside_threshold else "outside" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=list(labels),
        orientation="h",
        text=texts,
        textposition=pos,            # array of positions
        insidetextanchor="end",      # align inside labels to the right end
        cliponaxis=False,
        marker=dict(line=dict(color="white", width=1))
    ))
    fig.update_traces(
        insidetextfont=dict(color="white"),
        outsidetextfont=dict(color="#111"),
        hovertemplate="<b>%{y}</b><br>%{x:.2%}<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1.02], fixedrange=True),  # tiny headroom
        yaxis=dict(title=None, tickfont=dict(size=14), automargin=True),
        margin=dict(l=0, r=10, t=10, b=0),
        #height=40 * len(labels) + 40,
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_pianoroll_plotly_clean(pr: np.ndarray):
    """
    Display a piano-roll. Accepts (88, T) or (T, 88).
    Values may be 0..127 or 0..1; we auto-handle both.
    """
    if pr is None:
        st.info("No piano-roll to display.")
        return

    arr = np.asarray(pr, dtype=float)

    # Fix orientation if needed: expect (88, T)
    if arr.ndim != 2:
        st.error(f"Expected 2D array, got {arr.shape}")
        return
    if arr.shape[0] != 88 and arr.shape[1] == 88:
        arr = arr.T  # -> (88, T)

    if arr.shape[0] != 88:
        st.error(f"Expected 88 pitch rows, got {arr.shape[0]}")
        return

    # If values look normalized, scale to 0..127 for better contrast
    vmax = float(arr.max()) if arr.size else 0.0
    if vmax == 0.0:
        st.warning("Selected window has no non-zero velocities.")
        return
    if vmax <= 1.01:
        arr *= 127.0
        vmax = float(arr.max())

    # Clip and choose a contrast-friendly zmax
    arr = np.clip(arr, 0, 127)
    zmax = max(20.0, vmax)  # keep contrast even for soft notes

    # Build y-axis ticks only on C notes
    _NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def midi_to_name(n: int) -> str:
        return f"{_NOTE_NAMES[n % 12]}{(n // 12) - 1}"

    midi_low = 21
    c_midis  = list(range(24, 109, 12))  # C1..C8
    tick_vals = [m - midi_low for m in c_midis]
    tick_text = [midi_to_name(m) for m in c_midis]

    fig = px.imshow(
        arr,
        origin="lower",
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=0, zmax=zmax
    )
    fig.update_yaxes(
        title="Pitch",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=False, zeroline=False
    )
    fig.update_xaxes(title="Time frames", showgrid=False, zeroline=False)
    fig.update_layout(
        coloraxis_colorbar=dict(title="Velocity"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})





