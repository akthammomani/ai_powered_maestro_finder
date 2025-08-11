from pathlib import Path
import base64, tempfile
from music21 import converter
import streamlit as st, base64, uuid

def midi_to_musicxml_str(midi_path: str) -> str:
    """Convert a MIDI file to MusicXML text using music21."""
    s = converter.parse(midi_path)
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        out_path = tmp.name
    s.write("musicxml", fp=out_path)
    xml = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    try:
        Path(out_path).unlink(missing_ok=True)
    except Exception:
        pass
    return xml

def render_musicxml_osmd(xml_str: str, height: int = 620, compact: bool = True):
    import streamlit as st, base64, uuid
    from string import Template

    uid  = "osmd_" + uuid.uuid4().hex
    mode = "compact" if compact else "default"
    b64  = base64.b64encode(xml_str.encode("utf-8")).decode("ascii")

    html = Template(r"""
<div id="${uid}-wrap" style="width:250%;">
  <div style="display:flex; gap:8px; align-items:center; margin:0 0 8px;">
    <button id="${uid}-save-svg">Save SVG</button>
    <button id="${uid}-save-png">Save PNG</button>
  </div>
  <div id="${uid}" style="width:250%;"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.4/build/opensheetmusicdisplay.min.js"></script>
<script>
  const el  = document.getElementById("${uid}");
  const xml = atob("${b64}");

  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(el, {
    autoResize: false,
    backend: "svg"
  });
  osmd.setOptions({
    drawingParameters: "${mode}",
    drawPartNames: false,
    drawTitle: false,
    pageFormat: "Endless"
  });

  let refined = false;

  function fitAndWiden() {
    const svg = el.querySelector("svg");
    if (!svg) return;

    // container width (minus tiny gutter & padding)
    const cs   = getComputedStyle(el);
    const pad  = (parseFloat(cs.paddingLeft)||0) + (parseFloat(cs.paddingRight)||0);
    const colW = Math.max(0, el.clientWidth - pad - 0);

    // current drawn pixel width of the music
    const curPx = svg.getBoundingClientRect().width;

    if (colW > 0 && curPx > 0) {
      const factor = Math.max(0.5, Math.min(colW / curPx, 5)); // scale to fill column

      // Prefer widening measures (content width), else fall back to zoom
      let usedMeasureFactor = false;
      try {
        if (osmd.sheet && typeof osmd.sheet.MeasureWidthFactor !== "undefined") {
          osmd.sheet.MeasureWidthFactor = factor;
          // reclaim a bit of margin space
          osmd.rules.PageLeftMargin  = 2.0;
          osmd.rules.PageRightMargin = 2.0;
          usedMeasureFactor = true;
        }
      } catch (e) {}

      if (!usedMeasureFactor) {
        osmd.zoom = Math.max(0.05, Math.min(osmd.zoom * factor, 5));
      }
      osmd.render();
    }

    // Let CSS own the final SVG size (prevents future loops)
    const finalSVG = el.querySelector("svg");
    if (finalSVG) {
      finalSVG.removeAttribute("width");
      finalSVG.removeAttribute("height");
      finalSVG.style.width  = "100%";
      finalSVG.style.height = "auto";
      finalSVG.style.display = "block";
    }

    // One small refinement pass helps if fonts finished loading late
    if (!refined) { refined = true; setTimeout(fitAndWiden, 120); }
  }

  function startWhenVisible() {
    const w = el.clientWidth;
    if (!w || w < 240) { requestAnimationFrame(startWhenVisible); return; }
    osmd.load(xml).then(() => { osmd.render(); fitAndWiden(); });
  }

  // Render only after the element is actually visible (tabs are hidden initially)
  const io = new IntersectionObserver((entries, obs) => {
    if (entries.some(e => e.isIntersecting)) { obs.disconnect(); startWhenVisible(); }
  }, { threshold: 0.1 });
  io.observe(el);

  // ----- Save buttons -----
  function dl(name, blob) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  document.getElementById("${uid}-save-svg").onclick = () => {
    const svg = el.querySelector("svg"); if (!svg) return;
    const s = new XMLSerializer().serializeToString(svg);
    dl("score.svg", new Blob([s], {type: "image/svg+xml;charset=utf-8"}));
  };
  document.getElementById("${uid}-save-png").onclick = () => {
    const svg = el.querySelector("svg"); if (!svg) return;
    const s = new XMLSerializer().serializeToString(svg);
    const vb = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : null;
    const w  = vb && vb.width  ? vb.width  : svg.getBBox().width;
    const h  = vb && vb.height ? vb.height : svg.getBBox().height;
    const img = new Image();
    img.onload = () => {
      const scale = 2;
      const c = document.createElement("canvas");
      c.width  = Math.max(1, Math.round(w * scale));
      c.height = Math.max(1, Math.round(h * scale));
      const ctx = c.getContext("2d");
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.drawImage(img, 0, 0);
      c.toBlob(b => dl("score.png", b), "image/png");
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(s);
  };
</script>
""").substitute(uid=uid, b64=b64, mode=mode)

    st.components.v1.html(html, height=height, scrolling=False)
