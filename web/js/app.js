/** DGene web compiler — landing → split workspace (Claude-style artifact pane). */

const $ = (id) => document.getElementById(id);

function cleanSeq(s) {
  return String(s || "")
    .replace(/\s/g, "")
    .toUpperCase()
    .replace(/[^ACGT]/g, "");
}

function gcPercent(seq) {
  const L = seq.length;
  if (!L) return 0;
  const g = (seq.match(/G/g) || []).length + (seq.match(/C/g) || []).length;
  return (100 * g) / L;
}

function wallaceTm(seq) {
  const at = (seq.match(/A/g) || []).length + (seq.match(/T/g) || []).length;
  const gc = (seq.match(/G/g) || []).length + (seq.match(/C/g) || []).length;
  return 2 * at + 4 * gc;
}

function toFasta(seq, name = "compiled_sequence") {
  const w = 80;
  const lines = [];
  for (let i = 0; i < seq.length; i += w) lines.push(seq.slice(i, i + w));
  return `>${name}\n${lines.join("\n")}\n`;
}

function toGenbank(seq, name = "compiled_sequence") {
  const d = new Date();
  const months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
  ];
  const dateStr = `${String(d.getUTCDate()).padStart(2, "0")}-${months[d.getUTCMonth()]}-${d.getUTCFullYear()}`;
  const locus = `LOCUS       ${name.slice(0, 16).padEnd(16)}${String(seq.length).padStart(11)} bp    DNA     circular SYN ${dateStr.toUpperCase()}`;
  const def = "DEFINITION  Synthetic compiler generated DNA construct.";
  const acc = "ACCESSION   .";
  const ver = "VERSION     .";
  const src = "SOURCE      Synthetic DNA construct";
  const org = "  ORGANISM  Synthetic DNA construct";
  const feat = `FEATURES             Location/Qualifiers\n     source          1..${seq.length}\n                     /organism="Synthetic DNA construct"`;
  let origin = "ORIGIN\n";
  const low = seq.toLowerCase();
  for (let start = 0; start < low.length; start += 60) {
    const chunk = low.slice(start, start + 60);
    const groups = [];
    for (let j = 0; j < chunk.length; j += 10) groups.push(chunk.slice(j, j + 10));
    origin += `${String(start + 1).padStart(9)} ${groups.join(" ")}\n`;
  }
  origin += "//\n";
  return [locus, def, acc, ver, src, org, feat, origin].join("\n");
}

function downloadText(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

const PLASMID_GEOM = {
  cx: 300,
  cy: 300,
  rBackboneInner: 195,
  rBackboneOuter: 205,
  rFeatureInner: 178,
  rFeatureOuter: 222,
  rTickInner: 205,
  rTickOuterMinor: 211,
  rTickOuterMajor: 215,
  rTickLabel: 226,
  rSiteTickInner: 222,
  rSiteTickOuter: 240,
  rSiteLabel: 250,
  rLeaderInner: 222,
  rLeaderElbow: 252,
  rLeaderTextX: 262,
};

/** Build a feature ribbon path with an arrowhead at the strand-end.
 *  Coordinates are in the rotated plasmid space (origin at center). */
function featureArrowPath(rIn, rOut, a0, a1, strand) {
  if (a1 <= a0) return "";
  const sweep = a1 - a0;
  const arrowAngle = Math.min(sweep * 0.45, 0.18);
  const headIn = strand >= 0 ? a1 - arrowAngle : a0 + arrowAngle;
  const a0Body = a0;
  const a1Body = strand >= 0 ? a1 - arrowAngle : a0 + arrowAngle;
  const large = a1Body - a0Body > Math.PI ? 1 : 0;

  const cos = Math.cos;
  const sin = Math.sin;
  const x0o = rOut * cos(a0Body);
  const y0o = rOut * sin(a0Body);
  const x1o = rOut * cos(a1Body);
  const y1o = rOut * sin(a1Body);
  const x1i = rIn * cos(a1Body);
  const y1i = rIn * sin(a1Body);
  const x0i = rIn * cos(a0Body);
  const y0i = rIn * sin(a0Body);

  const rTip = (rIn + rOut) / 2;
  const aTip = strand >= 0 ? a1 : a0;
  const xTip = rTip * cos(aTip);
  const yTip = rTip * sin(aTip);

  const rArrowOut = rOut + (rOut - rIn) * 0.18;
  const rArrowIn = rIn - (rOut - rIn) * 0.18;

  if (strand >= 0) {
    const xWingOut = rArrowOut * cos(headIn);
    const yWingOut = rArrowOut * sin(headIn);
    const xWingIn = rArrowIn * cos(headIn);
    const yWingIn = rArrowIn * sin(headIn);
    return [
      `M ${x0o} ${y0o}`,
      `A ${rOut} ${rOut} 0 ${large} 1 ${x1o} ${y1o}`,
      `L ${xWingOut} ${yWingOut}`,
      `L ${xTip} ${yTip}`,
      `L ${xWingIn} ${yWingIn}`,
      `L ${x1i} ${y1i}`,
      `A ${rIn} ${rIn} 0 ${large} 0 ${x0i} ${y0i}`,
      "Z",
    ].join(" ");
  } else {
    const xWingOut = rArrowOut * cos(headIn);
    const yWingOut = rArrowOut * sin(headIn);
    const xWingIn = rArrowIn * cos(headIn);
    const yWingIn = rArrowIn * sin(headIn);
    return [
      `M ${xTip} ${yTip}`,
      `L ${xWingOut} ${yWingOut}`,
      `L ${x0o} ${y0o}`,
      `A ${rOut} ${rOut} 0 ${large} 1 ${x1o} ${y1o}`,
      `L ${x1i} ${y1i}`,
      `A ${rIn} ${rIn} 0 ${large} 0 ${x0i} ${y0i}`,
      `L ${xWingIn} ${yWingIn}`,
      "Z",
    ].join(" ");
  }
}

const ENZYMES = {
  EcoRI: "GAATTC",
  BamHI: "GGATCC",
  HindIII: "AAGCTT",
  NdeI: "CATATG",
  XhoI: "CTCGAG",
  SpeI: "ACTAGT",
  PstI: "CTGCAG",
  SalI: "GTCGAC",
  NcoI: "CCATGG",
  KpnI: "GGTACC",
  XbaI: "TCTAGA",
  NotI: "GCGGCCGC",
};

function findRestrictionSites(sequence) {
  const sites = [];
  for (const [name, motif] of Object.entries(ENZYMES)) {
    let i = -1;
    while ((i = sequence.indexOf(motif, i + 1)) !== -1) {
      sites.push({ name, position: i + 1 });
      if (sites.length > 16) return sites;
    }
  }
  sites.sort((a, b) => a.position - b.position);
  return sites;
}

/** Heuristic feature catalog: realistic SnapGene-style segments scaled to length.
 *  Replace this with model-emitted features when available. */
function inferFeatures(L) {
  const catalog = [
    { label: "J23100",   sub: "promoter",   pStart: 0.00, pEnd: 0.16, strand: +1, color: "#22c55e" },
    { label: "lacO",     sub: "operator",   pStart: 0.16, pEnd: 0.23, strand: +1, color: "#f59e0b" },
    { label: "B0034",    sub: "RBS",        pStart: 0.23, pEnd: 0.29, strand: +1, color: "#a855f7" },
    { label: "sfGFP",    sub: "CDS",        pStart: 0.29, pEnd: 0.86, strand: +1, color: "#0ea5e9" },
    { label: "B0015",    sub: "terminator", pStart: 0.86, pEnd: 1.00, strand: -1, color: "#ef4444" },
  ];
  return catalog.map((f) => {
    const start = Math.max(1, Math.round(f.pStart * L) + (f.pStart === 0 ? 0 : 1));
    const end = Math.max(start + 1, Math.round(f.pEnd * L));
    return { ...f, start, end };
  });
}

function bpToAngle(bp, L) {
  return (2 * Math.PI * bp) / L;
}

/** Convert (radius, bp) in unrotated viewport space (with -90deg offset so 0 bp is at top). */
function polarToXY(r, bp, L) {
  const a = bpToAngle(bp, L) - Math.PI / 2;
  return { x: PLASMID_GEOM.cx + r * Math.cos(a), y: PLASMID_GEOM.cy + r * Math.sin(a) };
}

function tickStep(L) {
  if (L <= 150) return { minor: 5, major: 25 };
  if (L <= 400) return { minor: 10, major: 50 };
  if (L <= 1200) return { minor: 25, major: 100 };
  if (L <= 4000) return { minor: 100, major: 500 };
  if (L <= 12000) return { minor: 250, major: 1000 };
  return { minor: 500, major: 2500 };
}

function renderTicks(L) {
  const ticks = $("plasmidTicks");
  ticks.innerHTML = "";
  const { minor, major } = tickStep(L);
  for (let bp = 0; bp < L; bp += minor) {
    const isMajor = bp % major === 0;
    const r0 = PLASMID_GEOM.rTickInner;
    const r1 = isMajor ? PLASMID_GEOM.rTickOuterMajor : PLASMID_GEOM.rTickOuterMinor;
    const a = bp;
    const p0 = polarToXY(r0, a, L);
    const p1 = polarToXY(r1, a, L);
    ticks.appendChild(svgEl("line", {
      x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y,
      class: isMajor ? "plasmid-tick-major" : "plasmid-tick",
    }));
    if (isMajor && bp !== 0) {
      const lp = polarToXY(PLASMID_GEOM.rTickLabel, a, L);
      const txt = svgEl("text", {
        x: lp.x, y: lp.y + 3, "text-anchor": "middle",
        class: "plasmid-tick-label",
      });
      txt.textContent = String(bp);
      ticks.appendChild(txt);
    }
  }
}

function renderRestrictionSites(sites, L, tip, frame) {
  const g = $("plasmidSites");
  g.innerHTML = "";
  for (const site of sites) {
    const a = site.position;
    const p0 = polarToXY(PLASMID_GEOM.rSiteTickInner, a, L);
    const p1 = polarToXY(PLASMID_GEOM.rSiteTickOuter, a, L);
    const lp = polarToXY(PLASMID_GEOM.rSiteLabel, a, L);
    g.appendChild(svgEl("line", {
      x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y, class: "plasmid-site-tick",
    }));
    const angle = bpToAngle(a, L) - Math.PI / 2;
    const deg = (angle * 180) / Math.PI;
    const rotateDeg = deg > 90 || deg < -90 ? deg + 180 : deg;
    const anchor = deg > 90 || deg < -90 ? "end" : "start";
    const txt = svgEl("text", {
      x: lp.x, y: lp.y,
      transform: `rotate(${rotateDeg} ${lp.x} ${lp.y})`,
      "text-anchor": anchor,
      "dominant-baseline": "middle",
      class: "plasmid-site-label",
    });
    txt.textContent = `${site.name} (${site.position})`;
    txt.addEventListener("mouseenter", (ev) => {
      tip.hidden = false;
      tip.innerHTML = `<strong>${site.name}</strong><br/>cut site · ${site.position} bp`;
      moveTooltipAt(tip, frame, ev);
    });
    txt.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
    txt.addEventListener("mouseleave", () => { tip.hidden = true; });
    g.appendChild(txt);
  }
}

function moveTooltipAt(tip, frame, ev) {
  const rect = frame.getBoundingClientRect();
  tip.style.left = `${ev.clientX - rect.left + 12}px`;
  tip.style.top = `${ev.clientY - rect.top + 12}px`;
}

function renderFeaturesAndLabels(features, L, tip, frame) {
  const arcs = $("plasmidRotate");
  const labels = $("plasmidLabels");
  arcs.innerHTML = "";
  labels.innerHTML = "";

  const { rFeatureInner: rIn, rFeatureOuter: rOut } = PLASMID_GEOM;

  for (const f of features) {
    const a0 = bpToAngle(f.start - 1, L);
    const a1 = bpToAngle(f.end, L);
    if (a1 <= a0) continue;

    const d = featureArrowPath(rIn, rOut, a0, a1, f.strand);
    const path = svgEl("path", {
      d, fill: f.color, class: "plasmid-seg",
    });
    arcs.appendChild(path);

    const midBp = (f.start - 1 + f.end) / 2;
    const midA = bpToAngle(midBp, L) - Math.PI / 2;
    const onRight = Math.cos(midA) >= 0;

    const pStart = polarToXY(PLASMID_GEOM.rLeaderInner, midBp, L);
    const pElbow = polarToXY(PLASMID_GEOM.rLeaderElbow, midBp, L);
    const labelX = onRight ? PLASMID_GEOM.cx + PLASMID_GEOM.rLeaderTextX : PLASMID_GEOM.cx - PLASMID_GEOM.rLeaderTextX;
    const pText = { x: labelX, y: pElbow.y };

    labels.appendChild(svgEl("path", {
      d: `M ${pStart.x} ${pStart.y} L ${pElbow.x} ${pElbow.y} L ${pText.x} ${pText.y}`,
      class: "plasmid-leader",
    }));

    const anchor = onRight ? "start" : "end";
    const dx = onRight ? 4 : -4;
    const tName = svgEl("text", {
      x: pText.x + dx, y: pText.y - 2,
      "text-anchor": anchor,
      class: "plasmid-feature-label",
    });
    tName.textContent = f.label;
    labels.appendChild(tName);

    const tSub = svgEl("text", {
      x: pText.x + dx, y: pText.y + 12,
      "text-anchor": anchor,
      class: "plasmid-feature-sub",
    });
    tSub.textContent = `${f.sub} · ${f.start}–${f.end}`;
    labels.appendChild(tSub);

    const onEnter = (ev) => {
      document.querySelectorAll(".plasmid-seg.is-active").forEach((el) => el.classList.remove("is-active"));
      path.classList.add("is-active");
      tip.hidden = false;
      tip.innerHTML = `<strong>${f.label}</strong> · ${f.sub}<br/>${f.start}–${f.end} bp · ${f.strand >= 0 ? "+" : "−"} strand`;
      moveTooltipAt(tip, frame, ev);
    };
    path.addEventListener("mouseenter", onEnter);
    path.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
    path.addEventListener("mouseleave", () => {
      path.classList.remove("is-active");
      tip.hidden = true;
    });
  }
}

function renderPlasmid(sequence, name = "construct") {
  const tip = $("featureTooltip");
  const frame = $("vizFrame");
  const bpEl = $("bpLabel");
  const nameEl = $("centerName");
  const L = sequence.length;

  nameEl.textContent = name;
  if (L < 40) {
    bpEl.textContent = "— bp";
    $("plasmidRotate").innerHTML = "";
    $("plasmidLabels").innerHTML = "";
    $("plasmidTicks").innerHTML = "";
    $("plasmidSites").innerHTML = "";
    return;
  }
  bpEl.textContent = `${L} bp`;

  const features = inferFeatures(L);
  const sites = findRestrictionSites(sequence);

  renderTicks(L);
  renderFeaturesAndLabels(features, L, tip, frame);
  renderRestrictionSites(sites, L, tip, frame);
}

function attachPlasmidNav(svg, viewport) {
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dragging = false;
  let lx = 0;
  let ly = 0;
  const cx = PLASMID_GEOM.cx;
  const cy = PLASMID_GEOM.cy;

  function apply() {
    viewport.setAttribute(
      "transform",
      `translate(${cx + tx},${cy + ty}) scale(${scale}) translate(${-cx},${-cy})`
    );
  }

  svg.addEventListener(
    "wheel",
    (e) => {
      // Only zoom on Cmd/Ctrl+scroll so normal page scrolling still works.
      if (!(e.metaKey || e.ctrlKey)) return;
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0015);
      scale = Math.min(4, Math.max(0.55, scale * factor));
      apply();
    },
    { passive: false }
  );

  svg.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    tx += e.clientX - lx;
    ty += e.clientY - ly;
    lx = e.clientX;
    ly = e.clientY;
    apply();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
}

async function streamThought(el, text, msPerWord = 38) {
  const words = text.split(/\s+/).filter(Boolean);
  el.textContent = "";
  for (let i = 0; i < words.length; i++) {
    el.textContent += (i ? " " : "") + words[i];
    await new Promise((r) => setTimeout(r, msPerWord));
  }
}

function showWorkspace() {
  $("landing").hidden = true;
  $("workspace").hidden = false;
  document.body.classList.add("in-workspace");
}

function showLanding() {
  $("landing").hidden = false;
  $("workspace").hidden = true;
  document.body.classList.remove("in-workspace");
  $("statusHint").textContent = "⌘ or Ctrl + Enter to compile";
  $("userBubble").hidden = true;
  $("thoughtText").textContent = "";
  const badge = $("streamBadge");
  badge.textContent = "…";
  badge.classList.remove("done");
  $("seqPre").textContent = "";
  $("bpLabel").textContent = "— bp";
  $("centerName").textContent = "construct";
  $("plasmidRotate").innerHTML = "";
  $("plasmidLabels").innerHTML = "";
  $("plasmidTicks").innerHTML = "";
  $("plasmidSites").innerHTML = "";
  $("metrics").hidden = true;
  $("seqCard").hidden = true;
  $("passesCard").hidden = true;
  $("candidatesCard").hidden = true;
  $("paretoSection").hidden = true;
  $("passesList").innerHTML = "";
  $("candidatesList").innerHTML = "";
  $("paretoPoints").innerHTML = "";
  $("paretoAxes").innerHTML = "";
  const extrasMeta = document.getElementById("extrasMeta");
  if (extrasMeta) extrasMeta.textContent = "";
  $("mLen").textContent = "—";
  $("mGc").textContent = "—";
  $("mTm").textContent = "—";
  state.candidates = [];
  state.selectedId = null;
  lastSequence = "";
}

function setModelLabel(model) {
  const m = model || "mock";
  $("modelPill").textContent = `Model: ${m}`;
  $("modelPillNav").textContent = `Model · ${m}`;
}

let lastSequence = "";

const state = {
  candidates: [],
  selectedId: null,
};

const PASS_GLYPH = { ok: "✓", warn: "!", error: "✕" };

function getCandidate(id) {
  return state.candidates.find((c) => c.id === id) || null;
}

function passMetricFor(passes, passId, key = "metric_raw") {
  const p = passes.find((x) => x.pass_id === passId);
  return p ? p[key] : null;
}

/** Render the compiler-passes log. animate=true reveals one row at a time. */
function renderPasses(passes, { animate } = { animate: false }) {
  const list = $("passesList");
  list.innerHTML = "";

  const counts = passes.reduce(
    (acc, p) => ((acc[p.status] = (acc[p.status] || 0) + 1), acc),
    {}
  );
  const totalMs = passes.reduce((s, p) => s + (p.duration_ms || 0), 0);
  $("passesCardMeta").textContent =
    `${passes.length} passes · ${(counts.ok || 0)}✓ ${(counts.warn || 0)}! ${(counts.error || 0)}✕ · ${totalMs.toFixed(1)} ms`;

  passes.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "pass-row";
    li.dataset.status = p.status;
    li.style.animationDelay = animate ? `${i * 90}ms` : "0ms";

    li.innerHTML = `
      <span class="pass-icon" aria-hidden="true">${PASS_GLYPH[p.status] || "·"}</span>
      <div class="pass-body">
        <span class="pass-name">${escapeHtml(p.pass_id)}</span>
        <span class="pass-summary">${escapeHtml(p.summary || p.name)}</span>
      </div>
      <span class="pass-duration">${(p.duration_ms || 0).toFixed(1)}ms</span>
    `;

    if (p.diagnostics && p.diagnostics.length) {
      li.addEventListener("click", () => {
        const opened = li.classList.toggle("is-open");
        const existing = li.querySelector(".pass-diagnostics");
        if (existing) existing.remove();
        if (opened) {
          const div = document.createElement("div");
          div.className = "pass-diagnostics";
          div.innerHTML = p.diagnostics
            .map((d) => {
              const pos =
                d.start != null
                  ? ` <span class="pass-diag-pos">@ ${d.start}${d.end && d.end !== d.start ? `–${d.end}` : ""} bp</span>`
                  : "";
              return `<div class="pass-diag" data-sev="${d.severity}">${escapeHtml(d.message)}${pos}</div>`;
            })
            .join("");
          li.appendChild(div);
        }
      });
    } else {
      li.style.cursor = "default";
    }

    list.appendChild(li);
  });

  $("passesCard").hidden = false;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderCandidates(candidates, selectedId) {
  const list = $("candidatesList");
  list.innerHTML = "";
  const paretoCount = candidates.filter((c) => c.is_pareto).length;
  $("candidatesCardMeta").textContent = `${candidates.length} generated · ${paretoCount} on Pareto front`;

  for (const c of candidates) {
    const li = document.createElement("li");
    li.className = "candidate-row";
    if (c.is_pareto) li.classList.add("is-pareto");
    if (c.id === selectedId) li.classList.add("is-selected");
    li.dataset.id = c.id;

    const expr = (c.scores.expression * 100).toFixed(0);
    const burden = ((1 - c.scores.low_burden) * 100).toFixed(0);
    const gc = (c.scores.gc_balance * 100).toFixed(0);
    const clean = (c.scores.cleanliness * 100).toFixed(0);

    li.innerHTML = `
      <span class="candidate-pareto" title="${c.is_pareto ? "Pareto-optimal" : "Dominated"}">${c.is_pareto ? "★" : "○"}</span>
      <div class="candidate-meta">
        <span class="candidate-strategy">${escapeHtml(c.strategy_name || c.id)}</span>
        <span class="candidate-scoreline">
          <span>exp <em>${expr}</em></span>
          <span>burden <em>${burden}</em></span>
          <span>gc <em>${gc}</em></span>
          <span>clean <em>${clean}</em></span>
        </span>
      </div>
      <span class="candidate-composite" title="Composite score">${c.scores.composite.toFixed(2)}</span>
    `;

    li.addEventListener("click", () => selectCandidate(c.id));
    list.appendChild(li);
  }
  $("candidatesCard").hidden = false;
}

/** Render the Pareto chart: x = expression, y = low_burden, both in [0,1]. */
function renderParetoChart(candidates, selectedId) {
  const axes = $("paretoAxes");
  const points = $("paretoPoints");
  axes.innerHTML = "";
  points.innerHTML = "";

  const W = 480, H = 320;
  const PAD = { l: 50, r: 28, t: 22, b: 44 };
  const px = (x) => PAD.l + x * (W - PAD.l - PAD.r);
  const py = (y) => H - PAD.b - y * (H - PAD.t - PAD.b);

  // grid + axes
  for (let t = 0; t <= 1.0001; t += 0.25) {
    axes.appendChild(svgEl("line", {
      x1: px(t), y1: py(0), x2: px(t), y2: py(1), class: "pareto-grid",
    }));
    axes.appendChild(svgEl("line", {
      x1: px(0), y1: py(t), x2: px(1), y2: py(t), class: "pareto-grid",
    }));
    const xt = svgEl("text", {
      x: px(t), y: py(0) + 14, "text-anchor": "middle", class: "pareto-tick-label",
    });
    xt.textContent = t.toFixed(2);
    axes.appendChild(xt);
    const yt = svgEl("text", {
      x: px(0) - 8, y: py(t) + 3, "text-anchor": "end", class: "pareto-tick-label",
    });
    yt.textContent = t.toFixed(2);
    axes.appendChild(yt);
  }
  axes.appendChild(svgEl("line", {
    x1: px(0), y1: py(0), x2: px(1), y2: py(0), class: "pareto-axis",
  }));
  axes.appendChild(svgEl("line", {
    x1: px(0), y1: py(0), x2: px(0), y2: py(1), class: "pareto-axis",
  }));

  // Pareto front line (sorted by x, only Pareto points)
  const front = candidates
    .filter((c) => c.is_pareto)
    .map((c) => ({ id: c.id, x: c.scores.expression, y: c.scores.low_burden }))
    .sort((a, b) => a.x - b.x);
  if (front.length >= 2) {
    const d = front
      .map((p, i) => `${i ? "L" : "M"} ${px(p.x)} ${py(p.y)}`)
      .join(" ");
    axes.appendChild(svgEl("path", { d, class: "pareto-front-line" }));
  }

  const tip = $("paretoTip");
  const frame = tip.parentElement;

  for (const c of candidates) {
    const cx = px(c.scores.expression);
    const cy = py(c.scores.low_burden);
    const isSelected = c.id === selectedId;
    const r = isSelected ? 8 : 6;

    const circle = svgEl("circle", {
      cx, cy, r,
      class: `pareto-point${c.is_pareto ? " is-pareto" : ""}${isSelected ? " is-selected" : ""}`,
    });
    circle.addEventListener("mouseenter", (ev) => {
      tip.hidden = false;
      tip.innerHTML =
        `<strong>${escapeHtml(c.strategy_name || c.id)}</strong><br/>` +
        `composite ${c.scores.composite.toFixed(3)}<br/>` +
        `exp ${c.scores.expression.toFixed(2)} · burden ${(1 - c.scores.low_burden).toFixed(2)}<br/>` +
        `gc ${c.scores.gc_balance.toFixed(2)} · clean ${c.scores.cleanliness.toFixed(2)}`;
      moveParetoTip(tip, frame, ev);
    });
    circle.addEventListener("mousemove", (ev) => moveParetoTip(tip, frame, ev));
    circle.addEventListener("mouseleave", () => { tip.hidden = true; });
    circle.addEventListener("click", () => selectCandidate(c.id));
    points.appendChild(circle);

    const lbl = svgEl("text", {
      x: cx + 9, y: cy - 9, class: "pareto-point-label",
    });
    lbl.textContent = `#${c.rank}`;
    points.appendChild(lbl);
  }

  $("paretoSection").hidden = false;
}

function moveParetoTip(tip, frame, ev) {
  const rect = frame.getBoundingClientRect();
  tip.style.left = `${ev.clientX - rect.left + 12}px`;
  tip.style.top = `${ev.clientY - rect.top + 12}px`;
}

/** Switch the active candidate and re-render every dependent surface. */
function selectCandidate(id, { animatePasses = false } = {}) {
  const cand = getCandidate(id);
  if (!cand) return;
  state.selectedId = id;

  // Assistant bubble: replace thought instantly (no streaming on switch).
  $("thoughtText").textContent = cand.thought;

  // Plasmid map + sequence + metrics tied to selected candidate.
  const seq = cleanSeq(cand.sequence);
  lastSequence = seq;
  renderPlasmid(seq, cand.strategy || "compiled_construct");
  $("mLen").textContent = `${seq.length} bp`;
  $("mGc").textContent = `${gcPercent(seq).toFixed(2)}%`;
  const cai = passMetricFor(cand.passes, "cai");
  $("mTm").textContent = cai ? `CAI ${cai}` : `${wallaceTm(seq).toFixed(0)} °C`;
  $("seqPre").textContent = seq;

  renderPasses(cand.passes, { animate: animatePasses });
  renderCandidates(state.candidates, id);
  renderParetoChart(state.candidates, id);

  // Reveal everything that's gated until a result exists.
  $("metrics").hidden = false;
  $("seqCard").hidden = false;
}

async function compile() {
  const prompt = $("prompt").value.trim();
  const btn = $("compileBtn");
  const hint = $("statusHint");
  const badge = $("streamBadge");

  if (!prompt) {
    hint.textContent = "Add a short design brief first.";
    return;
  }

  showWorkspace();

  $("userBubbleText").textContent = prompt;
  $("userBubble").hidden = false;
  $("thoughtText").textContent = "Calling compiler…";
  badge.textContent = "Running";
  badge.classList.remove("done");
  $("metrics").hidden = true;
  $("seqCard").hidden = true;
  $("passesCard").hidden = true;
  $("candidatesCard").hidden = true;
  $("paretoSection").hidden = true;
  $("passesList").innerHTML = "";
  $("candidatesList").innerHTML = "";

  btn.disabled = true;
  hint.textContent = "Compiling…";

  try {
    const res = await fetch("/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, n: 4 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);

    setModelLabel(data.model);

    state.candidates = Array.isArray(data.candidates) ? data.candidates : [];
    if (!state.candidates.length) throw new Error("Compiler returned no candidates");

    state.selectedId = data.best_id || state.candidates[0].id;
    const best = getCandidate(state.selectedId);

    const extrasMeta = document.getElementById("extrasMeta");
    if (extrasMeta) {
      const paretoCount = state.candidates.filter((c) => c.is_pareto).length;
      extrasMeta.textContent = `${state.candidates.length} candidates · ${paretoCount}★ · ${data.model}`;
    }

    // Wire up everything for the BEST candidate first; reveal as we go.
    const seq = cleanSeq(best.sequence);
    lastSequence = seq;
    renderPlasmid(seq, best.strategy || "compiled_construct");
    $("mLen").textContent = `${seq.length} bp`;
    $("mGc").textContent = `${gcPercent(seq).toFixed(2)}%`;
    const cai = passMetricFor(best.passes, "cai");
    $("mTm").textContent = cai ? `CAI ${cai}` : `${wallaceTm(seq).toFixed(0)} °C`;
    $("seqPre").textContent = seq;

    // Animate compiler passes + render candidates/Pareto in parallel with thought stream.
    renderPasses(best.passes, { animate: true });
    renderCandidates(state.candidates, state.selectedId);
    renderParetoChart(state.candidates, state.selectedId);

    const thoughtEl = $("thoughtText");
    thoughtEl.textContent = "";
    badge.textContent = "Reasoning";
    await streamThought(thoughtEl, best.thought);
    badge.textContent = "Done";
    badge.classList.add("done");
    hint.textContent = `Ready · ${state.candidates.length} candidates`;

    $("metrics").hidden = false;
    $("seqCard").hidden = false;
    $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  } catch (e) {
    console.error(e);
    hint.textContent = e.message || "Compile failed";
    $("thoughtText").textContent = `Something went wrong: ${hint.textContent}`;
    badge.textContent = "Error";
    badge.classList.remove("done");
  } finally {
    btn.disabled = false;
  }
}

$("compileBtn").addEventListener("click", compile);
$("newDesignBtn").addEventListener("click", showLanding);
$("topnavHome").addEventListener("click", (e) => {
  if (!$("workspace").hidden) {
    e.preventDefault();
    showLanding();
  }
});

$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    compile();
  }
});

$("dlFasta").addEventListener("click", () => {
  if (!lastSequence) return;
  downloadText("compiled_sequence.fasta", toFasta(lastSequence));
});
$("dlGb").addEventListener("click", () => {
  if (!lastSequence) return;
  downloadText("compiled_sequence.gb", toGenbank(lastSequence));
});
$("seqCopyBtn").addEventListener("click", async () => {
  if (!lastSequence) return;
  const btn = $("seqCopyBtn");
  const label = $("seqCopyLabel");
  try {
    await navigator.clipboard.writeText(lastSequence);
    label.textContent = "Copied";
    btn.classList.add("is-copied");
    setTimeout(() => {
      label.textContent = "Copy";
      btn.classList.remove("is-copied");
    }, 1500);
  } catch {
    label.textContent = "Failed";
    setTimeout(() => {
      label.textContent = "Copy";
    }, 1500);
  }
});

const plasmidViewport = $("viewport");
const plasmidSvgEl = $("plasmidSvg");
if (plasmidViewport && plasmidSvgEl) attachPlasmidNav(plasmidSvgEl, plasmidViewport);
