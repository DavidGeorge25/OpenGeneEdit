/** OpenGeneEdit web compiler — landing → split workspace (Claude-style artifact pane). */

const $ = (id) => document.getElementById(id);

function cleanSeq(s) {
  return String(s || "")
    .replace(/\s/g, "")
    .toUpperCase()
    .replace(/[^ACGT]/g, "");
}

/** Canonical DNA string for copy / FASTA / GenBank; keeps ``lastSequence`` in sync when repaired. */
function dnaSequenceForTools() {
  const pre = $("seqPre");
  const c = getCandidate(state.selectedId);
  let raw = "";
  if (c != null && c.sequence != null && String(c.sequence).length > 0) {
    raw = c.sequence;
  } else if (lastSequence) {
    raw = lastSequence;
  } else if (pre && pre.textContent) {
    raw = pre.textContent;
  }
  if (!String(raw).trim()) {
    const rows = state.candidates || [];
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (row != null && row.sequence != null && String(row.sequence).trim() !== "") {
        raw = row.sequence;
        break;
      }
    }
  }
  const seq = cleanSeq(raw);
  if (seq) lastSequence = seq;
  return seq;
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
  const U = typeof URL !== "undefined" ? URL : typeof webkitURL !== "undefined" ? webkitURL : null;
  if (!U) {
    exportDebug("downloadText: no URL / webkitURL API");
    const hint = $("statusHint");
    if (hint) hint.textContent = "Download API unavailable in this browser.";
    return;
  }
  const url = U.createObjectURL(blob);
  exportDebug("downloadText", filename, `${text.length} chars`, url.slice(0, 48) + "…");

  if (typeof navigator !== "undefined" && navigator.msSaveOrOpenBlob) {
    navigator.msSaveOrOpenBlob(blob, filename);
    U.revokeObjectURL(url);
    return;
  }

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  a.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;";
  document.body.appendChild(a);

  requestAnimationFrame(() => {
    a.click();
    setTimeout(() => {
      if (a.parentNode) document.body.removeChild(a);
      U.revokeObjectURL(url);
    }, 400);
  });
}

/** Set ``?export_debug=1`` or ``localStorage.DGENE_EXPORT_DEBUG=1`` for export traces. */
function exportDebugEnabled() {
  try {
    return (
      new URLSearchParams(window.location.search).get("export_debug") === "1" ||
      (typeof localStorage !== "undefined" && localStorage.getItem("DGENE_EXPORT_DEBUG") === "1")
    );
  } catch {
    return false;
  }
}

function exportDebug(...args) {
  if (exportDebugEnabled()) console.info("[OpenGeneEdit export]", ...args);
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
/** Closed annulus sector (no arrowhead) — base ring under feature ribbons. */
function plainAnnulusSectorPath(rIn, rOut, a0, a1) {
  const sweep = a1 - a0;
  if (sweep <= 1e-9) return "";
  const large = sweep > Math.PI ? 1 : 0;
  const c = Math.cos;
  const s = Math.sin;
  const x0o = rOut * c(a0);
  const y0o = rOut * s(a0);
  const x1o = rOut * c(a1);
  const y1o = rOut * s(a1);
  const x1i = rIn * c(a1);
  const y1i = rIn * s(a1);
  const x0i = rIn * c(a0);
  const y0i = rIn * s(a0);
  return [
    `M ${x0o} ${y0o}`,
    `A ${rOut} ${rOut} 0 ${large} 1 ${x1o} ${y1o}`,
    `L ${x1i} ${y1i}`,
    `A ${rIn} ${rIn} 0 ${large} 0 ${x0i} ${y0i}`,
    "Z",
  ].join(" ");
}

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

/** Default heuristic ribbon when no parsed parts are available from the backend. */
const DEFAULT_FEATURE_CATALOG = [
  { label: "J23100", sub: "promoter", pStart: 0.0, pEnd: 0.16, strand: +1, color: "#16a34a" },
  { label: "lacO", sub: "operator", pStart: 0.16, pEnd: 0.23, strand: +1, color: "#d97706" },
  { label: "B0034", sub: "rbs", pStart: 0.23, pEnd: 0.29, strand: +1, color: "#9333ea" },
  { label: "sfGFP", sub: "cds", pStart: 0.29, pEnd: 0.86, strand: +1, color: "#0284c7" },
  { label: "B0015", sub: "terminator", pStart: 0.86, pEnd: 1.0, strand: -1, color: "#dc2626" },
];

const SUB_COLOR = {
  promoter: "#16a34a",
  operator: "#d97706",
  rbs: "#9333ea",
  cds: "#0284c7",
  terminator: "#dc2626",
  feature: "#64748b",
  backbone: "#475569",
};

/** Aligns with ``_part_type_to_map_sub`` in ``circuit_rag_first.py`` (iGEM registry strings). */
function partTypeToMapSub(partType) {
  const t = String(partType || "").trim().toLowerCase();
  if (!t) return "";
  if (t.includes("promoter")) return "promoter";
  if (t.includes("terminator")) return "terminator";
  if (t.includes("rbs") || t.includes("ribosome")) return "rbs";
  if (t === "cds" || t.includes("coding") || t.includes("protein domain") || t === "orf") return "cds";
  if (t.includes("operator")) return "operator";
  if (t.includes("origin")) return "backbone";
  return "";
}

function normalizeMapSlotsForPlasmid(mapSlots) {
  if (!Array.isArray(mapSlots) || !mapSlots.length) return null;
  const drawn = mapSlots.filter((s) => s && s.ok !== false);
  return drawn.length ? drawn : mapSlots;
}

function mergeBpIntervals(intervals) {
  if (!intervals.length) return [];
  const sorted = [...intervals].sort((a, b) => a.start - b.start);
  const out = [{ start: sorted[0].start, end: sorted[0].end }];
  for (let i = 1; i < sorted.length; i++) {
    const cur = sorted[i];
    const last = out[out.length - 1];
    if (cur.start <= last.end + 1) last.end = Math.max(last.end, cur.end);
    else out.push({ start: cur.start, end: cur.end });
  }
  return out;
}

/** Linear [1..L] bp not covered by merged intervals (for circular plasmid map). */
function backboneGapIntervals(L, merged) {
  if (!L || L < 1) return [];
  if (!merged.length) return [{ start: 1, end: L }];
  const gaps = [];
  let expect = 1;
  for (const intv of merged) {
    const s = intv.start;
    const e = intv.end;
    if (s > expect) gaps.push({ start: expect, end: s - 1 });
    expect = Math.max(expect, e + 1);
  }
  if (expect <= L) gaps.push({ start: expect, end: L });
  return gaps;
}

function backboneGapFeatures(L, gaps) {
  return gaps.map((g) => ({
    label: "Backbone · unannotated",
    sub: "backbone",
    pStart: (g.start - 1) / L,
    pEnd: g.end / L,
    strand: +1,
    color: SUB_COLOR.backbone,
    start: g.start,
    end: g.end,
    unverified: false,
    coordSuspect: false,
    isBackboneGap: true,
  }));
}

/** Feature arcs for the plasmid map: parsed ``map_slots`` from the model when present,
 *  otherwise a generic catalog (historical demo layout). */
function inferFeatures(L, mapSlots) {
  const slots = normalizeMapSlotsForPlasmid(mapSlots);
  if (slots) {
    const n = slots.length;
    const raw = [];
    for (let i = 0; i < n; i++) {
      const s = slots[i] || {};
      const fromRegistryType = partTypeToMapSub(s.part_type);
      let sub = String(s.sub || "").toLowerCase();
      if (!sub || sub === "feature") {
        sub = fromRegistryType || sub || "feature";
      }
      const label =
        String(s.label || s.part_name || s.normalized_name || "part").trim() || "part";
      const strand = sub === "terminator" ? -1 : +1;
      const color = SUB_COLOR[sub] || SUB_COLOR.feature;
      let start;
      let end;
      const sb = s.start_bp != null ? Number(s.start_bp) : NaN;
      const eb = s.end_bp != null ? Number(s.end_bp) : NaN;
      if (Number.isFinite(sb) && Number.isFinite(eb) && eb >= sb && sb >= 1) {
        start = Math.max(1, Math.min(L, Math.round(sb)));
        end = Math.max(1, Math.min(L, Math.round(eb)));
        if (end < start) continue;
      } else {
        const pStart = i / n;
        const pEnd = (i + 1) / n;
        start = Math.max(1, Math.round(pStart * L) + (pStart === 0 ? 0 : 1));
        end = Math.max(start, Math.round(pEnd * L));
      }
      const pStart = (start - 1) / L;
      const pEnd = end / L;
      const unverified = s.verified === false;
      const span = end - start + 1;
      const coordSuspect =
        sub !== "cds" &&
        sub !== "backbone" &&
        span > Math.max(120, L * 0.42);
      raw.push({ label, sub, pStart, pEnd, strand, color, start, end, unverified, coordSuspect });
    }
    raw.sort((a, b) => a.start - b.start || a.end - b.end);
    const merged = mergeBpIntervals(raw.map((f) => ({ start: f.start, end: f.end })));
    const gaps = backboneGapIntervals(L, merged);
    const underlay = backboneGapFeatures(L, gaps);
    return [...underlay, ...raw];
  }
  return DEFAULT_FEATURE_CATALOG.map((f) => {
    const start = Math.max(1, Math.round(f.pStart * L) + (f.pStart === 0 ? 0 : 1));
    const end = Math.max(start + 1, Math.round(f.pEnd * L));
    return { ...f, start, end };
  });
}

function bpToAngle(bp, L) {
  if (!L || L < 1) return 0;
  const b = Math.max(0, Math.min(L, Number(bp) || 0));
  return (2 * Math.PI * b) / L;
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

function renderRestrictionSites(sites, L, sequence, tip, frame) {
  const g = $("plasmidSites");
  const seq = String(sequence || "");
  g.innerHTML = "";
  const clusters = clusterRestrictionSitesForLabels(sites);
  const MIN_ANGLE_SEP = 0.1;
  const R_EXTRA_STEP = 16;
  const placed = [];

  const sortedClusters = [...clusters].sort((a, b) => a.labelBp - b.labelBp);

  for (const cl of sortedClusters) {
    const a = cl.labelBp;
    const angleUnwrapped = bpToAngle(a, L);
    let tier = 0;
    while (placed.some((p) => p.tier === tier && angleDistRad(angleUnwrapped, p.angleRad) < MIN_ANGLE_SEP)) {
      tier++;
    }
    placed.push({ angleRad: angleUnwrapped, tier });

    const rLabel = PLASMID_GEOM.rSiteLabel + tier * R_EXTRA_STEP;
    const p0 = polarToXY(PLASMID_GEOM.rSiteTickInner, a, L);
    const p1 = polarToXY(PLASMID_GEOM.rSiteTickOuter + tier * 6, a, L);
    const lp = polarToXY(rLabel, a, L);

    g.appendChild(svgEl("line", {
      x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y, class: "plasmid-site-tick",
    }));

    const angle = angleUnwrapped - Math.PI / 2;
    const deg = (angle * 180) / Math.PI;
    const rotateDeg = deg > 90 || deg < -90 ? deg + 180 : deg;
    const anchor = deg > 90 || deg < -90 ? "end" : "start";
    const posStr = cl.posLo === cl.posHi ? String(cl.posLo) : `${cl.posLo}–${cl.posHi}`;
    const labelNames = cl.names.join(" · ");

    const txt = svgEl("text", {
      x: lp.x,
      y: lp.y,
      transform: `rotate(${rotateDeg} ${lp.x} ${lp.y})`,
      "text-anchor": anchor,
      "dominant-baseline": "middle",
      class: "plasmid-site-label plasmid-site-hit",
    });
    txt.textContent = `${labelNames} (${posStr})`;

    function tooltipHtmlForCluster() {
      const blocks = [];
      for (const site of cl.sites) {
        const motif = ENZYMES[site.name] || "";
        const siteSeg =
          motif && seq.length ? plasmidSegmentSeq(seq, site.position, site.position + motif.length - 1) : "";
        if (motif && siteSeg) {
          const prev = formatSeqTooltipPreview(siteSeg, 120);
          const safeSeqAttr = escapeHtml(siteSeg);
          blocks.push(
            `<div class="feature-tooltip-head"><strong>${escapeHtml(site.name)}</strong><br/>` +
              `site · ${site.position} bp · ${escapeHtml(motif)} (5′→3′)</div>` +
              `<pre class="feature-tooltip-seq">${escapeHtml(prev)}</pre>` +
              `<button type="button" class="feature-tooltip-copy" data-copy-seq="${safeSeqAttr}">` +
              `<span class="feature-tooltip-copy-label">Copy</span> site</button>`,
          );
        } else {
          blocks.push(
            `<div class="feature-tooltip-head"><strong>${escapeHtml(site.name)}</strong><br/>` +
              `cut site · ${site.position} bp</div>`,
          );
        }
      }
      return blocks.join("");
    }

    txt.addEventListener("mouseenter", (ev) => {
      cancelPlasmidTipHide();
      txt.classList.add("is-active");
      tip.classList.add("tooltip--rich");
      tip.hidden = false;
      tip.innerHTML = tooltipHtmlForCluster();
      moveTooltipAt(tip, frame, ev);
    });
    txt.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
    txt.addEventListener("mouseleave", () => {
      txt.classList.remove("is-active");
      schedulePlasmidTipHide(tip);
    });
    g.appendChild(txt);
  }
}

function moveTooltipAt(tip, frame, ev) {
  const rect = frame.getBoundingClientRect();
  tip.style.left = `${ev.clientX - rect.left + 12}px`;
  tip.style.top = `${ev.clientY - rect.top + 12}px`;
}

let plasmidTipHideTimer = null;

function cancelPlasmidTipHide() {
  if (plasmidTipHideTimer) {
    clearTimeout(plasmidTipHideTimer);
    plasmidTipHideTimer = null;
  }
}

function schedulePlasmidTipHide(tip) {
  cancelPlasmidTipHide();
  plasmidTipHideTimer = setTimeout(() => {
    tip.hidden = true;
    document
      .querySelectorAll(".plasmid-seg.is-active, .plasmid-site-hit.is-active")
      .forEach((el) => el.classList.remove("is-active"));
  }, 220);
}

/** Bind once per viz frame: sticky hover for DNA tooltips + copy control. */
function bindPlasmidTipFrame(tip, frame) {
  if (!tip || !frame || frame.dataset.plasmidTipBound === "1") return;
  frame.dataset.plasmidTipBound = "1";
  tip.addEventListener("mouseenter", cancelPlasmidTipHide);
  tip.addEventListener("mouseleave", () => {
    tip.hidden = true;
    document
      .querySelectorAll(".plasmid-seg.is-active, .plasmid-site-hit.is-active")
      .forEach((el) => el.classList.remove("is-active"));
  });
  frame.addEventListener("click", async (e) => {
    const btn = e.target.closest(".feature-tooltip-copy");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const raw = btn.getAttribute("data-copy-seq") || "";
    if (!raw) return;
    try {
      await navigator.clipboard.writeText(raw);
      const lab = btn.querySelector(".feature-tooltip-copy-label");
      if (lab) lab.textContent = "Copied";
      btn.classList.add("is-copied");
      setTimeout(() => {
        if (lab) lab.textContent = "Copy";
        btn.classList.remove("is-copied");
      }, 1600);
    } catch (err) {
      console.warn(err);
    }
  });
}

function plasmidSegmentSeq(seq, start, end) {
  const s = String(seq || "");
  if (!s.length) return "";
  const i0 = Math.max(0, Number(start) - 1);
  const i1 = Math.min(s.length, Number(end));
  if (i1 <= i0) return "";
  return s.slice(i0, i1);
}

function formatSeqTooltipPreview(seq, maxPreviewChars) {
  const cap = maxPreviewChars > 80 ? maxPreviewChars : 360;
  const piece = seq.length > cap ? seq.slice(0, cap) : seq;
  const w = 10;
  const parts = [];
  for (let i = 0; i < piece.length; i += w) parts.push(piece.slice(i, i + w));
  let out = parts.join(" ");
  if (seq.length > cap) out += `\n… ${seq.length} bp total (copy gets full segment)`;
  return out;
}

function buildFeatureTooltipHtml(f, subHuman, seg) {
  const safeSegAttr = escapeHtml(seg);
  const prev = formatSeqTooltipPreview(seg, 400);
  const unvNote = f.unverified
    ? `<br/><span class="feature-tooltip-note">* Sequence not verified (iGEM / NCBI); model or low-confidence DNA.</span>`
    : "";
  const coordNote = f.coordSuspect
    ? `<br/><span class="feature-tooltip-note">Fragment span looks unusually wide for this part type; <code>map_slots</code> coordinates may need review.</span>`
    : "";
  return (
    `<div class="feature-tooltip-head"><strong>${escapeHtml(f.label)}</strong> · ${escapeHtml(subHuman)}<br/>` +
    `${f.start}–${f.end} bp · ${f.strand >= 0 ? "+" : "−"} strand${unvNote}${coordNote}</div>` +
    `<pre class="feature-tooltip-seq">${escapeHtml(prev)}</pre>` +
    `<button type="button" class="feature-tooltip-copy" data-copy-seq="${safeSegAttr}">` +
    `<span class="feature-tooltip-copy-label">Copy</span> segment</button>`
  );
}

function humanizeFeatureSub(sub) {
  const s = String(sub || "feature").toLowerCase();
  if (s === "rbs") return "RBS";
  if (s === "cds") return "CDS";
  if (s === "backbone") return "Backbone";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Shortest angular distance on the circle (radians). */
function angleDistRad(a, b) {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
}

/** Vertically stack labels that share the same side so two-line blocks do not overlap. */
function packPlasmidFeatureLabelYs(rows, yMin, yMax) {
  if (!rows.length) return;
  rows.sort((x, y) => x.idealY - y.idealY);
  const range = Math.max(1, yMax - yMin);
  let sep = 21;
  const minSep = 13;
  const need = (rows.length - 1) * sep;
  if (need > range * 0.92 && rows.length > 1) {
    sep = Math.max(minSep, Math.floor((range * 0.92) / (rows.length - 1)));
  }
  const ys = rows.map((r) => r.idealY);
  for (let i = 1; i < ys.length; i++) {
    ys[i] = Math.max(ys[i], ys[i - 1] + sep);
  }
  let shift = 0;
  if (ys[ys.length - 1] > yMax) shift -= ys[ys.length - 1] - yMax;
  if (ys[0] + shift < yMin) shift += yMin - (ys[0] + shift);
  for (let i = 0; i < ys.length; i++) {
    rows[i].packedY = ys[i] + shift;
  }
  for (let rep = 0; rep < 6; rep++) {
    let moved = false;
    if (rows[0].packedY < yMin) {
      const d = yMin - rows[0].packedY;
      for (const r of rows) r.packedY += d;
      moved = true;
    }
    if (rows[rows.length - 1].packedY > yMax) {
      const d = rows[rows.length - 1].packedY - yMax;
      for (const r of rows) r.packedY -= d;
      moved = true;
    }
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].packedY < rows[i - 1].packedY + sep) {
        rows[i].packedY = rows[i - 1].packedY + sep;
        moved = true;
      }
    }
    if (!moved) break;
  }
}

/** Merge restriction sites that are adjacent on the sequence map so labels are not duplicated on top of each other. */
function clusterRestrictionSitesForLabels(sites, bpMerge = 32) {
  if (!sites.length) return [];
  const sorted = [...sites].sort((a, b) => a.position - b.position);
  const clusters = [];
  let cur = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].position - cur[cur.length - 1].position <= bpMerge) cur.push(sorted[i]);
    else {
      clusters.push(cur);
      cur = [sorted[i]];
    }
  }
  clusters.push(cur);
  return clusters.map((c) => {
    const positions = c.map((s) => s.position);
    const lo = Math.min(...positions);
    const hi = Math.max(...positions);
    const labelBp = Math.round(positions.reduce((a, p) => a + p, 0) / positions.length);
    return {
      sites: c,
      names: [...new Set(c.map((s) => s.name))],
      labelBp,
      posLo: lo,
      posHi: hi,
    };
  });
}

function renderFeaturesAndLabels(features, L, sequence, tip, frame) {
  const arcs = $("plasmidRotate");
  const labels = $("plasmidLabels");
  const seq = String(sequence || "");
  arcs.innerHTML = "";
  labels.innerHTML = "";

  const { rFeatureInner: rIn, rFeatureOuter: rOut } = PLASMID_GEOM;

  const trackEps = 1e-4;
  const dTrack = plainAnnulusSectorPath(rIn, rOut, 0, 2 * Math.PI - trackEps);
  if (dTrack) {
    arcs.appendChild(svgEl("path", {
      d: dTrack,
      class: "plasmid-track-base",
      fill: "#e4e9f2",
    }));
  }

  const labelRows = [];
  const Y_PACK_MIN = 36;
  const Y_PACK_MAX = 564;

  for (const f of features) {
    const a0 = bpToAngle(f.start - 1, L);
    const a1 = bpToAngle(f.end, L);
    if (a1 <= a0) continue;

    const usePlainArc = f.sub === "terminator";
    const d = usePlainArc
      ? plainAnnulusSectorPath(rIn, rOut, a0, a1)
      : featureArrowPath(rIn, rOut, a0, a1, f.strand);
    if (!d) continue;
    const path = svgEl("path", {
      d,
      fill: f.color,
      class: f.isBackboneGap ? "plasmid-seg plasmid-seg--gap" : "plasmid-seg",
    });
    arcs.appendChild(path);

    if (f.isBackboneGap) {
      path.addEventListener("mouseenter", (ev) => {
        cancelPlasmidTipHide();
        tip.classList.remove("tooltip--rich");
        tip.hidden = false;
        tip.innerHTML =
          `<div class="feature-tooltip-head"><strong>Unannotated region</strong><br/>` +
          `${f.start}–${f.end} bp · no <code>map_slots</code> entry (backbone / linker).</div>`;
        moveTooltipAt(tip, frame, ev);
      });
      path.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
      path.addEventListener("mouseleave", () => schedulePlasmidTipHide(tip));
      continue;
    }

    const midBp = (f.start - 1 + f.end) / 2;
    const midA = bpToAngle(midBp, L) - Math.PI / 2;
    const onRight = Math.cos(midA) >= 0;

    const pStart = polarToXY(PLASMID_GEOM.rLeaderInner, midBp, L);
    const pElbow = polarToXY(PLASMID_GEOM.rLeaderElbow, midBp, L);
    const labelX = onRight ? PLASMID_GEOM.cx + PLASMID_GEOM.rLeaderTextX : PLASMID_GEOM.cx - PLASMID_GEOM.rLeaderTextX;

    labelRows.push({
      f,
      path,
      midBp,
      onRight,
      pStart,
      pElbow,
      labelX,
      idealY: pElbow.y,
      packedY: pElbow.y,
    });
  }

  const rightRows = labelRows.filter((r) => r.onRight);
  const leftRows = labelRows.filter((r) => !r.onRight);
  packPlasmidFeatureLabelYs(rightRows, Y_PACK_MIN, Y_PACK_MAX);
  packPlasmidFeatureLabelYs(leftRows, Y_PACK_MIN, Y_PACK_MAX);

  for (const row of labelRows) {
    const { f, path } = row;
    const pText = { x: row.labelX, y: row.packedY };

    labels.appendChild(svgEl("path", {
      d: `M ${row.pStart.x} ${row.pStart.y} L ${row.pElbow.x} ${row.pElbow.y} L ${pText.x} ${pText.y}`,
      class: "plasmid-leader",
    }));

    const onRight = row.onRight;
    const anchor = onRight ? "start" : "end";
    const dx = onRight ? 4 : -4;
    const tName = svgEl("text", {
      x: pText.x + dx,
      y: pText.y - 2,
      "text-anchor": anchor,
      class: "plasmid-feature-label plasmid-feature-hit",
    });
    const tspanLabel = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    tspanLabel.textContent = f.label;
    tName.appendChild(tspanLabel);
    if (f.unverified) {
      const tspanAst = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
      tspanAst.setAttribute("class", "plasmid-unverified-ast");
      tspanAst.textContent = "*";
      tName.appendChild(tspanAst);
    }
    labels.appendChild(tName);

    const tSub = svgEl("text", {
      x: pText.x + dx,
      y: pText.y + 12,
      "text-anchor": anchor,
      class: "plasmid-feature-sub plasmid-feature-hit",
    });
    const subHuman = humanizeFeatureSub(f.sub);
    tSub.textContent = `${subHuman} · ${f.start}–${f.end}${f.coordSuspect ? " · coords?" : ""}`;
    labels.appendChild(tSub);

    const seg = plasmidSegmentSeq(seq, f.start, f.end);
    const showSeq = seg.length > 0;

    const showTip = (ev) => {
      cancelPlasmidTipHide();
      document.querySelectorAll(".plasmid-seg.is-active").forEach((el) => el.classList.remove("is-active"));
      path.classList.add("is-active");
      tip.classList.add("tooltip--rich");
      tip.hidden = false;
      const unvNote = f.unverified
        ? `<br/><span class="feature-tooltip-note">* Sequence not verified (iGEM / NCBI); model or low-confidence DNA.</span>`
        : "";
      const coordNote = f.coordSuspect
        ? `<br/><span class="feature-tooltip-note">Unusually wide span for this part type — verify <code>map_slots</code> coordinates.</span>`
        : "";
      tip.innerHTML = showSeq
        ? buildFeatureTooltipHtml(f, subHuman, seg)
        : `<div class="feature-tooltip-head"><strong>${escapeHtml(f.label)}</strong> · ${escapeHtml(subHuman)}<br/>` +
          `${f.start}–${f.end} bp · ${f.strand >= 0 ? "+" : "−"} strand${unvNote}${coordNote}<br/>` +
          `<span class="feature-tooltip-note">No sequence loaded for this map.</span></div>`;
      moveTooltipAt(tip, frame, ev);
    };

    path.addEventListener("mouseenter", (ev) => showTip(ev));
    path.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
    path.addEventListener("mouseleave", () => {
      path.classList.remove("is-active");
      schedulePlasmidTipHide(tip);
    });

    for (const el of [tName, tSub]) {
      el.addEventListener("mouseenter", (ev) => showTip(ev));
      el.addEventListener("mousemove", (ev) => moveTooltipAt(tip, frame, ev));
      el.addEventListener("mouseleave", () => {
        path.classList.remove("is-active");
        schedulePlasmidTipHide(tip);
      });
    }
  }
}

function renderPlasmid(sequence, name = "construct", mapSlots = null) {
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

  const features = inferFeatures(L, mapSlots);
  const sites = findRestrictionSites(sequence);

  renderTicks(L);
  bindPlasmidTipFrame(tip, frame);
  renderFeaturesAndLabels(features, L, sequence, tip, frame);
  renderRestrictionSites(sites, L, sequence, tip, frame);
}

/** Zoom/pan the plasmid: use document-capture wheel + pointer handlers so parent ``.artifact-body``
 *  scroll containers do not eat trackpad/mouse wheel, and pan keeps working when the cursor leaves
 *  the frame mid-drag. Shift+wheel does not zoom (page scroll). */
let plasmidNavDocBound = false;
function attachPlasmidNav(svg, viewport) {
  if (!svg || !viewport || plasmidNavDocBound) return;
  plasmidNavDocBound = true;
  const frame = $("vizFrame") || svg.parentElement;
  if (!frame) return;

  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dragging = false;
  let activePointerId = null;
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

  function onWheel(e) {
    if (!frame.contains(e.target)) return;
    if (e.shiftKey) return;
    e.preventDefault();
    e.stopPropagation();
    const factor = Math.exp(-e.deltaY * 0.0015);
    scale = Math.min(4, Math.max(0.55, scale * factor));
    apply();
  }

  document.addEventListener("wheel", onWheel, { passive: false, capture: true });

  function onPointerDown(e) {
    if (!frame.contains(e.target)) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const root = e.target;
    if (root && typeof root.closest === "function") {
      if (root.closest("#featureTooltip")) return;
      if (root.closest(".feature-tooltip-copy")) return;
    }
    dragging = true;
    activePointerId = e.pointerId;
    try {
      frame.setPointerCapture(e.pointerId);
    } catch (_) {
      /* ignore */
    }
    lx = e.clientX;
    ly = e.clientY;
  }

  function onPointerMove(e) {
    if (!dragging || e.pointerId !== activePointerId) return;
    tx += e.clientX - lx;
    ty += e.clientY - ly;
    lx = e.clientX;
    ly = e.clientY;
    apply();
  }

  function endPan(e) {
    if (!dragging || e.pointerId !== activePointerId) return;
    dragging = false;
    activePointerId = null;
    try {
      frame.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* ignore */
    }
  }

  document.addEventListener("pointerdown", onPointerDown, { capture: true });
  document.addEventListener("pointermove", onPointerMove, { capture: true });
  document.addEventListener("pointerup", endPan, { capture: true });
  document.addEventListener("pointercancel", endPan, { capture: true });
  frame.addEventListener("lostpointercapture", () => {
    dragging = false;
    activePointerId = null;
  });

  frame.addEventListener("dblclick", (e) => {
    if (!frame.contains(e.target)) return;
    e.preventDefault();
    scale = 1;
    tx = 0;
    ty = 0;
    apply();
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
  if (compileAbort) {
    compileAbort.abort();
    compileAbort = null;
  }
  stopCompilePipelineVisual();

  $("landing").hidden = false;
  $("workspace").hidden = true;
  document.body.classList.remove("in-workspace");
  $("statusHint").textContent = "⌘ or Ctrl + Enter to compile";
  $("userBubble").hidden = true;
  $("thoughtText").textContent = "";
  const teLand = $("thoughtEyebrow");
  const twLand = $("thoughtWrap");
  if (teLand) teLand.hidden = false;
  if (twLand) twLand.hidden = false;
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
  const ragEl0 = $("ragCard");
  if (ragEl0) ragEl0.hidden = true;
  $("passesCard").hidden = true;
  const expertQaReset = $("expertQaCard");
  if (expertQaReset) {
    expertQaReset.hidden = true;
    const eqb = $("expertQaBody");
    const eqm = $("expertQaMeta");
    if (eqb) eqb.innerHTML = "";
    if (eqm) eqm.textContent = "";
  }
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
  state.snapshotIdForBar = null;
  lastSequence = "";
  clearSnapshotFromUrl();
  updateSnapshotBar(null);
}

function setModelLabel(model) {
  const m = model || "Gemma-4";
  $("modelPill").textContent = `Model: ${m}`;
  $("modelPillNav").textContent = `Model · ${m}`;
}

/** Sync nav pills + footer with server /api/health (hosted Gemma-4 vs local GGUF). */
function applyBackendMeta(d) {
  if (!d) return;
  if (d.model) setModelLabel(d.model);

  const kRaw = d.backend_kind;
  if (kRaw == null || String(kRaw) === "") return;

  const k = String(kRaw);
  const hostedEl = document.getElementById("footHostedLine");
  const plugEl = document.getElementById("footGgufPlug");
  const ftEl = document.getElementById("footFtLine");

  if (k === "fine_tuned") {
    if (hostedEl) hostedEl.hidden = true;
    if (plugEl) plugEl.hidden = true;
    if (ftEl) {
      ftEl.hidden = false;
      const gf = $("footGgufName");
      if (gf && d.gguf_file) gf.textContent = String(d.gguf_file);
    }
  } else {
    if (hostedEl) hostedEl.hidden = false;
    if (plugEl) plugEl.hidden = false;
    if (ftEl) ftEl.hidden = true;
    const mid = $("footApiModelId");
    if (mid) mid.textContent = d.api_model_id ? String(d.api_model_id) : "hosted";
  }
}

async function refreshBackendMeta() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    applyBackendMeta(data);
  } catch {
    /* keep static HTML defaults */
  }
}

let lastSequence = "";

const state = {
  candidates: [],
  selectedId: null,
  /** Set after successful compile or snapshot restore; drives “Copy link” + URL query. */
  snapshotIdForBar: null,
};

/** 32-char hex ids written by ``server.save_compile_snapshot``. */
const SNAPSHOT_ID_RE = /^[0-9a-f]{32}$/;

function buildSnapshotPageUrl(snapshotId) {
  const u = new URL(window.location.href);
  u.searchParams.set("snapshot", snapshotId);
  const q = u.searchParams.toString();
  return `${u.pathname}?${q}`;
}

function replaceUrlWithSnapshot(snapshotId) {
  if (!snapshotId || !SNAPSHOT_ID_RE.test(snapshotId)) return;
  history.replaceState({ snapshot: snapshotId }, "", buildSnapshotPageUrl(snapshotId));
}

function clearSnapshotFromUrl() {
  const u = new URL(window.location.href);
  if (!u.searchParams.has("snapshot")) return;
  u.searchParams.delete("snapshot");
  const q = u.searchParams.toString();
  history.replaceState({}, "", u.pathname + (q ? `?${q}` : ""));
}

function updateSnapshotBar(snapshotId) {
  const bar = $("snapshotBar");
  const fb = $("snapshotCopyFeedback");
  const show = snapshotId && SNAPSHOT_ID_RE.test(snapshotId);
  if (bar) bar.hidden = !show;
  if (fb) fb.textContent = "";
}

/** Paint workspace from a compile API / snapshot GET payload (no compile pipeline). */
function applyCompileResultPayload(data) {
  lastPartialPassesSig = null;
  passFixState.spinningPassId = null;
  passFixState.lastFix = null;
  setModelLabel(data.model);
  state.candidates = Array.isArray(data.candidates) ? data.candidates : [];
  if (!state.candidates.length) throw new Error("No candidates in snapshot");

  state.selectedId =
    data.best_id != null && data.best_id !== "" ? data.best_id : state.candidates[0].id;
  let best = getCandidate(state.selectedId);
  if (!best) {
    best = state.candidates[0];
    state.selectedId = best.id;
  }
  if (!best) throw new Error("best candidate missing");

  const promptVal = typeof data.prompt === "string" ? data.prompt : "";
  if ($("prompt")) $("prompt").value = promptVal;
  $("userBubbleText").textContent = promptVal;
  $("userBubble").hidden = false;

  const extrasMeta = document.getElementById("extrasMeta");
  if (extrasMeta) {
    const paretoCount = state.candidates.filter((c) => c.is_pareto).length;
    extrasMeta.textContent = `${state.candidates.length} candidates · ${paretoCount}★ · ${data.model || ""}`;
  }

  const seq = cleanSeq(best.sequence);
  lastSequence = seq;
  const mapSlots = best.rag && Array.isArray(best.rag.map_slots) ? best.rag.map_slots : null;
  renderPlasmid(seq, best.strategy || "compiled_construct", mapSlots);
  $("mLen").textContent = `${seq.length} bp`;
  $("mGc").textContent = `${gcPercent(seq).toFixed(2)}%`;
  const cai = passMetricFor(best.passes, "cai");
  $("mTm").textContent = cai ? `CAI ${cai}` : `${wallaceTm(seq).toFixed(0)} °C`;
  $("seqPre").textContent = seq;

  renderPasses(best.passes, { animate: false });
  renderCandidates(state.candidates, state.selectedId);
  renderParetoChart(state.candidates, state.selectedId);
  renderRagPanel(best.rag);
  renderExpertQa(best.rag);

  $("thoughtText").textContent = best.thought || "";
  $("streamBadge").textContent = "Restored";
  $("streamBadge").classList.add("done");

  $("metrics").hidden = false;
  $("seqCard").hidden = false;
  $("ragCard").hidden = false;
  $("passesCard").hidden = false;
  $("candidatesCard").hidden = false;
  $("paretoSection").hidden = false;
}

/**
 * Applies a `/api/compile` job `result` (partial or final) to the workspace maps/lists.
 */
function paintLiveCompileWorkspace(data, { partial = false, animatePasses = true } = {}) {
  setModelLabel(data.model);
  const incoming = Array.isArray(data.candidates) ? data.candidates : [];
  if (!incoming.length) return;

  const prevSelected = state.selectedId;
  state.candidates = incoming;

  function resolveCandId(raw) {
    if (raw === undefined || raw === null || raw === "") return null;
    const sid = String(raw);
    const hit = state.candidates.find((c) => String(c.id) === sid);
    return hit ? hit.id : null;
  }

  let chosen = resolveCandId(prevSelected);
  if (!chosen) chosen = resolveCandId(data.best_id);
  if (!chosen && state.candidates.length) chosen = state.candidates[0].id;
  state.selectedId = chosen;

  let best = getCandidate(state.selectedId);
  if (!best && state.candidates.length) {
    best = state.candidates[0];
    state.selectedId = best.id;
  }
  if (!best) return;

  const extrasMeta = document.getElementById("extrasMeta");
  if (extrasMeta) {
    const paretoCount = state.candidates.filter((c) => c.is_pareto).length;
    if (
      partial &&
      data.variants_ready != null &&
      data.variants_total != null &&
      data.variants_total > 0
    ) {
      extrasMeta.textContent = `${data.variants_ready}/${data.variants_total} variants ready · ${paretoCount}★ · ${data.model}`;
    } else {
      extrasMeta.textContent = `${state.candidates.length} candidates · ${paretoCount}★ · ${data.model}`;
    }
  }

  const seq = cleanSeq(best.sequence);
  lastSequence = seq;
  const mapSlots = best.rag && Array.isArray(best.rag.map_slots) ? best.rag.map_slots : null;
  renderPlasmid(seq, best.strategy || "compiled_construct", mapSlots);
  $("mLen").textContent = `${seq.length} bp`;
  $("mGc").textContent = `${gcPercent(seq).toFixed(2)}%`;
  const cai = passMetricFor(best.passes, "cai");
  $("mTm").textContent = cai ? `CAI ${cai}` : `${wallaceTm(seq).toFixed(0)} °C`;
  $("seqPre").textContent = seq;

  const passSig = `${state.selectedId}:${JSON.stringify(best.passes)}`;
  let skipPassesRender = false;
  if (partial) {
    if (passSig === lastPartialPassesSig) skipPassesRender = true;
    else lastPartialPassesSig = passSig;
  } else {
    lastPartialPassesSig = null;
  }
  if (!skipPassesRender) {
    renderPasses(best.passes, { animate: animatePasses });
  }
  renderCandidates(state.candidates, state.selectedId);
  renderParetoChart(state.candidates, state.selectedId);
  renderRagPanel(best.rag);
  renderExpertQa(best.rag);

  $("thoughtText").textContent = best.thought || "";

  $("metrics").hidden = false;
  $("seqCard").hidden = false;
  const ragEl = $("ragCard");
  if (ragEl) ragEl.hidden = false;
  $("passesCard").hidden = false;
  $("candidatesCard").hidden = false;
  $("paretoSection").hidden = false;
}

async function maybeRestoreCompileSnapshot() {
  let id = new URLSearchParams(window.location.search).get("snapshot");
  id = id ? String(id).trim().toLowerCase() : "";
  if (!SNAPSHOT_ID_RE.test(id)) return;

  try {
    const res = await fetch(`/api/snapshot?id=${encodeURIComponent(id)}`);
    let payload = {};
    try {
      payload = await res.json();
    } catch {
      payload = {};
    }
    if (!res.ok) {
      const hint = $("statusHint");
      if (hint) {
        if (res.status === 404)
          hint.textContent = "Saved link not found (new server folder or snapshots cleared).";
        else if (res.status === 503) hint.textContent = "Design snapshots disabled on server.";
        else hint.textContent = payload.error ? String(payload.error) : "Could not restore design.";
      }
      clearSnapshotFromUrl();
      return;
    }

    showWorkspace();
    stopCompilePipelineVisual();
    applyCompileResultPayload(payload);

    const sid = payload.snapshot_id || id;
    state.snapshotIdForBar = sid;
    updateSnapshotBar(sid);
    replaceUrlWithSnapshot(sid);

    $("statusHint").textContent = `Opened saved design (${state.candidates.length} variants)`;
  } catch (e) {
    console.error(e);
    const hint = $("statusHint");
    if (hint) hint.textContent = "Network error loading saved design.";
    clearSnapshotFromUrl();
  }
}

async function bootstrapApp() {
  await refreshBackendMeta();
  await maybeRestoreCompileSnapshot();
}

const PASS_GLYPH = { ok: "✓", warn: "!", error: "✕" };

/** Pass IDs that support targeted /api/fix (fix_type matches pass_id). */
const PASS_FIX_TYPES = new Set(["repeats", "type_iis", "cai", "rbs"]);

const passFixState = {
  spinningPassId: null,
  /** Set after a successful fix while viewing `newCandidateId`; cleared on other selection. */
  lastFix: null,
};

/** Skip redundant partial pass repaints (stops COMPILER PASSES flicker while variants stream). */
let lastPartialPassesSig = null;

function getCandidate(id) {
  if (id === undefined || id === null) return null;
  const sid = String(id);
  return state.candidates.find((c) => String(c.id) === sid) || null;
}

function passMetricFor(passes, passId, key = "metric_raw") {
  const p = passes.find((x) => x.pass_id === passId);
  return p ? p[key] : null;
}

/** Render the compiler-passes log. animate=true reveals one row at a time. */
function renderPasses(passes, { animate } = { animate: false }) {
  const list = $("passesList");
  list.innerHTML = "";

  const lastFix = passFixState.lastFix;
  const spinPid = passFixState.spinningPassId;
  const selectedCand = getCandidate(state.selectedId);
  const showFixOutcome =
    lastFix &&
    selectedCand &&
    selectedCand.id === lastFix.newCandidateId;

  const counts = passes.reduce(
    (acc, p) => ((acc[p.status] = (acc[p.status] || 0) + 1), acc),
    {}
  );
  const totalMs = passes.reduce((s, p) => s + (p.duration_ms || 0), 0);
  $("passesCardMeta").textContent =
    `${passes.length} passes · ${(counts.ok || 0)}✓ ${(counts.warn || 0)}! ${(counts.error || 0)}✕ · ${totalMs.toFixed(1)} ms`;

  passes.forEach((p, i) => {
    let displayStatus = p.status;
    let displaySummary = p.summary || p.name || "";
    let glyph = PASS_GLYPH[p.status] || "·";

    const fixOverride =
      showFixOutcome && lastFix.passId === p.pass_id ? lastFix : null;
    if (fixOverride) {
      if (fixOverride.stillFlagged) {
        displayStatus = "warn";
        displaySummary = "Fix attempted — still flagged";
        glyph = PASS_GLYPH.warn;
      } else if (fixOverride.passCleared) {
        displayStatus = "ok";
        displaySummary = `Fixed in candidate #${fixOverride.candNum}`;
        glyph = PASS_GLYPH.ok;
      } else {
        displayStatus = "ok";
        displaySummary = `Improved in candidate #${fixOverride.candNum} — pass may still warn`;
        glyph = PASS_GLYPH.ok;
      }
    }

    const li = document.createElement("li");
    li.className = "pass-row";
    if (!animate) li.classList.add("pass-row--static");
    li.dataset.status = displayStatus;
    li.style.animationDelay = animate ? `${i * 90}ms` : "0ms";

    let fixCellInner = "";
    if (spinPid === p.pass_id) {
      fixCellInner =
        '<span class="pass-fix-spinner" role="status" aria-label="Fix running"></span>';
    } else if (
      p.status === "warn" &&
      PASS_FIX_TYPES.has(p.pass_id) &&
      !(fixOverride && fixOverride.passCleared)
    ) {
      fixCellInner = `<button type="button" class="pass-fix-btn" data-fix-pass="${escapeHtml(p.pass_id)}">Fix →</button>`;
    }

    li.innerHTML = `
      <span class="pass-icon" aria-hidden="true">${glyph}</span>
      <div class="pass-body">
        <span class="pass-name">${escapeHtml(p.pass_id)}</span>
        <span class="pass-summary">${escapeHtml(displaySummary)}</span>
      </div>
      <span class="pass-fix-cell">${fixCellInner}</span>
      <span class="pass-duration">${(p.duration_ms || 0).toFixed(1)}ms</span>
    `;

    const fixBtn = li.querySelector(".pass-fix-btn");
    if (fixBtn) {
      fixBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        runFixFromPass(fixBtn.getAttribute("data-fix-pass"));
      });
    }

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

async function runFixFromPass(fixType) {
  const hint = $("statusHint");
  const original_prompt = $("prompt").value.trim();
  const cand = getCandidate(state.selectedId);
  if (!cand || !original_prompt) {
    if (hint) hint.textContent = "Need a design brief and candidate to run a fix.";
    return;
  }
  if (!PASS_FIX_TYPES.has(fixType)) return;

  beginFixTerminalTrace(fixType);

  passFixState.spinningPassId = fixType;
  renderPasses(cand.passes, { animate: false });
  if (hint) hint.textContent = `Fix running (${fixType}) · see Compiler output → Live trace`;

  try {
    appendFixTerminalLine("POST /api/fix (single candidate, ~same cost as one compile variant)…");
    const res = await fetch("/api/fix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_prompt,
        current_sequence: cleanSeq(cand.sequence),
        fix_type: fixType,
        candidates: state.candidates,
      }),
    });
    appendFixTerminalLine(
      `HTTP ${res.status} ${res.ok ? "OK" : "error"} · reading body…`,
    );
    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }
    if (!res.ok) throw new Error(data.error || res.statusText);
    if (!data.fix || !data.fix.new_candidate_id) {
      throw new Error("Invalid fix response");
    }

    appendFixTerminalLine(
      `Response OK · new candidate #${data.fix.new_candidate_index} · id ${data.fix.new_candidate_id}`,
    );

    passFixState.spinningPassId = null;
    passFixState.lastFix = {
      passId: data.fix.fix_type,
      newCandidateId: data.fix.new_candidate_id,
      candNum: data.fix.new_candidate_index,
      stillFlagged: !!data.fix.still_flagged,
    };
    state.candidates = Array.isArray(data.candidates) ? data.candidates : [];
    state.selectedId = data.fix.new_candidate_id;

    paintLiveCompileWorkspace(data, { partial: false, animatePasses: false });

    const nid = data.fix.new_candidate_id;
    const row = document.querySelector(`#candidatesList .candidate-row[data-id="${CSS.escape(nid)}"]`);
    if (row) row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const pareto = $("paretoSection");
    if (pareto && !pareto.hidden) {
      pareto.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const sid =
      data.snapshot_id != null && SNAPSHOT_ID_RE.test(String(data.snapshot_id).trim().toLowerCase())
        ? String(data.snapshot_id).trim().toLowerCase()
        : null;
    if (sid) {
      state.snapshotIdForBar = sid;
      replaceUrlWithSnapshot(sid);
      updateSnapshotBar(sid);
    }

    if (hint) hint.textContent = "Fix complete — new candidate added to the Pareto set.";

    const badge = $("streamBadge");
    if (badge) {
      badge.textContent = data.fix.still_flagged ? "Fix done · review passes" : "Fixed";
      badge.classList.add("done");
    }

    const extra = [
      `Merged & re-ranked: ${data.candidates.length} candidate(s)`,
      data.fix.still_flagged
        ? `Pass “${data.fix.fix_type}” still flagged on new sequence — try Fix again or edit the brief`
        : `Pass “${data.fix.fix_type}” cleared on new sequence (see Compiler passes)`,
    ];
    endFixTerminalTrace({
      tickerText: data.fix.still_flagged
        ? `Fix · finished · candidate #${data.fix.new_candidate_index} · review warnings`
        : `Fix · finished · candidate #${data.fix.new_candidate_index} · done`,
      extraLogLines: extra,
    });
  } catch (e) {
    console.error(e);
    passFixState.spinningPassId = null;
    const c2 = getCandidate(state.selectedId);
    if (c2) renderPasses(c2.passes, { animate: false });
    if (hint) hint.textContent = `Fix failed: ${e.message || e}`;
    const badge = $("streamBadge");
    if (badge) {
      badge.textContent = "Fix failed";
      badge.classList.remove("done");
    }
    endFixTerminalTrace({
      tickerText: `Fix · failed · ${e.message || e}`,
      extraLogLines: [`Error: ${e.message || e}`],
    });
  }
}

function renderExpertQa(rag) {
  const card = $("expertQaCard");
  const metaEl = $("expertQaMeta");
  const body = $("expertQaBody");
  if (!card || !body) return;
  const lint = rag && rag.expert_lint;
  const rev = rag && rag.expert_review;
  if (!lint && !rev) {
    card.hidden = true;
    body.innerHTML = "";
    if (metaEl) metaEl.textContent = "";
    return;
  }
  card.hidden = false;
  if (metaEl) {
    if (lint && !lint.error) {
      const g = lint.grade || "—";
      const sc = lint.score != null ? lint.score : "—";
      metaEl.textContent = `${g} · score ${sc}`;
    } else {
      metaEl.textContent = "";
    }
  }
  const chunks = [];
  if (lint && !lint.error) {
    chunks.push(`<p class="expert-qa-summary">${escapeHtml(lint.summary || "")}</p>`);
    if (Array.isArray(lint.issues) && lint.issues.length) {
      chunks.push('<ul class="expert-qa-issues">');
      for (const iss of lint.issues) {
        const sev = iss.severity === "error" ? "error" : "warn";
        chunks.push(`<li class="is-${sev}">${escapeHtml(iss.message || "")}</li>`);
      }
      chunks.push("</ul>");
    }
  } else if (lint && lint.error) {
    chunks.push(`<p class="expert-qa-err">${escapeHtml(lint.error)}</p>`);
  }
  if (rev && !rev.error) {
    if (rev.summary) {
      const v = rev.verdict ? ` · ${rev.verdict}` : "";
      chunks.push(
        `<p class="expert-qa-rev"><strong>Model review</strong>${escapeHtml(v)}: ${escapeHtml(rev.summary)}</p>`,
      );
    }
    if (Array.isArray(rev.concerns) && rev.concerns.length) {
      chunks.push('<ul class="expert-qa-concerns">');
      for (const c of rev.concerns) {
        chunks.push(`<li>${escapeHtml(c)}</li>`);
      }
      chunks.push("</ul>");
    }
  } else if (rev && rev.error) {
    chunks.push(`<p class="expert-qa-err">${escapeHtml(rev.error)}</p>`);
  }
  body.innerHTML = chunks.join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderRagPanel(rag) {
  const card = $("ragCard");
  const summary = $("ragSummary");
  const list = $("ragList");
  if (!card || !summary || !list) return;

  if (!rag || rag.enabled === false) {
    card.hidden = true;
    summary.textContent = "";
    list.innerHTML = "";
    return;
  }

  if (rag.error) {
    card.hidden = false;
    summary.textContent = `RAG unavailable: ${rag.error}`;
    list.innerHTML = "";
    return;
  }

  const parts = Array.isArray(rag.parts) ? rag.parts : [];
  if (!rag.applied || parts.length === 0) {
    card.hidden = true;
    summary.textContent = "";
    list.innerHTML = "";
    return;
  }

  card.hidden = false;
  const minS = rag.min_similarity != null ? rag.min_similarity : 0.6;
  const minPromo = rag.min_similarity_promoter;
  const promoNote =
    minPromo != null && Number(minPromo) > Number(minS)
      ? `, promoters ≥ ${Number(minPromo).toFixed(2)}`
      : "";
  const nIgem = parts.filter((p) => p.sequence_source === "registry").length;
  const nNcbi = parts.filter((p) => p.sequence_source === "ncbi").length;
  const nModel = parts.length - nIgem - nNcbi;
  summary.textContent = `${nIgem} iGEM (sim ≥ ${minS}${promoNote}) · ${nNcbi} NCBI Gene · ${nModel} model / unverified.`;

  list.innerHTML = "";
  for (const p of parts) {
    const li = document.createElement("li");
    li.className = p.verified ? "is-verified" : "is-unverified";
    const title = document.createElement("div");
    title.className = "rag-part-title";
    if (p.sequence_source === "ncbi" && p.verified) {
      const gn = p.ncbi_gene_name || p.part_name || "—";
      title.textContent = `${gn} · NCBI Gene`;
    } else if (p.verified) {
      title.textContent = `${p.part_name || "—"} · ${p.part_type || ""}`;
    } else if (p.part_name) {
      title.textContent = `${p.part_name} · model DNA kept (${p.reject_reason === "below_similarity_threshold" ? "below threshold" : "no verified hit"})`;
    } else {
      title.textContent = "Model DNA kept (no verified registry hit)";
    }
    const meta = document.createElement("span");
    meta.className = "rag-part-meta";
    const sim =
      p.similarity != null && p.similarity !== ""
        ? Number(p.similarity).toFixed(3)
        : "—";
    const q = p.query || p.retrieval_query || "";
    if (p.sequence_source === "ncbi") {
      const org = p.ncbi_organism ? String(p.ncbi_organism) : "";
      const acc = p.ncbi_accession ? `${p.ncbi_accession}:${p.ncbi_range || ""}` : "";
      meta.textContent = [q, org, acc].filter(Boolean).join(" · ");
    } else {
      meta.textContent = p.verified ? `${q} · sim ${sim}` : `${q} · best sim ${sim}`;
    }
    li.appendChild(title);
    li.appendChild(meta);
    list.appendChild(li);
  }
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

    const fixLabel =
      c.fix_badge ? `<span class="candidate-fix-badge">${escapeHtml(c.fix_badge)}</span>` : "";
    li.innerHTML = `
      <span class="candidate-pareto" title="${c.is_pareto ? "Pareto-optimal" : "Dominated"}">${c.is_pareto ? "★" : "○"}</span>
      <div class="candidate-meta">
        <span class="candidate-strategy">${escapeHtml(c.strategy_name || c.id)}${fixLabel}</span>
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

/** Client-side compile pipeline animation (real work is one POST — phases are staged UX). */
const COMPILE_PHASES = [
  {
    title: "Neural handshake",
    sub: "Generative Language API · Gemma-4 binding",
    ticker: "Establishing session to hosted inference…",
    bar: 10,
    atMs: 0,
  },
  {
    title: "Output contract",
    sub: "<|channel>thought · ATCG-only DNA strip",
    ticker: "Locking OpenGeneEdit compiler output envelope…",
    bar: 21,
    atMs: 950,
  },
  {
    title: "Design sampling",
    sub: "Sequential completions · temperature ladder",
    ticker: "Drawing diverse candidate constructs from your brief…",
    bar: 46,
    atMs: 2600,
  },
  {
    title: "Construct extraction",
    sub: "Thought strip · alphabet gate · parse",
    ticker: "Extracting sequences & validating bases…",
    bar: 64,
    atMs: 6800,
  },
  {
    title: "Static passes",
    sub: "ORF · GC · repeats · hygiene screens",
    ticker: "Scoring candidates for lab plausibility…",
    bar: 81,
    atMs: 13800,
  },
  {
    title: "Rank & Pareto",
    sub: "Composite score · non-dominated front",
    ticker: "Ordering candidates & picking default best…",
    bar: 94,
    atMs: 23500,
  },
];

let compilePipelineCleanup = null;
/** @type {AbortController | null} */
let compileAbort = null;
/** @type {ReturnType<typeof setInterval> | null} */
let fixTraceInterval = null;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

/** In docked live compile, overflow is on `.compile-terminal-scroll`, not the `<pre>`. */
function scrollCompileOutputToLatest() {
  const debugPre = $("compileDebugLog");
  const livePre = $("compileLivePre");
  const wrap =
    (debugPre && debugPre.closest(".compile-terminal-scroll")) ||
    (livePre && livePre.closest(".compile-terminal-scroll")) ||
    null;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (wrap) wrap.scrollTop = wrap.scrollHeight;
      if (debugPre) debugPre.scrollTop = debugPre.scrollHeight;
      if (livePre) livePre.scrollTop = livePre.scrollHeight;
    });
  });
}

function hideRegistryIndexBanner() {
  const banner = $("registryIndexBanner");
  if (banner) banner.hidden = true;
}

/**
 * When Chroma first embeds the iGEM JSONL, the server emits `rag · indexing…` and
 * `rag · indexed cur/total…`. Show a red explainer so first-time hosted users do not
 * assume the app is stuck.
 */
function updateRegistryIndexBannerFromCompileLines(lines) {
  const banner = $("registryIndexBanner");
  const detail = $("registryIndexBannerDetail");
  if (!banner) return;
  const text = Array.isArray(lines) ? lines.join("\n") : "";
  const indexingStart = /rag · indexing iGEM parts/i.test(text);
  let lastCur = 0;
  let lastTot = 0;
  const re = /rag · indexed (\d+)\/(\d+) parts/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    lastCur = parseInt(m[1], 10);
    lastTot = parseInt(m[2], 10);
  }
  const indexingProgress = lastTot > 0 && lastCur < lastTot;
  const indexingDone = lastTot > 0 && lastCur >= lastTot;
  if (indexingDone) {
    banner.hidden = true;
    return;
  }
  if (indexingStart || indexingProgress) {
    banner.hidden = false;
    if (detail) {
      if (indexingProgress) {
        detail.textContent = `Embedding registry parts: ${lastCur.toLocaleString()} / ${lastTot.toLocaleString()}. This is normal on first use and can take several minutes on this server. Later compiles skip this step until the server is redeployed.`;
      } else {
        detail.textContent =
          "This is normal on first use: the app builds a local iGEM search index. It can take several minutes on this host. Later compiles reuse the index and skip this step until the server is redeployed.";
      }
    }
  }
}

/** Latest line in ticker; scrollback in monospace panel */
function updateLiveCompileTrace(lines) {
  const pre = $("compileDebugLog");
  const ticker = $("compileTicker");
  const safe = Array.isArray(lines) ? lines : [];
  updateRegistryIndexBannerFromCompileLines(safe);
  if (pre) {
    pre.textContent = safe.join("\n");
    pre.hidden = safe.length === 0;
    pre.scrollTop = pre.scrollHeight;
  }
  if (ticker && safe.length) {
    ticker.textContent = safe[safe.length - 1];
  }
  scrollCompileOutputToLatest();
}

function fixTraceTimeStamp() {
  return new Date().toLocaleTimeString(undefined, { hour12: false });
}

/** Open the docked terminal and show that a targeted fix is running (compile panel is often hidden after job ends). */
function beginFixTerminalTrace(fixType) {
  const pipe = $("compilePipeline");
  const idle = $("terminalIdleHint");
  const pre = $("compileDebugLog");
  const ticker = $("compileTicker");
  const chip = $("compileChip");
  const elapsedEl = $("compileElapsed");
  const counter = $("compileStepCounter");
  const slot = $("compileStepSlot");
  const barWrap = document.querySelector(".compile-pipeline-bar");
  const fill = $("compilePipelineFill");

  if (fixTraceInterval) {
    clearInterval(fixTraceInterval);
    fixTraceInterval = null;
  }
  if (idle) idle.hidden = true;
  if (pipe) {
    pipe.hidden = false;
    pipe.setAttribute("aria-busy", "true");
  }
  if (chip) chip.textContent = "Fix";
  if (counter) counter.textContent = "Targeted fix";
  if (slot) slot.innerHTML = "";
  if (fill) fill.style.width = "50%";
  if (barWrap) barWrap.setAttribute("aria-valuenow", "50");

  const t0 = Date.now();
  if (elapsedEl) elapsedEl.textContent = "0:00";
  fixTraceInterval = setInterval(() => {
    if (elapsedEl) elapsedEl.textContent = formatElapsed(Date.now() - t0);
  }, 380);

  if (pre) {
    pre.textContent = `${fixTraceTimeStamp()}  Fix · start · ${fixType} · same compile pipeline · awaiting server…`;
    pre.hidden = false;
  }
  if (ticker) {
    ticker.textContent = `Fix · running (${fixType}) · inference + passes…`;
  }
  const badge = $("streamBadge");
  if (badge) {
    badge.textContent = "Fix…";
    badge.classList.remove("done");
  }
  scrollCompileOutputToLatest();
}

function appendFixTerminalLine(message) {
  const pre = $("compileDebugLog");
  const ticker = $("compileTicker");
  if (!pre) return;
  pre.textContent += `\n${fixTraceTimeStamp()}  ${message}`;
  pre.hidden = false;
  if (ticker) {
    const short = message.length > 140 ? `${message.slice(0, 137)}…` : message;
    ticker.textContent = short;
  }
  scrollCompileOutputToLatest();
}

function endFixTerminalTrace({ tickerText, extraLogLines = [] }) {
  if (fixTraceInterval) {
    clearInterval(fixTraceInterval);
    fixTraceInterval = null;
  }
  const pipe = $("compilePipeline");
  const ticker = $("compileTicker");
  const pre = $("compileDebugLog");
  const chip = $("compileChip");
  const fill = $("compilePipelineFill");
  const barWrap = document.querySelector(".compile-pipeline-bar");

  if (pipe) pipe.setAttribute("aria-busy", "false");
  if (chip) chip.textContent = "Gemma-4";
  if (fill) fill.style.width = "100%";
  if (barWrap) barWrap.setAttribute("aria-valuenow", "100");
  const counter = $("compileStepCounter");
  if (counter) counter.textContent = `Step 1 / ${COMPILE_PHASES.length}`;
  if (ticker && tickerText) ticker.textContent = tickerText;
  for (const ln of extraLogLines) {
    if (pre) pre.textContent += `\n${fixTraceTimeStamp()}  ${ln}`;
  }
  if (pre) pre.hidden = !String(pre.textContent || "").trim();
  scrollCompileOutputToLatest();
}

async function pollCompileJob(jobId, signal, onLines, onStreams, onPartialResult) {
  const url = `/api/compile/status?job_id=${encodeURIComponent(jobId)}`;
  while (true) {
    const res = await fetch(url, { signal });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    if (typeof onLines === "function") onLines(data.lines || []);
    if (typeof onStreams === "function") onStreams(data.streams || {});
    if (
      data.result &&
      data.result.partial === true &&
      typeof onPartialResult === "function"
    ) {
      onPartialResult(data.result);
    }
    if (data.done) {
      const errRaw = data.error;
      if (errRaw != null && errRaw !== "") {
        const errStr =
          typeof errRaw === "string" ? errRaw.trim() : String(errRaw).trim();
        throw new Error(errStr || "Compile failed (empty server error).");
      }
      return data.result;
    }
    await sleep(380);
  }
}

/** Partial raw output per candidate (streamGenerateContent SSE). */
function updateLiveStreams(streams) {
  const wrap = $("compileLiveStream");
  const pre = $("compileLivePre");
  if (!wrap || !pre) return;
  const keys = Object.keys(streams || {});
  if (!keys.length) {
    wrap.hidden = true;
    pre.textContent = "";
    return;
  }
  wrap.hidden = false;
  wrap.open = true;
  keys.sort();
  pre.textContent = keys.map((k) => `── ${k} ──\n${streams[k]}`).join("\n\n");
  pre.scrollTop = pre.scrollHeight;
  scrollCompileOutputToLatest();
}

/** Real backend trace (no fake timed phases). */
function startLiveCompilePipeline() {
  stopCompilePipelineVisual();

  const pipe = $("compilePipeline");
  const thoughtEyebrow = $("thoughtEyebrow");
  const thoughtWrap = $("thoughtWrap");
  const fill = $("compilePipelineFill");
  const barWrap = document.querySelector(".compile-pipeline-bar");
  const ticker = $("compileTicker");
  const elapsedEl = $("compileElapsed");
  const chip = $("compileChip");
  const slot = $("compileStepSlot");
  const counter = $("compileStepCounter");
  const pre = $("compileDebugLog");
  const idleHint = $("terminalIdleHint");

  if (thoughtEyebrow) thoughtEyebrow.hidden = true;
  if (thoughtWrap) thoughtWrap.hidden = true;
  if (idleHint) idleHint.hidden = true;
  if (pipe) {
    pipe.hidden = false;
    pipe.setAttribute("aria-busy", "true");
  }
  if (chip) chip.textContent = "Gemma-4";
  if (counter) counter.textContent = "Live backend trace";
  if (slot) slot.innerHTML = "";
  if (ticker) ticker.textContent = "Connecting to compiler job…";
  if (pre) {
    pre.textContent = "";
    pre.hidden = true;
  }
  if (fill) fill.style.width = "32%";
  if (barWrap) barWrap.setAttribute("aria-valuenow", "32");

  const t0 = Date.now();
  const elapsedIv = setInterval(() => {
    if (elapsedEl) elapsedEl.textContent = formatElapsed(Date.now() - t0);
  }, 380);

  const barNudge = setInterval(() => {
    if (!fill) return;
    const w = 22 + ((Date.now() / 2800) % 1) * 58;
    fill.style.width = `${w.toFixed(0)}%`;
    if (barWrap) barWrap.setAttribute("aria-valuenow", String(Math.round(w)));
  }, 420);

  compilePipelineCleanup = () => {
    clearInterval(elapsedIv);
    clearInterval(barNudge);
    hideRegistryIndexBanner();
    if (pipe) {
      pipe.hidden = true;
      pipe.setAttribute("aria-busy", "false");
    }
    if (thoughtEyebrow) thoughtEyebrow.hidden = false;
    if (thoughtWrap) thoughtWrap.hidden = false;
    if (fill) fill.style.width = "4%";
    if (barWrap) barWrap.setAttribute("aria-valuenow", "0");
    if (elapsedEl) elapsedEl.textContent = "0:00";
    if (pre) {
      pre.textContent = "";
      pre.hidden = true;
    }
    const live = $("compileLiveStream");
    const livePre = $("compileLivePre");
    if (live) {
      live.hidden = true;
      live.open = false;
    }
    if (livePre) livePre.textContent = "";
    if (counter) counter.textContent = `Step 1 / ${COMPILE_PHASES.length}`;
    const ch = $("compileChip");
    if (ch) ch.textContent = "Gemma-4";
    const tick = $("compileTicker");
    if (tick) tick.textContent = "Idle · compile to stream logs";
    const idleH = $("terminalIdleHint");
    if (idleH) idleH.hidden = false;
  };
}

function stopCompilePipelineVisual() {
  if (typeof compilePipelineCleanup === "function") {
    compilePipelineCleanup();
    compilePipelineCleanup = null;
  }
}

function renderPipelineStepSlot(slot, counter, i, mode) {
  const ph = COMPILE_PHASES[i];
  if (!ph || !slot) return;
  const isDone = mode === "done";
  const glyph = isDone ? "✓" : "●";
  const cls = isDone ? "is-done" : "is-active";
  slot.innerHTML = `
    <div class="compile-step compile-step--solo ${cls}">
      <span class="compile-step-glyph" aria-hidden="true">${glyph}</span>
      <span class="compile-step-body">
        <span class="compile-step-title">${escapeHtml(ph.title)}</span>
        <span class="compile-step-sub">${escapeHtml(ph.sub)}</span>
      </span>
    </div>`;
  if (counter) counter.textContent = `Step ${i + 1} / ${COMPILE_PHASES.length}`;
}

function finishCompilePipelineSuccess() {
  const slot = $("compileStepSlot");
  const counter = $("compileStepCounter");
  const fill = $("compilePipelineFill");
  const barWrap = document.querySelector(".compile-pipeline-bar");
  const ticker = $("compileTicker");
  const last = COMPILE_PHASES.length - 1;
  renderPipelineStepSlot(slot, counter, last, "done");
  if (fill) fill.style.width = "100%";
  if (barWrap) barWrap.setAttribute("aria-valuenow", "100");
  if (ticker) ticker.textContent = "Pipeline complete · streaming reasoning…";
}

function startCompilePipeline(numCandidates) {
  stopCompilePipelineVisual();

  const pipe = $("compilePipeline");
  const thoughtEyebrow = $("thoughtEyebrow");
  const thoughtWrap = $("thoughtWrap");
  const fill = $("compilePipelineFill");
  const barWrap = document.querySelector(".compile-pipeline-bar");
  const ticker = $("compileTicker");
  const elapsedEl = $("compileElapsed");
  const chip = $("compileChip");
  const slot = $("compileStepSlot");
  const counter = $("compileStepCounter");

  if (thoughtEyebrow) thoughtEyebrow.hidden = true;
  if (thoughtWrap) thoughtWrap.hidden = true;
  if (pipe) {
    pipe.hidden = false;
    pipe.setAttribute("aria-busy", "true");
  }
  if (chip) chip.textContent = "Gemma-4";

  let phasePtr = 0;
  let variantRot = 1;

  function applyMetrics(i) {
    phasePtr = i;
    const ph = COMPILE_PHASES[i];
    if (!ph) return;
    if (fill) fill.style.width = `${ph.bar}%`;
    if (barWrap) barWrap.setAttribute("aria-valuenow", String(Math.round(ph.bar)));
    if (ticker) ticker.textContent = ph.ticker;
    if (chip) {
      if (i <= 1) chip.textContent = "Gemma-4";
      else if (i >= 2 && i <= 4) chip.textContent = `Variant ${variantRot}/${numCandidates}`;
      else chip.textContent = "Rank";
    }
  }

  renderPipelineStepSlot(slot, counter, 0, "active");
  applyMetrics(0);

  const timers = [];
  const transitionTimers = [];

  COMPILE_PHASES.forEach((ph, i) => {
    if (i === 0) return;
    timers.push(
      setTimeout(() => {
        renderPipelineStepSlot(slot, counter, i - 1, "done");
        transitionTimers.push(
          setTimeout(() => {
            renderPipelineStepSlot(slot, counter, i, "active");
            applyMetrics(i);
          }, 420)
        );
      }, ph.atMs)
    );
  });

  const t0 = Date.now();
  const elapsedIv = setInterval(() => {
    if (elapsedEl) elapsedEl.textContent = formatElapsed(Date.now() - t0);
  }, 380);

  const candIv = setInterval(() => {
    if (phasePtr >= 2 && phasePtr <= 4) {
      variantRot = (variantRot % numCandidates) + 1;
      if (chip) chip.textContent = `Variant ${variantRot}/${numCandidates}`;
    }
  }, 2400);

  compilePipelineCleanup = () => {
    timers.forEach((id) => clearTimeout(id));
    transitionTimers.forEach((id) => clearTimeout(id));
    clearInterval(elapsedIv);
    clearInterval(candIv);
    if (pipe) {
      pipe.hidden = true;
      pipe.setAttribute("aria-busy", "false");
    }
    if (thoughtEyebrow) thoughtEyebrow.hidden = false;
    if (thoughtWrap) thoughtWrap.hidden = false;
    if (fill) fill.style.width = "4%";
    if (barWrap) barWrap.setAttribute("aria-valuenow", "0");
    if (elapsedEl) elapsedEl.textContent = "0:00";
    if (slot) slot.innerHTML = "";
    const ch = $("compileChip");
    if (ch) ch.textContent = "Gemma-4";
  };
}

/** Switch the active candidate and re-render every dependent surface. */
function selectCandidate(id, { animatePasses = false } = {}) {
  if (passFixState.lastFix && id !== passFixState.lastFix.newCandidateId) {
    passFixState.lastFix = null;
  }
  const cand = getCandidate(id);
  if (!cand) return;
  state.selectedId = id;

  // Assistant bubble: replace thought instantly (no streaming on switch).
  $("thoughtText").textContent = cand.thought;

  // Plasmid map + sequence + metrics tied to selected candidate.
  const seq = cleanSeq(cand.sequence);
  lastSequence = seq;
  const mapSlots = cand.rag && Array.isArray(cand.rag.map_slots) ? cand.rag.map_slots : null;
  renderPlasmid(seq, cand.strategy || "compiled_construct", mapSlots);
  $("mLen").textContent = `${seq.length} bp`;
  $("mGc").textContent = `${gcPercent(seq).toFixed(2)}%`;
  const cai = passMetricFor(cand.passes, "cai");
  $("mTm").textContent = cai ? `CAI ${cai}` : `${wallaceTm(seq).toFixed(0)} °C`;
  $("seqPre").textContent = seq;

  renderPasses(cand.passes, { animate: animatePasses });
  renderCandidates(state.candidates, id);
  renderParetoChart(state.candidates, id);
  renderRagPanel(cand.rag);
  renderExpertQa(cand.rag);

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
  clearSnapshotFromUrl();
  state.snapshotIdForBar = null;
  state.selectedId = null;
  state.candidates = [];
  lastSequence = "";
  if ($("seqPre")) $("seqPre").textContent = "";
  lastPartialPassesSig = null;
  passFixState.spinningPassId = null;
  passFixState.lastFix = null;
  updateSnapshotBar(null);

  $("userBubbleText").textContent = prompt;
  $("userBubble").hidden = false;
  $("thoughtText").textContent = "";
  badge.textContent = "Running";
  badge.classList.remove("done");
  $("metrics").hidden = true;
  $("seqCard").hidden = true;
  const ragEl1 = $("ragCard");
  if (ragEl1) ragEl1.hidden = true;
  $("passesCard").hidden = true;
  const expertQaEl0 = $("expertQaCard");
  if (expertQaEl0) expertQaEl0.hidden = true;
  $("candidatesCard").hidden = true;
  $("paretoSection").hidden = true;
  $("passesList").innerHTML = "";
  $("candidatesList").innerHTML = "";

  const nCand = 4;
  startLiveCompilePipeline();
  compileAbort = new AbortController();

  btn.disabled = true;
  hint.textContent =
    "Live trace below · variants may appear one at a time as each finishes inferencing + registry pass";

  try {
    const res = await fetch("/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, n: nCand, progress: true }),
      signal: compileAbort.signal,
    });

    let data;
    let openedWorkspaceEarly = false;
    const onPartialResult = (partial) => {
      if (!openedWorkspaceEarly) {
        openedWorkspaceEarly = true;
        finishCompilePipelineSuccess();
        stopCompilePipelineVisual();
      }
      paintLiveCompileWorkspace(partial, { partial: true, animatePasses: false });
      const vr = partial.variants_ready ?? "—";
      const vt = partial.variants_total ?? "—";
      hint.textContent = `Showing best of ${vr}/${vt} · more generating…`;
      badge.textContent = `Partial (${vr}/${vt})`;
      badge.classList.remove("done");
    };

    if (res.status === 202) {
      const meta = await res.json();
      if (!meta.job_id) throw new Error("Server did not return job_id");
      data = await pollCompileJob(
        meta.job_id,
        compileAbort.signal,
        updateLiveCompileTrace,
        updateLiveStreams,
        onPartialResult
      );
    } else {
      data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
    }

    state.candidates = Array.isArray(data.candidates) ? data.candidates : [];
    if (!state.candidates.length) throw new Error("Compiler returned no candidates");

    if (!openedWorkspaceEarly) {
      finishCompilePipelineSuccess();
      await sleep(420);
      stopCompilePipelineVisual();
    }

    paintLiveCompileWorkspace(data, {
      partial: false,
      animatePasses: openedWorkspaceEarly ? false : true,
    });

    const activeCand = getCandidate(state.selectedId);
    if (!activeCand) throw new Error("missing best candidate");

    const sid =
      data.snapshot_id != null && SNAPSHOT_ID_RE.test(String(data.snapshot_id).trim().toLowerCase())
        ? String(data.snapshot_id).trim().toLowerCase()
        : null;
    if (sid) {
      state.snapshotIdForBar = sid;
      replaceUrlWithSnapshot(sid);
      updateSnapshotBar(sid);
    } else {
      state.snapshotIdForBar = null;
      updateSnapshotBar(null);
    }

    const thoughtEl = $("thoughtText");
    if (openedWorkspaceEarly) {
      thoughtEl.textContent = activeCand.thought || "";
      badge.textContent = "Done";
      badge.classList.add("done");
    } else {
      thoughtEl.textContent = "";
      badge.textContent = "Reasoning";
      await streamThought(thoughtEl, activeCand.thought);
      badge.textContent = "Done";
      badge.classList.add("done");
    }
    hint.textContent = `Ready · ${state.candidates.length} candidates`;
    $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
  } catch (e) {
    if (e.name === "AbortError") {
      hint.textContent = "Compile cancelled";
      badge.textContent = "…";
      $("thoughtText").textContent = "";
      stopCompilePipelineVisual();
      return;
    }
    console.error(e);
    const errMsg =
      (e && typeof e.message === "string" && e.message.trim()) ||
      (e != null ? String(e) : "") ||
      "Compile failed";
    hint.textContent = errMsg;
    stopCompilePipelineVisual();
    $("thoughtText").textContent = `Something went wrong: ${errMsg}`;
    badge.textContent = "Error";
    badge.classList.remove("done");
  } finally {
    compileAbort = null;
    btn.disabled = false;
  }
}

const snapshotCopyBtn = $("snapshotCopyBtn");
if (snapshotCopyBtn) {
  snapshotCopyBtn.addEventListener("click", async () => {
    const sid = state.snapshotIdForBar;
    const fb = $("snapshotCopyFeedback");
    if (!sid || !SNAPSHOT_ID_RE.test(sid)) return;
    const href = `${window.location.origin}${buildSnapshotPageUrl(sid)}`;
    try {
      await navigator.clipboard.writeText(href);
      if (fb) fb.textContent = "Copied";
      snapshotCopyBtn.classList.add("is-copied");
      setTimeout(() => {
        if (fb) fb.textContent = "";
        snapshotCopyBtn.classList.remove("is-copied");
      }, 1600);
    } catch {
      if (fb) fb.textContent = "Copy failed";
      setTimeout(() => {
        if (fb) fb.textContent = "";
      }, 2000);
    }
  });
}

const compileBtnEl = $("compileBtn");
if (compileBtnEl) compileBtnEl.addEventListener("click", compile);

const newDesignBtnEl = $("newDesignBtn");
if (newDesignBtnEl) newDesignBtnEl.addEventListener("click", showLanding);

/** Optional nav control (not always present in index.html). Must not throw — a null ref aborts the whole module. */
const topnavHomeEl = $("topnavHome");
if (topnavHomeEl) {
  topnavHomeEl.addEventListener("click", (e) => {
    if (!$("workspace").hidden) {
      e.preventDefault();
      showLanding();
    }
  });
}

const promptEl = $("prompt");
if (promptEl) {
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      compile();
    }
  });
}

function onExportClick(format) {
  const hint = $("statusHint");
  const c = getCandidate(state.selectedId);
  exportDebug("onExportClick", format, {
    selectedId: state.selectedId,
    sequenceFieldBp: c && c.sequence != null ? String(c.sequence).length : 0,
    lastSequenceBp: lastSequence ? lastSequence.length : 0,
    seqPreBp: $("seqPre") && $("seqPre").textContent ? $("seqPre").textContent.length : 0,
    candidates: state.candidates ? state.candidates.length : 0,
  });
  const seq = dnaSequenceForTools();
  if (!seq) {
    exportDebug("onExportClick: empty sequence after dnaSequenceForTools");
    if (hint) {
      hint.textContent =
        "No DNA loaded for export yet. If a compile is still running, wait until at least one variant finishes.";
    }
    return;
  }
  if (format === "fasta") downloadText("compiled_sequence.fasta", toFasta(seq));
  else if (format === "genbank") downloadText("compiled_sequence.gb", toGenbank(seq));
}

/** Delegation: clicks on label text inside buttons still reach `#dlFasta` / `#dlGb`. */
document.body.addEventListener("click", (e) => {
  const el = e.target;
  if (!(el instanceof Element)) return;
  const fasta = el.closest("#dlFasta");
  const gb = el.closest("#dlGb");
  if (fasta) {
    e.preventDefault();
    onExportClick("fasta");
  } else if (gb) {
    e.preventDefault();
    onExportClick("genbank");
  }
});

const seqCopyBtnEl = $("seqCopyBtn");
if (seqCopyBtnEl) {
  seqCopyBtnEl.addEventListener("click", async () => {
    const hint = $("statusHint");
    const seq = dnaSequenceForTools();
    if (!seq) {
      if (hint) {
        hint.textContent =
          "No DNA loaded to copy yet. If a compile is still running, wait for a finished variant.";
      }
      return;
    }
    const btn = $("seqCopyBtn");
    const label = $("seqCopyLabel");
    try {
      await navigator.clipboard.writeText(seq);
      if (label) label.textContent = "Copied";
      if (btn) btn.classList.add("is-copied");
      setTimeout(() => {
        if (label) label.textContent = "Copy";
        if (btn) btn.classList.remove("is-copied");
      }, 1500);
    } catch {
      if (label) label.textContent = "Failed";
      setTimeout(() => {
        if (label) label.textContent = "Copy";
      }, 1500);
    }
  });
}

const plasmidSvgEl = $("plasmidSvg");
const plasmidViewport = $("viewport");
if (plasmidSvgEl && plasmidViewport) attachPlasmidNav(plasmidSvgEl, plasmidViewport);

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrapApp);
} else {
  bootstrapApp();
}
