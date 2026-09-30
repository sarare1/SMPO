const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const OUT = process.argv[2] || "SMPO_Team_Presentation.pptx";

// Palette: graphite steel + safety amber (industrial), teal for "good", red for "risk"
const DARK = "1B2631", STEEL = "34495E", LIGHT = "EEF2F5", AMBER = "E8910C",
  TEAL = "138D75", RED = "C0392B", WHITE = "FFFFFF", TEXT = "1B2631", MUTED = "5F6B78",
  PALE = "B8C4D0";
const HF = "Calibri", BF = "Calibri", MONO = "Courier New";

async function icon(name, color = WHITE, size = 256) {
  const el = React.createElement(fa[name], { color: "#" + color, size: String(size) });
  const svg = ReactDOMServer.renderToStaticMarkup(el);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function iconCircle(slide, name, x, y, d, fill = AMBER, fg = WHITE) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  const p = d * 0.25;
  slide.addImage({ data: await icon(name, fg), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}

function card(slide, x, y, w, h, fill = LIGHT) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
}

function T(slide, text, opts) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: BF, color: TEXT, margin: 0, valign: "top" }, opts));
}

let pageNo = 0;
function header(slide, section, title) {
  pageNo++;
  slide.background = { color: WHITE };
  T(slide, section.toUpperCase(), { x: 0.5, y: 0.3, w: 9, h: 0.25, fontSize: 10, bold: true, color: AMBER, charSpacing: 2 });
  T(slide, title, { x: 0.5, y: 0.55, w: 9, h: 0.6, fontSize: 28, bold: true, fontFace: HF, color: DARK, valign: "middle" });
  T(slide, `SMPO  ·  ${pageNo + 1}`, { x: 8.0, y: 5.3, w: 1.5, h: 0.2, fontSize: 9, color: MUTED, align: "right" });
}

function bullets(items, opts = {}) {
  return items.map((it, i) => {
    const o = { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: opts.gap ?? 4 };
    if (typeof it === "string") return { text: it, options: o };
    return { text: it.text, options: Object.assign(o, it.options || {}) };
  });
}

const tableHead = (cells) => cells.map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: STEEL } } }));

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.title = "Smart Manufacturing Process Optimization";

  // ───────────────────────── 1. Title
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    T(s, "TEAM BRIEFING  ·  ARCHITECTURE, PROCESSES & SOLUTION", { x: 0.6, y: 0.55, w: 8.8, h: 0.3, fontSize: 11, bold: true, color: AMBER, charSpacing: 2 });
    T(s, "Smart Manufacturing\nProcess Optimization", { x: 0.6, y: 1.0, w: 8.8, h: 1.5, fontSize: 40, bold: true, color: DARK, fontFace: HF, valign: "middle" });
    T(s, "AI-driven predictive maintenance and workflow optimisation — from factory sensor data to a prioritised, explained action plan for the shift supervisor.",
      { x: 0.6, y: 2.6, w: 7.6, h: 0.75, fontSize: 15, color: MUTED });
    const steps = [["Predict", "FaChartLine"], ["Explain", "FaSearch"], ["Optimise", "FaCogs"], ["Decide", "FaRobot"], ["Act", "FaUserCheck"]];
    for (let i = 0; i < steps.length; i++) {
      const x = 0.6 + i * 1.75;
      await iconCircle(s, steps[i][1], x, 3.9, 0.5, AMBER);
      T(s, steps[i][0], { x: x + 0.6, y: 3.9, w: 1.05, h: 0.5, fontSize: 14, bold: true, color: DARK, valign: "middle" });
    }
    T(s, "Domain: Manufacturing & Industrial IoT", { x: 0.6, y: 4.85, w: 6, h: 0.3, fontSize: 11, color: MUTED, italic: true });
    s.addNotes("Welcome. This session walks through the Smart Manufacturing Process Optimization project: the problem, how the solution is architected, how data flows through it, and what results we get. The five words at the bottom are the backbone of the whole system: Predict, Explain, Optimise, Decide, Act.");
  }

  // ───────────────────────── 2. Agenda
  {
    const s = pres.addSlide();
    header(s, "Agenda", "What we'll cover");
    const items = [
      ["Problem & solution overview", "Why factories lose output and our 5-stage answer"],
      ["Datasets", "AI4I 2020 + Azure PdM, and why"],
      ["Architecture & request flow", "Layers, agent run sequence, factory replay"],
      ["Data pipeline & ML models", "Features, LightGBM, anomaly detection, results"],
      ["Optimisation & business case", "Setpoints, CP-SAT scheduling, OEE"],
      ["Decision agents (LLM layer)", "Agents, tools, backends, modes, safeguards"],
      ["Dashboard & operations", "7 pages, setup, config, testing"],
      ["Design decisions & roadmap", "Trade-offs, limitations, next steps"],
    ];
    for (let i = 0; i < items.length; i++) {
      const col = i < 4 ? 0 : 1, row = i % 4;
      const x = 0.5 + col * 4.6, y = 1.45 + row * 0.95;
      s.addShape("ellipse", { x, y, w: 0.55, h: 0.55, fill: { color: i % 2 ? STEEL : AMBER }, line: { color: i % 2 ? STEEL : AMBER } });
      T(s, String(i + 1).padStart(2, "0"), { x, y, w: 0.55, h: 0.55, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
      T(s, items[i][0], { x: x + 0.7, y: y - 0.02, w: 3.7, h: 0.32, fontSize: 15, bold: true });
      T(s, items[i][1], { x: x + 0.7, y: y + 0.3, w: 3.7, h: 0.3, fontSize: 11.5, color: MUTED });
    }
    s.addNotes("Eight sections. We'll go from the business problem down into architecture and models, then back up to business value, operations and the roadmap.");
  }

  // ───────────────────────── 3. Problem
  {
    const s = pres.addSlide();
    header(s, "01 · Problem", "Factories lose output to three linked problems");
    const cards = [
      ["FaClock", RED, "Machine downtime", "Components fail without warning. An unplanned repair stops a machine for 24 h vs 4 h for a planned one.",
        "Predict component failures 24 h and 7 days ahead; flag abnormal sensor behaviour."],
      ["FaThermometerHalf", AMBER, "Quality losses", "Cycles fail through heat build-up, power overload, overstrain or tool wear.",
        "Predict cycle failure and its mode, explain the physical cause, find safer setpoints."],
      ["FaCalendarAlt", STEEL, "Reactive workflows", "Maintenance is reactive or calendar-based — ignoring risk, crew capacity and production load.",
        "Schedule maintenance to minimise expected cost under crew and load constraints."],
    ];
    for (let i = 0; i < 3; i++) {
      const [ic, col, title, sym, fix] = cards[i];
      const x = 0.5 + i * 3.1, y = 1.45, w = 2.8;
      card(s, x, y, w, 3.7);
      await iconCircle(s, ic, x + 0.25, y + 0.25, 0.6, col);
      T(s, title, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.3, fontSize: 16, bold: true });
      T(s, "SYMPTOM", { x: x + 0.25, y: y + 1.45, w: w - 0.5, h: 0.2, fontSize: 9, bold: true, color: MUTED, charSpacing: 1 });
      T(s, sym, { x: x + 0.25, y: y + 1.68, w: w - 0.5, h: 0.85, fontSize: 11.5 });
      T(s, "WHAT WE DO", { x: x + 0.25, y: y + 2.55, w: w - 0.5, h: 0.2, fontSize: 9, bold: true, color: TEAL, charSpacing: 1 });
      T(s, fix, { x: x + 0.25, y: y + 2.78, w: w - 0.5, h: 0.8, fontSize: 11.5, bold: true, color: TEAL });
    }
    s.addNotes("Three problems that feed each other: downtime, quality losses and reactive maintenance. The key number is 24 hours of downtime for an unplanned failure versus 4 hours when planned. Every capability in the system maps to one of these three columns.");
  }

  // ───────────────────────── 4. Solution overview
  {
    const s = pres.addSlide();
    header(s, "02 · Solution overview", "From sensor data to action in five stages");
    const st = [
      ["FaChartLine", "Predict", "LightGBM + Isolation Forest: component failures (24 h, 7 d), cycle failures & modes"],
      ["FaSearch", "Explain", "SHAP feature contributions + known physical failure rules"],
      ["FaCogs", "Optimise", "Model-verified setpoint search; OR-Tools CP-SAT weekly maintenance plan"],
      ["FaRobot", "Decide", "LLM agents call models as tools → structured, prioritised action plan"],
      ["FaUserCheck", "Act", "Dashboard: accept / reject every recommendation; every decision logged"],
    ];
    for (let i = 0; i < 5; i++) {
      const x = 0.5 + i * 1.84, w = 1.66, y = 1.45;
      card(s, x, y, w, 2.45);
      await iconCircle(s, st[i][0], x + 0.2, y + 0.2, 0.5, i === 3 ? AMBER : STEEL);
      T(s, String(i + 1), { x: x + w - 0.5, y: y + 0.2, w: 0.3, h: 0.5, fontSize: 20, bold: true, color: PALE, align: "right", valign: "middle" });
      T(s, st[i][1], { x: x + 0.2, y: y + 0.8, w: w - 0.4, h: 0.35, fontSize: 15, bold: true });
      T(s, st[i][2], { x: x + 0.2, y: y + 1.17, w: w - 0.35, h: 1.2, fontSize: 10.5, color: MUTED });
      if (i < 4) T(s, "›", { x: x + w - 0.02, y: y + 1.0, w: 0.2, h: 0.4, fontSize: 22, bold: true, color: AMBER, align: "center", valign: "middle" });
    }
    card(s, 0.5, 4.1, 9.0, 1.05, DARK);
    await iconCircle(s, "FaShieldAlt", 0.7, 4.3, 0.65, AMBER);
    T(s, "Core principle: the LLM never predicts failures.", { x: 1.55, y: 4.22, w: 7.8, h: 0.35, fontSize: 15, bold: true, color: WHITE });
    T(s, "Trained models, optimiser and scheduler are the source of truth. The LLM interprets their output, connects evidence and communicates the decision. Setpoint recommendations are re-scored by the failure model before they are shown.",
      { x: 1.55, y: 4.57, w: 7.8, h: 0.55, fontSize: 10.5, color: PALE });
    s.addNotes("The pipeline: predict, explain, optimise, decide, act. Agents run on a free local model via Ollama or on Claude. The most important design principle: the LLM is a decision and communication layer, not a predictor. Numbers always come from measurable, testable models.");
  }

  // ───────────────────────── 5. Datasets
  {
    const s = pres.addSlide();
    header(s, "03 · Datasets", "Two complementary public datasets");
    const rows = [
      tableHead(["", "AI4I 2020 (UCI #601)", "Microsoft Azure PdM"]),
      ["Scope", "Machining line, one row per production cycle", "100 machines, hourly, calendar year 2015"],
      ["Size", "10,000 cycles, 3.39% failures", "876,100 telemetry rows · 3,919 errors · 3,286 replacements · 761 failures"],
      ["Signals", "Air/process temp, speed, torque, tool wear, quality variant L/M/H", "Voltage, rotation, pressure, vibration; error codes; model & age"],
      ["Labels", "Failure + modes: TWF, HDF, PWF, OSF, RNF", "Failure time per component (comp1–comp4)"],
      ["Used for", "Process risk, failure-mode diagnosis, setpoint optimisation, quality KPI", "Fleet health, component forecasts, anomalies, maintenance planning, availability"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: j === 0 ? { bold: true, color: STEEL } : {} })));
    s.addTable(rows, { x: 0.5, y: 1.4, w: 6.25, colW: [0.95, 2.65, 2.65], fontFace: BF, fontSize: 10.5, color: TEXT,
      border: { type: "solid", pt: 0.5, color: "D5DCE3" }, fill: { color: WHITE }, valign: "middle", margin: 0.06, rowH: 0.6 });
    card(s, 7.0, 1.4, 2.5, 3.75);
    T(s, "Evaluated, not used", { x: 7.2, y: 1.55, w: 2.1, h: 0.3, fontSize: 13, bold: true });
    T(s, [
      { text: "NASA C-MAPSS", options: { bold: true, breakLine: true } },
      { text: "Turbofan engines, not factory machines", options: { color: MUTED, breakLine: true } },
      { text: " ", options: { fontSize: 5, breakLine: true } },
      { text: "SECOM (UCI)", options: { bold: true, breakLine: true } },
      { text: "Quality only; 590 anonymised sensors", options: { color: MUTED, breakLine: true } },
      { text: " ", options: { fontSize: 5, breakLine: true } },
      { text: "Bosch Production Line", options: { bold: true, breakLine: true } },
      { text: "~14 GB, anonymised, not explainable", options: { color: MUTED } },
    ], { x: 7.2, y: 1.95, w: 2.1, h: 2.2, fontSize: 10.5 });
    T(s, "Both are synthetic (real physics / real plant patterns) — expect lower scores on real plant data.",
      { x: 7.2, y: 4.3, w: 2.1, h: 0.75, fontSize: 9.5, italic: true, color: RED });
    s.addNotes("AI4I explains WHY a cycle fails — failure modes map to physics, so predictions translate into process fixes. Azure gives the fleet time series with errors, maintenance and failures that we need for downtime, scheduling and OEE. Other datasets were rejected for domain mismatch, lack of explainability or size. Caveat: both are synthetic.");
  }

  // ───────────────────────── 6. Architecture layers
  {
    const s = pres.addSlide();
    header(s, "04 · Architecture", "Layered architecture");
    const cols = [
      ["Data layer", "src/data", STEEL, [["download.py", "UCI + GitHub mirror"], ["features.py", "physics features, rolling stats, error counts, component age, labels"]]],
      ["ML layer", "src/models", STEEL, [["train.py", "LightGBM, Isolation Forest"], ["predictor.py", "ProcessModel, FleetModel, SHAP"]]],
      ["Optimisation", "src/optimization", STEEL, [["process_optimizer.py", "what-if + setpoint search"], ["scheduler.py", "CP-SAT maintenance plan"], ["kpis.py", "OEE + business case"]]],
      ["Agent layer", "src/agents", AMBER, [["tools.py", "7 strict tools"], ["orchestrator.py", "agent modes, rule-based fallback"], ["runners", "Claude · ollama_runner.py"]]],
      ["Presentation", "app/", TEAL, [["Streamlit", "dashboard, 7 pages"], ["decisions.jsonl", "accept / reject log"]]],
    ];
    const w = 1.6, gap = 0.25, y0 = 1.4;
    for (let i = 0; i < cols.length; i++) {
      const [name, path, col, mods] = cols[i];
      const x = 0.5 + i * (w + gap);
      card(s, x, y0, w, 3.05);
      s.addShape("roundRect", { x, y: y0, w, h: 0.6, rectRadius: 0.08, fill: { color: col }, line: { color: col } });
      T(s, name, { x: x + 0.1, y: y0 + 0.05, w: w - 0.2, h: 0.3, fontSize: 12.5, bold: true, color: WHITE, align: "center" });
      T(s, path, { x: x + 0.1, y: y0 + 0.32, w: w - 0.2, h: 0.22, fontSize: 8.5, color: WHITE, fontFace: MONO, align: "center" });
      for (let j = 0; j < mods.length; j++) {
        const my = y0 + 0.72 + j * 0.76;
        s.addShape("roundRect", { x: x + 0.08, y: my, w: w - 0.16, h: 0.68, rectRadius: 0.05, fill: { color: WHITE }, line: { color: "D5DCE3", width: 0.75 } });
        T(s, mods[j][0], { x: x + 0.14, y: my + 0.05, w: w - 0.28, h: 0.2, fontSize: 7.5, bold: true, fontFace: MONO, color: DARK });
        T(s, mods[j][1], { x: x + 0.14, y: my + 0.27, w: w - 0.28, h: 0.38, fontSize: 8.5, color: MUTED });
      }
      if (i < cols.length - 1)
        s.addShape("line", { x: x + w + 0.02, y: y0 + 1.5, w: gap - 0.04, h: 0, line: { color: AMBER, width: 2, endArrowType: "triangle" } });
    }
    // Simulation feeding agents
    const sx = 0.5 + 2 * (w + gap);
    s.addShape("roundRect", { x: sx, y: 4.6, w: w * 2 + gap, h: 0.55, rectRadius: 0.06, fill: { color: WHITE }, line: { color: AMBER, width: 1.25, dashType: "dash" } });
    T(s, [{ text: "Simulation  ", options: { bold: true } }, { text: "src/simulation/stream.py — FactoryState replays held-out data → feeds agent tools", options: { color: MUTED } }],
      { x: sx + 0.12, y: 4.62, w: w * 2 + gap - 0.24, h: 0.5, fontSize: 9.5, valign: "middle" });
    T(s, "Models, optimiser and KPIs also feed the dashboard directly — the UI works even with no LLM.",
      { x: 0.5, y: 4.62, w: 2.9, h: 0.5, fontSize: 9.5, italic: true, color: MUTED, valign: "middle" });
    s.addNotes("Five layers, each a Python package. Data produces features; ML trains LightGBM and Isolation Forest models and wraps them in ProcessModel and FleetModel with SHAP. The optimisation layer turns predictions into setpoints, a maintenance plan and KPIs. The agent layer exposes all of that as seven strict tools to an orchestrator that can run Claude or Ollama, with a rule-based fallback. The simulation replays held-out data as a live factory. Streamlit sits on top.");
  }

  // ───────────────────────── 7. Request flow (sequence)
  {
    const s = pres.addSlide();
    header(s, "04 · Architecture", "Request flow for an agent run");
    const lanes = ["Supervisor\n(dashboard)", "Orchestrator", "ToolBox", "Models / optimiser\n/ scheduler", "LLM\n(Ollama or Claude)"];
    const cx = [1.2, 3.05, 4.9, 6.75, 8.6], lw = 1.6;
    for (let i = 0; i < 5; i++) {
      const col = i === 0 ? TEAL : i === 4 ? AMBER : STEEL;
      s.addShape("roundRect", { x: cx[i] - lw / 2, y: 1.3, w: lw, h: 0.5, rectRadius: 0.06, fill: { color: col }, line: { color: col } });
      T(s, lanes[i], { x: cx[i] - lw / 2, y: 1.3, w: lw, h: 0.5, fontSize: 9.5, bold: true, color: WHITE, align: "center", valign: "middle" });
      s.addShape("line", { x: cx[i], y: 1.8, w: 0, h: 3.4, line: { color: "C3CCD5", width: 1, dashType: "dash" } });
    }
    const msgs = [
      [0, 1, "Run agent analysis (current factory time)"],
      [1, 2, "Gather evidence / tool calls"],
      [2, 3, "Snapshot, diagnose, optimise, schedule"],
      [3, 2, "Probabilities, SHAP, candidates, plan", true],
      [2, 1, "Compact JSON", true],
      [1, 4, "Evidence + role prompt"],
      [4, 1, "Report / structured plan (JSON schema)", true],
      [1, 0, "Action plan + reasoning trace", true],
    ];
    for (let k = 0; k < msgs.length; k++) {
      const [a, b, label, ret] = msgs[k];
      const y = 2.12 + k * 0.37;
      const x1 = Math.min(cx[a], cx[b]), x2 = Math.max(cx[a], cx[b]);
      const line = { color: ret ? MUTED : DARK, width: 1.25 };
      if (ret) line.dashType = "dash";
      if (b > a) line.endArrowType = "triangle"; else line.beginArrowType = "triangle";
      s.addShape("line", { x: x1 + 0.03, y, w: x2 - x1 - 0.06, h: 0, line });
      const lw2 = Math.max(x2 - x1 - 0.2, 2.9), mid = (x1 + x2) / 2;
      T(s, `${k + 1}  ${label}`, { x: mid - lw2 / 2, y: y - 0.2, w: lw2, h: 0.18, fontSize: 8.5, color: ret ? MUTED : DARK, align: "center", fit: "shrink" });
    }
    s.addShape("roundRect", { x: 0.45, y: 5.0, w: 3.0, h: 0.32, rectRadius: 0.05, fill: { color: "E3F2EE" }, line: { color: TEAL, width: 0.75 } });
    T(s, "9  Accept / reject each action → data/decisions.jsonl", { x: 0.5, y: 5.0, w: 2.9, h: 0.32, fontSize: 8.5, bold: true, color: TEAL, valign: "middle" });
    s.addNotes("One agent run end to end. The supervisor triggers analysis for the current factory time. The orchestrator gathers evidence through the ToolBox, which calls models, optimiser and scheduler and returns compact JSON. Only then is the LLM called, with evidence plus a role prompt, and it returns a JSON-schema-constrained plan. The supervisor accepts or rejects every action and that decision is logged.");
  }

  // ───────────────────────── 8. Factory simulation
  {
    const s = pres.addSlide();
    header(s, "04 · Architecture", "Factory simulation: replaying held-out data");
    const c = [
      ["FaIndustry", "Fleet clock", "Azure test period (1 Sep 2015 onwards) advanced in 3-hour steps across 100 machines."],
      ["FaCogs", "Machining line", "AI4I 20% test cycles — including every failing cycle — replayed as the production line."],
    ];
    for (let i = 0; i < 2; i++) {
      const x = 0.5 + i * 3.05;
      card(s, x, 1.45, 2.8, 2.3);
      await iconCircle(s, c[i][0], x + 0.25, 1.7, 0.6, STEEL);
      T(s, c[i][1], { x: x + 0.25, y: 2.45, w: 2.3, h: 0.35, fontSize: 16, bold: true });
      T(s, c[i][2], { x: x + 0.25, y: 2.85, w: 2.3, h: 0.8, fontSize: 11.5, color: MUTED });
    }
    card(s, 6.6, 1.45, 2.9, 3.7, DARK);
    await iconCircle(s, "FaLock", 6.85, 1.7, 0.6, AMBER);
    T(s, "No look-ahead", { x: 6.85, y: 2.45, w: 2.4, h: 0.35, fontSize: 16, bold: true, color: WHITE });
    T(s, bullets([
      "Models & agents see only the state at the current clock position",
      "Actual future failures appear only in a “ground truth” expander for demo evaluation",
      "Ground truth is never passed to the agents",
    ]), { x: 6.85, y: 2.85, w: 2.45, h: 2.2, fontSize: 11, color: PALE });
    card(s, 0.5, 3.95, 5.85, 1.2);
    T(s, "Sidebar controls", { x: 0.75, y: 4.07, w: 5.3, h: 0.3, fontSize: 13, bold: true });
    T(s, "Factory clock slider · +3 h · +1 day · production-cycle picker · active decision engine (Ollama / Claude / rule-based).",
      { x: 0.75, y: 4.4, w: 5.4, h: 0.65, fontSize: 11, color: MUTED });
    s.addNotes("The dashboard treats held-out test data as a live factory. The fleet clock walks through the Azure test period in 3-hour steps; the machining line replays the AI4I test cycles. This means the demo is honest: nothing the agents see comes from the future.");
  }

  // ───────────────────────── 9. Data pipeline & features
  {
    const s = pres.addSlide();
    header(s, "05 · Data pipeline", "Three commands from raw data to trained models");
    const steps = [
      ["python -m src.data.download", "→ data/raw/*.csv  (~80 MB)"],
      ["python -m src.data.features", "→ ai4i / azure_features.parquet  (~10 s)"],
      ["python -m src.models.train", "→ models/*.joblib, reports/metrics.json  (~1 min)"],
    ];
    for (let i = 0; i < 3; i++) {
      const x = 0.5 + i * 3.05;
      s.addShape("roundRect", { x, y: 1.4, w: 2.85, h: 0.8, rectRadius: 0.06, fill: { color: DARK }, line: { color: DARK } });
      T(s, steps[i][0], { x: x + 0.12, y: 1.47, w: 2.65, h: 0.3, fontSize: 10, bold: true, fontFace: MONO, color: AMBER });
      T(s, steps[i][1], { x: x + 0.12, y: 1.8, w: 2.65, h: 0.3, fontSize: 9, color: PALE });
      if (i < 2) T(s, "›", { x: x + 2.85, y: 1.4, w: 0.2, h: 0.8, fontSize: 22, bold: true, color: AMBER, align: "center", valign: "middle" });
    }
    card(s, 0.5, 2.45, 4.4, 2.7);
    T(s, "AI4I features — per production cycle", { x: 0.7, y: 2.57, w: 4.0, h: 0.3, fontSize: 13, bold: true });
    T(s, bullets([
      { text: "temp_diff_k: process − air temp (HDF when < 8.6 K at low speed)" },
      { text: "power_w: torque × rpm × 2π/60 (PWF outside 3,500–9,000 W)" },
      { text: "strain_minnm: tool wear × torque (OSF > 11k/12k/13k for L/M/H)" },
      { text: "strain_ratio, type_code, raw setpoints" },
      { text: "Physics-rule flags → hard, explainable evidence; covers tool wear (flag from 190 min)", options: { bold: true, color: TEAL } },
    ]), { x: 0.7, y: 2.92, w: 4.05, h: 2.15, fontSize: 10.5 });
    card(s, 5.1, 2.45, 4.4, 2.7);
    T(s, "Azure features — per machine, every 3 h", { x: 5.3, y: 2.57, w: 4.0, h: 0.3, fontSize: 13, bold: true });
    T(s, bullets([
      "Telemetry: volt, rotate, pressure, vibration + 3 h / 24 h rolling mean & std",
      "Errors: count of error1–error5 in the last 24 h",
      "Component age: days since last replacement (strictly before t — no leakage)",
      "Machine: model, age",
      "Labels: component fails within 24 h / 7 days (strictly after t)",
    ]), { x: 5.3, y: 2.92, w: 2.75, h: 2.15, fontSize: 10.5 });
    T(s, [{ text: "292,100", options: { fontSize: 22, bold: true, color: AMBER, breakLine: true } }, { text: "rows × 41 columns", options: { fontSize: 10, color: MUTED } }],
      { x: 8.15, y: 3.3, w: 1.25, h: 0.9, align: "center" });
    s.addNotes("The pipeline is three commands. On the AI4I side we engineer physics features that mirror the documented failure rules, and we also implement those rules as explicit flags. On the Azure side we build rolling telemetry statistics, error counts and component ages, with careful handling of time to avoid leakage.");
  }

  // ───────────────────────── 10. Predictive models
  {
    const s = pres.addSlide();
    header(s, "06 · Predictive models", "Seven model families, all explainable");
    const rows = [
      tableHead(["Model", "Algorithm", "Output"]),
      ["Machine failure (process)", "LightGBM", "P(cycle fails); alert threshold tuned on out-of-fold predictions"],
      ["Failure mode × 4 (TWF, HDF, PWF, OSF)", "LightGBM", "P(each mode)"],
      ["Component failure within 24 h × 4", "LightGBM", "P(comp fails in 24 h) → drives urgent alerts"],
      ["Component failure within 7 days × 4", "LightGBM", "P(comp fails in 7 d) → drives weekly scheduling"],
      ["Anomaly detector", "Isolation Forest (24 h telemetry stats)", "Score: 0 = typical, > 1 = beyond 99th pct of normal"],
      ["Explanations", "SHAP TreeExplainer", "Per-prediction feature contributions (log-odds)"],
    ];
    s.addTable(rows, { x: 0.5, y: 1.4, w: 9.0, colW: [2.9, 2.4, 3.7], fontFace: BF, fontSize: 11, color: TEXT,
      border: { type: "solid", pt: 0.5, color: "D5DCE3" }, valign: "middle", margin: 0.06, rowH: 0.38 });
    card(s, 0.5, 4.35, 9.0, 0.8);
    await iconCircle(s, "FaLightbulb", 0.7, 4.47, 0.55, AMBER);
    T(s, "No class re-weighting: keeps probabilities calibrated for cost calculations; class imbalance is handled by tuning the alert threshold instead.",
      { x: 1.45, y: 4.45, w: 7.9, h: 0.6, fontSize: 11.5, valign: "middle" });
    s.addNotes("All supervised models are LightGBM. We deliberately avoid class re-weighting because the scheduler uses the probabilities in cost formulas, so they must be calibrated. An Isolation Forest flags abnormal telemetry, and SHAP explains every prediction.");
  }

  // ───────────────────────── 11. Evaluation
  {
    const s = pres.addSlide();
    header(s, "06 · Predictive models", "Held-out evaluation results");
    s.addChart(pres.charts.BAR, [{ name: "PR-AUC", labels: ["Machine failure", "Heat dissipation", "Power", "Overstrain", "Tool wear"], values: [0.891, 1.0, 0.982, 0.971, 0.061] }], {
      x: 0.4, y: 1.3, w: 4.9, h: 3.9, barDir: "bar", showTitle: true, title: "Machining line (AI4I) — PR-AUC, 80/20 split",
      titleFontSize: 12, titleColor: DARK, titleFontFace: BF, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: DARK,
      dataLabelFormatCode: "0.000", chartColors: [STEEL], catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10,
      valAxisMaxVal: 1.15, valAxisMinVal: 0, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
    });
    const cards = [
      ["0.996–1.000", "PR-AUC, 24 h component models", "Near-perfect: synthetic data has strong pre-failure signals"],
      ["~10×", "better than base rate, 7-day models", "PR-AUC 0.35–0.52 vs 2.4–4.9% base rate — enough to rank for weekly planning"],
      ["11.4% vs 1.0%", "anomaly flags before failures vs overall", "Readings flagged in the 24 h before a failure"],
    ];
    for (let i = 0; i < 3; i++) {
      const y = 1.4 + i * 1.02;
      card(s, 5.55, y, 3.95, 0.9);
      T(s, cards[i][0], { x: 5.7, y: y + 0.08, w: 1.55, h: 0.75, fontSize: 17, bold: true, color: AMBER, valign: "middle", fit: "shrink" });
      T(s, [{ text: cards[i][1], options: { bold: true, breakLine: true } }, { text: cards[i][2], options: { color: MUTED } }],
        { x: 7.3, y: y + 0.08, w: 2.1, h: 0.76, fontSize: 9, valign: "middle" });
    }
    T(s, "Tool wear is random between 200–240 min — unpredictable from one cycle. The physics rule covers it. Fleet: time split, train Jan–Aug 2015, test Sep onwards.",
      { x: 5.55, y: 4.5, w: 3.95, h: 0.65, fontSize: 9.5, italic: true, color: MUTED });
    s.addNotes("Machine-failure model: PR-AUC 0.891, precision 0.949, recall 0.824 at threshold 0.47. HDF, PWF and OSF are nearly perfect because they follow deterministic physics. Tool wear is effectively random and no model can predict it, which is exactly why we keep the physics rules. On the fleet, 24-hour models are near perfect (synthetic data), and 7-day models are about ten times better than the base rate — good for ranking.");
  }

  // ───────────────────────── 12. Process optimiser
  {
    const s = pres.addSlide();
    header(s, "07 · Optimisation", "Process setpoint optimiser");
    const rows = [
      ["FaSlidersH", "Search space", "Speed −15% to +25%, torque −30% to +50% (5% steps), optional tool change; kept within the training range"],
      ["FaTachometerAlt", "Constraint", "Speed ≥ min_throughput_ratio × current speed (default 0.9) — speed is the throughput proxy"],
      ["FaBullseye", "Objective", "Failure prob. + 0.5 × throughput loss + small change-size penalty + tool-change penalty → smallest fix that works"],
      ["FaCheckCircle", "Verification", "Every candidate scored by the trained failure model; simulate() re-scores any what-if"],
    ];
    for (let i = 0; i < rows.length; i++) {
      const y = 1.4 + i * 0.95;
      await iconCircle(s, rows[i][0], 0.5, y, 0.5, STEEL);
      T(s, rows[i][1], { x: 1.15, y: y - 0.02, w: 4.3, h: 0.28, fontSize: 13, bold: true });
      T(s, rows[i][2], { x: 1.15, y: y + 0.27, w: 4.3, h: 0.6, fontSize: 10.5, color: MUTED });
    }
    card(s, 5.8, 1.4, 3.7, 1.55, DARK);
    T(s, "67 / 68", { x: 6.0, y: 1.5, w: 3.3, h: 0.75, fontSize: 44, bold: true, color: AMBER });
    T(s, "failing test cycles brought below 20% risk", { x: 6.0, y: 2.3, w: 3.3, h: 0.5, fontSize: 12, color: WHITE });
    card(s, 5.8, 3.15, 3.7, 2.0);
    T(s, "EXAMPLE — HEAT-DISSIPATION FAILURE", { x: 6.0, y: 3.27, w: 3.3, h: 0.22, fontSize: 9, bold: true, color: MUTED, charSpacing: 1 });
    T(s, [{ text: "Speed 1363 → 1431 rpm ", options: { bold: true } }, { text: "(+5%)", options: { color: MUTED } }], { x: 6.0, y: 3.55, w: 3.3, h: 0.3, fontSize: 13 });
    T(s, [{ text: "99.9%", options: { color: RED, bold: true } }, { text: "  →  ", options: { color: MUTED } }, { text: "0.01%", options: { color: TEAL, bold: true } }],
      { x: 6.0, y: 3.9, w: 3.3, h: 0.5, fontSize: 26 });
    T(s, "Moves speed above the 1380 rpm heat-dissipation limit.", { x: 6.0, y: 4.5, w: 3.3, h: 0.5, fontSize: 10.5, color: MUTED });
    s.addNotes("The setpoint optimiser is a model-verified search. It enumerates speed and torque adjustments within the range seen in training, keeps throughput above 90%, and prefers the smallest change that fixes the risk. Every candidate is re-scored by the failure model, so nothing is recommended without a predicted risk reduction. It fixes 67 of 68 failing test cycles.");
  }

  // ───────────────────────── 13. Scheduler
  {
    const s = pres.addSlide();
    header(s, "07 · Optimisation", "Maintenance scheduler (OR-Tools CP-SAT)");
    T(s, bullets([
      { text: "Jobs: every (machine, component) with 24 h or 7-day risk ≥ 5%" },
      { text: "Decision: replace on day d of the horizon, or don't schedule" },
      { text: "Cost of replacing on day d: planned cost + planned downtime × day's load + P(fails before d) × unplanned cost" },
      { text: "Cost of not scheduling: P(fails within horizon) × unplanned cost" },
      { text: "Constraint: at most crew_per_day jobs per day" },
    ], { gap: 6 }), { x: 0.5, y: 1.4, w: 4.6, h: 2.6, fontSize: 11.5 });
    card(s, 0.5, 4.1, 4.6, 1.05);
    T(s, "FAILURE TIMING", { x: 0.7, y: 4.2, w: 4.2, h: 0.2, fontSize: 9, bold: true, color: MUTED, charSpacing: 1 });
    T(s, "P(fail < d) = 1 − (1 − p24h)(1 − h)^(d−1)", { x: 0.7, y: 4.43, w: 4.2, h: 0.28, fontSize: 10.5, fontFace: MONO, bold: true });
    T(s, "h = constant daily hazard implied by the 7-day probability", { x: 0.7, y: 4.75, w: 4.2, h: 0.3, fontSize: 9.5, color: MUTED });
    s.addChart(pres.charts.BAR, [{ name: "Expected cost", labels: ["Optimised plan", "Run-to-failure"], values: [187.6, 326.8] }], {
      x: 5.4, y: 1.3, w: 4.1, h: 2.85, barDir: "col", showTitle: true, title: "Expected cost, week of 26 Sep 2015", titleFontSize: 11, titleColor: DARK,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '"$"0.0"k"', dataLabelFontSize: 10, dataLabelColor: DARK,
      chartColors: [TEAL, RED], valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: MUTED,
      showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 380, barGapWidthPct: 60,
    });
    card(s, 5.4, 4.3, 4.1, 0.85, DARK);
    T(s, [{ text: "$139,174 ", options: { fontSize: 22, bold: true, color: AMBER } }, { text: "expected saving that week", options: { fontSize: 11, color: WHITE } }],
      { x: 5.6, y: 4.35, w: 3.8, h: 0.75, valign: "middle" });
    s.addNotes("The scheduler turns probabilities into a plan. For every at-risk component it decides whether and on which day to replace, trading planned cost and downtime weighted by production load against the risk of failing before that day, subject to crew capacity. On the 26 September example it schedules 14 jobs and cuts expected cost from about 327k to 188k dollars.");
  }

  // ───────────────────────── 14. KPIs & business case
  {
    const s = pres.addSlide();
    header(s, "07 · Optimisation", "OEE and the business case");
    const tiles = [["Availability", "96.9%", "Azure downtime over 100 machine-years"], ["Performance", "99.3%", "Mean rotation vs 450 rpm nominal"], ["Quality", "96.6%", "AI4I cycles without failure"], ["OEE today", "92.9%", "Availability × Performance × Quality"]];
    for (let i = 0; i < 4; i++) {
      const x = 0.5 + i * 2.3, last = i === 3;
      card(s, x, 1.4, 2.1, 1.45, last ? STEEL : LIGHT);
      T(s, tiles[i][0], { x: x + 0.18, y: 1.5, w: 1.8, h: 0.25, fontSize: 11, bold: true, color: last ? WHITE : MUTED });
      T(s, tiles[i][1], { x: x + 0.18, y: 1.78, w: 1.8, h: 0.55, fontSize: 28, bold: true, color: last ? WHITE : DARK });
      T(s, tiles[i][2], { x: x + 0.18, y: 2.35, w: 1.8, h: 0.42, fontSize: 9, color: last ? PALE : MUTED });
    }
    card(s, 0.5, 3.05, 9.0, 2.1, DARK);
    T(s, "WITH PREDICTIVE MAINTENANCE  (24 h models detect 99.7% of failures; 70% converted to planned work)", { x: 0.75, y: 3.17, w: 8.6, h: 0.25, fontSize: 9.5, bold: true, color: AMBER, charSpacing: 1 });
    const stats = [["531", "failures prevented / yr"], ["10,623 h", "downtime saved / yr"], ["~$14.1M", "saved per year"], ["94.1%", "OEE (from 92.9%)"]];
    for (let i = 0; i < 4; i++) {
      const x = 0.75 + i * 2.2;
      T(s, stats[i][0], { x, y: 3.55, w: 2.1, h: 0.7, fontSize: 32, bold: true, color: WHITE });
      T(s, stats[i][1], { x, y: 4.25, w: 2.1, h: 0.3, fontSize: 11, color: PALE });
    }
    T(s, "All costs and downtime hours are assumptions in src/config.py — edit them for your plant.", { x: 0.75, y: 4.7, w: 8.6, h: 0.3, fontSize: 9.5, italic: true, color: PALE });
    s.addNotes("OEE is 92.9% today. With predictive maintenance, assuming 70% of detected failures become planned work, we project 531 failures prevented, over 10,000 downtime hours saved and about 14 million dollars a year, lifting OEE to 94.1%. Stress that these rely on configurable cost assumptions.");
  }

  // ───────────────────────── 15. Agents
  {
    const s = pres.addSlide();
    header(s, "08 · Decision agents", "Six agents, seven strict tools");
    const ag = [
      ["FaSatelliteDish", "Monitoring", "get_fleet_overview, get_machine_details", "Prioritised watch-list"],
      ["FaStethoscope", "Diagnosis", "get_process_status, get_fleet_overview, get_machine_details", "Failure mode → mechanism → evidence"],
      ["FaCalendarAlt", "Maintenance planning", "plan_maintenance, get_fleet_overview, get_plant_kpis", "Day-by-day plan, cost vs run-to-failure"],
      ["FaSlidersH", "Process optimisation", "get_process_status, optimize_process_setpoints, simulate_process_change", "Verified setpoint change"],
      ["FaSitemap", "Coordinator", "none — JSON-schema output", "De-duplicated, prioritised action plan"],
      ["FaComments", "Chat assistant", "all 7 tools", "Answers questions, runs what-ifs"],
    ];
    for (let i = 0; i < 6; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const x = 0.5 + col * 3.05, y = 1.4 + row * 1.5, w = 2.85;
      card(s, x, y, w, 1.42);
      await iconCircle(s, ag[i][0], x + 0.15, y + 0.15, 0.45, i === 4 ? AMBER : STEEL);
      T(s, ag[i][1], { x: x + 0.7, y: y + 0.15, w: w - 0.8, h: 0.45, fontSize: 13, bold: true, valign: "middle" });
      T(s, ag[i][2], { x: x + 0.15, y: y + 0.64, w: w - 0.3, h: 0.45, fontSize: 8, fontFace: MONO, color: STEEL, fit: "shrink" });
      T(s, ag[i][3], { x: x + 0.15, y: y + 1.12, w: w - 0.3, h: 0.25, fontSize: 10, bold: true, color: TEAL });
    }
    T(s, "Tools return compact JSON with strict schemas; unknown args are dropped; tool errors go back to the model instead of crashing. Plan schema: headline, situation_summary, actions[] (priority, category, urgency, target, action, rationale, expected_impact, source_agents), risks_and_caveats[].",
      { x: 0.5, y: 4.5, w: 9.0, h: 0.65, fontSize: 9.5, color: MUTED });
    s.addNotes("Four specialists — monitoring, diagnosis, maintenance planning and process optimisation — each have a narrow tool set. A coordinator merges their reports into one de-duplicated plan using a JSON schema. A chat assistant has access to all seven tools for ad-hoc questions and what-ifs.");
  }

  // ───────────────────────── 16. Backends & modes
  {
    const s = pres.addSlide();
    header(s, "08 · Decision agents", "Pluggable LLM backends and agent modes");
    const b = [
      tableHead(["", "ollama (default)", "claude"]),
      ["Model", "qwen3:4b", "claude-opus-5-5"],
      ["Cost", "Free, offline, no token limits", "Pay per token"],
      ["Setup", "ollama pull qwen3:4b", "ANTHROPIC_API_KEY"],
      ["Default mode", "narrate", "tools (parallel)"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: j === 0 ? { bold: true, color: STEEL } : {} })));
    s.addTable(b, { x: 0.5, y: 1.4, w: 3.7, colW: [0.95, 1.4, 1.35], fontFace: BF, fontSize: 9.5, color: TEXT,
      border: { type: "solid", pt: 0.5, color: "D5DCE3" }, valign: "middle", margin: 0.05, rowH: 0.36 });
    const m = [
      tableHead(["Mode", "What the LLM does", "Calls", "CPU time*"]),
      ["tools", "Specialists choose & call own tools; coordinator merges", "10–30", "> 10 min / call"],
      ["evidence", "Code runs tools; LLM writes 4 reports + merged plan", "5", "~8 min / report"],
      ["compact", "Code runs all tools; LLM writes whole plan", "1", "612 s"],
      ["narrate", "Models build actions; LLM writes headline, summary, caveats", "1", "86 s ✓"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: Object.assign(j === 0 ? { bold: true, fontFace: MONO } : {}, i === 4 ? { fill: { color: "E3F2EE" } } : {}) })));
    s.addTable(m, { x: 4.45, y: 1.4, w: 5.05, colW: [0.85, 2.5, 0.6, 1.1], fontFace: BF, fontSize: 9.5, color: TEXT,
      border: { type: "solid", pt: 0.5, color: "D5DCE3" }, valign: "middle", margin: 0.05, rowH: 0.36 });
    T(s, "* qwen3:4b on Intel i5-1135G7, no GPU, 20 GB RAM", { x: 4.45, y: 3.47, w: 5.05, h: 0.2, fontSize: 8.5, italic: true, color: MUTED });
    card(s, 0.5, 3.8, 9.0, 1.35);
    await iconCircle(s, "FaMicrochip", 0.7, 3.98, 0.55, AMBER);
    T(s, "Why narrate is the CPU default", { x: 1.45, y: 3.9, w: 7.8, h: 0.3, fontSize: 13, bold: true });
    T(s, "On CPU the model reads ~30–36 tokens/s and generates ~2–8 tokens/s, slowing as context grows. narrate keeps the prompt near 1k tokens and output near 200. On a GPU or with Claude, use compact or tools. llama3.2 (3B) was rejected: it claimed a what-if simulation it never ran — so lighter modes run tools in code.",
      { x: 1.45, y: 4.22, w: 7.9, h: 0.88, fontSize: 10.5, color: MUTED });
    s.addNotes("The LLM backend is pluggable: free local Ollama or Claude, sharing the same tools and plan schema. Agent modes trade LLM autonomy for speed. On a laptop CPU only narrate is practical, at about a minute and a half. Cursor was considered but has no public API for calling its models.");
  }

  // ───────────────────────── 17. Safeguards
  {
    const s = pres.addSlide();
    header(s, "08 · Decision agents", "Reliability safeguards");
    const g = [
      ["FaAnchor", "Grounding", "Every prompt requires numbers to come from tool results; lighter modes run the tools (incl. what-if verification) in code."],
      ["FaFileAlt", "Structured output", "Plans generated against a JSON schema; Ollama output validated with jsonschema and retried once."],
      ["FaUndoAlt", "Graceful fallback", "No credits, Ollama down, timeout → deterministic rule-based planner; dashboard shows why."],
      ["FaHistory", "Append-only history", "Chat turns appended; a failed turn is rolled back so history stays valid."],
      ["FaShieldAlt", "Claude-specific", "Server-side refusal fallback; refusal / max_tokens / pause_turn handling; typed error mapping."],
      ["FaUserCheck", "Human in the loop", "Supervisor accepts or rejects every action; logged to data/decisions.jsonl."],
    ];
    for (let i = 0; i < 6; i++) {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 0.5 + col * 4.6, y = 1.45 + row * 1.25;
      await iconCircle(s, g[i][0], x, y, 0.6, i === 5 ? TEAL : STEEL);
      T(s, g[i][1], { x: x + 0.78, y: y - 0.02, w: 3.6, h: 0.3, fontSize: 14, bold: true });
      T(s, g[i][2], { x: x + 0.78, y: y + 0.3, w: 3.6, h: 0.75, fontSize: 10.5, color: MUTED });
    }
    s.addNotes("Six safeguards make the LLM layer production-minded: grounding in tool output, schema-validated output, a rule-based fallback so the dashboard works without any LLM, consistent chat history, Claude-specific error handling, and a human who approves every action.");
  }

  // ───────────────────────── 18. Dashboard
  {
    const s = pres.addSlide();
    header(s, "09 · Dashboard", "Streamlit dashboard — seven pages");
    const p = [
      ["FaTachometerAlt", "Fleet overview", "KPI tiles, OEE ± PdM, 7-day risk heatmap, business case, watch-list"],
      ["FaHeartbeat", "Machine health", "Component risk, SHAP drivers, 14-day telemetry, errors, replacements"],
      ["FaSlidersH", "Process optimizer", "Diagnosis, mode probabilities, physics flags, candidates, what-if"],
      ["FaCalendarAlt", "Maintenance plan", "Horizon & crew sliders, crew load vs capacity, cost vs run-to-failure"],
      ["FaRobot", "Agent action plan", "Run agents; prioritised actions with accept/reject; reasoning trace"],
      ["FaComments", "Assistant", "Chat with tool access; suggested questions; tool-call trace"],
      ["FaChartBar", "Model performance", "All evaluation metrics"],
    ];
    for (let i = 0; i < 7; i++) {
      const col = i % 4, row = Math.floor(i / 4);
      const x = 0.5 + col * 2.3, y = 1.4 + row * 1.9, w = 2.1;
      card(s, x, y, w, 1.75);
      await iconCircle(s, p[i][0], x + 0.18, y + 0.18, 0.5, i === 4 ? AMBER : STEEL);
      T(s, p[i][1], { x: x + 0.18, y: y + 0.78, w: w - 0.3, h: 0.3, fontSize: 12.5, bold: true });
      T(s, p[i][2], { x: x + 0.18, y: y + 1.08, w: w - 0.3, h: 0.62, fontSize: 9.5, color: MUTED });
    }
    card(s, 0.5 + 3 * 2.3, 3.3, 2.1, 1.75, DARK);
    T(s, "streamlit run app/streamlit_app.py", { x: 7.58, y: 3.45, w: 1.8, h: 0.6, fontSize: 9, fontFace: MONO, color: AMBER, bold: true });
    T(s, "Opens at http://localhost:8501 (use port 8502 if busy)", { x: 7.58, y: 4.1, w: 1.8, h: 0.8, fontSize: 9, color: PALE });
    s.addNotes("Seven pages, from fleet-level KPIs down to a single machine or production cycle, plus the agent action plan, a chat assistant and a model-performance page. The sidebar holds the factory clock, cycle picker and decision engine selector.");
  }

  // ───────────────────────── 19. Setup & config
  {
    const s = pres.addSlide();
    header(s, "10 · Setup & configuration", "Running it and tuning it");
    card(s, 0.5, 1.4, 4.4, 3.75, DARK);
    T(s, [
      "pip install -r requirements.txt", "cp .env.example .env", "ollama pull qwen3:4b", "",
      "python -m src.data.download", "python -m src.data.features", "python -m src.models.train", "",
      "streamlit run app/streamlit_app.py",
    ].map((l, i, a) => ({ text: l || " ", options: { breakLine: i < a.length - 1 } })), { x: 0.7, y: 1.6, w: 4.0, h: 2.6, fontSize: 10.5, fontFace: MONO, color: WHITE });
    T(s, "Python 3.11+ · ~1 GB disk · Ollama with ~4 GB free RAM. For Claude: LLM_PROVIDER=claude + ANTHROPIC_API_KEY.",
      { x: 0.7, y: 4.35, w: 4.0, h: 0.7, fontSize: 9.5, color: PALE });
    const cfg = [
      tableHead([".env", "Default"]),
      ["LLM_PROVIDER", "ollama"], ["AGENT_MODE", "narrate / tools"], ["OLLAMA_MODEL", "qwen3:4b"], ["OLLAMA_NUM_CTX", "8192"], ["CLAUDE_EFFORT", "medium"],
    ].map((r, i) => i === 0 ? r : [{ text: r[0], options: { fontFace: MONO } }, r[1]]);
    s.addTable(cfg, { x: 5.1, y: 1.4, w: 4.4, colW: [2.2, 2.2], fontFace: BF, fontSize: 9.5, color: TEXT,
      border: { type: "solid", pt: 0.5, color: "D5DCE3" }, valign: "middle", margin: 0.05, rowH: 0.28 });
    card(s, 5.1, 3.25, 4.4, 1.9);
    T(s, "Business assumptions (src/config.py)", { x: 5.3, y: 3.35, w: 4.0, h: 0.3, fontSize: 12, bold: true });
    T(s, bullets([
      "Unplanned failure $12,000 · planned $1,500 · downtime $800/h",
      "Downtime: unplanned 24 h · planned 4 h",
      "Crew 4 jobs/day · horizon 7 days · nominal 450 rpm",
      "Daily load 1.0, 1.2, 1.2, 1.0, 0.8, 0.5, 0.4 · PdM realisation 0.7",
    ], { gap: 3 }), { x: 5.3, y: 3.68, w: 4.05, h: 1.4, fontSize: 9.5 });
    s.addNotes("Setup is a handful of commands. Configuration splits into .env for the LLM backend and src/config.py for business assumptions. Changing the cost and load assumptions changes the scheduler and business case, so plant teams should review them.");
  }

  // ───────────────────────── 20. Testing & layout
  {
    const s = pres.addSlide();
    header(s, "11 · Quality & code", "Testing and project layout");
    T(s, [{ text: "32", options: { fontSize: 44, bold: true, color: AMBER, breakLine: true } }, { text: "pytest tests — no network, API or LLM calls (clients mocked)", options: { fontSize: 11, color: MUTED } }],
      { x: 0.5, y: 1.35, w: 4.3, h: 1.2 });
    const t = [
      ["test_features.py", "Derived features and each physics rule"],
      ["test_optimization.py", "Failure timing, scheduler capacity & urgency, optimiser fixes an HDF failure"],
      ["test_agents.py", "Tool schemas, Claude/Ollama loops, JSON retry, modes, fallback, chat rollback"],
      ["test_app.py", "Every dashboard page renders headlessly (AppTest)"],
    ];
    for (let i = 0; i < 4; i++) {
      const y = 2.6 + i * 0.64;
      card(s, 0.5, y, 4.3, 0.56);
      T(s, t[i][0], { x: 0.65, y: y + 0.05, w: 4.0, h: 0.2, fontSize: 9.5, bold: true, fontFace: MONO, color: STEEL });
      T(s, t[i][1], { x: 0.65, y: y + 0.27, w: 4.05, h: 0.25, fontSize: 9, color: MUTED });
    }
    card(s, 5.1, 1.4, 4.4, 3.75, DARK);
    T(s, [
      "SMPO/", "+- src/", "|  +- config.py", "|  +- data/         download, features", "|  +- models/       train, predictor",
      "|  +- optimization/ optimizer, scheduler,", "|  |                kpis", "|  +- simulation/   stream.py", "|  +- agents/       tools, prompts,", "|                   orchestrator, ollama_runner",
      "+- app/             streamlit_app, pages", "+- notebooks/       01_eda.ipynb", "+- tests/", "+- data/, models/   (generated)", "+- reports/metrics.json",
    ].map((l, i, a) => ({ text: l, options: { breakLine: i < a.length - 1 } })), { x: 5.3, y: 1.6, w: 4.1, h: 3.45, fontSize: 10, fontFace: MONO, color: WHITE });
    s.addNotes("32 tests run offline with mocked LLM clients, covering features, optimisation, the agent loops including failure paths, and headless rendering of every dashboard page. The code layout mirrors the architecture layers one to one.");
  }

  // ───────────────────────── 21. Design decisions
  {
    const s = pres.addSlide();
    header(s, "12 · Design decisions", "Key trade-offs we made");
    const d = [
      ["Two datasets", "AI4I explains why a cycle fails; Azure gives fleet time series and maintenance history"],
      ["LLM as decision layer, not predictor", "Predictions stay measurable and testable"],
      ["LightGBM without re-weighting", "Calibrated probabilities for cost maths; tune the threshold instead"],
      ["Time-based split for the fleet", "Test months lie in the future — no look-ahead"],
      ["7-day classifier, not days-to-failure", "Regressor barely beat the median (MAE 15 days)"],
      ["Physics rules alongside ML", "Covers unpredictable tool wear; hard evidence for explanations"],
      ["Optimiser verifies with the model", "No recommendation without a predicted risk reduction"],
      ["CP-SAT scheduling", "Exact risk vs load vs crew optimisation in < 1 s"],
      ["Pluggable LLM + rule-based fallback", "Free local or stronger Claude; dashboard works with no LLM"],
    ];
    for (let i = 0; i < d.length; i++) {
      const col = i % 3, row = Math.floor(i / 3);
      const x = 0.5 + col * 3.05, y = 1.4 + row * 1.27, w = 2.85;
      card(s, x, y, w, 1.12);
      T(s, d[i][0], { x: x + 0.18, y: y + 0.12, w: w - 0.3, h: 0.3, fontSize: 11.5, bold: true, color: DARK, fit: "shrink" });
      T(s, d[i][1], { x: x + 0.18, y: y + 0.45, w: w - 0.3, h: 0.6, fontSize: 9.5, color: MUTED });
    }
    s.addNotes("These are the decisions most worth defending in a review. The common thread: keep numbers in measurable models, keep the LLM for reasoning and communication, and make sure the system degrades gracefully.");
  }

  // ───────────────────────── 22. Limitations & next steps
  {
    const s = pres.addSlide();
    header(s, "13 · Roadmap", "Limitations and next steps");
    card(s, 0.5, 1.4, 4.35, 3.75);
    await iconCircle(s, "FaExclamationTriangle", 0.7, 1.55, 0.5, RED);
    T(s, "Limitations", { x: 1.35, y: 1.55, w: 3.3, h: 0.5, fontSize: 16, bold: true, valign: "middle" });
    T(s, bullets([
      "Both datasets synthetic — real 24 h scores will be lower",
      "Datasets describe different plants; joined at decision layer only",
      "Cost, downtime and load figures are assumptions",
      "Scheduler double-charges downtime for same-machine, same-day jobs",
      "Small local models can misstate facts in prose",
      "No authentication; decisions logged to a local file",
    ], { gap: 5 }), { x: 0.7, y: 2.2, w: 4.0, h: 2.85, fontSize: 11 });
    card(s, 5.15, 1.4, 4.35, 3.75, DARK);
    await iconCircle(s, "FaRocket", 5.35, 1.55, 0.5, AMBER);
    T(s, "Next steps", { x: 6.0, y: 1.55, w: 3.3, h: 0.5, fontSize: 16, bold: true, color: WHITE, valign: "middle" });
    T(s, bullets([
      "Connect a real source (OPC UA / MQTT historian) in place of replay",
      "Calibrate 7-day probabilities (isotonic); retrain on real history",
      "Group same-machine jobs into a single stop",
      "Feed accept/reject decisions back into scheduling & prompts",
      "Run agents on GPU or Claude with compact / tools mode",
      "Add sign-in (st.login) and a database for the decision log",
    ], { gap: 5 }), { x: 5.35, y: 2.2, w: 4.0, h: 2.85, fontSize: 11, color: WHITE });
    s.addNotes("Be candid about limitations: synthetic data, assumed costs, and no auth yet. The roadmap addresses each one — real plant connectivity, calibration, scheduler grouping, closing the feedback loop from supervisor decisions, stronger LLM modes, and production hardening.");
  }

  // ───────────────────────── 23. Closing
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    T(s, "KEY TAKEAWAYS", { x: 0.6, y: 0.55, w: 8.8, h: 0.3, fontSize: 11, bold: true, color: AMBER, charSpacing: 2 });
    const k = [
      ["FaChartLine", "Models are the source of truth", "LightGBM, Isolation Forest, SHAP and physics rules produce every number."],
      ["FaCogs", "Optimisation turns risk into action", "Verified setpoints and a CP-SAT plan — ~$139k saved in a sample week."],
      ["FaUserCheck", "LLM explains, humans decide", "Grounded agents, schema output, rule-based fallback, full decision log."],
    ];
    for (let i = 0; i < 3; i++) {
      const y = 1.1 + i * 0.95;
      await iconCircle(s, k[i][0], 0.6, y, 0.6, AMBER);
      T(s, k[i][1], { x: 1.4, y: y - 0.02, w: 7.8, h: 0.32, fontSize: 16, bold: true, color: DARK });
      T(s, k[i][2], { x: 1.4, y: y + 0.3, w: 7.8, h: 0.3, fontSize: 12, color: MUTED });
    }
    T(s, "Questions & discussion", { x: 0.6, y: 4.2, w: 8.8, h: 0.7, fontSize: 32, bold: true, color: DARK });
    s.addNotes("Recap the three takeaways, then open the floor. Good discussion prompts: which real data source should we connect first, and who owns the business-cost assumptions?");
  }

  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
