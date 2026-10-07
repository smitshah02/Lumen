/* Lumen local UI.
 *
 * Talks only to the API that served this page: /health, /ready, /ask and
 * /review/{thread_id}. Conversation history lives in this tab's memory and is
 * display only: every question is an independent request, and nothing earlier
 * in the chat is sent to the model.
 *
 * All text from the API is written as text nodes, never parsed as HTML.
 */
"use strict";

const ASK_TIMEOUT_MS = 10 * 60 * 1000;
const STATUS_REFRESH_MS = 30 * 1000;
const MAX_SUBJECT = 2147483647;
const CITE_RE = /\[([SLGPA]\d+)\]/g;

const ERRORS = {
  subject_not_found: "Patient not found.",
  model_unavailable: "Local model is unavailable.",
  ollama_unavailable: "Local model is unavailable.",
  database_unavailable: "The database is unavailable.",
  generation_failed: "Lumen could not generate an answer for this question.",
  validation_error: "The request was not valid. Check the subject ID and the question (500 characters at most).",
  review_not_found: "No paused review exists for this answer.",
  review_not_pending: "This review has already been completed.",
  forbidden: "This API only serves requests from this machine.",
  internal_error: "Lumen hit an internal error. Details are in the server log.",
  timeout: "The request timed out.",
  network: "Cannot reach the Lumen API. Lumen may still be starting; please try again shortly.",
  malformed: "Lumen returned an unexpected response.",
};
const NODE_NAMES = {
  triage: "Triage", lab_lookup: "Lab lookup (structured)", encounter_lookup: "Admissions lookup (structured)",
  structured_lookup: "Orders / coded records lookup (structured)",
  patient_retrieval: "Patient retrieval", guideline_retrieval: "Guideline retrieval",
  literature_retrieval: "Literature retrieval", synthesis: "Synthesis", verification: "Verification",
  human_review: "Human review", finalize: "Finalize", refuse: "Refuse (out of scope)",
};
const SOURCE_KINDS = { S: "Patient note", L: "Structured lab result", A: "Admissions record", R: "Structured record", G: "Guideline", P: "Literature" };
const REVIEW_STATUS = {
  auto_approved: "Verified automatically", pending: "Awaiting human review", reviewed: "Approved by reviewer",
  rejected: "Rejected by reviewer", escalated: "Escalated", failed: "Failed",
};

const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

const state = { subject: null, busy: false, selected: null, focusLabel: null };

// ---------------------------------------------------------------- API ----
class UiError extends Error {
  constructor(code, status, data) { super(code); this.code = code; this.status = status; this.data = data; }
}

async function api(method, path, body, timeoutMs) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs || 30000);
  let res;
  try {
    res = await fetch(path, {
      method, signal: ctl.signal,
      headers: body ? { "content-type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    console.error("Lumen request failed", method, path, e);
    throw new UiError(e && e.name === "AbortError" ? "timeout" : "network");
  } finally {
    clearTimeout(timer);
  }
  let data;
  try { data = await res.json(); } catch (e) {
    console.error("Lumen returned a non-JSON body", method, path, res.status);
    throw new UiError("malformed", res.status);
  }
  if (!res.ok) {
    console.error("Lumen API error", method, path, res.status, data);
    throw new UiError((data && typeof data.error === "string" && data.error) || "http_" + res.status, res.status, data);
  }
  if (!data || typeof data !== "object") throw new UiError("malformed", res.status);
  return data;
}

function explain(err) {
  if (!(err instanceof UiError)) { console.error(err); return ERRORS.malformed; }
  if (ERRORS[err.code]) return ERRORS[err.code];
  if (err.status === 404) return "Not found.";
  if (err.status === 409) return ERRORS.review_not_pending;
  if (err.status === 422) return ERRORS.validation_error;
  if (err.status === 503) return "Lumen is still starting. Please try again shortly.";
  return ERRORS.internal_error;
}

// ------------------------------------------------------------- status ----
function setChip(id, label, kind) {
  const chip = $(id);
  chip.className = "chip " + (kind || "");
  chip.querySelector("b").textContent = label;
}

async function refreshStatus() {
  if (state.busy) return;                       // a probe mid-answer would only report a busy model
  let healthy = false, ready = null;
  try { healthy = (await fetch("/health")).ok; } catch (e) { /* shown as down below */ }
  try { ready = await (await fetch("/ready")).json(); } catch (e) { /* 503 still has a body; this is no body at all */ }
  const deps = (ready && ready.dependencies) || {};
  const all = (keys) => keys.every((k) => deps[k] === undefined || deps[k] === "ok");
  const known = ready && ready.dependencies;

  setChip("st-api", healthy ? (ready && ready.status === "ready" ? "Ready" : "Not ready") : "Down",
          healthy ? (ready && ready.status === "ready" ? "ok" : "warn") : "bad");
  setChip("st-db", known ? (all(["database", "schema", "extension", "corpus", "ingestion", "index"]) ? "Ready" : "Not ready") : "Unknown",
          known ? (all(["database", "schema", "extension", "corpus", "ingestion", "index"]) ? "ok" : "bad") : "");
  setChip("st-models", known ? (all(["retrieval_models", "ollama", "model"]) ? "Ready" : "Not ready") : "Unknown",
          known ? (all(["retrieval_models", "ollama", "model"]) ? "ok" : "bad") : "");
  const tracing = (ready && ready.tracing) || {};
  const tstate = tracing.state || (tracing.enabled === false ? "off" : "unknown");
  const tlabel = { active: "Active", pending: "On (idle)", off: "Off" }[tstate] || tstate;
  setChip("st-tracing", tlabel.charAt(0).toUpperCase() + tlabel.slice(1),
          tstate === "active" || tstate === "pending" ? "ok" : tstate === "off" || tstate === "unknown" ? "" : "warn");

  const plane = $("plane");
  const name = ready && ready.data_plane;
  plane.textContent = "";
  plane.append(el("i", "dot " + (name ? "ok" : "")),
               name === "research" ? "MIMIC-IV research environment"
               : name === "demo" ? "Synthetic demo environment" : "Environment unknown");
}

// ------------------------------------------------------------ patient ----
function parseSubject(raw) {
  const text = String(raw || "").trim();
  if (!/^\d{1,10}$/.test(text)) return null;
  const n = Number(text);
  return n >= 1 && n <= MAX_SUBJECT ? n : null;
}

function subjectChanged() {
  const input = $("subject"), hint = $("subject-hint");
  const next = parseSubject(input.value);
  input.classList.toggle("invalid", next === null && input.value.trim() !== "");
  hint.classList.toggle("bad", next === null && input.value.trim() !== "");
  hint.textContent = next === null && input.value.trim() !== ""
    ? "Subject ID must be a whole number." : "Required. Each question is an independent request for this patient.";
  if (next === state.subject) return;
  const had = state.subject !== null && $("messages").querySelector(".msg");
  state.subject = next;
  state.selected = null;
  $("messages").textContent = "";
  renderDetails();
  if (next !== null) {
    addNote(had ? "Patient changed to subject " + next + ". The previous conversation was cleared."
                : "Asking about subject " + next + ".");
  } else {
    showEmpty();
  }
}

// --------------------------------------------------------------- chat ----
function showEmpty() {
  const box = $("messages");
  if (box.children.length) return;
  box.append(el("div", "empty", "Enter a subject ID, then ask a question about that patient's record."));
}
function clearEmpty() { const e = $("messages").querySelector(".empty"); if (e) e.remove(); }
function addNote(text) { clearEmpty(); $("messages").append(el("div", "note", text)); }
function scrollDown() { const box = $("messages"); box.scrollTop = box.scrollHeight; }

function setBusy(busy) {
  state.busy = busy;
  $("send").disabled = busy;
  $("subject").disabled = busy;                 // an answer must land in the conversation it was asked in
}

async function send(event) {
  event.preventDefault();
  if (state.busy) return;
  const queryBox = $("query");
  const query = queryBox.value.trim();
  subjectChanged();
  if (state.subject === null) {
    $("subject-hint").textContent = "Enter a numeric subject ID before asking a question.";
    $("subject-hint").classList.add("bad");
    $("subject").focus();
    return;
  }
  if (!query) return;

  clearEmpty();
  $("messages").append(el("div", "msg user", query));
  queryBox.value = "";
  const msg = { subject: state.subject, query, node: el("div", "msg assistant"), started: performance.now() };
  msg.node.addEventListener("click", () => { state.focusLabel = null; select(msg); });
  $("messages").append(msg.node);
  setBusy(true);
  renderBubble(msg);
  msg.timer = setInterval(() => {
    const t = msg.node.querySelector(".elapsed");
    if (t) t.textContent = "Thinking… " + ((performance.now() - msg.started) / 1000).toFixed(1) + "s";
  }, 100);
  scrollDown();

  try {
    const resp = await api("POST", "/ask", { subject_id: msg.subject, query }, ASK_TIMEOUT_MS);
    if (typeof resp.answer !== "string" || typeof resp.status !== "string") throw new UiError("malformed");
    msg.resp = resp;
  } catch (err) {
    msg.error = explain(err);
  } finally {
    clearInterval(msg.timer);
    setBusy(false);
  }
  renderBubble(msg);
  select(msg);
  scrollDown();
  if (msg.resp && msg.resp.status === "human_review_required") loadPending(msg);
  refreshStatus();
}

function renderAnswer(container, text, msg) {
  container.textContent = "";
  LumenReview.structuredLabels = ((msg && msg.resp && msg.resp.sources) || []).map((s) => s.label);
  for (const part of LumenReview.citationRuns(text)) {
    if (!part.labels) { container.append(part.text); continue; }
    // Several markers in a row become one chip; the answer text keeps them all.
    const labels = part.labels, first = labels[0], many = labels.length > 1;
    const chip = el("button", "cite cite-" + first[0], many ? labels.length + " sources" : first);
    chip.type = "button";
    chip.title = many ? "Sources: " + labels.join(", ") : "Show source " + first;
    chip.addEventListener("click", (ev) => {
      ev.stopPropagation();
      state.focusLabel = first;
      select(msg);
      showTab("evidence");
    });
    container.append(chip);
  }
}

function renderBubble(msg) {
  const node = msg.node;
  node.textContent = "";
  node.append(el("div", "who", "Lumen"));

  if (!msg.resp && !msg.error) {                 // still running
    const thinking = el("div", "thinking");
    const dots = el("span", "dots");
    dots.append(el("span"), el("span"), el("span"));
    thinking.append(dots, el("span", "elapsed", "Thinking… 0.0s"));
    node.append(thinking);
    return;
  }
  if (msg.error) {
    node.append(el("div", "body error", msg.error));
    return;
  }

  const r = msg.resp;
  if (r.answer_is_draft) node.append(el("div", "banner draft", "DRAFT — HUMAN REVIEW REQUIRED"));
  else if (r.review_status === "rejected") node.append(el("div", "banner rejected", "Rejected by reviewer — no answer released"));
  else if (r.review_status === "reviewed") {
    node.append(el("div", "banner approved", "Approved by reviewer" +
                   (msg.overridden ? " — " + LumenReview.plural(msg.overridden, "verifier flag", "verifier flags") + " overridden" : "")));
  }
  else if (r.status === "refused") node.append(el("div", "banner neutral", "Out of scope — not answered"));
  // An admission was asked for but the answer was not limited to it: say so.
  const scope = r.admission_scope;
  if (scope && scope.requested && !scope.applied) {
    const warning = el("div", "banner scope", "Admission scope was not applied to this answer.");
    if (scope.reason) warning.append(el("div", "scope-reason", scope.reason));
    node.append(warning);
  }

  const body = el("div", "body");
  renderAnswer(body, r.answer || "(no answer text)", msg);
  node.append(body);

  if (r.status === "human_review_required" || msg.reviewNote) node.append(renderReview(msg));

  const meta = el("div", "meta");
  meta.append(el("span", null, REVIEW_STATUS[r.review_status] || r.review_status || r.status));
  if (typeof r.latency_ms === "number") meta.append(el("span", null, fmtMs(r.latency_ms)));
  meta.append(el("span", null, (r.sources || []).length + " source(s)"));
  node.append(meta);
}

// ------------------------------------------------------------- review ----
async function loadPending(msg) {
  try {
    msg.pending = await api("GET", "/review/" + encodeURIComponent(msg.resp.thread_id));
  } catch (err) {
    msg.pendingError = explain(err);
  }
  if (state.selected === msg) renderDetails();
}

function renderReview(msg) {
  const box = el("div", "review");
  box.addEventListener("click", (ev) => ev.stopPropagation());
  const view = LumenReview.view(msg);
  if (view.open) {
    if (view.warning) box.append(el("div", "msgline warn", view.warning));
    box.append(el("div", "msgline", "The flagged claims and their cited sources are in the Evidence tab."));
    const note = el("textarea");
    note.placeholder = "Reviewer note (optional)";
    note.maxLength = 1000;
    note.value = msg.noteDraft || "";
    note.disabled = !!msg.deciding;
    note.addEventListener("input", () => { msg.noteDraft = note.value; });
    const actions = el("div", "actions");
    if (view.question) actions.append(el("span", "msgline warn", view.question));
    for (const spec of view.buttons) {
      const button = el("button", "btn " + spec.kind, spec.label);
      button.type = "button";
      button.disabled = !!msg.deciding;
      button.addEventListener("click", () => {
        const decision = LumenReview.act(msg, spec.action);     // null: nothing is sent
        if (decision) decide(msg, decision); else renderBubble(msg);
      });
      actions.append(button);
    }
    if (msg.deciding) actions.append(el("span", "msgline", "Resuming from the saved checkpoint…"));
    box.append(note, actions);
  }
  if (msg.reviewNote) box.append(el("div", "msgline" + (msg.reviewBad ? " bad" : ""), msg.reviewNote));
  if (msg.decision && msg.decision.review_status === "reviewed" && LumenReview.overrideNote(msg)) {
    box.append(el("div", "msgline warn", LumenReview.overrideNote(msg)));
  }
  return box;
}

async function decide(msg, decision) {
  if (msg.deciding || msg.closed) return;        // one submission per review
  msg.deciding = true;
  msg.reviewNote = "";
  renderBubble(msg);
  try {
    const out = await api("POST", "/review/" + encodeURIComponent(msg.resp.thread_id),
                          { decision, reviewer_note: (msg.noteDraft || "").slice(0, 1000) }, ASK_TIMEOUT_MS);
    msg.closed = true;
    msg.decision = out;
    msg.resp = Object.assign({}, msg.resp, {
      status: out.status, review_status: out.review_status, answer: typeof out.answer === "string" ? out.answer : "",
      answer_is_draft: false, needs_human_review: !!out.needs_human_review,
      node_trail: Array.isArray(out.node_trail) ? out.node_trail : msg.resp.node_trail,
    });
    msg.reviewNote = "Decision recorded: " + decision + ". The paused run resumed and finished.";
    msg.reviewBad = false;
  } catch (err) {
    msg.reviewBad = true;
    if (err instanceof UiError && (err.code === "review_not_pending" || err.code === "review_not_found")) {
      msg.closed = true;                         // nothing left to decide; never offer the buttons again
      const was = err.data && err.data.detail && err.data.detail.review_status;
      msg.reviewNote = explain(err) + (was ? " Final status: " + (REVIEW_STATUS[was] || was) + "." : "");
    } else {
      msg.reviewNote = explain(err) + " The review is still pending; you can try again.";
    }
  } finally {
    msg.deciding = false;
  }
  renderBubble(msg);
  if (state.selected === msg) renderDetails();
}

// ------------------------------------------------------------ details ----
function select(msg) {
  if (state.selected && state.selected.node) state.selected.node.classList.remove("selected");
  state.selected = msg;
  msg.node.classList.add("selected");
  renderDetails();
}

function showTab(name) {
  for (const tab of document.querySelectorAll(".tab")) tab.classList.toggle("active", tab.dataset.tab === name);
  for (const pane of ["evidence", "workflow", "performance"]) $("pane-" + pane).classList.toggle("hidden", pane !== name);
}

function fmtMs(ms) {
  if (typeof ms !== "number" || !isFinite(ms)) return "—";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
}
function fmtDate(value) { return value ? String(value).slice(0, 10) : "—"; }

function kv(pairs) {
  const dl = el("dl", "kv");
  for (const [k, v] of pairs) {
    if (v === undefined || v === null || v === "") continue;
    dl.append(el("dt", null, k), el("dd", null, v));
  }
  return dl;
}

function claimRow(claim, verdict, why) {
  const row = el("div", "claim " + (verdict ? "ok" : "no"));
  row.append(el("span", "mark", verdict ? "✓" : "✗"), claim);
  if (why) row.append(el("span", "why", why));
  return row;
}

function renderEvidence(msg) {
  const pane = $("pane-evidence");
  pane.textContent = "";
  const r = msg && msg.resp;
  if (!r) { pane.append(el("p", "placeholder", "Select an answer to see its sources.")); return; }
  const cites = Array.isArray(r.citations) ? r.citations : [];
  const sources = Array.isArray(r.sources) ? r.sources : [];
  const reviewed = msg.decision && msg.decision.review_status;

  const pending = msg.pending && Array.isArray(msg.pending.flagged) ? msg.pending.flagged : null;
  if (r.status === "human_review_required" || pending) {
    pane.append(el("div", "section-title", "Flagged for review"));
    if (msg.pendingError && !pending) pane.append(el("p", "placeholder", msg.pendingError));
    if (!pending && !msg.pendingError) pane.append(el("p", "placeholder", "Loading flagged claims…"));
    for (const f of pending || []) {
      const card = el("div", "source");
      card.append(claimRow(f.claim || "", false, f.note || ""));
      card.append(kv([["Cited as", f.label ? "[" + f.label + "]" : "no citation"]]));
      if (f.source_text) {
        const more = el("details", "excerpt");
        more.append(el("summary", null, "Cited source text"), el("pre", null, f.source_text));
        card.append(more);
      }
      pane.append(card);
    }
  }

  pane.append(el("div", "section-title", "Sources (" + sources.length + ")"));
  if (!sources.length) pane.append(el("p", "placeholder", "This answer cites no sources."));
  for (const s of sources) {
    const label = String(s.label || "");
    const mine = cites.filter((c) => c.label === label);
    const card = el("div", "source" + (state.focusLabel === label ? " focus" : ""));
    card.dataset.label = label;
    const head = el("h4");
    head.append(el("span", "cite cite-" + label[0], label), SOURCE_KINDS[label[0]] || s.source_type || "Source");
    card.append(head);
    card.append(kv([
      ["Type", s.note_type || s.source_type], ["Date", fmtDate(s.charttime)],
      ["Admission", s.hadm_id], ["Note", s.note_id], ["Chunk", s.chunk_id > 0 ? s.chunk_id : null],
      ["Citation verified", mine.length ? (mine.every((c) => c.verified) ? "Yes" : "No") : "Not cited in the answer"],
    ]));
    for (const c of mine) card.append(claimRow(c.claim, !!c.verified));
    pane.append(card);
  }

  const orphan = cites.filter((c) => !c.label || !sources.some((s) => s.label === c.label));
  if (orphan.length) {
    pane.append(el("div", "section-title", "Claims without a valid source"));
    for (const c of orphan) pane.append(claimRow(c.claim, !!c.verified));
  }
  if (reviewed) {
    pane.append(el("p", "placeholder", "Verification marks above are from before the review. Reviewer decision: " +
                   (REVIEW_STATUS[reviewed] || reviewed) + "."));
  }

  const focus = state.focusLabel && pane.querySelector('.source[data-label="' + state.focusLabel + '"]');
  if (focus) focus.scrollIntoView({ block: "nearest" });
}

function renderWorkflow(msg) {
  const pane = $("pane-workflow");
  pane.textContent = "";
  const r = msg && msg.resp;
  if (!r) { pane.append(el("p", "placeholder", "Select an answer to see the nodes it ran.")); return; }
  const trail = Array.isArray(r.node_trail) ? r.node_trail : [];
  const list = el("ol", "flow");
  trail.forEach((name, i) => {
    if (i) list.append(el("li", "arrow", "↓"));
    const row = el("li");
    row.append(el("span", "tick", "✓"), NODE_NAMES[name] || String(name), el("small", null, name));
    list.append(row);
  });
  if (r.status === "human_review_required") {
    if (trail.length) list.append(el("li", "arrow", "↓"));
    const row = el("li", "paused");
    row.append(el("span", "tick", "⚠"), "Human review required — run is paused", el("small", null, "checkpointed"));
    list.append(row);
  }
  if (!trail.length) pane.append(el("p", "placeholder", "The API returned no node trail."));
  pane.append(list);
}

function renderPerformance(msg) {
  const pane = $("pane-performance");
  pane.textContent = "";
  const r = msg && msg.resp;
  if (!r) { pane.append(el("p", "placeholder", "Select an answer to see its timings.")); return; }
  const t = (r.timings && typeof r.timings === "object") ? r.timings : {};
  const models = (r.models && typeof r.models === "object") ? r.models : {};
  const ms = (key) => (typeof t[key] === "number" ? fmtMs(t[key]) : null);
  const rows = [
    ["Total", fmtMs(typeof r.latency_ms === "number" ? r.latency_ms : t.total_ms)],
    ["Retrieval", ms("retrieval_ms")], ["Reranking", ms("retrieval_reranking_ms")],
    ["Retriever load", typeof t.retriever_load_ms === "number" && t.retriever_load_ms > 50 ? ms("retriever_load_ms") : null],
    ["LLM", ms("llm_ms")],
    ["LLM calls", typeof t.llm_calls === "number"
      ? t.llm_calls + " (main " + (t.llm_main_calls || 0) + ", fast " + (t.llm_fast_calls || 0) + ")" : null],
    ["Deterministic answer", t.deterministic_answer ? "Yes" : null],
    ["Main model", models.main], ["Fast model", models.fast],
    ["Review status", REVIEW_STATUS[r.review_status] || r.review_status],
    ["Query type", r.query_type], ["Temporal mode", r.temporal_mode],
    ["Complexity", t.query_complexity], ["Classified by", t.classified_by],
    ["Thread", r.thread_id], ["Request", r.request_id],
  ];
  const table = el("table", "perf");
  for (const [k, v] of rows) {
    if (v === undefined || v === null || v === "") continue;
    const tr = el("tr");
    tr.append(el("td", null, k), el("td", null, v));
    table.append(tr);
  }
  pane.append(table);
  if (msg.decision) {
    pane.append(el("p", "placeholder", "Timings are from the original run. The review resumed the same thread; " +
                   "retrieval and synthesis were not run again."));
  }
}

function renderDetails() {
  renderEvidence(state.selected);
  renderWorkflow(state.selected);
  renderPerformance(state.selected);
}

// --------------------------------------------------------------- boot ----
$("composer").addEventListener("submit", send);
$("query").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $("composer").requestSubmit(); }
});
$("subject").addEventListener("change", subjectChanged);
$("subject").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); subjectChanged(); $("query").focus(); } });
for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => showTab(tab.dataset.tab));
showEmpty();
renderDetails();
refreshStatus();
setInterval(refreshStatus, STATUS_REFRESH_MS);
