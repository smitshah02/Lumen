/* Review-control logic for the Lumen UI, kept free of the DOM so it can be
 * tested under Node.
 *
 * The verifier says: "this draft contains unsupported or uncited claims."
 * Approving does not change that judgement. It records that a reviewer saw
 * the flags and chose to release the draft unchanged, so approving a flagged
 * draft takes two deliberate steps and nothing is sent on the first one.
 */
"use strict";

const LumenReview = {
  /* How many claims the verifier flagged on this paused draft. */
  flaggedCount(msg) {
    const pending = msg && msg.pending && Array.isArray(msg.pending.flagged) ? msg.pending.flagged.length : null;
    const reported = msg && msg.resp && typeof msg.resp.flagged_claims === "number" ? msg.resp.flagged_claims : 0;
    return pending !== null ? pending : reported;
  },

  plural(n, one, many) { return n + " " + (n === 1 ? one : many); },

  /* What the review box should show right now. */
  view(msg) {
    const open = !!(msg && msg.resp && msg.resp.status === "human_review_required" && !msg.closed);
    if (!open) return { open: false, buttons: [] };
    const n = LumenReview.flaggedCount(msg);
    if (n === 0) {
      return { open: true, count: 0, warning: "", question: "",
               buttons: [{ action: "approve", label: "Approve", kind: "approve" },
                         { action: "reject", label: "Reject", kind: "reject" }] };
    }
    const warning = "This draft contains " + LumenReview.plural(n, "flagged claim", "flagged claims") +
                    ". Approving will release " + (n === 1 ? "it" : "them") + " unchanged.";
    if (msg.confirming) {
      return { open: true, count: n, warning,
               question: "Release this draft despite " + LumenReview.plural(n, "verifier flag", "verifier flags") + "?",
               buttons: [{ action: "confirm", label: "Confirm approval", kind: "approve" },
                         { action: "cancel", label: "Cancel", kind: "" }] };
    }
    return { open: true, count: n, warning, question: "",
             buttons: [{ action: "approve", label: "Approve draft as-is", kind: "approve" },
                       { action: "reject", label: "Reject", kind: "reject" }] };
  },

  /* Apply a button press. Returns the decision to POST, or null when nothing
   * may be sent: a first click on a flagged draft, a cancel, or a review that
   * is already being submitted or is closed. */
  act(msg, action) {
    if (!msg || msg.deciding || msg.closed) return null;
    const allowed = LumenReview.view(msg).buttons.some((b) => b.action === action);
    if (!allowed) return null;
    if (action === "reject") { msg.confirming = false; return "reject"; }
    if (action === "cancel") { msg.confirming = false; return null; }
    if (action === "confirm") { msg.confirming = false; msg.overridden = LumenReview.flaggedCount(msg); return "approve"; }
    if (LumenReview.flaggedCount(msg) > 0) { msg.confirming = true; return null; }   // first click: ask, send nothing
    msg.overridden = 0;
    return "approve";
  },

  /* Shown after an approval, so an override is never mistaken for verification. */
  overrideNote(msg) {
    const n = msg && msg.overridden;
    if (!n) return "";
    return "Reviewer override: " + LumenReview.plural(n, "verifier flag was", "verifier flags were") +
           " approved as-is. The verifier's judgement is unchanged.";
  },
};

/* Split an answer into prose and citation runs, for display only. A run is
 * one or more markers with nothing but whitespace between them: "[S1] [S2]"
 * is one run, "[S1], and ... [S2]" is two. The answer text itself is never
 * changed; every label of a run is kept, in order, without repeats. */
/* Structured-record labels ("R1") of the answer being rendered. "[R1]" is a
 * citation only when the response has an R1 source; otherwise it stays text. */
LumenReview.structuredLabels = [];
LumenReview.citationRuns = function (text) {
  const parts = [];
  let last = 0;
  const known = LumenReview.structuredLabels.filter((l) => /^R\d+$/.test(l));
  const one = "\\[(?:[SLGPA]\\d+" + known.map((l) => "|" + l).join("") + ")\\]";
  for (const run of String(text).matchAll(new RegExp(one + "(?:\\s*" + one + ")*", "g"))) {
    if (run.index > last) parts.push({ text: text.slice(last, run.index) });
    parts.push({ labels: [...new Set(run[0].match(/[SLGPAR]\d+/g))] });
    last = run.index + run[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last) });
  return parts;
};

if (typeof module !== "undefined") module.exports = LumenReview;
