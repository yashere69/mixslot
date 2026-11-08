"""Build site/index.html: the benchmark as an interactive page (no external assets).

python tools/build_site.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mixslot.bench import LEVELS, examples, run

ROOT = Path(__file__).resolve().parent.parent

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mixslot: Hinglish transcript noise vs tool arguments</title>
<meta name="description" content="How speech-recognition errors in Hinglish corrupt order ids,
phone numbers and PIN codes passed to a voice agent's tools, and how much normalising recovers.">
<style>
:root{--bg:#fbfaf7;--fg:#17212b;--muted:#5b6672;--line:#e3dfd6;--ok:#1f6f5c;--bad:#b3412e;
--skip:#c9c2b3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.6rem;margin:0 0 6px} p{margin:0 0 14px} .muted{color:var(--muted)}
.panel{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0}
label{font-weight:600} input[type=range]{width:100%;accent-color:var(--ok)}
.row{display:grid;grid-template-columns:110px 1fr 70px;gap:10px;align-items:center;margin:10px 0}
.bar{display:flex;height:22px;border-radius:6px;overflow:hidden;background:var(--line)}
.bar span{display:block;height:100%}
.c{background:var(--ok)} .w{background:var(--bad)} .a{background:var(--skip)}
.legend span{margin-right:16px;font-size:.9rem;white-space:nowrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
select{font:inherit;padding:4px 8px}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}
td code{white-space:nowrap} .wrap{overflow-x:auto}
.miss{color:var(--bad)} .hit{color:var(--ok)}
</style></head><body><main>
<h1>Hinglish transcript noise vs tool arguments</h1>
<p class="muted">A voice agent hears <em>"order kay see saat do paanch do"</em>. Does its tool
get <code>KC7252</code>? Drag the noise level to see how often the order id, phone number, PIN
code or amount comes out <b>correct</b>, <b>wrong</b>, or is <b>refused</b> so the agent can ask
again.</p>
<div class="panel">
<label for="lvl">Noise level: <output id="lv">0.30</output></label>
<input id="lvl" type="range" min="0" max="__MAX__" step="1" value="6">
<p style="margin-top:12px"><label for="slot">Slot</label>
<select id="slot"><option value="all">all slots</option><option>order_id</option>
<option>phone</option><option>pin</option><option>amount</option></select></p>
<div id="bars" aria-live="polite"></div>
<p class="legend"><span><i class="c"></i>correct</span><span><i class="w"></i>wrong value
sent to the tool</span><span><i class="a"></i>refused, agent asks again</span></p>
</div>
<div class="panel"><h2 style="font-size:1.1rem;margin:0 0 8px">Sample transcripts at noise 0.3</h2>
<div class="wrap"><table><thead><tr><th>Transcript</th><th>Wanted</th><th>Naive</th>
<th>Normalised</th></tr></thead><tbody id="ex"></tbody></table></div></div>
<p class="muted">The noise model is synthetic (n=__N__ utterances, seed __SEED__) and, checked
against a real recogniser, optimistic. On Whisper base transcripts of Hindi TTS speech the
normaliser got 11.1% of values right and passed on a wrong one 11.6% of the time
(<a href="https://github.com/yashere69/mixslot/blob/main/results/asr.md">results/asr.md</a>).
Treat this page as the shape of the problem, not a recogniser's error rate.
Source: <a href="https://github.com/yashere69/mixslot">github.com/yashere69/mixslot</a>.</p>
</main>
<script>
const DATA = __DATA__, LEVELS = __LEVELS__, EX = __EX__;
const lvl = document.getElementById("lvl"), slot = document.getElementById("slot");
function counts(row, s){ return s === "all" ? row.total : row.slots[s]; }
function render(){
  const level = LEVELS[+lvl.value];
  document.getElementById("lv").textContent = level.toFixed(2);
  let html = "";
  for (const name of ["naive", "normalised"]) {
    const row = DATA.rows.find(r => r.level === level && r.extractor === name);
    const c = counts(row, slot.value), n = c.correct + c.wrong + c.abstain;
    const pc = k => (100 * c[k] / n).toFixed(1);
    html += `<div class="row"><strong>${name}</strong><div class="bar" role="img"
      aria-label="${name}: ${pc("correct")}% correct, ${pc("wrong")}% wrong, ${pc("abstain")}% refused">
      <span class="c" style="width:${pc("correct")}%"></span><span class="w"
      style="width:${pc("wrong")}%"></span><span class="a" style="width:${pc("abstain")}%"></span>
      </div><span>${pc("correct")}%</span></div>`;
  }
  document.getElementById("bars").innerHTML = html;
}
const esc = s => String(s ?? "refused").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
document.getElementById("ex").innerHTML = EX.map(e => `<tr><td>${esc(e.transcript)}</td>
  <td><code>${e.gold}</code></td>
  <td class="${e.naive === e.gold ? "hit" : "miss"}"><code>${esc(e.naive)}</code></td>
  <td class="${e.normalised === e.gold ? "hit" : "miss"}"><code>${esc(e.normalised)}</code></td></tr>`).join("");
lvl.addEventListener("input", render); slot.addEventListener("change", render); render();
</script></body></html>
"""


def main() -> None:
    data = run()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "curve.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    page = (
        PAGE.replace("__DATA__", json.dumps(data))
        .replace("__LEVELS__", json.dumps(LEVELS))
        .replace("__EX__", json.dumps(examples(), ensure_ascii=False))
        .replace("__MAX__", str(len(LEVELS) - 1))
        .replace("__N__", str(data["n"]))
        .replace("__SEED__", str(data["seed"]))
    )
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "index.html").write_text(page, encoding="utf-8")
    print("wrote site/index.html and results/curve.json")


if __name__ == "__main__":
    main()
