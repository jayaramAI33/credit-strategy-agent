"""Build a self-contained replay page (one HTML file, no internet needed) from agent audit logs.

Usage:
  python replay/build_replay.py <run_dir_act1> [<run_dir_act2>] [--out outputs/replay/demo_replay.html]

The page is a fixed 16:9 stage (designed at 960x540) scaled to fill the window, so text is the
same share of the frame however it is recorded, and stays readable on a phone at 720p.
Each run is an "act": a full-screen GOAL, then (first act only) the RISK APPETITE full-screen,
then cards labelled AGENT DECIDED / CODE CALCULATED / CHECK REJECTED / HUMAN beside a chart of
every plan tested, paced for reading. Later acts are condensed (goal, opening plan, one summary
card, outcome). A closing screen ends the replay. All figures come from the audit log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csa import audit  # noqa: E402

CAPTIONS = {
    "goal": "A business goal arrives in plain English.",
    "goal2": "Now a bigger ask, from the same agent under the same appetite.",
    "appetite": "The agent reads the risk appetite. It is locked: no tool can change it.",
    "simulate": "Code tests each plan against every limit. The agent reads the result and decides.",
    "rejected": "A check rejects any figure it cannot trace to a tool result. The agent fixes it.",
    "propose": "It proposes a small randomised test, not a rollout.",
    "stop": "Then it stops. A person must approve.",
    "human": "A person decides. Only test cells change; the current policy does not.",
    "refuse": "Asked for more than the appetite allows, it refuses, names the limit and makes a counter-offer.",
    "end": "Every number on screen came from deterministic code, not the language model.",
}
DISCLAIMER = "Personal prototype · public Lending Club data, 2012Q4–2015 · no employer data"

# Reading pace
WORDS_PER_SEC = 4.0
BASE_MS, MIN_MS, MAX_MS = 2500, 3500, 16000
SCREEN_MS = {"goal": 6000, "appetite": 15000, "end": 12000}
OUTCOME_MIN_MS = 9000

END_SCREEN = {
    "kind": "screen", "screen": "end", "label": "CREDIT STRATEGY AGENT",
    "title": "Growth inside a locked risk appetite, with a person deciding.",
    "lines": ["Revised: fixed its own plan when approvals fell",
              "Rejected: a plan that passed every limit but could still raise the bad rate",
              "Refused: “Grow 15%”, with a counter-offer",
              "Stopped: for a person to approve the test"],
    "note": "Built by Jayaram Ranganathan · 20 years, 11+ in product  ·  "
            "personal prototype on public data · github.com/jayaramAI33/credit-strategy-agent",
    "caption": CAPTIONS["end"],
}


def _fmt(v, nd=2, sign=False):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _clip(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " …"


def _limits_text(appetite: dict) -> list[str]:
    L = appetite["limits"]
    st = L["stressed_portfolio_bad_rate_delta_pp"]
    return [
        f"Portfolio bad rate may rise by at most {L['portfolio_bad_rate_delta_pp']['max']} pp",
        f"… still at most {st['max']} pp if every swap-in performs {st['swap_in_shock_pp']} pp worse",
        "Swap-ins no riskier than the riskiest customers approved today",
        f"No pocket (decline reason) above {L['swap_in_concentration_pct']['max']}% of swapped-in loans",
        f"Each segment at least {L['segment_min_volume_pct']['min']}% of applications",
        f"Net approval growth at most +{L['net_approval_growth_pct']['max']}%",
        f"Test cells at most {L['test_cell_share_pct']['max']}% of a segment",
        "Green → credit manager · Amber → senior risk · Red → refused",
    ]


def _sim_lines(r: dict) -> list[str]:
    bad = [c for c in r["limit_checks"] if c["status"] == "breach"]
    ci = r.get("delta_bad_rate_ci95_pp")
    lines = [f"In: {', '.join(r['swap_ins']) or '—'}  ·  Out: {', '.join(r['swap_outs']) or '—'}",
             f"Approvals {_fmt(r['approval_growth_pct'], 2, True)}%  ·  bad rate {_fmt(r['delta_bad_rate_pp'], 3, True)} pp"
             + (f" (95% range {_fmt(ci[0], 3, True)} to {_fmt(ci[1], 3, True)})" if ci else ""),
             f"Stress test {_fmt(r['stressed_delta_bad_rate_pp'], 3, True)} pp"]
    lines += [f"✖ {c['check']}: {c['value']} vs limit {c['limit']}" for c in bad]
    return lines


def steps_from_run(run_dir: Path, act_no: int) -> tuple[list[dict], dict]:
    run_dir = Path(run_dir)
    ev = audit.read(run_dir)
    steps, meta, pending = [], {}, {}
    for e in ev:
        t = e["event"]
        if t == "run_start":
            meta.update(goal=e["goal"], model=e["model"])
            steps.append({"kind": "screen", "screen": "goal", "label": "THE BUSINESS GOAL", "title": f"“{e['goal']}”",
                          "lines": [], "note": DISCLAIMER, "caption": CAPTIONS["goal" if act_no == 1 else "goal2"]})
        elif t in ("agent_note", "agent_text"):
            steps.append({"kind": "agent", "label": "AGENT DECIDED", "title": "", "lines": [_clip(e["text"].strip(), 60)]})
        elif t == "tool_call":
            pending = e
        elif t == "tool_error":
            steps.append({"kind": "rejected", "label": "CHECK REJECTED", "title": f"{e['tool']} refused",
                          "lines": [_clip(e["error"], 35)], "caption": CAPTIONS["rejected"]})
        elif t == "tool_result":
            tool, r = e["tool"], e["result"]
            name = {"get_baseline": "Baseline: current policy", "find_segments": "Candidate segments",
                    "simulate_swap_set": "Plan tested", "draft_test_proposal": "Test proposal drafted",
                    "export_strategy": "Decision table exported"}.get(tool, tool)
            s = {"kind": "code", "label": "CODE CALCULATED", "title": f"{r['result_id']} · {name}", "lines": [],
                 "tool": tool}
            if tool == "get_risk_appetite":
                meta["appetite"] = r["appetite"]
                if act_no == 1:
                    s = {"kind": "screen", "screen": "appetite",
                         "label": f"{r['result_id']} · THE RISK APPETITE · READ-ONLY, SET BY RISK",
                         "title": "Limits the agent cannot change", "lines": _limits_text(r["appetite"]),
                         "caption": CAPTIONS["appetite"]}
                else:
                    s.update(title=f"{r['result_id']} · Same locked risk appetite, read-only")
            elif tool == "get_baseline":
                s["lines"] = [f"2015: {_fmt(r['approval_rate_pct'])}% approved, "
                              f"approved bad rate {_fmt(r['approved_bad_rate_pct'])}%"]
            elif tool == "find_segments":
                kind = "swap-in (growth)" if r["kind"] == "swap_in" else "swap-out (loss control)"
                s["lines"] = [f"{r['count']} {kind} candidates"]
            elif tool == "simulate_swap_set":
                s.update(lines=_sim_lines(r), band=r["band"], caption=CAPTIONS["simulate"],
                         point={"id": r["result_id"], "x": r["delta_bad_rate_pp"], "xs": r["stressed_delta_bad_rate_pp"],
                                "y": r["approval_growth_pct"], "band": r["band"]})
            elif tool == "draft_test_proposal":
                cells = r["test_cells"]
                s.update(lines=[f"{len(cells)} randomised test cells, each {cells[0]['test_share_pct']}% of its segment",
                                "Stop rule: pause a cell if its bad rate passes the swap-in limit"] if cells else ["No cells"],
                         caption=CAPTIONS["propose"], band=r["band"])
            elif tool == "export_strategy":
                s["lines"] = [f"Decision table with {r['cells']} cells saved. Not deployed."]
            elif tool == "submit_for_approval":
                p = json.loads((run_dir / "pending_approval.json").read_text())
                e2 = p["proposal"]["expected_impact_if_rolled_out"]
                s = {"kind": "stop", "label": "STOPPED", "title": "Pending human approval", "band": p["band"],
                     "lines": [f"Approvals {_fmt(e2['approval_growth_pct'], 2, True)}%  ·  bad rate "
                               f"{_fmt(e2['delta_bad_rate_pp'], 3, True)} pp  ·  needs {p['route'].replace('_', ' ')}",
                               "Note to the committee: " + _clip(p["recommendation"], 40)],
                     "caption": CAPTIONS["stop"]}
            elif tool == "report_refusal":
                f = json.loads((run_dir / "refusal.json").read_text())
                co = f.get("counter_offer") or {}
                s = {"kind": "refuse", "label": "REFUSED", "title": "Goal declined", "band": co.get("band"),
                     "lines": [f"Binding: {', '.join(x.replace('_', ' ') for x in f['binding_limit_ids'])}",
                               f"Counter-offer: approvals {_fmt(co.get('approval_growth_pct'), 2, True)}%, bad rate "
                               f"{_fmt(co.get('delta_bad_rate_pp'), 3, True)} pp ({co.get('band', '')})",
                               _clip(f["explanation"], 40)],
                     "caption": CAPTIONS["refuse"]}
            steps.append(s)
        elif t == "human_decision":
            verb = {"approve": "APPROVED", "reject": "REJECTED"}.get(e["decision"], e["decision"].upper())
            steps.append({"kind": "human", "label": "HUMAN",
                          "title": f"{verb} by {e['approver']} ({e['role'].replace('_', ' ')})",
                          "lines": ["Test cells enabled. Current policy unchanged."
                                    if e["strategy_updated"] else "No change to the strategy."],
                          "caption": CAPTIONS["human"]})
        elif t == "run_end":
            meta.update(elapsed_s=e["elapsed_s"], cost=e["est_cost_usd"], calls=e["api_calls"])
    return steps, meta


ROUTINE_TOOLS = {"get_baseline", "find_segments", "export_strategy"}


def trim_routine(steps: list[dict]) -> list[dict]:
    """Drop machine-work cards (baseline, segment lists, export) and the short agent notes that
    only announce them. Longer notes, where the agent reasons, are kept."""
    drop = {k for k, s in enumerate(steps) if s.get("tool") in ROUTINE_TOOLS}
    for k in sorted(drop):
        j = k - 1
        while j >= 0 and steps[j]["kind"] == "agent" and len(steps[j]["lines"][0].split()) <= 20:
            drop.add(j); j -= 1
    return [s for k, s in enumerate(steps) if k not in drop]


def condense(steps: list[dict]) -> list[dict]:
    """Highlight cut for a later act: the goal, the agent's opening plan, ONE summary card of every
    plan it tested (all plotted at once), and the final outcome. Labelled as condensed."""
    goal = next(s for s in steps if s["kind"] == "screen")
    opening = next((s for s in steps if s["kind"] == "agent"), None)
    pts = [s["point"] for s in steps if s.get("point")]
    final = [s for s in steps if s["kind"] in ("stop", "refuse", "human")]
    out = [goal] + ([opening] if opening else [])
    if pts:
        widest = max(pts, key=lambda p: p["y"])
        ok = [p for p in pts if p["band"] != "red"]
        best = max(ok, key=lambda p: p["y"]) if ok else None
        lines = [f"Widest: {widest['id']}, {_fmt(widest['y'], 2, True)}% approvals at "
                 f"{_fmt(widest['x'], 3, True)} pp bad rate ({widest['band']})"]
        if best:
            lines.append(f"Best inside the appetite: {best['id']}, {_fmt(best['y'], 2, True)}% approvals, "
                         f"{_fmt(best['x'], 3, True)} pp ({best['band']})")
        lines.append("Condensed for this replay; every step is in the audit log.")
        out.append({"kind": "code", "label": "CODE CALCULATED",
                    "title": f"{pts[0]['id']}–{pts[-1]['id']} · {len(pts)} plans tested (condensed)",
                    "lines": lines, "points": pts,
                    "caption": f"It tests {len(pts)} plans. None reaches the goal inside the appetite."})
    return out + final


def _ms(st: dict) -> int:
    if st["kind"] == "screen":
        return SCREEN_MS[st["screen"]]
    words = len(st.get("title", "").split()) + sum(len(x.split()) for x in st["lines"] if x)
    ms = max(MIN_MS, min(MAX_MS, int(BASE_MS + 1000 * words / WORDS_PER_SEC)))
    return max(ms, OUTCOME_MIN_MS) if st["kind"] in ("stop", "refuse", "human") else ms


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Strategy Agent replay</title>
<style>
:root{--bg:#f4f5f7;--panel:#fff;--ink:#14171c;--mute:#5b6370;--line:#dfe3e8;--agent:#4b3fd1;--code:#0f766e;
--rej:#b42318;--human:#9a3412;--green:#15803d;--amber:#b45309;--red:#b42318;}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#14171c}
body{display:flex;align-items:center;justify-content:center;font-family:"Segoe UI",system-ui,sans-serif;overflow:hidden}
#stage{position:relative;width:960px;height:540px;background:var(--bg);color:var(--ink);overflow:hidden;flex:none}
header{position:absolute;left:0;right:0;top:0;height:42px;display:flex;justify-content:space-between;align-items:center;padding:0 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header .t{font-size:15px;font-weight:700}header .sub{font-size:11px;color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:640px}
.ctrl button{font:600 11px "Segoe UI",sans-serif;border:1px solid var(--line);background:#fff;border-radius:6px;padding:3px 8px;margin-left:4px;cursor:pointer}
main{position:absolute;left:14px;right:14px;top:52px;bottom:76px;display:grid;grid-template-columns:1.12fr 1fr;gap:14px}
#feed{display:flex;flex-direction:column;gap:9px;height:100%;overflow:hidden;justify-content:flex-end}
.card{flex:none;background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--mute);border-radius:9px;padding:9px 12px;opacity:0;transform:translateY(8px);transition:all .6s}
.card.show{opacity:1;transform:none}
.card .lab{font-size:11.5px;font-weight:800;letter-spacing:.06em}
.card .ttl{font-size:18px;font-weight:700;margin:1px 0 2px;line-height:1.25}
.card .ln{font-size:17px;line-height:1.35;color:#23282f}
.k-agent{border-left-color:var(--agent)}.k-agent .lab{color:var(--agent)}
.k-code{border-left-color:var(--code)}.k-code .lab{color:var(--code)}
.k-rejected{border-left-color:var(--rej);background:#fff6f5}.k-rejected .lab{color:var(--rej)}
.k-stop,.k-refuse{border-left-color:var(--ink);background:#fbfbf6}.k-stop .lab,.k-refuse .lab{color:var(--ink)}
.k-human{border-left-color:var(--human);background:#fff8f1}.k-human .lab{color:var(--human)}
.band{display:inline-block;padding:1px 8px;border-radius:99px;color:#fff;font-size:11px;font-weight:800;margin-left:7px}
.b-green{background:var(--green)}.b-amber{background:var(--amber)}.b-red{background:var(--red)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:9px 12px;height:100%;display:flex;flex-direction:column}
.panel .h{font-size:14px;font-weight:700}.panel .note{font-size:11.5px;color:var(--mute);line-height:1.3}
#svg{flex:1;min-height:0}#svg svg{width:100%;height:100%}
#screen{position:absolute;left:14px;right:14px;top:52px;bottom:76px;display:none;align-items:center;justify-content:center;background:var(--bg)}
#screen.on{display:flex}
#screen .box{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px 30px;opacity:0;transition:opacity .8s}
#screen .box.show{opacity:1}
#screen .lab{font-size:12.5px;font-weight:800;letter-spacing:.08em;color:#0f766e}
#screen h2{font-size:28px;line-height:1.2;margin:6px 0 10px}
#screen ul{margin:0;padding-left:22px;font-size:17.5px;line-height:1.42;color:#23282f}
#screen .note{margin-top:10px;font-size:12.5px;color:var(--mute)}
#cap{position:absolute;left:0;right:0;bottom:0;height:66px;background:#14171c;color:#fff;padding:10px 18px}
#cap .c{font-size:20px;font-weight:700;line-height:1.25}#cap small{display:block;font-size:11.5px;color:#c9ced6;margin-top:3px}
</style></head><body>
<div id="stage">
<header><div><div class="t">Credit Strategy Agent — run replay</div><div class="sub" id="sub"></div></div>
<div class="ctrl"><button id="play">Pause</button><button id="next">Next ▶</button><button id="restart">Restart</button></div></header>
<div id="screen"><div class="box"><div class="lab" id="slab"></div><h2 id="stitle"></h2><ul id="slist"></ul><div class="note" id="snote"></div></div></div>
<main><section id="feed"></section>
<div class="panel"><div class="h">Plans tested (2015 data, change vs current policy)</div><div id="svg"></div>
<div class="note">● measured · ○ stress test (swap-ins 2 pp worse) · dashed: +0.2 pp limit · x: bad-rate change (pp) · y: approvals change (%)</div></div></main>
<div id="cap"></div>
</div>
<script>
const DATA = __DATA__;
let i = 0, timer = null, playing = true;
const stage = document.getElementById('stage'), feed = document.getElementById('feed'), cap = document.getElementById('cap'), scr = document.getElementById('screen');
function fit(){const s=Math.min(innerWidth/960, innerHeight/540);stage.style.zoom=s;}
addEventListener('resize', fit); fit();
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function bandChip(b){return b?`<span class="band b-${b}">${b.toUpperCase()}</span>`:''}
function setAct(a){const m=DATA.acts[a];document.getElementById('sub').textContent=`${m.title} · ${m.model} · real run ${m.elapsed_s} s, about $${m.cost} (slowed down for reading)`;}
function setCap(st){cap.innerHTML=`<div class="c">${esc(st.caption)}</div><small>${esc(DATA.acts[st.act].title)}</small>`;}
function drawChart(act){const pts=DATA.steps.slice(0,i).filter(s=>s.act===act).flatMap(s=>s.points||(s.point?[s.point]:[]));
  const W=600,H=420,P=56;const xs=pts.flatMap(p=>[p.x,p.xs]).concat([0,0.25,-0.1]);const ys=pts.map(p=>p.y).concat([0,6]);
  const x0=Math.min(...xs)-0.06,x1=Math.max(...xs)+0.06,y0=Math.min(...ys,0),y1=Math.max(...ys)*1.12+0.5;
  const X=v=>P+(v-x0)/(x1-x0)*(W-P-14),Y=v=>H-P-(v-y0)/(y1-y0)*(H-P-14);
  const col={green:'#15803d',amber:'#b45309',red:'#b42318'};
  let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"><rect x="${X(0.2)}" y="14" width="${Math.max(0,W-14-X(0.2))}" height="${H-P-14}" fill="#b4231814"/>`;
  s+=`<line x1="${X(0.2)}" x2="${X(0.2)}" y1="14" y2="${H-P}" stroke="#b42318" stroke-width="2" stroke-dasharray="7 5"/><text x="${X(0.2)+6}" y="36" font-size="20" fill="#b42318">limit</text>`;
  s+=`<line x1="${X(0)}" x2="${X(0)}" y1="14" y2="${H-P}" stroke="#9aa3ae" stroke-width="1.5"/><line x1="${P}" x2="${W-14}" y1="${Y(0)}" y2="${Y(0)}" stroke="#9aa3ae" stroke-width="1.5"/>`;
  s+=`<circle cx="${X(0)}" cy="${Y(0)}" r="8" fill="#14171c"/><text x="${X(0)-12}" y="${Y(0)-12}" font-size="19" text-anchor="end">today</text>`;
  pts.forEach(p=>{s+=`<line x1="${X(p.x)}" x2="${X(p.xs)}" y1="${Y(p.y)}" y2="${Y(p.y)}" stroke="${col[p.band]}" stroke-width="2.5" opacity=".6"/>`;
    s+=`<circle cx="${X(p.xs)}" cy="${Y(p.y)}" r="8" fill="#fff" stroke="${col[p.band]}" stroke-width="3"/>`;
    s+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="12" fill="${col[p.band]}"/><text x="${X(p.x)-12}" y="${Y(p.y)-18}" font-size="21" font-weight="700">${p.id}</text>`;});
  const tick=(v,a)=>a==='x'?`<text x="${X(v)}" y="${H-P+26}" font-size="18" text-anchor="middle" fill="#5b6370">${v.toFixed(1)}</text>`:`<text x="${P-10}" y="${Y(v)+6}" font-size="18" text-anchor="end" fill="#5b6370">${v.toFixed(0)}%</text>`;
  for(let v=Math.ceil(x0*5)/5;v<=x1;v+=0.2)s+=tick(v,'x');for(let v=Math.ceil(y0);v<=y1;v+=Math.max(1,Math.round((y1-y0)/5)))s+=tick(v,'y');
  document.getElementById('svg').innerHTML=s+'</svg>';}
function showScreen(st){scr.classList.add('on');document.querySelector('main').style.visibility='hidden';
  const box=scr.querySelector('.box');box.classList.remove('show');
  document.getElementById('slab').textContent=st.label;document.getElementById('stitle').textContent=st.title;
  document.getElementById('slist').innerHTML=(st.lines||[]).map(l=>`<li>${esc(l)}</li>`).join('');
  document.getElementById('snote').textContent=st.note||'';
  requestAnimationFrame(()=>requestAnimationFrame(()=>box.classList.add('show')));}
function show(){if(i>=DATA.steps.length){stop();return;}
  const st=DATA.steps[i];if(i===0||DATA.steps[i-1].act!==st.act){feed.innerHTML='';setAct(st.act);}
  if(st.kind==='screen'){showScreen(st);}
  else{scr.classList.remove('on');document.querySelector('main').style.visibility='visible';
    const d=document.createElement('div');d.className=`card k-${st.kind}`;
    d.innerHTML=`<div class="lab">${esc(st.label)}${bandChip(st.band)}</div>`+(st.title?`<div class="ttl">${esc(st.title)}</div>`:'')+
      st.lines.filter(Boolean).map(l=>`<div class="ln">${esc(l)}</div>`).join('');
    feed.appendChild(d);
    const used=()=>[...feed.children].reduce((h,c)=>h+c.offsetHeight+9,-9);
    while(feed.children.length>1&&used()>feed.clientHeight)feed.removeChild(feed.firstChild);
    requestAnimationFrame(()=>requestAnimationFrame(()=>d.classList.add('show')));}
  if(st.caption)setCap(st);
  i++;drawChart(st.act);}
function tick(){show();if(playing&&i<DATA.steps.length){const st=DATA.steps[i-1]||{};timer=setTimeout(tick,st.ms||4000);}else stop();}
function stop(){playing=false;clearTimeout(timer);document.getElementById('play').textContent='Play';}
document.getElementById('play').onclick=()=>{if(playing){stop();}else{playing=true;document.getElementById('play').textContent='Pause';tick();}};
document.getElementById('next').onclick=()=>{stop();show();};
document.getElementById('restart').onclick=()=>{stop();i=0;feed.innerHTML='';playing=true;document.getElementById('play').textContent='Pause';tick();};
document.addEventListener('keydown',e=>{if(e.key===' '||e.key==='ArrowRight'){e.preventDefault();document.getElementById('next').click();}});
timer = setTimeout(tick, 1500);
</script></body></html>"""


def build(run_dirs: list[Path], out: Path):
    steps, acts = [], {}
    for n, rd in enumerate(run_dirs, 1):
        act = f"act{n}"
        s, meta = steps_from_run(rd, n)
        s = condense(s) if n > 1 else trim_routine(s)
        if n == len(run_dirs):
            s.append(dict(END_SCREEN))
        for st in s:
            st["act"], st["ms"] = act, _ms(st)
        steps += s
        acts[act] = {"title": f"Run {n}: “{meta.get('goal', '')}”", "model": meta.get("model"),
                     "elapsed_s": meta.get("elapsed_s"), "calls": meta.get("calls"), "cost": meta.get("cost")}
    data = {"steps": steps, "acts": acts}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    total = sum(s["ms"] for s in steps) / 1000
    print(f"wrote {out} · {len(steps)} steps · about {total / 60:.1f} min of playback")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = ROOT / "outputs" / "replay" / "demo_replay.html"
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--out") + 1]]
    if not args:
        sys.exit(__doc__)
    build([Path(a) if Path(a).is_absolute() else ROOT / "outputs" / "runs" / a for a in args], out)
