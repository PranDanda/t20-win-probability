"use strict";
const DATA = "data/";
const NS = "http://www.w3.org/2000/svg";
const LIST_LIMIT = 150;
let INDEX = [];
const cache = new Map();

const fmtPct = p => (p < 0.01 ? "<1" : p > 0.99 ? ">99" : Math.round(p * 100)) + "%";
const fmtDate = s => new Date(s + "T12:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

async function getMatch(id) {
  if (!cache.has(id)) cache.set(id, fetch(`${DATA}matches/${id}.json`).then(r => r.json()));
  return cache.get(id);
}

function ballText(m, k) {
  const b = m.balls;
  if (b.text[k]) return b.text[k];
  const bowler = m.names[b.bowler[k]], batter = m.names[b.batter[k]];
  const r = b.runs[k];
  return r === 0 ? `Dot ball: ${bowler} to ${batter}` : `${r} run${r === 1 ? "" : "s"}: ${batter} off ${bowler}`;
}

/* ---------- Win-probability chart ---------- */
function drawMatch(container, m, theme) {
  container.innerHTML = "";
  const onBoard = theme === "board";
  // Narrow screens get a smaller drawing space, so labels stay readable when the SVG is scaled down
  const narrow = (container.clientWidth || 960) < 600;
  const W = narrow ? 480 : 960, H = narrow ? (onBoard ? 340 : 300) : (onBoard ? 400 : 360), L = 16, R = 16, T = 34, B = 40;
  const col = onBoard
    ? { line: css("--board-ink"), dim: css("--board-dim"), grid: "rgba(242,244,239,.18)", c1: "#ef9aa0", c2: "#a7c8ea", bg: css("--board") }
    : { line: css("--ink"), dim: css("--ink-2"), grid: css("--rule"), c1: css("--red"), c2: css("--blue"), bg: css("--paper") };
  const p = m.balls.p.concat([m.final]);
  const n = p.length;
  const firstInn2 = m.balls.inn.indexOf(2);
  const gap = 14;
  const plotW = W - L - R - gap;
  const xs = k => L + (k / (n - 1)) * plotW + (firstInn2 >= 0 && k >= firstInn2 ? gap : 0);
  const ys = v => T + (1 - v) * (H - T - B);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, tabindex: 0, "aria-label": `${m.team1} v ${m.team2} win probability; use arrow keys to step through balls` }, container);
  const uid = Math.random().toString(36).slice(2, 8);
  const defs = el("defs", {}, svg);
  el("rect", { x: 0, y: 0, width: W, height: ys(0.5) }, el("clipPath", { id: `top${uid}` }, defs));
  el("rect", { x: 0, y: ys(0.5), width: W, height: H }, el("clipPath", { id: `bot${uid}` }, defs));

  for (const v of [0.25, 0.5, 0.75]) {
    el("line", { x1: L, x2: W - R, y1: ys(v), y2: ys(v), stroke: col.grid, "stroke-width": v === 0.5 ? 1.5 : 1,
      "stroke-dasharray": v === 0.5 ? "" : "3 5" }, svg);
  }
  // team labels at top (team1 favoured) and bottom (team2 favoured)
  const lab = (txt, y, c) => { const t = el("text", { x: L, y, fill: c, "font-size": 17, "font-weight": 600 }, svg); t.textContent = txt; };
  lab(`${m.team1} win  ▲`, T - 12, col.c1);
  lab(`${m.team2} win  ▼`, H - B + 30, col.c2);
  const pct = (txt, y) => { const t = el("text", { x: W - R, y: y + 5, fill: col.dim, "font-size": 14, "text-anchor": "end" }, svg); t.textContent = txt; };
  pct("75%", ys(0.75)); pct("50%", ys(0.5)); pct("25%", ys(0.25));

  // innings divider and over ticks
  if (firstInn2 > 0) {
    const xd = (xs(firstInn2 - 1) + xs(firstInn2)) / 2;
    el("line", { x1: xd, x2: xd, y1: T, y2: H - B, stroke: col.grid, "stroke-width": 1 }, svg);
  }
  m.balls.over.forEach((o, k) => {
    const [ov, b] = o.split(".").map(Number);
    if (b === 1 && ov % 5 === 0 && ov > 0) {
      const t = el("text", { x: xs(k), y: H - B + 14, fill: col.dim, "font-size": 13, "text-anchor": "middle" }, svg);
      t.textContent = ov;
    }
  });

  // area between line and 50%, coloured by the side it favours
  let d = `M${xs(0)},${ys(0.5)}`;
  p.forEach((v, k) => { d += ` L${xs(k)},${ys(v)}`; });
  d += ` L${xs(n - 1)},${ys(0.5)} Z`;
  el("path", { d, fill: col.c1, opacity: onBoard ? 0.22 : 0.16, "clip-path": `url(#top${uid})` }, svg);
  el("path", { d, fill: col.c2, opacity: onBoard ? 0.22 : 0.16, "clip-path": `url(#bot${uid})` }, svg);
  const line = p.map((v, k) => `${k ? "L" : "M"}${xs(k)},${ys(v)}`).join(" ");
  el("path", { d: line, fill: "none", stroke: col.line, "stroke-width": 2, "stroke-linejoin": "round" }, svg);

  for (const k of m.balls.wkt) {
    el("circle", { cx: xs(k + 1), cy: ys(p[k + 1]), r: 4, fill: col.line, stroke: col.bg, "stroke-width": 2 }, svg);
  }
  for (const k of m.swings) {
    el("circle", { cx: xs(k + 1), cy: ys(p[k + 1]), r: 9, fill: "none", stroke: col.line, "stroke-width": 1.5, opacity: 0.8 }, svg);
  }

  // hover / keyboard read-out
  const guide = el("line", { y1: T, y2: H - B, stroke: col.line, "stroke-width": 1, opacity: 0, "stroke-dasharray": "2 3" }, svg);
  const dot = el("circle", { r: 5, fill: col.line, stroke: col.bg, "stroke-width": 2, opacity: 0 }, svg);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
  const tip = document.getElementById("tip");
  let cur = -1;
  function show(k, cx, cy) {
    cur = k;
    const x = xs(k), y = ys(p[k]);
    guide.setAttribute("x1", x); guide.setAttribute("x2", x); guide.setAttribute("opacity", 0.7);
    dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("opacity", 1);
    let html;
    if (k === n - 1) {
      html = `<b>Result</b>${m.winner} won`;
    } else {
      const inn = m.balls.inn[k], bat = inn === 1 ? m.team1 : m.team2;
      html = `<b>${inn === 1 ? "1st" : "2nd"} innings, over ${m.balls.over[k]}</b>${bat} ${m.balls.score[k]} before this ball<br>` +
             `${m.team1} ${fmtPct(p[k])}, ${m.team2} ${fmtPct(1 - p[k])}<br>${ballText(m, k)}`;
    }
    tip.innerHTML = html; tip.hidden = false;
    const r = svg.getBoundingClientRect();
    const px = cx ?? r.left + (x / W) * r.width, py = cy ?? r.top + (y / H) * r.height;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(window.innerWidth - tw - 8, Math.max(8, px + 14)) + "px";
    tip.style.top = Math.max(8, py - tip.offsetHeight - 14) + "px";
  }
  function hide() { tip.hidden = true; guide.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); }
  function nearest(evt) {
    const r = svg.getBoundingClientRect();
    const sx = ((evt.clientX - r.left) / r.width) * W;
    let best = 0, bd = Infinity;
    for (let k = 0; k < n; k++) { const dd = Math.abs(xs(k) - sx); if (dd < bd) { bd = dd; best = k; } }
    return best;
  }
  hit.addEventListener("pointermove", e => show(nearest(e), e.clientX, e.clientY));
  hit.addEventListener("pointerdown", e => show(nearest(e), e.clientX, e.clientY));
  hit.addEventListener("pointerleave", hide);
  svg.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      show(Math.max(0, Math.min(n - 1, (cur < 0 ? 0 : cur) + (e.key === "ArrowRight" ? 1 : -1))));
    } else if (e.key === "Escape") hide();
  });
  svg.addEventListener("blur", hide);
}

/* ---------- Match detail ---------- */
function swingItems(m) {
  return m.swings.map(k => {
    const d = m.balls.d[k];
    const gainer = d > 0 ? m.team1 : m.team2;
    const before = d > 0 ? m.balls.p[k] : 1 - m.balls.p[k];
    const after = before + Math.abs(d);
    const li = document.createElement("li");
    li.innerHTML = `<span class="delta" style="color:var(${d > 0 ? "--red" : "--blue"})">+${Math.round(Math.abs(d) * 100)}</span>` +
      `<span class="what">${ballText(m, k)}. ${gainer} ${fmtPct(before)} to ${fmtPct(after)}.` +
      ` <span class="over">${m.balls.inn[k] === 1 ? "1st" : "2nd"} innings, over ${m.balls.over[k]}</span></span>`;
    return li;
  });
}

async function openMatch(id, push = true) {
  const box = document.getElementById("detail");
  box.innerHTML = `<p class="empty">Loading…</p>`;
  const m = await getMatch(id);
  box.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = `${m.team1} v ${m.team2}`;
  const sub = document.createElement("p");
  sub.className = "sub";
  sub.textContent = `${m.comp}, ${fmtDate(m.date)}, ${m.venue.split(",")[0]}. ${m.team1} ${m.score1}, ${m.team2} ${m.score2}. ${m.winner} won.`;
  const chart = document.createElement("div");
  chart.className = "chart";
  const sh = document.createElement("h4");
  sh.style.cssText = "font:600 1.25rem var(--display);margin:1.25rem 0 0";
  sh.textContent = "Biggest swings (percentage points)";
  const ul = document.createElement("ol");
  ul.className = "swings";
  swingItems(m).forEach(li => ul.appendChild(li));
  box.append(h, sub, chart, sh, ul);
  drawMatch(chart, m, "paper");
  document.querySelectorAll(".match-list button").forEach(b => b.setAttribute("aria-current", String(b.dataset.id === id)));
  if (push) history.replaceState(null, "", `#m=${id}`);
}

/* ---------- Match list ---------- */
function renderList() {
  const comp = document.getElementById("f-comp").value;
  const team = document.getElementById("f-team").value.trim().toLowerCase();
  const sort = document.getElementById("f-sort").value;
  let rows = INDEX.filter(r => (!comp || r.comp === comp) &&
    (!team || r.team1.toLowerCase().includes(team) || r.team2.toLowerCase().includes(team)));
  if (sort === "comeback") rows = rows.slice().sort((a, b) => a.low - b.low);
  if (sort === "drama") rows = rows.slice().sort((a, b) => b.drama - a.drama);
  const ol = document.getElementById("match-list");
  ol.innerHTML = "";
  for (const r of rows.slice(0, LIST_LIMIT)) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button"; b.dataset.id = r.id;
    const extra = sort === "comeback" ? `, ${r.winner} were down to ${fmtPct(r.low)}` : "";
    b.innerHTML = `<span class="teams">${r.team1} v ${r.team2}</span>` +
      `<span class="meta">${r.comp}, ${fmtDate(r.date)}${extra}</span>`;
    b.addEventListener("click", () => openMatch(r.id));
    li.appendChild(b); ol.appendChild(li);
  }
  if (rows.length > LIST_LIMIT) {
    const li = document.createElement("li"); li.className = "more";
    li.textContent = `Showing ${LIST_LIMIT} of ${rows.length.toLocaleString()} matches. Filter by team or competition to find others.`;
    ol.appendChild(li);
  } else if (!rows.length) {
    const li = document.createElement("li"); li.className = "more";
    li.textContent = "No matches found. Check the team name or choose another competition.";
    ol.appendChild(li);
  }
}

/* ---------- Accuracy section ---------- */
function drawCalibration(container, cal) {
  const W = 420, H = 420, L = 64, R = 12, T = 12, B = 44;
  const xs = v => L + v * (W - L - R), ys = v => T + (1 - v) * (H - T - B);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: xs(0), x2: xs(1), y1: ys(v), y2: ys(v), stroke: css("--rule") }, svg);
    const t = el("text", { x: L - 8, y: ys(v) + 5, "text-anchor": "end", fill: css("--ink-2"), "font-size": 14 }, svg); t.textContent = `${v * 100}%`;
    const u = el("text", { x: xs(v), y: H - B + 20, "text-anchor": "middle", fill: css("--ink-2"), "font-size": 14 }, svg); u.textContent = `${v * 100}%`;
  }
  el("line", { x1: xs(0), y1: ys(0), x2: xs(1), y2: ys(1), stroke: css("--ink-2"), "stroke-dasharray": "4 4" }, svg);
  const d = cal.map((c, i) => `${i ? "L" : "M"}${xs(c.pred)},${ys(c.actual)}`).join(" ");
  el("path", { d, fill: "none", stroke: css("--blue"), "stroke-width": 2 }, svg);
  for (const c of cal) el("circle", { cx: xs(c.pred), cy: ys(c.actual), r: 5, fill: css("--blue"), stroke: css("--paper"), "stroke-width": 2 }, svg);
  const xl = el("text", { x: (xs(0) + xs(1)) / 2, y: H - 6, "text-anchor": "middle", fill: css("--ink-2"), "font-size": 15 }, svg); xl.textContent = "Predicted win chance";
  const yl = el("text", { x: 14, y: (ys(0) + ys(1)) / 2, "text-anchor": "middle", fill: css("--ink-2"), "font-size": 15, transform: `rotate(-90 14 ${(ys(0) + ys(1)) / 2})` }, svg); yl.textContent = "How often they won";
}

function renderAccuracy(s) {
  const names = { "coin flip": "Coin flip (always 50%)", "lookup table": "Similar past situations",
                  "logistic, scoreboard only": "Scoreboard only", "FINAL model": "This model" };
  const tb = document.querySelector("#results-table tbody");
  for (const [k, v] of Object.entries(s.test)) {
    const tr = document.createElement("tr");
    if (k === "FINAL model") tr.className = "ours";
    tr.innerHTML = `<td>${names[k] ?? k}</td><td>${v.log_loss.toFixed(3)}</td><td>${(v.accuracy * 100).toFixed(1)}%</td>`;
    tb.appendChild(tr);
  }
  drawCalibration(document.getElementById("cal-chart"), s.calibration);
  const a = s.market.all, f = s.market["franchise leagues"], t = s.market["T20 internationals"];
  document.getElementById("market-text").textContent =
    `On ${a.matches.toLocaleString()} test matches with Betfair prices, the market's pre-match odds scored ${a.market.toFixed(3)} against the model's ${a.model.toFixed(3)}. ` +
    `The market knows the team sheets, injuries and the toss; the model only knows history. Averaging the two did worse than the market alone, ` +
    `and backing the model's apparent value picks would have lost about 25p for every £1 staked. ` +
    `In franchise leagues even the market barely beats a coin flip before a ball is bowled (${f.market.toFixed(3)} against 0.693); in internationals it's far sharper (${t.market.toFixed(3)}).`;
}

/* ---------- Boot ---------- */
async function main() {
  const [idx, summary] = await Promise.all([
    fetch(DATA + "index.json").then(r => r.json()),
    fetch(DATA + "summary.json").then(r => r.json()),
  ]);
  INDEX = idx;
  const comps = [...new Set(INDEX.map(r => r.comp))].sort();
  const sel = document.getElementById("f-comp");
  for (const c of comps) { const o = document.createElement("option"); o.value = o.textContent = c; sel.appendChild(o); }
  ["f-comp", "f-sort"].forEach(id => document.getElementById(id).addEventListener("change", renderList));
  document.getElementById("f-team").addEventListener("input", renderList);
  renderList();
  renderAccuracy(summary);

  const hero = INDEX.find(r => r.date === "2025-04-15" && r.team1 === "Punjab Kings") ?? INDEX.slice().sort((a, b) => a.low - b.low)[0];
  const hm = await getMatch(hero.id);
  drawMatch(document.getElementById("hero-chart"), hm, "board");
  document.getElementById("hero-foot").textContent =
    `${hm.comp}, ${fmtDate(hm.date)}, ${hm.venue.split(",")[0]}. ${hm.team1} ${hm.score1}, ${hm.team2} ${hm.score2}. ` +
    `Rings mark the three biggest swings; dots mark wickets.`;

  const m = location.hash.match(/m=(\d+)/);
  openMatch(m ? m[1] : hero.id, false);
}
main().catch(err => {
  document.getElementById("detail").innerHTML = `<p class="empty">The match data didn't load (${err.message}). Refresh the page to try again.</p>`;
});
