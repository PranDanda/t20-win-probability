"""Link test-set matches (2025-26) to Betfair MATCH_ODDS markets.

Rules (fixed before any model-vs-market comparison):
  - candidate markets: 2 runners, start time within 1 day of the Cricsheet match date,
    no women's markets (names ending ' W' or containing 'Women')
  - team names normalised (lower case, punctuation and 'cc'/'cricket club' removed, known
    renames applied); similarity = difflib ratio, both teams must score >= 0.72 under the
    better of the two pairings; the best-scoring candidate wins
  - if more than one market passes for the same match (series with games on consecutive days,
    double-headers), the match is excluded as ambiguous. Rule added after the first linking run
    showed 2.6% winner disagreements, BEFORE any model-vs-market numbers were computed.
  - check: Betfair's settled winner must equal Cricsheet's winner
Output: data/processed/bf_links.parquet
"""
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RENAMES = {"bengaluru": "bangalore", "u.a.e.": "united arab emirates", "uae": "united arab emirates",
           "usa": "united states of america", "png": "papua new guinea"}
THRESH = 0.72


def norm(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(cc|cricket club|t20|xi)\b", " ", s)
    for a, b in RENAMES.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s)
    return " ".join(s.split())


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def main():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    f = pd.read_parquet(PROC / "features_v4.parquet", columns=["match_id", "split"])
    test_ids = set(f.loc[f["split"] == "test", "match_id"].astype(str))
    t = m[m["match_id"].isin(test_ids)].copy()

    mk = pd.read_parquet(PROC / "bf_markets.parquet")
    women = mk["team_a"].str.contains(r"(?: W$|Women)", na=False) | mk["team_b"].str.contains(r"(?: W$|Women)", na=False)
    mk = mk[(mk["n_runners"] == 2) & ~women].copy()
    mk["day"] = mk["market_time"].dt.tz_convert(None).dt.normalize()

    links = []
    for r in t.itertuples():
        cand = mk[(mk["day"] >= r.date - pd.Timedelta(days=1)) & (mk["day"] <= r.date + pd.Timedelta(days=1))]
        best, n_pass = None, 0
        for c in cand.itertuples():
            s1 = min(sim(r.team1, c.team_a), sim(r.team2, c.team_b))
            s2 = min(sim(r.team1, c.team_b), sim(r.team2, c.team_a))
            score, swapped = (s1, False) if s1 >= s2 else (s2, True)
            if score >= THRESH:
                n_pass += 1
                if best is None or score > best[0]:
                    best = (score, swapped, c)
        if best is None:
            links.append({"match_id": r.match_id, "matched": False, "reason": "no market"})
            continue
        if n_pass > 1:   # same teams, more than one market within a day: can't tell which
            links.append({"match_id": r.match_id, "matched": False, "reason": "ambiguous"})
            continue
        score, swapped, c = best
        id_team1, id_team2 = (c.id_b, c.id_a) if swapped else (c.id_a, c.id_b)
        bf_winner = None
        if pd.notna(c.winner_id):
            bf_winner = r.team1 if c.winner_id == id_team1 else r.team2 if c.winner_id == id_team2 else None
        links.append({"match_id": r.match_id, "matched": True, "market_id": c.market_id,
                      "score": score, "bf_event": c.event_name, "competition": r.competition,
                      "team1": r.team1, "team2": r.team2, "runner_team1": id_team1,
                      "runner_team2": id_team2, "inplay_pt": c.inplay_pt,
                      "cs_winner": r.winner, "bf_winner": bf_winner})
    L = pd.DataFrame(links)
    L.to_parquet(PROC / "bf_links.parquet", index=False)
    ok = L[L["matched"]]
    print("Unlinked reasons:", L.loc[~L["matched"], "reason"].value_counts().to_dict())
    print(f"Test matches: {len(L):,}; linked to a Betfair market: {len(ok):,} ({len(ok)/len(L):.1%})")
    print("By competition (linked / total):")
    tot = t.groupby("competition").size()
    lk = ok.groupby("competition").size()
    print(pd.DataFrame({"linked": lk, "total": tot}).fillna(0).astype(int).assign(
        share=lambda x: (x.linked / x.total).round(2)).to_string())
    both = ok[ok["bf_winner"].notna()]
    agree = (both["bf_winner"] == both["cs_winner"]).mean()
    print(f"\nWinner check: Betfair and Cricsheet agree on {agree:.1%} of {len(both):,} linked, settled matches")
    print("Lowest-similarity links accepted:")
    print(ok.nsmallest(6, "score")[["score", "team1", "team2", "bf_event"]].round(2).to_string(index=False))
    dis = both[both["bf_winner"] != both["cs_winner"]]
    if len(dis):
        print("\nDisagreements:")
        print(dis[["team1", "team2", "bf_event", "cs_winner", "bf_winner"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
