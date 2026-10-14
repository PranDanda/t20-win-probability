"""Parse Betfair historic MATCH_ODDS files (BASIC tier) into two tables.

data/processed/bf_markets.parquet : one row per market (teams, start, in-play start, winner)
data/processed/bf_prices.parquet  : every last-traded-price update (market, time, runner, ltp)
"""
import bz2
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "betfair" / "extracted"
PROC = ROOT / "data" / "processed"


def parse_file(path):
    market, prices = {}, []
    inplay_pt = None
    with bz2.open(path, "rt") as fh:
        for line in fh:
            msg = json.loads(line)
            pt = msg["pt"]
            for mc in msg.get("mc", []):
                md = mc.get("marketDefinition")
                if md:
                    market.update({
                        "market_id": mc["id"], "event_id": md.get("eventId"),
                        "event_name": md.get("eventName"), "market_type": md.get("marketType"),
                        "market_time": md.get("marketTime"), "country": md.get("countryCode"),
                        "runners": [(r["id"], r["name"]) for r in md.get("runners", [])],
                        "status": md.get("status"),
                        "winner_id": next((r["id"] for r in md.get("runners", [])
                                           if r.get("status") == "WINNER"), market.get("winner_id")),
                    })
                    if md.get("inPlay") and inplay_pt is None:
                        inplay_pt = pt
                    market["last_def_pt"] = pt
                for rc in mc.get("rc", []):
                    if "ltp" in rc:
                        prices.append((mc["id"], pt, rc["id"], rc["ltp"]))
    market["inplay_pt"] = inplay_pt
    return market, prices


def main():
    files = sorted(SRC.rglob("*.bz2"))
    print(f"{len(files):,} market files", flush=True)
    markets, prices = [], []
    for i, f in enumerate(files):
        m, p = parse_file(f)
        markets.append(m); prices += p
        if i % 2000 == 0:
            print(f"  {i:,}", flush=True)
    mk = pd.DataFrame(markets)
    mk = mk[mk["market_type"] == "MATCH_ODDS"]
    mk["n_runners"] = mk["runners"].map(len)
    mk["team_a"] = mk["runners"].map(lambda r: r[0][1] if len(r) > 0 else None)
    mk["team_b"] = mk["runners"].map(lambda r: r[1][1] if len(r) > 1 else None)
    mk["id_a"] = mk["runners"].map(lambda r: r[0][0] if len(r) > 0 else None)
    mk["id_b"] = mk["runners"].map(lambda r: r[1][0] if len(r) > 1 else None)
    mk["market_time"] = pd.to_datetime(mk["market_time"], utc=True)
    mk = mk.drop(columns="runners")
    pr = pd.DataFrame(prices, columns=["market_id", "pt", "runner_id", "ltp"])
    mk.to_parquet(PROC / "bf_markets.parquet", index=False)
    pr.to_parquet(PROC / "bf_prices.parquet", index=False)
    print(f"\nMarkets: {len(mk):,}  (2-runner: {(mk.n_runners == 2).sum():,}, 3-runner incl. draw: "
          f"{(mk.n_runners == 3).sum():,})")
    print(f"Price updates: {len(pr):,}")
    print(f"With an in-play start: {mk['inplay_pt'].notna().mean():.1%}; "
          f"with a settled winner: {mk['winner_id'].notna().mean():.1%}")
    print(mk[["event_name", "market_time", "team_a", "team_b"]].sample(8, random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()
