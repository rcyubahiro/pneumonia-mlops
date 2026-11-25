"""Turn results/containers_N_stats.csv into a Markdown table for the README."""
import csv
from pathlib import Path

rows = []
for n in (1, 2, 3):
    f = Path("results") / f"containers_{n}_stats.csv"
    if not f.exists():
        continue
    for r in csv.DictReader(f.open()):
        if r["Name"] == "/predict":
            rows.append((n, r["Request Count"], r["Failure Count"], r["Average Response Time"],
                         r["Median Response Time"], r["95%"], r["99%"], r["Requests/s"]))

header = "| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) | Req/s |"
print(header)
print("|" + "---|" * 8)
for n, req, fail, avg, med, p95, p99, rps in rows:
    print(f"| {n} | {req} | {fail} | {float(avg):.0f} | {med} | {p95} | {p99} | {float(rps):.1f} |")
