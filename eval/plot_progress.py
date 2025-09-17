# eval/plot_progress.py
import os, csv, collections, matplotlib.pyplot as plt

def load_rows(csv_path):
    rows=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        rd=csv.reader(f); header=next(rd)
        for r in rd:
            phase=r[0].split("_seed")[0]
            wr=float(r[5]); lo=float(r[6]); hi=float(r[7])
            rows.append((phase,wr,lo,hi))
    return rows

def aggregate(rows):
    agg=collections.defaultdict(list)
    for phase,wr,lo,hi in rows:
        agg[phase].append((wr,lo,hi))
    out=[]
    for p,lst in agg.items():
        wr=sum(x[0] for x in lst)/len(lst); lo=sum(x[1] for x in lst)/len(lst); hi=sum(x[2] for x in lst)/len(lst)
        out.append((p,wr,lo,hi))
    # sort theo số trong tên bc_xxxxxx, và đẩy dagger_final (nếu có) về cuối
    def key(p):
        if p[0].startswith("bc_"):
            return (0, int(p[0].split("_")[1]))
        return (1, 10**9 if p[0]=="dagger_final" else 10**8)
    out.sort(key=key)
    return out

def plot_bars(csv_path:str, out_png:str):
    rows=aggregate(load_rows(csv_path))
    names=[p for p,_,_,_ in rows]
    wr =[w*100 for _,w,_,_ in rows]
    lo =[l*100 for _,_,l,_ in rows]
    hi =[h*100 for _,_,_,h in rows]

    x=range(len(names))
    plt.figure(figsize=(12,5))
    plt.bar(x, wr, alpha=0.85)
    # error bars (yerr symmetrical by band width to mid)
    yerr=[ [(wr[i]-lo[i]) for i in range(len(wr))], [(hi[i]-wr[i]) for i in range(len(wr))] ]
    plt.errorbar(x, wr, yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=2, alpha=0.8)

    plt.title("Tiến bộ win-rate của agent vs Heuristic theo giai đoạn (BC → DAgger)")
    plt.ylabel("Win-rate (%)")
    plt.xlabel("Giai đoạn (0..1,000,000; bước 20,000) + dagger_final")
    plt.xticks([i for i in x if (i%5==0 or i==len(x)-1)], [names[i] for i in x if (i%5==0 or i==len(x)-1)], rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160)
    print(f"[OK] saved {out_png}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/progress_vs_heuristic.csv")
    ap.add_argument("--out", default="results/progress_vs_heuristic.png")
    args=ap.parse_args()
    plot_bars(args.csv,args.out)
