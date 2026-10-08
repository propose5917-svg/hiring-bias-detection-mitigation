"""Evidence figures for the pitch, from our own runs. Uses whatever conditions exist.
Discovery figures use all 672 pairs; mitigation figures use the held-out TEST split only
(steering directions were estimated on TRAIN)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import report, load, DF, RUNS

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "figure.facecolor": "white", "axes.grid": True, "grid.color": "#e8e7e3",
                     "grid.linewidth": 0.6})
M15, M05 = "Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct"
have = lambda m, t: os.path.exists(f"{RUNS}/{m}_{t}.npz")
LABEL = {"base": "Original model", "threshold_eq": "Output threshold fix", "prompt_debias": "\"Ignore race\" prompt",
         "steer_lstar": "Erase at 1 layer (ℓ*)", "steer_all": "Erase at all layers", "proxy": "Name hidden, proxy kept"}

# ---- 1. behavioral evidence: twin-resume scatter
b = load(M15, "base"); full = report(M15, ["base"])
thr = full["threshold"]; sw, sb = b["s_white"], b["s_black"]
flip = (sw > thr) != (sb > thr)
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.scatter(sw[~flip], sb[~flip], s=14, c=GRAY, alpha=.55, lw=0, label="Same decision")
ax.scatter(sw[flip], sb[flip], s=34, c=ORANGE, edgecolor="white", lw=1, label=f"Decision flipped ({flip.sum()}/{len(sw)})")
lo, hi = min(sw.min(), sb.min()), max(sw.max(), sb.max())
ax.plot([lo, hi], [lo, hi], c=MUTED, lw=1, ls="--")
ax.axvline(thr, c=MUTED, lw=.8, ls=":"); ax.axhline(thr, c=MUTED, lw=.8, ls=":")
ax.set_xlabel("Score with White-associated name  (logit Yes − No)")
ax.set_ylabel("Score with Black-associated name")
c = full["conds"]["base"]
ax.set_title(f"Same resume, only the name changes\nmean gap {c['score_gap']:+.3f}, p={max(c['p_score_gap'], 2e-4):.4f}, Δ={100*c['delta']:.1f}pp", loc="left")
ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}/1_twin_resume_scatter.png", dpi=200)

# ---- 2. activation evidence by layer (finding 3), both models if available (finding 2)
fig, ax = plt.subplots(figsize=(7, 4.2))
summary = {"1.5B_all": full}
for m, col, name in ((M15, BLUE, "Qwen 1.5B"), (M05, AQUA, "Qwen 0.5B")):
    if not have(m, "base"): continue
    r = full if m == M15 else report(m, ["base"]); summary[m] = r
    P, B95 = np.array(r["base_presence_by_layer"]), np.array(r["null_B95_by_layer"])
    x = np.linspace(0, 1, len(P))
    ax.plot(x, P, c=col, lw=2, label=f"{name}: bias ‖v_ℓ‖/E‖h_ℓ‖")
    ax.fill_between(x, 0, B95, color=col, alpha=.12, lw=0)
    ax.annotate(f"{P[-1]/max(B95[-1],1e-9):.0f}× noise", (1, P[-1]), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", color=INK, fontsize=9)
ax.set_xlabel("Relative depth (0 = embeddings, 1 = final layer)"); ax.set_ylabel("Activation-space bias")
ax.set_title("Bias is encoded internally and grows in late layers\nshaded = 95th pct. of shuffled-label noise", loc="left")
ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}/2_activation_bias_by_layer.png", dpi=200)

# ---- 3 & 4. mitigation comparison on the held-out test split
tags = ["base", "threshold_eq"] + [t for t in ("prompt_debias", "steer_lstar", "steer_all", "proxy") if have(M15, t)]
test = DF.split.values == "test"
rt = report(M15, tags, mask=test); summary["1.5B_test"] = rt
rows = []
for t in tags:
    cnd = rt["conds"][t]
    rows.append(dict(method=LABEL[t], delta_pp=100 * cnd["delta"], flip_rate_pct=100 * cnd["flip_rate"],
                     score_gap=cnd["score_gap"], p=cnd["p_score_gap"], retention_lstar=cnd["retention_lstar"],
                     retention_final=cnd["retention_last"], cos_lstar=cnd["cos_lstar"],
                     probe_final_pct=100 * cnd["probe_last"], bal_acc_pct=100 * cnd["bacc"]))
ro = f"{RUNS}/{M15}_readout_test.json"
if os.path.exists(ro):
    for k, nm in (("steer_all+readout", "All layers + re-estimated readout erase"), ("base+readout", "Readout erase only (cross-fit)")):
        c2 = json.load(open(ro))[k]
        rows.append(dict(method=nm, delta_pp=100 * c2["delta"], flip_rate_pct=100 * c2["flip_rate"], score_gap=c2["score_gap"],
                         p=c2["p_score_gap"], retention_lstar=np.nan, retention_final=c2["retention_final"], cos_lstar=np.nan,
                         probe_final_pct=100 * c2["probe_final"], bal_acc_pct=100 * c2["bacc"]))
tab = pd.DataFrame(rows); tab.to_csv(f"{FIG}/results_table_test.csv", index=False)
print(tab.round(3).to_string())

fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
y = np.arange(len(tab))[::-1]
for ax, col, key, xl in ((axs[0], ORANGE, "flip_rate_pct", "Decisions flipped by name swap (%)"),
                         (axs[1], BLUE, "retention_final", "Activation bias retained (final layer)\n1 = untouched, 0 = removed"),
                         (axs[2], AQUA, "probe_final_pct", "Race decodable from final layer (%)\n50 = chance")):
    ax.barh(y, tab[key], color=col, height=.6)
    for yi, v in zip(y, tab[key]): ax.text(v, yi, f" {v:.2f}" if key == "retention_final" else f" {v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
axs[0].set_yticks(y, tab.method); axs[2].set_xlim(40, 102)
fig.suptitle("Held-out test resumes (336 twin pairs): which fix removes which bias?", x=.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/3_methods_comparison.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4.2))
for t, col, ls in (("threshold_eq", GRAY, "--"), ("prompt_debias", ORANGE, "-"), ("steer_lstar", BLUE, "-"), ("steer_all", AQUA, "-")):
    if t not in rt["conds"]: continue
    R = np.array(rt["conds"][t]["retention_by_layer"])[1:]
    ax.plot(np.arange(1, len(R) + 1), R, c=col, lw=2, ls=ls, label=LABEL[t])
ax.axhline(0, c=MUTED, lw=.8); ax.axvline(rt["lstar"], c=MUTED, lw=.8, ls=":")
ax.set_xlabel("Layer"); ax.set_ylabel("Retention fraction")
ax.set_title("Where does bias survive? (held-out test split)\n1 = same as original model, 0 = at noise level", loc="left")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/4_retention_by_layer.png", dpi=200)
json.dump(summary, open(f"{FIG}/summary.json", "w"), indent=1)
