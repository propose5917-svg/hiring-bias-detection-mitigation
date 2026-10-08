"""Round-2 erasure at the readout, cross-fitted on the held-out split.
logit(Yes)-logit(No) = w·h_final (linear), so projecting h_final is exact post hoc.
Direction re-estimated ON THE ALREADY-STEERED MODEL from half the test resumes, applied to the other half."""
import sys, json, numpy as np, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from analyze import load, presence, null_presence, behavioral, probe_acc, DF, RUNS
mid = sys.argv[1]; m = mid.split("/")[-1]
path = snapshot_download(mid); tok = AutoTokenizer.from_pretrained(mid)
yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
import glob
W = None
for f in glob.glob(f"{path}/*.safetensors"):
    with safe_open(f, "pt") as s:
        k = "lm_head.weight" if "lm_head.weight" in s.keys() else ("model.embed_tokens.weight" if "model.embed_tokens.weight" in s.keys() else None)
        if k: E = s.get_tensor(k); W = (E[yes] - E[no]).float().numpy()
test = DF.split.values == "test"; ids = DF.resume_id.values
base = load(m, "base"); thr = np.median(np.r_[base["s_white"][test], base["s_black"][test]])
out = {}
for tag in ("base", "steer_all"):
    r = load(m, tag); hw, hb = r["h_white"][:, -1].copy(), r["h_black"][:, -1].copy()
    sw0 = hw @ W; print(tag, "readout reproduces scores, corr =", np.corrcoef(sw0[test], r["s_white"][test])[0, 1].round(4))
    uid = np.unique(ids[test]); A = np.isin(ids, uid[::2]) & test; B = np.isin(ids, uid[1::2]) & test
    for est, app in ((A, B), (B, A)):
        u = hw[est].mean(0) - hb[est].mean(0); u /= np.linalg.norm(u)
        hw[app] -= np.outer(hw[app] @ u, u); hb[app] -= np.outer(hb[app] @ u, u)
    sw, sb = hw @ W, hb @ W
    shift = np.median(np.r_[r["s_white"][test], r["s_black"][test]]) - np.median(np.r_[sw[test], sb[test]])
    beh = behavioral(sw[test] + shift, sb[test] + shift, np.median(np.r_[r["s_white"][test], r["s_black"][test]]), DF.qualified.values[test])
    P, _ = presence(hw[test][:, None], hb[test][:, None]); P0, _ = null_presence(hw[test][:, None], hb[test][:, None], 20)
    Pb, _ = presence(base["h_white"][test][:, -1:], base["h_black"][test][:, -1:]); Pn, _ = null_presence(base["h_white"][test][:, -1:], base["h_black"][test][:, -1:], 20)
    pr = probe_acc(hw[test][:, None], hb[test][:, None], 0, ids[test])
    out[tag + "+readout"] = dict(**{k: float(v) for k, v in beh.items()}, presence_final=float(P[0]), null_final=float(P0[0]),
                                 retention_final=float((P[0] - Pn[0]) / (Pb[0] - Pn[0])), probe_final=float(pr))
    print(tag + "+readout", {k: round(v, 4) for k, v in out[tag + "+readout"].items()})
json.dump(out, open(f"{RUNS}/{m}_readout_test.json", "w"), indent=1)
