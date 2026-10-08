"""Estimate bias directions on the TRAIN split only (test split stays held-out).
hidden_states[i] (i>=1) = output of decoder layer i-1; hidden_states[L] is after the final norm.
writes <model>_steer_lstar_dirs.npz (one layer) and <model>_steer_all_dirs.npz (every layer)."""
import sys, numpy as np, pandas as pd
from analyze import load, presence, null_presence, DF, RUNS

model = sys.argv[1]
b = load(model, "base")
tr = DF.split.values == "train"
P, v = presence(b["h_white"][tr], b["h_black"][tr])
P0, _ = null_presence(b["h_white"][tr], b["h_black"][tr])
L = P.shape[0] - 1
lstar = int(np.argmax((P - P0)[1:L])) + 1          # restrict to pre-norm layers for the single-layer hook
unit = lambda x: x / np.linalg.norm(x)
np.savez(f"{RUNS}/{model}_steer_lstar_dirs.npz", **{str(lstar - 1): unit(v[lstar])})
np.savez(f"{RUNS}/{model}_steer_all_dirs.npz",
         **{str(i - 1): unit(v[i]) for i in range(1, L)}, norm=unit(v[L]))
print("train l* =", lstar, "of", L, "presence", P[lstar].round(4), "null", P0[lstar].round(4))
