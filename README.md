# Hiring Bias Audit for LLM Resume Screeners

**Does an AI resume screener treat candidates differently because of their name, and can we remove that bias from both its decisions and its internal representations?**

Built for the 180DC Bristol × BDSS Datathon *"Exploring Bias in Hiring"* (Oct 2026).

## The problem

A hiring model can be biased in two ways:

- **Behavioral bias:** its *decisions* differ between demographic groups.
- **Activation-space bias:** group information is still *encoded inside* the model's hidden layers, even if the decisions look fair.

The datathon's research brief found that removing the first kind does not remove the second. The challenge was to build a way for a hiring tool to check itself for both.

## What I did

1. **Built "twin" resumes from real data**
   - Took 336 resumes (24 job fields) from a public resume dataset.
   - Paired each with a real LinkedIn job posting in the same field (qualified) and one from an unrelated field (unqualified), giving 672 resume–job pairs.
   - Rendered every pair twice: once with a White-associated name and once with a Black-associated name (name lists from Bertrand & Mullainathan, 2004). Gender is matched and every other character is identical.
   - Because the twins differ only by name, any difference in the model's output is caused by the name. This means no real hiring outcomes are needed.

2. **Used an open LLM as the resume screener**
   - Asked Qwen2.5-Instruct (0.5B and 1.5B) "Should this candidate be advanced to an interview? Yes/No".
   - Recorded the Yes-vs-No score and the hidden state at **every layer** of the model.

3. **Measured both kinds of bias**
   - Behavioral: the gap in acceptance rates between name groups (Δ), how often swapping the name flips the decision, and a paired permutation test.
   - Activation-space: the size of the average White-vs-Black difference in each layer's hidden state, compared with a noise floor from shuffled labels. Also a linear classifier that tries to predict race from the hidden state.
   - Utility: balanced accuracy on qualified vs unqualified pairs, so a fix that makes the model useless is caught.

4. **Tried to remove the bias and re-tested it on unseen resumes**
   - Output-only fixes: equalising decision thresholds; telling the model to "ignore race" in the prompt.
   - Internal fix: estimate the model's own "race direction" in its hidden state, then subtract it during inference: `h ← h − (h·u)u`. This was done at one layer, at every layer, and at the final output layer.
   - Directions were learned on one half of the resumes and evaluated on the other half.

## Key findings

- **Names change decisions.** With only the name swapped, 26 of 672 decisions (3.9%) flipped (p < 0.001).
- **Race is readable inside the model.** A simple linear classifier predicted the name group from the final layer with 99.9% accuracy. The signal was strongest in the last layers.
- **Looking fair is not being fair.** The smaller model had almost no decision gap (0.15 pp), but race was still 98% decodable from its hidden states.
- **Fixing outputs leaves the inside untouched.** Threshold equalisation brought the decision gap to 0, but the internal bias stayed at 100% of its original size. The "ignore race" prompt made the internal bias larger.
- **Bias moves when you remove it.** Removing the race direction at one layer cut the internal bias to 14% at that layer, but it grew back to 46% by the final layer. Removing it at every layer erased the original direction, but bias re-formed in a new one.
- **No method fully passed.** The best result (removal at the most biased layer) made the decision gap non-significant with no accuracy loss, but did not remove the internal bias.

### Results on unseen resumes (Qwen 1.5B, 336 twin pairs)

| Method | Decision gap Δ | Gap significant? | Internal bias left | Race predictable from model | Balanced accuracy |
|---|---|---|---|---|---|
| Original model | 0.9 pp | yes (p < 0.001) | 100% | 99.9% | 67.1% |
| Equalise thresholds | 0.0 pp | no | 100% | 99.9% | 67.0% |
| "Ignore race" prompt | 1.2 pp | yes | 117% | 99.9% | 69.3% |
| Remove at most biased layer | 0.0 pp | no (p = 0.10) | 46% | 95.8% | 67.0% |
| Remove at every layer | 3.0 pp | yes (reversed direction) | 107% | 100% | 67.3% |
| Remove at final output layer | 0.6 pp | yes (p = 0.02) | 48% | 63.2% | 65.2% |

"Internal bias left" is the retention fraction: 100% means unchanged and 0% means at noise level. For "race predictable", 50% would be chance.

**Takeaway:** a hiring model should never be cleared on its outputs alone. Any fix must be re-tested on both outputs and internals every time, and blocked if it fails.

## Code

| File | What it does |
|---|---|
| `build_dataset.py` | Matches resumes to job postings and builds the 672 name-swapped twin pairs |
| `run_model.py` | Runs the LLM as a screener, records hidden states at every layer, and applies the bias-removal hooks |
| `make_dirs.py` | Learns the model's bias direction at each layer from the training half |
| `analyze.py` | Computes all bias and accuracy metrics |
| `readout_erasure.py` | Re-estimates and removes bias at the final output layer |
| `make_figures.py` | Draws the result charts |
| `run_all.sh` | Runs the full experiment for one model |

## How to run

Tested on a MacBook (Apple M3, 16 GB). CUDA and CPU also work. One condition with the 1.5B model takes about 16 minutes.

Data (not included in this repo):
- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (`Resume.csv`)
- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) (`postings.csv`)

```bash
pip install torch transformers accelerate safetensors huggingface_hub numpy pandas scikit-learn matplotlib
RESUME_CSV=path/to/Resume.csv POSTINGS_CSV=path/to/postings.csv python build_dataset.py
./run_all.sh Qwen/Qwen2.5-1.5B-Instruct
python analyze.py Qwen2.5-1.5B-Instruct base,threshold_eq,prompt_debias,steer_lstar,steer_all test
python readout_erasure.py Qwen/Qwen2.5-1.5B-Instruct
python make_figures.py
```

## Limitations

- This is a simulation with small open LLMs, not a real hiring system or real hiring outcomes.
- Race is signalled only through first names and surnames.
- Other proxies, such as university, clubs or postcode, were not tested.
- The removal method is linear, so non-linear encodings of race can survive it.
- The internal checks need access to the model's weights, so they do not work on closed API models.

## Tools

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · matplotlib
