# Training status and development handoff — 12 September 2026

Evidence reviewed 12 September; documentation finalized 13 September 2026.

## Decision in brief

Keep the current geometry-neutral, backbone-readout design as the baseline.
The old CB collapse and the approximately 0.599 real-sidechain plateau are no
longer present in this run. Do **not** declare all virtual-slot or representation
problems solved: rare protein chemistry and late-slot real-versus-V decisions
remain weak. Two independently reproduced encoder feature defects and a verified
epoch-truncation issue also remain. Address these with controlled comparisons
before inventing another V representation.

This review changes documentation only. It does not modify model code, resume
training, regenerate PT files, or authorize a new long run.

## 1. Evidence and exact run identity

| Item | Evidence |
|---|---|
| Log | `logs/September2026/v_full_100epoch_20260901_214648.log` |
| Output directory | `outputs/v_full_100epoch_20260901_214648/` |
| W&B run | `xj4i2lzd`, `multidomain-full-geometry-neutral-v1` |
| Start | 1 September 2026, approximately 21:47, Europe/Istanbul |
| Finish | 12 September 2026, approximately 04:03, Europe/Istanbul |
| Completion | Explicit `Trainer.fit stopped: max_epochs=100 reached` and success message |
| Logged source SHA | `e6b1037589eb659cfcb255784bec42c8ecad9dfb` in W&B metadata |
| Review source HEAD | `952a3f6`; subsequent commits concern W&B/docs and adapter tests |
| Seed / domains | 42; protein, RNA, UniMol; metabolite disabled |
| Final tables | Last validation at zero-based epoch 99; **not held-out test results** |

The log size was 228,656,752 bytes at review. Its final state was checked against
the host process table; no matching training worker remained. The source SHA
alone does not prove the launch worktree was clean: preserve the resolved W&B
config, checkpoint hyperparameters, and local files as well as Git history.

Retain **both** `last.ckpt` and
`best-checkpoint-epoch=99-val_total_loss=0.7948.ckpt`. The latter was saved at
an earlier validation within epoch 99 (global step 1,903,244). Its score is not
the final validation loss of 0.795997. Both files are 36,201,730 bytes.

The output's W&B directory contains `config.yaml`, `wandb-summary.json`,
`wandb-metadata.json`, `requirements.txt`, `output.log`, and the raw `.wandb`
history. They are part of the experiment, not disposable console noise.

### Data and loss context

| Split | Protein graphs | RNA graphs | UniMol graphs |
|---|---:|---:|---:|
| Train selection | 32,034 | 6,833 | 686,908 |
| Validation | 3,983 | 818 | 85,920 |
| Test allocation | 3,983 | 905 | 85,835 |

The initial train plan has 19,081 batches and validation has 2,362. A test allocation
exists, but `core/train_pretrain.py` calls `trainer.fit`, not `trainer.test`.
At epoch 99 the sampler prepares **19,122** batches, but Lightning consumes only
**19,081**. The log reports 31,968 protein, 6,819 RNA and 685,503 UniMol samples
actually tracked: **1,485 selected samples in 41 batches are not consumed**
(approximately 0.205% of that epoch's selection). A full-pass sampler plan is
therefore not proof of full-pass training. See the trainer-loop finding below.

Final validation losses: total **0.795997**, long-range **0.406115**, short-range
**0.230160**, MLM **0.159722**. Per-domain MLM values are protein **0.226229**,
RNA **0.133893**, and UniMol **0.003915**. These tasks have different masking
and target difficulty; the small UniMol loss does not establish useful transfer.
Final train total loss is approximately **0.780**. No divergence is evident
from these end-of-run values; they do not by themselves prove generalization.

The current objective is:

```text
total = long-range loss + short-range loss + MLM
MLM   = mean CE(real masked targets) + 0.2 × mean CE(V masked targets)
```

Real targets include backbone, real sidechain/base, and other domains' masked
atoms. There is no active element-balanced term, occupancy head, or masked-real-
slot distance loss. The final epoch's first-batch representation-gradient audit
reports relative norm magnitudes of approximately 33.3% long-range, 19.8%
short-range, and 46.9% MLM. These are **one-batch norm comparisons**, not a
measured allocation of all training gradients. The old “98% distance / 2% MLM”
claim is not supported for this run.

### Relevant development sequence

| Source checkpoint | Change or decision still relevant |
|---|---|
| `70660f4`, 25 August | Distance targets constrained to valid graph-local pairs |
| `47d8671`, 1 September | Dynamic Tier 3 restored to the production `pos_code` contract |
| `4f8855f`, 1 September | Coordination restricted to explicit bonds or a covalent-distance proxy; batch-wide precedence remains defective |
| `2440558`, 1 September | Tier-3 distance attributes synchronized; local RBF aggregation separated from distant edges |
| `a0bab33`, 1 September | Graph-local M attention and Flash-Varlen value projection fixed |
| `7c2323e` / `daf1a4e`, 1 September | Tier-1 preservation, strict contracts, compact UniMol bond forwarding |
| `e6b1037`, before launch | Experimental slot-distance and element-balanced terms removed; task weights set to 1.0 |
| Full run, 1–12 September | Current invariant ESA and latent fixed-slot readout trained for 100 epochs |
| `99ff035` / `952a3f6` | W&B/docs settings finalized; standalone adapter validator migrated to pytest |

The current documentation review adds reproducible evidence and the migration
handoff. The three defects reported here have **not** been repaired by this task.
Archived notes may describe different stages; use the run's resolved config to
determine which stage produced a particular result.

## 2. What the model currently sees

```text
visible real atoms + masked real backbone atoms
                   |
             compact physical graph
                   |
       ESA: graph-local M / molecule-global S
                   |
       encoded backbone of the owning block
                   |
  mean(backbone states) + domain-local slot embedding
                   |
                LayerNorm
                   |
      shared atom classifier: C / N / O / S / ... / V
```

Visible V and **all** masked sidechain/base slots are outside the encoder graph.
They do not contribute coordinates, physical edges, RBF features, or attention
nodes. Ten protein or eleven RNA output slots are reconstructed after ESA from
their own backbone. Unmasked real sidechain/base atoms remain physical context.
The cache still has fixed 14/23 slots; its placeholder positions are not input
geometry. No PT rebuild is needed for this runtime design.

The backbone can receive information from neighboring atoms through M and from
the molecule through S. This is a valid route for contextual learning, not a
guarantee that every prediction uses nearby chemistry. Equivariant features
remain disabled; no coordinate-generation capability is claimed.

Sources: [`virtual_slot_graph.py`](../../core/virtual_slot_graph.py),
[`pretraining_model.py`](../../core/pretraining_model.py),
[`molecular_encoder.py`](../../core/molecular_encoder.py),
[`pretraining_tasks.py`](../../core/pretraining_tasks.py), and
[`masked_layers.py`](../../esa/masked_layers.py).

## 3. Final learning results

Use recall/fraction-correct and support in the atom-name tables. Their near-1.0
precision and one-versus-rest accuracy are affected by element-to-position
mapping and are not independent evidence of chemical recognition. “Exact” below
means every evaluated target element in a block was correct; it is not an
amino-acid classifier or a coordinate-quality metric.

| Measure | Earlier epoch-45 review | Final epoch 99 |
|---|---:|---:|
| Protein real sidechain, SCr | 83.02% | **85.77%** |
| Protein real sidechain N/O/S | 55.77% | **62.77%** |
| Protein V recall | 89.06% | **91.05%** |
| Protein real-only exact block | 65.61% | **71.22%** |
| Protein exact block including V | 44.24% | **53.11%** |
| Protein CB recall | 99.49% | **99.72%** |
| RNA real base, BaseR | 90.08% | **93.04%** |
| RNA real base N/O/S | 88.22% | **91.55%** |
| RNA V recall | 94.69% | **94.28%** |
| RNA real-only exact block | 74.29% | **82.28%** |
| RNA exact block including V | 74.09% | **81.87%** |

These are different validation snapshots, not paired retraining experiments.
Protein has 82,316 evaluated masked residues; RNA has 10,477 nucleotides.
The real-base diagnostic rounds to 0.9305 while the block table gives 0.9304:
their handling of a small modified/unknown-position subset differs. The block
table denominator is 97,343 real-base targets; the named sidechain group has
97,315, with another 28 assigned to Unknown. Do not silently mix denominators.

### Is the model still only using the slot prior?

No: the current result exceeds the logged per-slot majority baseline substantially.

| Domain | Real-slot model / prior | All-slot model / prior |
|---|---|---|
| Protein | 85.77% / 59.95% | 89.04% / 79.84% |
| RNA | about 93.05% / 57.72% | about 93.24% / 59.64% |

In the final four-batch audit, removing the protein owner context reduces real-
slot accuracy by 21.82 percentage points; shuffling it reduces accuracy by
22.12 points. Cross-entropy rises by 1.367 and 1.673, respectively. For RNA the
accuracy drops are 44.33 and 32.99 points, with CE rises of 1.202 and 2.317.

That is positive evidence of **owner-context dependence**. It is limited to
669 protein real targets / 155 blocks and 97 RNA targets / 11 blocks. It does
not isolate geometric neighborhood use from backbone shape, global composition,
or other cues. The RNA counterfactual especially needs a larger fixed sample.

## 4. Protein: where failures remain

### Every amino-acid class

| AA | Real sidechain correct | Real-only exact block | Exact including V |
|---|---:|---:|---:|
| ALA | 99.12% | 99.12% | 59.57% |
| ARG | 92.94% | 78.64% | 78.64% |
| ASN | 73.45% | 43.10% | 42.26% |
| ASP | 91.93% | 72.93% | 63.07% |
| CYS | **58.14%** | **16.75%** | 14.21% |
| GLN | **67.27%** | **20.47%** | 20.39% |
| GLU | 90.45% | 72.57% | 72.03% |
| GLY | N/A — no real sidechain | 100.00% | 67.04% |
| HIS | **61.53%** | **8.87%** | 6.87% |
| ILE | 94.89% | 82.67% | 73.06% |
| LEU | 95.87% | 89.62% | 67.09% |
| LYS | 97.63% | 92.13% | 37.50% |
| MET | 78.10% | **24.51%** | 24.31% |
| PHE | 82.74% | 62.68% | 35.78% |
| PRO | 99.92% | 99.83% | 99.13% |
| SER | **70.05%** | **41.00%** | 32.34% |
| THR | 83.77% | 55.33% | 51.98% |
| TRP | **57.97%** | **3.96%** | 3.96% |
| TYR | 76.01% | **30.81%** | 30.03% |
| VAL | 97.02% | 91.51% | 35.17% |

ALA/LYS/VAL/LEU often get real atoms right but fail the V-inclusive block test:
predictions place extra real elements in padding. CYS/HIS/GLN/TRP fail even
without evaluating V targets. That second group cannot be dismissed as an
unimportant metric penalty. However, real-only scoring still allows a wrong
prediction of V on a real target, so it does **not** isolate V competition.

### Atom positions and interpretation

| Family | Final recall (support in parentheses) | Interpretation / priority |
|---|---|---|
| Backbone | N/CA/C/O 100%, each 82,316 | Easy fixed-role targets; not proof of context use |
| Early carbons | CB 99.72% (75,875); CG 97.58% (47,810); CG1 99.60% (10,494); CG2 93.54% (14,846) | Old CB collapse absent; strong carbon priors remain |
| Middle carbons | CD 93.51% (20,287); CD1 92.38% (19,764); CD2 88.69% (16,931); CE 95.81% (6,281) | Generally usable element recovery |
| Aromatic carbons | CE1 67.33% (8,034); CE2 65.27% (7,395); CE3 61.05% (1,312); CZ 78.18% (10,451); CZ2 58.77% (1,312) | Ring/length distinction remains weaker |
| TRP tail | CZ3 **6.02%**, CH2 **5.18%**, NE1 **5.87%**, each 1,312 | Severe late-slot occupancy/chemistry failure; high priority |
| Ring/amide nitrogen | ND1 **12.92%** (1,951); ND2 **44.01%** (3,483); NE2 **17.12%** (4,609) | HIS/ASN/GLN chemistry remains poorly represented by this task |
| Basic nitrogen | NE 94.51% (4,368); NZ 92.37% (4,208); NH1 79.78%, NH2 79.97% (each 4,368) | Nitrogen is not globally unlearnable |
| Carboxyl oxygen | OD1 67.32% (8,519); OD2 95.21% (5,036); OE1 70.74% (8,131); OE2 81.05% (5,473) | Position/class competition differs; one global oxygen weight is not a diagnosis |
| Hydroxyl oxygen | OG **41.09%** (4,558); OG1 **59.40%** (4,352); OH **30.92%** (2,827) | SER/THR/TYR functional groups deserve targeted analysis |
| Sulfur | SG **16.75%** (1,063); SD **24.51%** (2,073) | Rare but chemically important; do not ignore because aggregate loss is good |
| V | 91.05% (509,045) | Aggregate hides opposite errors at early and late slots |

The final slot histogram makes late-slot failure concrete: slot 9 has 1,312
real-C targets but only 188 predictions of C; slot 10 has 1,312 real-N targets
but only 215 predictions of N. Most predictions there are V. Conversely, slot 1
has 6,441 V targets but only 4,552 V predictions, consistent with extra-carbon
errors on short residues. Histograms are marginal counts, not a complete
target-to-prediction confusion matrix; obtain the latter before assigning each
atom's errors to C, V, or another element.

**Correction to the earlier interpretation:** TRP having no V targets does not
prove that its failure is unrelated to V. The same late output slots are V for
most other residues, so the shared classifier can still predict V on TRP.
Graph contamination is gone in the intended runtime path; output-class
competition is not.

## 5. RNA: substantially better, not uniformly solved

| Nucleotide | Real base correct | Real-only exact block | Exact including V |
|---|---:|---:|---:|
| A | 91.94% | 78.48% | 77.75% |
| C | 94.13% | 86.52% | 85.82% |
| G | 94.45% | 84.86% | 84.86% |
| U | 90.45% | 77.33% | 77.11% |

All twelve backbone positions are at 100% or 99.99% recall. Common base atoms
N9/C8/N7/C6/N1/C2/N3/C4/C5 are approximately 92.95–96.50%. N4 is 94.19%
(2,842 targets), O2 92.23% (5,082), O6 88.27% (3,036), and N2 85.44%
(3,036). The weakest canonical discriminators are **U O4: 77.73% (2,236)** and
**A N6: 78.59% (2,326)**. Their substantial support makes these real learning
gaps, not tiny-sample noise. Review base confusion and modified/incomplete
nucleotide handling before changing the RNA objective.

The RNA C7 entry is 100% on **one** example; Unknown is 0% on **28** examples.
Audit their identities and element/position mappings, but neither number can
justify a global architectural change. RNA's small real-exact versus all-exact
gap suggests less padding-only failure than protein; it does not exclude
real-to-V errors on the remaining base targets.

## 6. Two open encoder defects — reproduced, not inferred from scores

Run the small CPU reproduction from the repository root:

```bash
MPLCONFIGDIR=/tmp/mind_mpl python Docs/training/reproduce_encoder_findings.py
```

It constructs production-field PyG samples; it neither reads training PTs nor
trains a model. On Anzu it exited successfully with these observations:

| Defect | Reproduction | Why it matters | Required repair and test |
|---|---|---|---|
| Mask token mistaken for padding in RBF features | Raw Gaussian absolute sum 13.2821; masked-backbone RBF sum **0.0**; scalar-feature sum 28.9 | Retained real backbone loses its rich local-distance input precisely when asked to support masked-slot prediction. Some scalar and neighbor-mediated context remains, so this is not total geometry loss. | Use the actual dense-node validity mask for padding, retaining geometry-valid filtering. Test masked real nodes, true padding, placeholder perturbation, and mixed domains. |
| Explicit bond availability selected for the whole batch | Protein coordination alone `[1,2,2,1]`; the identical graph alongside bonded UniMol becomes **`[0,0,0,0]`** | Protein/RNA features can change merely because an unrelated batch member supplies explicit bonds. | Choose explicit bonds versus distance-proxy fallback per graph with an explicit availability contract. Test alone vs mixed, no-bond real molecules, compact reindexing, and no cross-graph edges. |

The first issue is at `_compute_geometric_features`'s `padding_mask =
x_dense.eq(0)` in `core/molecular_encoder.py`. Zero is also the MLM mask token.
The second is `_compute_chemical_coordination`'s batch-wide nonempty
`bond_edge_index` branch. `LazyUniversalDataset._normalize_sample_for_collate`
adds empty bond tensors to samples without bonds, and `pretraining_model.py`
forwards compact explicit bonds, so mixed-domain collation can reach this path.
The fraction of production batches affected and the learning impact were **not
measured** by this synthetic reproduction.

Existing coordination tests exercise explicit/inferred cases separately; the
RBF isolation test uses unmasked atom ids. They missed these combinations.
Git blame places the mask-token/padding pattern in the early encoder history;
the explicit-bond precedence dates to the September 1 coordination change.
Neither is evidence of malicious behavior or proof of the sole cause of the
remaining low recalls.

The distance-based “chemical coordination” is only a proxy: `0.5 < d <= 1.85 Å`
is not a complete bond graph or bond-order assignment. Do not describe it as
exact chemistry, and do not widen the cutoff merely to improve a score.

## 7. Important objective and evaluation limits

### Verified trainer-loop issue: variable plans meet a cached epoch limit

The final log records a 19,122-batch plan at lines 1,311,159–1,311,164 and only
19,081 consumed batches around lines 1,324,165–1,324,170. This is not merely a
callback percentage-rounding problem. The missing sample counts match all three
domains: 66 protein, 14 RNA and 1,405 UniMol.

`ImprovedDynamicBatchSampler.set_epoch` invalidates its plan and rebuilding can
change batch count. In installed Lightning 2.5.2, `FitLoop.setup_data` retains
the combined loader unless reload is requested; `Trainer.num_training_batches`
exposes its cached limit. `TrainingEpochLoop._num_ready_batches_reached` ends an
epoch at that number. The current trainer setup does not refresh the limit for
each rebuilt plan. A shorter plan can exhaust naturally (epoch 96 consumed
19,069 batches with full sample coverage), while longer plans get truncated.

This does not invalidate every learned metric or explain a severe atom collapse
by itself. It **does** invalidate “every sample once in every epoch” and needs
repair before the next full-pass production run. Test the trainer/sampler
integration over changing plan lengths; sampler-only `len == yielded` tests do
not cover Lightning's consumption limit. Ensure the epoch plan is set before
its length is consumed by the trainer, with correct validation/scheduler cadence.
Do not silently mutate private Lightning internals as an untested fix.

### Evaluation and supervision limits

1. **Different residues can have identical target strings.** Current protein
   ordering puts sidechain atoms in the shared priority order. ILE and LEU both
   yield `CCCCVVVVVV`; PRO and VAL both yield `CCCVVVVVVV`. Perfect element
   output therefore cannot uniquely identify every amino acid. Their visible
   geometry and embeddings may still differ, but this MLM does not explicitly
   supervise that distinction. Any future template-based residue decoder must
   group identical strings rather than award an arbitrary unique identity.
2. **Complete block accuracy is not 3D reconstruction.** The model predicts
   elements, not atom names, bond orders, torsions, coordinates, or chirality.
   Foundation-model usefulness must eventually be tested by transfer/probes;
   it is neither established nor disproved by these MLM scores alone.
3. **No same-split V-free causal comparison exists here.** Removing masked
   sidechain coordinates intentionally makes the problem harder. We cannot
   quantify a “normal V penalty” or promise that a new loss fixes it without a
   matched comparison. Backbone and environment do not uniquely determine every
   residue in every structure.
4. **Split independence needs an audit.** The current entry point uses seeded
   random splitting. A random sample split does not establish protein-family or
   RNA-family separation, nor exclude duplicate structures/conformers. Preserve
   the old split for comparability, then assess a stricter split separately.
5. **No fresh test evaluation yet.** Last-validation tables, best-checkpoint
   selection, and held-out test results must be reported separately. The current
   CLI has no verified evaluation-only/resume workflow; do not assume that a
   new `make train` command evaluates or resumes a saved checkpoint.

## 8. Atum: ordered next steps and acceptance gates

### A. Preserve and establish a frozen baseline

Follow [migration](../operations/MIGRATION.md). Preserve source, versioned docs,
all logs, best/last checkpoints, W&B state, exact dataset selection and metadata,
and environment exports. Do not rebuild current PTs or start 1M preprocessing
to troubleshoot these runtime issues.

The next agent should add a **Make-exposed evaluation-only workflow**, with
explicit checkpoint/config/split/seed/output paths and no optimizer updates.
Use an isolated output directory. First reproduce the last-checkpoint validation
tables; then evaluate the best checkpoint and the untouched test allocation.
Persist the selected sample identities and masking seeds for paired comparisons.

Add bounded diagnostics to that workflow, without changing the loss:

- true-element → predicted-element confusion by domain, slot, and residue;
- real→V, V→real, and real→wrong-real rates separately;
- invalid full-slot template rate; report protein target-string equivalence
  classes, not a fabricated unique residue classifier;
- larger owner-removed/shuffled audit and a matched visible-neighbor/context
  ablation, keeping owner geometry and graph boundaries controlled;
- support and confidence intervals across graphs, not only millions of
  correlated atom predictions.

The existing long run is already sufficient to justify these diagnostics.
Another blind 100-epoch run is not the next step.

### B. Repair the demonstrated runtime defects, one at a time

First establish that the trainer consumes the current epoch's entire sampler
plan, including variable-length epochs. Then fix masked-backbone RBF validity
and batch-local bond availability separately. For each: add a failing
production-contract regression test,
implement the smallest repair, and run geometry-neutral/attention/distance/
adapter tests. Preserve placeholder invariance and physical-edge isolation.
Do not change atom vocabulary, V weight, element balancing, or M/S scope in
these fixes.

Before expensive retraining, use frozen-checkpoint diagnostics to reveal any
feature-distribution shift. Such evaluation alone cannot establish training
benefit: compare old/fixed code on the **same** small dataset, masks, seeds,
steps, and domain mixture through user-started Make runs. Use fast checks for
crashes/routing, then a matched multidomain comparison for learning. Different
seeds or a fresh three-epoch run versus the 100-epoch checkpoint are not a causal
comparison. Expect correctness improvement, not guaranteed metric increases.

Acceptance: planned and consumed sample identities/counts match in every tested
epoch, finite loss/gradients, restored masked-node RBF, unchanged unrelated
graph coordination, no V/hidden-slot physical edges, healthy distance tasks,
and no material regression beyond repeat-run variation in real sidechain/base,
heteroatoms, exact blocks, or runtime resources. Record both improvements and
regressions. Commit/push only after the user-approved validation gates.

### C. Decide whether the remaining bottleneck needs objective changes

Prioritize CYS/MET sulfur, HIS/GLN/ASN nitrogen, TRP tail, SER/THR/TYR oxygen,
and RNA A-N6/U-O4 using the confusion evidence. Audit incomplete residues,
modified bases, rare labels, and split overlap. Verify actual mixed-batch bond
availability on a bounded production sample when execution is authorized.

Only then compare a narrowly motivated sampling/loss adjustment if the same
rare-class failure remains. Do not reintroduce macro element balancing or
masked-slot distance targets by default: previous experiments changed behavior
without establishing the desired contextual learning. No new occupancy head,
equivariant branch, or residue head is required to fix the two demonstrated bugs.

### D. Evaluate the foundation representation, then expand data

Use frozen-encoder probes or small controlled downstream tasks to test whether
embeddings distinguish chemistry, including element-string collisions. If later
users need residue or coordinate generation, fine-tuning may require additional
supervision/heads; current fixed-slot output preserves a useful interface but
does not guarantee such performance. Move to 1M proteins only after the baseline,
runtime fixes, dataset integrity, and practical memory/throughput gates are clear.

## Handoff summary for another agent

Start from the current source, not the archived CB-stacking plan. The long run
completed; no training needs babysitting. Keep invariant ESA with local M/global
S, geometry-neutral 14/23-slot caches, post-ESA owner readout, three task weights
of 1.0, and `virtual_atom_loss_weight: 0.2`. Do not revive removed experimental
losses. Reproduce the two CPU findings, preserve the frozen baseline, implement
an evaluation-only Make interface, and test each repair independently. The
remaining problem is **not one universal V bug**: physical routing is addressed,
late-slot V competition and rare chemistry remain, two geometry-feature defects
and a trainer epoch-limit defect are open, and downstream usefulness still needs
independent evaluation.
