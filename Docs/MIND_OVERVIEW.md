# MIND — A Plain-English Overview

*Molecular Intelligence for Novel Discovery*

This document explains what MIND is in simple terms, with enough technical substance
to hold up in an interview. Everything below is grounded in the actual code and config
(`core/pretraining_model.py`, `core/pretraining_config_multidomain.yaml`,
`data_loading/adapters/*`).

---

## 1. What is MIND?

**MIND is a "foundation model" for molecules.** Just like GPT is one big model that has
read enormous amounts of text and can then be adapted to many language tasks, MIND is one
big model that has "looked at" enormous numbers of 3D molecular structures — **proteins,
RNA, and small molecules (drug-like ligands)** — and learned a single, shared way of
representing all of them. The key idea is *universal*: instead of building a separate
specialist model for proteins, another for RNA, and another for small molecules, MIND
learns **one shared representation** that works across all molecule types. Under the hood
it's a **graph transformer** (an attention-based neural network that operates on atoms in
3D space) trained with **self-supervised learning** — meaning it teaches itself from the
structures alone, with no human labels.

---

## 2. What problem does it solve, and why?

Biology and chemistry are full of different "languages": a protein is a chain of amino
acids, RNA is a chain of nucleotides, and a drug molecule is a small cluster of atoms.
Historically, each gets its own specialist AI model. That's wasteful and limiting:

- **They can't share knowledge.** A protein model never learns from small-molecule data,
  even though both obey the *same laws of physics and chemistry* (bond lengths, atom
  packing, geometry).
- **Drug discovery is inherently cross-domain.** A drug (small molecule) binds to a
  protein; an RNA folds and interacts with proteins. A model that only understands one
  side of the interaction is half-blind.

MIND's bet is that **the underlying "grammar" of 3D molecular structure is universal**.
If you teach one model the shared rules — how atoms arrange in space, which atoms tend to
sit near which — it can transfer that understanding across proteins, RNA, and small
molecules. One model, many molecule types, shared learning.

---

## 3. How it works — the data

### The universal "14-atom" representation

The hard part of "one model for everything" is making different molecules look uniform to
the model. MIND does this by breaking every molecule into **residues/units, each described
by a fixed number of atom slots**:

- **Proteins:** each amino acid → exactly **14 atoms** (4 backbone atoms `N, CA, C, O`
  + up to 10 side-chain atoms). Tryptophan, the biggest amino acid, has 10 heavy
  side-chain atoms, so 14 covers every residue. Smaller residues get **virtual atoms** to
  fill empty slots. (See `protein_adapter.py`.)
- **RNA:** each nucleotide → a fixed **23-atom** layout (12 backbone + 11 base atoms).
  (See `rna_adapter.py`.)
- **Small molecules** (UniMol ligand dataset) and **metabolites/natural products**
  (COCONUT): kept as flat sets of atoms with their 3D coordinates.

A clever detail: the virtual filler atoms all share **one generic type ("V")**. This
prevents "sequence leakage" — the model can't cheat and guess a residue's identity just by
counting how many real vs. virtual atoms it has. Every molecule, regardless of type, ends
up as the same kind of object: **a list of atoms, each with an atomic number and an (x, y, z)
position.**

### Mixing all domains together (cross-modal batching)

MIND trains on multiple domains *at the same time*. In the multi-domain config, proteins
(from the AlphaFold database), small molecules (UniMol), and RNA are all enabled and mixed
into the **same training batches** ("cross-modal batches"). Because molecules vary wildly
in size, batches are built by a **dynamic, atom-aware sampler** that packs graphs up to a
budget of atoms per batch (`max_atoms_per_batch`) rather than a fixed number of molecules.

### Turning a molecule into a graph

The model sees each molecule as a **graph**: nodes = atoms, edges = relationships between
atoms. MIND builds edges in **three tiers** ("hybrid edge construction", inspired by recent
structure models):

1. **Sequence edges** — connect each residue to its neighbors along the chain
   (`sequence_neighbors_k = 3`). This captures backbone connectivity.
2. **Spatial edges** — connect atoms that are physically close in 3D (within a
   `cutoff_distance` of 5 Å, up to ~48 neighbors each). This captures folding/3D contacts.
3. **Random long-range edges** — a few (`num_random_edges = 8`) random connections between
   far-apart atoms (≥10 Å). These let information jump across the molecule so it isn't
   purely local. They're re-randomized each epoch so the model doesn't memorize them.

---

## 4. How it works — the model

### Attention over a *set/graph of atoms*, not text

MIND is in the **transformer family** (same attention mechanism that powers LLMs), but
instead of attending over a sequence of word tokens, it attends over a **set of atoms**.
The backbone is an **ESA (Edge-Set / Set-Attention) architecture**: a stack of attention
blocks where each atom can "look at" the other atoms and update its understanding based on
who's around it. It uses **multi-head attention** (16 heads per layer in the multi-domain
config) and **flash attention** (`flash_varlen`) so it can handle batches where molecules
have very different sizes efficiently, without wasteful padding.

### The layer stack — M / S / P

The model is 8 attention layers, all width 256, typed as
`["M", "S", "M", "S", "M", "S", "P", "P"]`:

- **S = Self-Attention Block (SAB):** atoms attend to each other freely.
- **M = Masked Attention Block:** attention restricted by the molecular graph (atoms
  mainly attend to their connected neighbors — encodes structure directly).
- **P = Pooling by Multihead Attention (PMA):** at the end, the variable-size set of atoms
  is pooled into a fixed-size summary, giving a whole-molecule embedding.

So the flow is: alternate "look at neighbors" (M) and "look at everyone" (S) layers to
build rich per-atom representations, then pool (P) into a molecule-level representation.

### Geometry: 3D coordinates + equivariant features

This is what makes MIND a *molecular* model rather than a generic graph model:

- Atom positions are **centered per molecule** (subtract the center of mass) so the model
  is **translation-invariant** — sliding a molecule across space doesn't change anything.
- 3D distances between atoms are turned into features via **Gaussian basis functions**
  (`gaussian_kernels = 128`), plus invariant geometric descriptors (coordination numbers,
  nearest-neighbor distances). These are rotation-invariant.
- On top of that, MIND uses **equivariant features** (`use_equivariant_features: true`,
  `equivariant_lmax = 2`) via spherical-harmonic-based blocks (Equiformer/NequIP-style).
  "Equivariant" means: if you **rotate the molecule, the geometric features rotate with
  it** in a physically consistent way. The model understands 3D orientation as a first-class
  citizen, not as accidental coordinates.

---

## 5. How it's trained (self-supervised — no labels)

MIND trains on **three self-supervised tasks at once** (no human annotations required —
the structures themselves provide the answers). From the config:

1. **Masked atom modeling (MLM) — "BERT for atoms."** Hide ~15% of the atoms' identities
   and ask the model to predict what they were from context (weight 0.3). For proteins
   there's a smarter variant — **block MLM** — where *whole residues* (all 14 atoms) are
   masked together, forcing the model to reason about structure rather than copying a
   neighbor atom in the same residue.
2. **Long-range distance prediction** (weight 1.5). Pick atom pairs anywhere in the
   molecule (up to ~1000 pairs, spanning 0–20 Å) and predict which **distance bin** they
   fall into. This teaches global 3D shape.
3. **Short-range distance prediction** (weight 1.0). For atoms connected by a graph edge
   (within 5 Å), predict their fine-grained distance bin. This teaches local chemistry.

(The code also contains a coordinate-denoising task that can be enabled.) Because distance
prediction uses **bins / relative distances**, the targets are unchanged by rotation or
translation — fully consistent with the equivariant design. The model is optimized end to
end (AdamW, cosine schedule with warmup) to minimize the weighted sum of these losses.

The point of pretraining: the model learns a **general-purpose understanding of molecular
structure**, which can later be **fine-tuned** on specific downstream tasks (e.g.
binding-affinity prediction) with far less data.

---

## 6. How MIND differs from a modern LLM (GPT, etc.)

| | **LLM (GPT)** | **MIND** |
|---|---|---|
| **Input unit** | Text tokens (words/sub-words) | Atoms with atomic number + 3D position |
| **Structure** | 1D sequence | 3D graph (sequence + spatial + random edges) |
| **Geometry** | None — order is all there is | Real 3D coordinates, distances, angles |
| **Symmetry** | None | Rotation/translation invariance & **equivariance** |
| **Order sensitivity** | Order matters | A molecule is a *set* — reorder the atoms and it's the same molecule (permutation-aware via attention) |
| **Training task** | Next-token prediction | Masked-atom prediction + distance prediction |
| **Domains** | One modality (text) | Cross-modal: proteins + RNA + small molecules together |

The single biggest conceptual difference: **an LLM has no notion that a rotated sentence is
the same sentence — because that idea is meaningless for text.** But a molecule rotated or
slid through space is *literally the same molecule*, and MIND is built so the model knows
this (invariance/equivariance). It reasons about a **3D set of atoms**, not a 1D string.

**But the similarities are real and worth stating:** both are transformers built on
attention; both are **pretrained self-supervised** on huge unlabeled data; and both follow
the **"foundation model then fine-tune"** paradigm. MIND is essentially "the BERT/GPT idea,
but for the language of 3D molecular structure instead of human text."

---

## 7. Interview talking points

- *"MIND is a foundation model for molecules — one transformer that learns a **shared,
  universal representation** of proteins, RNA, and small molecules, instead of a separate
  specialist model per molecule type."*
- *"The trick that makes 'one model for everything' work is a **uniform 14-atom-per-residue
  representation** so every molecule looks like the same object — a set of atoms with 3D
  coordinates — and we train all domains together in **cross-modal batches**."*
- *"It's a graph transformer with **multi-head attention over atoms** (an ESA / Set-Attention
  backbone), and crucially it's **geometry-aware**: translation-invariant and
  rotation-equivariant, so a rotated molecule is correctly treated as the same molecule —
  something an LLM has no concept of."*
- *"Training is **fully self-supervised, no labels**: masked-atom prediction (BERT-for-atoms,
  with residue-level block masking for proteins) plus long- and short-range **distance
  prediction** to learn both global shape and local chemistry."*
- *"Conceptually it's the same playbook as GPT/BERT — attention, self-supervised pretraining,
  then fine-tune — but the 'language' is the **3D structure of molecules** rather than text."*
