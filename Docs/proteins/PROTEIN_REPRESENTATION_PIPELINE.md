# Protein Representation Pipeline: Complete Data Flow

## Overview

This document explains the complete pipeline from raw protein PDB/CIF files through model training, with detailed examples at each step.

---

## Pipeline Summary

```
PDB/CIF File
    ↓
[1] ProteinAdapter: Create UniversalMolecule
    ↓
[2] cache_to_pyg.py: PyG Data Object + Edge Construction
    ↓
[3] LazyUniversalDataset: Chunk-based Loading
    ↓
[4] DataLoader: Batch Creation
    ↓
[5] Model Forward: Encoder + ESA + Tasks
    ↓
Loss Computation & Backpropagation
```

---

## EXAMPLE: Single Protein Molecule (2 Amino Acids)

We'll track a small protein through each pipeline stage.

### Raw Protein (Hypothetical)

```
Protein ID: P12345
Amino Acids: ALA-GLY (2 residues)

ALA (Alanine):
  - N  (x=1.0, y=2.0, z=3.0)
  - CA (x=1.5, y=2.5, z=3.5)
  - C  (x=2.0, y=3.0, z=4.0)
  - O  (x=2.5, y=3.5, z=4.5)
  - CB (x=1.2, y=2.2, z=3.2)

GLY (Glycine):
  - N  (x=3.0, y=4.0, z=5.0)
  - CA (x=3.5, y=4.5, z=5.5)
  - C  (x=4.0, y=5.0, z=6.0)
  - O  (x=4.5, y=5.5, z=6.5)
```

---

## STEP 1: ProteinAdapter - UniversalMolecule Creation

**Code:** `data_loading/adapters/protein_adapter.py`

### 1.1 Parse PDB/CIF File

```python
# protein_adapter.py: lines 315-344
def create_blocks(self, raw_item: Path) -> List[UniversalBlock]:
    """Convert PDB/CIF file to UniversalBlocks"""
    
    # Parse with BioPython
    structure = self.pdb_parser.get_structure(
        id=raw_item.stem,
        file=str(raw_item)
    )
    
    blocks: List[UniversalBlock] = []
    
    # Model -> Chain -> Residue hierarchy
    for model in structure:
        for chain in model:
            for residue in chain:
                # Each amino acid becomes one block
                ...
```

### 1.2 Create UniversalBlock for Each Amino Acid

#### **Block 0: ALA (Alanine)**

```python
# protein_adapter.py: lines 362-385
block_atoms: List[UniversalAtom] = []

for atom in residue:  # ALA residue atoms
    element = atom.element.strip().upper()  # 'N', 'C', 'O'
    
    if element == 'H':  # Skip hydrogen
        continue
    
    uni_atom = UniversalAtom(
        element=element,           # 'N', 'C', 'O', 'C', 'C'
        position=tuple(atom.get_coord().tolist()),
        pos_code=atom.get_name(),  # 'N', 'CA', 'C', 'O', 'CB'
        block_idx=len(blocks),     # 0 (ALA is first block)
        atom_idx_in_block=len(block_atoms),  # 0, 1, 2, 3, 4
        entity_idx=0               # Protein = entity 0  (This specifies the data type (RNA, metabolite, small molecules, protein).)
    )
    block_atoms.append(uni_atom)
```

**Result:**

```python
block_0 = UniversalBlock(
    symbol='ALA',
    atoms=[
        UniversalAtom(element='N',  position=(1.0, 2.0, 3.0), pos_code='N',  block_idx=0, atom_idx_in_block=0, entity_idx=0),
        UniversalAtom(element='C',  position=(1.5, 2.5, 3.5), pos_code='CA', block_idx=0, atom_idx_in_block=1, entity_idx=0),
        UniversalAtom(element='C',  position=(2.0, 3.0, 4.0), pos_code='C',  block_idx=0, atom_idx_in_block=2, entity_idx=0),
        UniversalAtom(element='O',  position=(2.5, 3.5, 4.5), pos_code='O',  block_idx=0, atom_idx_in_block=3, entity_idx=0),
        UniversalAtom(element='C',  position=(1.2, 2.2, 3.2), pos_code='CB', block_idx=0, atom_idx_in_block=4, entity_idx=0),
    ]
)
```

#### **Block 1: GLY (Glycine)**

```python
# Same process for GLY
block_1 = UniversalBlock(
    symbol='GLY',
    atoms=[
        UniversalAtom(element='N',  position=(3.0, 4.0, 5.0), pos_code='N',  block_idx=1, atom_idx_in_block=0, entity_idx=0),
        UniversalAtom(element='C',  position=(3.5, 4.5, 5.5), pos_code='CA', block_idx=1, atom_idx_in_block=1, entity_idx=0),
        UniversalAtom(element='C',  position=(4.0, 5.0, 6.0), pos_code='C',  block_idx=1, atom_idx_in_block=2, entity_idx=0),
        UniversalAtom(element='O',  position=(4.5, 5.5, 6.5), pos_code='O',  block_idx=1, atom_idx_in_block=3, entity_idx=0),
    ]
)
```

### 1.3 Create UniversalMolecule

```python
# protein_adapter.py: lines 405-421
def convert_to_universal(self, raw_item: Path) -> UniversalMolecule:
    blocks = self.create_blocks(raw_item)  # [block_0, block_1]
    
    return UniversalMolecule(
        id='P12345',           # Protein ID
        dataset_type='protein',
        blocks=[block_0, block_1],
        properties={}
    )
```

**Result:**

```python
universal_molecule = UniversalMolecule(
    id='P12345',
    dataset_type='protein',
    blocks=[
        UniversalBlock(symbol='ALA', atoms=[...5 atoms...]),
        UniversalBlock(symbol='GLY', atoms=[...4 atoms...])
    ],
    properties={}
)
```

**IMPORTANT NOTE - entity_idx:**
- `entity_idx=0`: All atoms belong to the protein (single entity)
- For protein-ligand complex: protein atoms would have `entity_idx=0`, ligand atoms `entity_idx=1`
- In batch, different proteins are distinguished by `batch` tensor, NOT by `entity_idx`
- `entity_idx` is for distinguishing multiple chemical entities WITHIN a single complex

---

## STEP 2: cache_to_pyg.py - PyG Data Object + Edge Construction

**Code:** `data_loading/cache_to_pyg.py`

### 2.1 Flatten All Atoms

```python
# cache_to_pyg.py: lines 419-431
atoms = mol.get_all_atoms()  # Get all atoms from UniversalMolecule

# Result: 9 atoms (5 ALA + 4 GLY)
# atoms[0] = N  (ALA)
# atoms[1] = CA (ALA)
# atoms[2] = C  (ALA)
# atoms[3] = O  (ALA)
# atoms[4] = CB (ALA)
# atoms[5] = N  (GLY)
# atoms[6] = CA (GLY)
# atoms[7] = C  (GLY)
# atoms[8] = O  (GLY)
```

### 2.2 Convert to Tensors

```python
# cache_to_pyg.py: lines 423-430
positions = torch.tensor([atom.position for atom in atoms], dtype=torch.float32)
# Shape: [9, 3]
# positions = torch.tensor([
#     [1.0, 2.0, 3.0],  # 0: N (ALA)
#     [1.5, 2.5, 3.5],  # 1: CA (ALA)
#     [2.0, 3.0, 4.0],  # 2: C (ALA)
#     [2.5, 3.5, 4.5],  # 3: O (ALA)
#     [1.2, 2.2, 3.2],  # 4: CB (ALA)
#     [3.0, 4.0, 5.0],  # 5: N (GLY)
#     [3.5, 4.5, 5.5],  # 6: CA (GLY)
#     [4.0, 5.0, 6.0],  # 7: C (GLY)
#     [4.5, 5.5, 6.5],  # 8: O (GLY)
# ])

atomic_numbers = torch.tensor(
    [self._element_to_atomic_number(atom.element) for atom in atoms],
    dtype=torch.long
)
# Shape: [9]
# atomic_numbers = torch.tensor([7, 6, 6, 8, 6, 7, 6, 6, 8])
#                               N  C  C  O  C  N  C  C  O

block_indices = torch.tensor([atom.block_idx for atom in atoms], dtype=torch.long)
# Shape: [9]
# block_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
#                              ALA-------------  GLY-------

entity_indices = torch.tensor([atom.entity_idx for atom in atoms], dtype=torch.long)
# Shape: [9]
# entity_indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
#                               All protein (entity 0)

pos_codes = [atom.pos_code for atom in atoms]
# pos_codes = ['N', 'CA', 'C', 'O', 'CB', 'N', 'CA', 'C', 'O']

block_symbols = [block.symbol for block in mol.blocks]
# block_symbols = ['ALA', 'GLY']
```

### 2.3 Edge Construction (SPARSE!)

```python
# cache_to_pyg.py: lines 434-446
if num_atoms > 1:
    # Hybrid edges or legacy radius_graph
    if self.use_hybrid_edges:
        edge_index = self._build_hybrid_edges(
            positions, block_indices, entity_indices, pos_codes
        )
    else:
        # LEGACY: radius_graph only
        edge_index = radius_graph(
            positions, 
            r=float(self.cutoff_distance),  # 5.0 Å
            batch=None, 
            loop=False,
            max_num_neighbors=int(self.max_neighbors)  # 32
        )
```

#### **How radius_graph Works**

```python
# PyTorch Geometric: torch_cluster.radius_graph
def radius_graph(pos, r, batch=None, loop=False, max_num_neighbors=32):
    """
    Find neighbor atoms within radius r for each atom.
    
    Args:
        pos: [N, 3] - Atom coordinates
        r: float - Cutoff distance (5.0 Å)
        batch: [N] - Batch indices (None = single molecule)
        loop: bool - Add self-loops? (False)
        max_num_neighbors: int - Max neighbors (32)
    
    Returns:
        edge_index: [2, E] - Sparse edge list (COO format)
    """
    edges = []
    
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i == j and not loop:
                continue
            
            distance = torch.norm(pos[i] - pos[j])
            if distance < r:
                edges.append([i, j])
                
                if len(edges) >= max_num_neighbors:
                    break
    
    return torch.tensor(edges).t()  # [2, E]
```

#### **Example: Neighbors for Atom 1 (CA-ALA)**

```python
# Atom 1: CA (ALA), position=(1.5, 2.5, 3.5)

# Calculate distances:
d(1, 0) = ||pos[1] - pos[0]|| = ||(1.5, 2.5, 3.5) - (1.0, 2.0, 3.0)|| = 0.87 Å  ✓ < 5.0
d(1, 2) = ||pos[1] - pos[2]|| = ||(1.5, 2.5, 3.5) - (2.0, 3.0, 4.0)|| = 0.87 Å  ✓ < 5.0
d(1, 3) = ||pos[1] - pos[3]|| = ||(1.5, 2.5, 3.5) - (2.5, 3.5, 4.5)|| = 1.73 Å ✓ < 5.0
d(1, 4) = ||pos[1] - pos[4]|| = ||(1.5, 2.5, 3.5) - (1.2, 2.2, 3.2)|| = 0.52 Å ✓ < 5.0
d(1, 5) = ||pos[1] - pos[5]|| = ||(1.5, 2.5, 3.5) - (3.0, 4.0, 5.0)|| = 2.60 Å ✓ < 5.0
d(1, 6) = ||pos[1] - pos[6]|| = ||(1.5, 2.5, 3.5) - (3.5, 4.5, 5.5)|| = 3.46 Å ✓ < 5.0
d(1, 7) = ||pos[1] - pos[7]|| = ||(1.5, 2.5, 3.5) - (4.0, 5.0, 6.0)|| = 4.33 Å ✓ < 5.0
d(1, 8) = ||pos[1] - pos[8]|| = ||(1.5, 2.5, 3.5) - (4.5, 5.5, 6.5)|| = 5.20 Å ✗ > 5.0

# Result: Atom 1 neighbors: [0, 2, 3, 4, 5, 6, 7]
# Edges: (1→0), (1→2), (1→3), (1→4), (1→5), (1→6), (1→7)
```

#### **All Edges (Hypothetical)**

```python
# radius_graph result (bidirectional)
edge_index = torch.tensor([
    # ALA internal edges
    [0, 1], [1, 0],  # N-CA
    [1, 2], [2, 1],  # CA-C
    [2, 3], [3, 2],  # C-O
    [1, 4], [4, 1],  # CA-CB
    [0, 4], [4, 0],  # N-CB
    
    # ALA-GLY backbone edges
    [2, 5], [5, 2],  # C(ALA)-N(GLY)
    
    # GLY internal edges
    [5, 6], [6, 5],  # N-CA
    [6, 7], [7, 6],  # CA-C
    [7, 8], [8, 7],  # C-O
], dtype=torch.long).t()

# Shape: [2, 20] (20 edges, bidirectional)
```

### 2.4 Edge Features (Distance)

```python
# cache_to_pyg.py: lines 478-483
def _calculate_edge_features(self, positions, edge_index):
    """Calculate edge distances"""
    row, col = edge_index  # Source, destination atom indices
    diff = positions[row] - positions[col]
    distances = torch.norm(diff, dim=1, keepdim=True)
    return distances

# Example:
# edge_attr = torch.tensor([
#     [0.87],  # Edge (0→1): N-CA distance
#     [0.87],  # Edge (1→0): CA-N distance
#     [0.87],  # Edge (1→2): CA-C distance
#     ...
# ])
# Shape: [20, 1]
```

### 2.5 Create PyG Data Object

```python
# cache_to_pyg.py: lines 451-465
data = Data(
    pos=positions,              # [9, 3] - Atom coordinates
    z=atomic_numbers,           # [9] - Atomic numbers
    edge_index=edge_index,      # [2, 20] - SPARSE edges!
    edge_attr=edge_attr,        # [20, 1] - Edge distances
    block_idx=block_indices,    # [9] - Block indices
    entity_idx=entity_indices,  # [9] - Entity indices
    pos_code=pos_codes,         # [9] - Position codes
    block_symbols=block_symbols,# ['ALA', 'GLY']
    mol_id='P12345',
    dataset_type='protein',
    num_nodes=9,
    num_edges=20,
)
```

**CRITICAL POINT:**

- `z`: **All atoms** present (9 atoms)
- `pos`: **All coordinates** present (9x3)
- `edge_index`: **Only connected edges** (20 edges, sparse!)
- `edge_attr`: **Only connected edge distances** (20x1)

**This PyG Data object is saved to disk (.pt file)**

---

## STEP 3: LazyUniversalDataset - Chunk-based Loading

**Code:** `data_loading/lazy_universal_dataset.py`

### 3.1 Chunk Files

```bash
# Chunked protein dataset on disk
proteins/
  processed_graphs_40k_chunk_0/processed/optimized_universal_protein_chunk_0.pt
  processed_graphs_40k_chunk_1/processed/optimized_universal_protein_chunk_1.pt
  ...
  processed_graphs_40k_chunk_49/processed/optimized_universal_protein_chunk_49.pt
```

### 3.2 Build Index Map

```python
# lazy_universal_dataset.py: lines 82-108
def _build_index_map(self):
    """Map each sample to its chunk location"""
    self.index_map = []
    
    for chunk_idx, chunk_path in enumerate(self.chunk_pt_files):
        # Read chunk metadata (header only, not full data!)
        _, slices = torch.load(chunk_path, map_location='cpu')
        num_samples_in_chunk = slices['z'].size(0) - 1
        
        for local_idx in range(num_samples_in_chunk):
            self.index_map.append({
                'chunk_file': chunk_path,
                'chunk_idx': chunk_idx,
                'local_idx': local_idx
            })
    
    # Example:
    # self.index_map[0] = {'chunk_file': 'chunk_0.pt', 'chunk_idx': 0, 'local_idx': 0}
    # self.index_map[800] = {'chunk_file': 'chunk_0.pt', 'chunk_idx': 0, 'local_idx': 800}
    # self.index_map[801] = {'chunk_file': 'chunk_1.pt', 'chunk_idx': 1, 'local_idx': 0}
```

### 3.3 Sample Retrieval (On-demand)

```python
# lazy_universal_dataset.py: lines 128-151
def __getitem__(self, idx):
    """Load requested sample lazily"""
    # Get chunk info from index map
    entry = self.index_map[idx]
    chunk_file = entry['chunk_file']
    local_idx = entry['local_idx']
    
    # Load chunk from cache (LRU cache)
    collated_data, slices = self._load_chunk(chunk_file)
    
    # Extract sample (PyG separate function)
    sample = self._extract_sample(collated_data, slices, local_idx)
    
    # Apply transform (MLM masking)
    if self.transform:
        sample = self.transform(sample)
    
    return sample
```

---

## STEP 4: DataLoader - Batch Creation

**Code:** `core/train_pretrain.py` + `data_loading/improved_dynamic_sampler.py`

### 4.1 Dynamic Batch Sampler

```python
# train_pretrain.py: lines 379-386
train_batch_sampler = ImprovedDynamicBatchSampler(
    train_dataset,
    max_atoms_per_batch=25000,  # Maximum atoms per batch
    shuffle_chunks=True,
    shuffle_within_chunk=True,
    seed=42,
    enable_cross_modal_batches=False  # Single-domain
)
```

### 4.2 Batch Creation Logic

```python
# improved_dynamic_sampler.py: __iter__()
def __iter__(self):
    """Create batches"""
    # 1. Group samples by chunk
    samples_by_chunk = self._organize_samples_by_chunk()
    
    # 2. Shuffle chunk order
    chunk_ids = list(samples_by_chunk.keys())
    random.shuffle(chunk_ids)
    
    # 3. Create batches for each chunk
    for chunk_id in chunk_ids:
        sample_indices = samples_by_chunk[chunk_id]
        
        current_batch = []
        current_atoms = 0
        
        for idx in sample_indices:
            # Get atom count from metadata (no disk I/O!)
            num_atoms = self._get_atom_count_fast(idx)
            
            if current_atoms + num_atoms > self.max_atoms_per_batch:
                # Batch full, yield it
                yield current_batch
                current_batch = [idx]
                current_atoms = num_atoms
            else:
                # Add to batch
                current_batch.append(idx)
                current_atoms += num_atoms
        
        # Last batch
        if current_batch:
            yield current_batch
```

### 4.3 Example Batch

```python
# Batch 0
batch_indices = [0, 1, 2, 3, 4]  # 5 proteins
# Total atoms: 4500 + 5200 + 3800 + 6100 + 5400 = 25000 atoms

# DataLoader uses these indices to get samples from LazyUniversalDataset
batch_samples = [dataset[idx] for idx in batch_indices]

# Merge with PyG collate function
batch = Batch.from_data_list(batch_samples)
```

### 4.4 Collated Batch Structure

```python
# batch object (PyG Batch)
batch = Batch(
    z=torch.tensor([...]),              # [25000] - All atoms' atomic numbers in batch
    pos=torch.tensor([...]),            # [25000, 3] - All atom coordinates
    edge_index=torch.tensor([...]),     # [2, 350000] - All edges (sparse!)
    edge_attr=torch.tensor([...]),      # [350000, 1] - Edge distances
    batch=torch.tensor([...]),          # [25000] - Which atom belongs to which graph
    block_idx=torch.tensor([...]),      # [25000] - Block indices
    entity_idx=torch.tensor([...]),     # [25000] - Entity indices
    pos_code=['N', 'CA', ...],          # [25000] - Position codes
    block_symbols=[['ALA', 'GLY', ...], ...],  # [5] - Block symbols per protein
    dataset_type=['protein', 'protein', ...],  # [5] - Dataset types
    num_nodes=25000,
    num_edges=350000,
    num_graphs=5,
)
```

**IMPORTANT:**

- `batch.z`: **25,000 atoms** (5 proteins x ~5000 atoms average)
- `batch.edge_index`: **350,000 edges** (only connected edges, sparse!)
- `batch.batch`: Indicates which atom belongs to which graph
  ```python
  batch.batch = torch.tensor([
      0, 0, 0, ..., 0,  # First protein (4500 atoms)
      1, 1, 1, ..., 1,  # Second protein (5200 atoms)
      2, 2, 2, ..., 2,  # Third protein (3800 atoms)
      3, 3, 3, ..., 3,  # Fourth protein (6100 atoms)
      4, 4, 4, ..., 4,  # Fifth protein (5400 atoms)
  ])
  ```

---

## STEP 5: Model Forward Pass

**Code:** `core/pretraining_model.py`

### 5.1 Encoder: Node Embeddings

```python
# pretraining_model.py: lines 773-793
def forward(self, batch):
    """Forward pass"""
    edge_index, batch_mapping = batch.edge_index, batch.batch
    pos = batch.pos
    
    # Input: Atomic numbers
    x = batch.z  # [25000] - Atomic numbers
    
    # Encode nodes (universal approach)
    node_embeddings = self.encoder(x, pos, batch)
    # node_embeddings: [25000, hidden_dim] (e.g. 512)
```

### 5.2 Encoder Details

```python
# pretraining_model.py: lines 156-189
def forward(self, x, pos=None, batch=None):
    """Universal molecular encoding"""
    # x: [25000] - Atomic numbers (torch.long)
    
    # 1. Atom embeddings
    encoded = self._encode_molecule_features(x)
    # encoded: [25000, hidden_dim]
    
    # 2. Geometric features (3D coordinates + edge info)
    if pos is not None:
        geometric_features = self._compute_geometric_features(x, pos, batch_mapping, batch)
        # geometric_features: [25000, hidden_dim]
        
        encoded = encoded + geometric_features
    
    return encoded  # [25000, hidden_dim]
```

### 5.3 Geometric Features: SPARSE vs DENSE

**CRITICAL POINT: Two paths exist!**

#### **PATH A: SPARSE COMPUTATION (RNA & Large Molecules)**

```python
# pretraining_model.py: lines 324-431
if use_sparse_computation:  # RNA or >500 atoms
    # ===== SPARSE PATH =====
    # Edge distances (ONLY CONNECTED EDGES)
    edge_src, edge_dst = edge_index[0], edge_index[1]  # [350000]
    edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)
    # edge_distances: [350000] - Only connected edge distances
    
    # Coordination features (scatter operations)
    close_mask = (edge_distances < 3.0) & (edge_distances > 0.5)
    close_coordination_sparse = scatter_add(
        close_mask.float(), edge_src, dim=0, dim_size=num_nodes
    )
    # close_coordination_sparse: [25000] - Close neighbor count per atom
    
    # RBF features (Gaussian kernels)
    edge_rbf = self.gaussian_layer(edge_distances, edge_types)
    # edge_rbf: [350000, 128] - RBF features per edge
    
    node_rbf_src = scatter_add(edge_rbf, edge_src, dim=0, dim_size=num_nodes)
    node_rbf_dst = scatter_add(edge_rbf, edge_dst, dim=0, dim_size=num_nodes)
    node_rbf_combined = node_rbf_src + node_rbf_dst
    # node_rbf_combined: [25000, 128] - Aggregated RBF per node
```

**SPARSE COMPUTATION ADVANTAGES:**

- Memory-efficient: Only operates on connected edges (350K edges)
- GPU-optimized: `scatter_add` uses CUDA kernels (parallel)
- No NxN matrix: Does not create dense distance matrix

**NOT DOING DENSE COMPUTATION:**

```python
# This code DOES NOT EXIST (in sparse path)
dist = torch.cdist(pos, pos)  # [25000, 25000] - HUGE!
# This would be 25000x25000 = 625 million elements = 2.5 GB float32!
# Guaranteed CUDA OOM!
```

#### **PATH B: DENSE COMPUTATION (Protein & Small Molecules)**

```python
# pretraining_model.py: lines 433-519
else:  # Dense path
    # ===== DENSE PATH =====
    # Convert batch to dense format with to_dense_batch
    x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
    pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)
    # x_dense: [5, 6100, 1] - Max protein size 6100 atoms (with padding)
    # pos_dense: [5, 6100, 3]
    
    # Pairwise distance matrix (DENSE!)
    delta_pos = pos_dense.unsqueeze(1) - pos_dense.unsqueeze(2)
    # delta_pos: [5, 6100, 6100, 3]
    
    dist = delta_pos.norm(dim=-1)
    # dist: [5, 6100, 6100] - NxN distance matrix per graph
    
    # Coordination features
    close_mask = (dist < 3.0) & (dist > 0.5)
    close_coordination = close_mask.sum(dim=-1).float()
    # close_coordination: [5, 6100]
    
    # BUT RBF FEATURES STILL USE SPARSE EDGE_INDEX!
    edge_src, edge_dst = batch.edge_index
    edge_distances = torch.norm(pos[edge_src] - pos[edge_dst], dim=1)
    edge_rbf = self.gaussian_layer(edge_distances, edge_types)
    node_rbf_combined = scatter_add(edge_rbf, edge_src, ...) + scatter_add(edge_rbf, edge_dst, ...)
```

**DENSE COMPUTATION:**

- Uses dense pairwise distance for coordination features
- Fast for small batches (with padding)
- Memory-intensive: NxN matrix (5 x 6100 x 6100 = 186M elements = 745 MB)
- RBF features STILL SPARSE: Uses `edge_index`!

**DIFFERENCE:**

| Feature | Sparse Path | Dense Path |
|---------|-------------|------------|
| **Coordination** | scatter_add (sparse edges) | dist.sum() (NxN matrix) |
| **RBF** | scatter_add (sparse edges) | scatter_add (sparse edges) |
| **Memory** | O(E) - Edge count | O(B × N²) - Batch × Nodes² |
| **Use case** | RNA, >500 atoms | Protein, <500 atoms |

---

## SUMMARY: Protein Data Flow

```
1. PDB/CIF File
   ↓
   [ProteinAdapter]
   ↓
2. UniversalMolecule
   - blocks: [ALA, GLY, SER, ...]
   - atoms: [N, CA, C, O, CB, ...]
   ↓
   [cache_to_pyg.py]
   ↓
3. PyG Data Object
   - z: [N] atomic numbers <- ALL ATOMS
   - pos: [N, 3] coordinates <- ALL ATOMS
   - edge_index: [2, E] <- ONLY CONNECTED EDGES (sparse!)
   - edge_attr: [E, 1] <- Edge distances
   - block_idx: [N] <- Block indices (amino acid ID)
   - entity_idx: [N] <- Entity indices (protein=0)
   ↓
   [LazyUniversalDataset]
   ↓
4. Batch (collated)
   - z: [25000] <- 5 proteins, total 25K atoms
   - edge_index: [2, 350000] <- Only connected edges!
   - batch: [25000] <- Which atom belongs to which protein
   ↓
   [Model Forward]
   ↓
5. Node Embeddings
   - Encoder: z -> embeddings [25000, hidden_dim]
   - Geometric features:
     * Sparse path: scatter_add on edge_index
     * Dense path: NxN distance matrix (coordination only)
   - ESA backbone: Attention [25000, graph_dim]
   ↓
6. Loss & Backprop
   - Long-range distance: Atom pairs prediction
   - MLM: Masked atom type prediction
```

---

## Code References

### Data Loading

- `protein_adapter.py:315-421` - PDB/CIF to UniversalMolecule
- `cache_to_pyg.py:416-471` - UniversalMolecule to PyG Data
- `cache_to_pyg.py:442-445` - radius_graph edge construction
- `lazy_universal_dataset.py:82-151` - Chunk-based loading

### Model

- `pretraining_model.py:156-189` - Universal encoder forward
- `pretraining_model.py:289-536` - Geometric features (sparse vs dense)
- `pretraining_model.py:773-954` - Model forward pass
- `pretraining_model.py:957-989` - Loss computation

### Training

- `train_pretrain.py:62-317` - Dataset loading
- `train_pretrain.py:320-575` - DataLoader creation
- `train_pretrain.py:578-765` - Training loop