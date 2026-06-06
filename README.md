# ternary-lattice-gc

**Lattice-based garbage collection for GPU object graphs with ternary liveness marking.**

[![crates.io](https://img.shields.io/crates/v/ternary-lattice-gc.svg)](https://crates.io/crates/ternary-lattice-gc)
[![docs.rs](https://docs.rs/ternary-lattice-gc/badge.svg)](https://docs.rs/ternary-lattice-gc)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why This Exists

GPU memory management is fundamentally different from CPU-side garbage collection. The GPU is a massively parallel device with limited ability to perform pointer chasing, and GPU memory is a scarce resource that's expensive to waste. Traditional mark-and-sweep GCs require atomic colour transitions that serialize work across threads — a terrible fit for GPU workloads.

**`ternary-lattice-gc`** models object liveness as a **join-semilattice** over three states: definitely alive (`+1`), maybe alive (`0`), and definitely dead (`−1`). This maps directly onto the classic **tri-color marking** abstraction used in production GCs (JVM, V8, Go), but with a key difference: liveness is **monotonic** — it only ever increases. This gives us:

- **Conflict-free concurrent propagation** — two threads can both "promote" a node's liveness without atomic CAS operations; the join operator is commutative, associative, and idempotent.
- **Incremental collection** — pause the GPU, mark a bounded budget of nodes, resume rendering. No stop-the-world pauses.
- **Natural cycle detection** — the fixed-point of liveness propagation reveals reference cycles automatically.

## How Tri-Color GC Maps to Ternary

The tri-color abstraction is the backbone of most modern garbage collectors. Every object is coloured:

| Colour | Traditional Meaning | Ternary Value | `Liveness` Variant |
|--------|-------------------|---------------|---------------------|
| **Black** | Reached and fully scanned | `+1` | `Alive` |
| **Gray** | Reached but not yet scanned | `0` | `MaybeAlive` |
| **White** | Not reached (candidate for collection) | `−1` | `Dead` |

The collection cycle works in phases:

1. **Mark roots gray** — all root objects transition from White → Gray.
2. **Propagate** — pop a gray node, colour it black, colour its children gray. Repeat until the gray worklist is empty.
3. **Sweep** — any node still white is unreachable and can be reclaimed.

Because `Dead ⊑ MaybeAlive ⊑ Alive` forms a total order, we can implement the join (least upper bound) as `max()`. This means:
- A node can go from `Dead → MaybeAlive → Alive`, but **never backwards**.
- Concurrent updates that both promote liveness are always safe — the result is the same regardless of order.

## The Algorithm

```
Phase 1: MARK ROOTS
  for each root in roots:
    node.liveness = MaybeAlive   // White → Gray
    gray_stack.push(root)

Phase 2: PROPAGATE (incremental, budgeted)
  while gray_stack not empty && budget > 0:
    node = gray_stack.pop()
    node.liveness = Alive        // Gray → Black
    for child in node.edges:
      if child.liveness == Dead: // Only promote White → Gray
        child.liveness = MaybeAlive
        gray_stack.push(child)
    budget--

Phase 3: FINALIZE
  dead = all nodes with liveness == Dead
  for each node in dead:
    run finalizer(node)          // User callback for cleanup
    remove node from graph
```

### Invariant: Monotonicity

The key invariant is that liveness is **monotone**: once a node becomes `MaybeAlive`, it can only transition to `Alive`, never back to `Dead`. This is the join-semilattice property:

```
Dead.join(MaybeAlive) = MaybeAlive
MaybeAlive.join(Alive) = Alive
Alive.join(Dead) = Alive        // can't regress!
```

This is what makes the approach safe for incremental and concurrent collection.

## Key Types

### `Liveness`

```rust
pub enum Liveness {
    Dead = -1,       // White: unreachable, candidate for reclamation
    MaybeAlive = 0,  // Gray: reachable but children not yet scanned
    Alive = 1,       // Black: reachable and fully scanned
}
```

Implements `Ord` (forming the lattice), provides `join()` (least upper bound), `from_i8()`, and `color_name()`.

### `LatticeNode`

```rust
pub struct LatticeNode {
    pub liveness: Liveness,
    pub edges: Vec<usize>,       // outgoing references
    pub tag: Option<String>,     // e.g. "texture", "buffer", "pipeline"
    pub gpu_handle: u64,         // opaque GPU resource handle
}
```

A node in the object graph. Each node carries its liveness state, a list of outgoing reference edges (indices), an optional type tag for classification, and a GPU handle for mapping back to the actual GPU resource.

### `LatticeGC`

The garbage collector itself. Manages a `HashMap<usize, LatticeNode>` and provides:

| Method | Description |
|--------|-------------|
| `add_node(node)` | Insert a node, returns its index |
| `add_edge(from, to)` | Add a directed reference |
| `remove_edge(from, to)` | Remove a reference |
| `mark_roots(&[id])` | Phase 1: reset all to Dead, mark roots as Gray |
| `propagate(budget)` | Phase 2: process up to `budget` gray nodes |
| `collect(&[roots])` | Full mark-sweep in one call |
| `collect_incremental(budget)` | Incremental: returns `(processed, finished)` |
| `finalize_dead()` | Run finalizers on dead nodes, remove them |
| `detect_cycles()` | DFS-based cycle detection |
| `is_alive(id)` / `is_dead(id)` | Query node state |

## Examples

### Basic Collection

```rust
use ternary_lattice_gc::{Liveness, LatticeNode, LatticeGC};

let mut gc = LatticeGC::new();

// Build a small GPU resource graph
let texture = gc.add_node(LatticeNode::with_handle(Liveness::Dead, 0x1000, "texture"));
let sampler = gc.add_node(LatticeNode::with_handle(Liveness::Dead, 0x2000, "sampler"));
let orphan  = gc.add_node(LatticeNode::with_handle(Liveness::Dead, 0x3000, "buffer"));

gc.add_edge(texture, sampler);
// orphan has no incoming references

// Run a full collection cycle
gc.collect(&[texture]);

assert!(gc.is_alive(texture));   // root → alive
assert!(gc.is_alive(sampler));   // reachable from root → alive
assert!(!gc.is_alive(orphan));   // unreachable → reclaimed
```

### Incremental Collection (Pause, Mark, Resume)

```rust
let mut gc = LatticeGC::new();
// ... build graph ...

gc.mark_roots(&[root]);

// Process in small increments (e.g., between frames)
loop {
    let (processed, finished) = gc.collect_incremental(16); // 16 nodes per tick
    if finished {
        break; // Collection complete, dead objects finalized
    }
    // ... render a frame, handle input, etc. ...
}
```

### Finalizer Callbacks

```rust
let mut gc = LatticeGC::new();
gc.set_finalizer(|id, node| {
    println!("Reclaiming GPU resource: handle={:#x}, tag={:?}",
             node.gpu_handle, node.tag);
    // Call vkDestroyBuffer, glDeleteTextures, etc.
});
```

### Cycle Detection

```rust
let mut gc = LatticeGC::new();
let a = gc.add_node(LatticeNode::new(Liveness::Alive));
let b = gc.add_node(LatticeNode::new(Liveness::Alive));
let c = gc.add_node(LatticeNode::new(Liveness::Alive));

gc.add_edge(a, b);
gc.add_edge(b, c);
gc.add_edge(c, a); // cycle!

let cycles = gc.detect_cycles();
assert!(!cycles.is_empty());
```

## Why This Matters for GPU Memory

GPU memory is a **finite, expensive resource**. A single 4K texture can consume 64MB. Unlike CPU memory, you can't just page things out to disk — if the GPU runs out of VRAM, things break.

The problems specific to GPU object graphs:

1. **Reference cycles are common.** Render passes reference textures, which reference samplers, which reference views back to textures. Ref-counting alone can't collect these cycles.

2. **Stop-the-world is unacceptable.** GPU workloads run at 60-144+ FPS. A 10ms GC pause is visible as a stutter. Incremental collection with bounded work per frame is essential.

3. **Concurrent mutation.** The CPU might be building a command buffer while the GPU is executing a previous one. The GC needs to handle concurrent liveness updates without expensive synchronization.

4. **Resource lifetime is data-driven.** A texture might be "maybe alive" (referenced by a pending command buffer) or "definitely alive" (actively bound in a running shader). The ternary model captures this distinction naturally.

The lattice-based approach addresses all four: cycles are detected by the propagation algorithm, pauses are bounded by the incremental budget, concurrent updates are safe due to monotonicity, and the three-state liveness captures the uncertainty inherent in GPU resource management.

## Performance Characteristics

| Operation | Complexity |
|-----------|------------|
| `add_node` | O(1) amortized |
| `add_edge` | O(edges) — duplicate check |
| `mark_roots` | O(nodes) — resets all to Dead |
| `propagate(b)` | O(b × max_edges) — bounded by budget |
| `detect_cycles` | O(nodes + edges) — standard DFS |
| `finalize_dead` | O(nodes) — scans for Dead |

Memory overhead: one `LatticeNode` per GPU object (typically hundreds to low thousands in a frame), plus the gray worklist (at most O(nodes) entries).

## Installation

```toml
[dependencies]
ternary-lattice-gc = "0.1"
```

## License

MIT
