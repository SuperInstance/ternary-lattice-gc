//! # Ternary Lattice GC
//!
//! Lattice-based garbage collection for GPU object graphs where each object carries
//! a **ternary liveness** value: `+1` (definitely alive), `0` (maybe alive), or `-1`
//! (definitely dead). These map directly to the classic tri-color marking scheme:
//!
//! | Ternary | Tri-color | Meaning                |
//! |---------|-----------|------------------------|
//! | +1      | Black     | Reachable, scanned     |
//! |  0      | Gray      | Reachable, unscanned   |
//! | -1      | White     | Unreachable / dead     |
//!
//! ## Why ternary + lattice?
//!
//! GPU memory management is fundamentally different from CPU GC: the GPU is a
//! massively parallel device with limited ability to perform pointer chasing.
//! Traditional mark-and-sweep GCs require atomic colour transitions that serialize
//! work. By modelling liveness as a **join-semilattice** (−1 ⊑ 0 ⊑ +1), we get:
//!
//! - **Monotonic propagation**: liveness only increases, so concurrent updates never conflict.
//! - **Incremental collection**: pause, propagate a bounded number of nodes, resume.
//! - **Cycle detection**: fixed-point iteration naturally detects reference cycles.
//!
//! ## Example
//!
//! ```rust
//! use ternary_lattice_gc::{Liveness, LatticeNode, LatticeGC};
//!
//! let mut gc = LatticeGC::new();
//!
//! // Root node (definitely alive)
//! let root = gc.add_node(LatticeNode::new(Liveness::Alive));
//!
//! // Child node (maybe alive — we haven't confirmed yet)
//! let child = gc.add_node(LatticeNode::new(Liveness::MaybeAlive));
//!
//! // Orphan node (definitely dead)
//! let orphan = gc.add_node(LatticeNode::new(Liveness::Dead));
//!
//! gc.add_edge(root, child);
//!
//! gc.mark_roots(&[root]);
//! gc.propagate(usize::MAX); // propagate all
//!
//! assert!(gc.is_alive(child));
//! assert!(!gc.is_alive(orphan));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Liveness (ternary value)
// ---------------------------------------------------------------------------

/// Ternary liveness state for a node in the object graph.
///
/// Forms a join-semilattice: `Dead ⊑ MaybeAlive ⊑ Alive`.
/// The join (least upper bound) of two states is the larger one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Liveness {
    /// Definitely dead / unreachable (tri-color: white, value −1).
    Dead = -1,
    /// Maybe alive / pending scan (tri-color: gray, value 0).
    MaybeAlive = 0,
    /// Definitely alive / reachable (tri-color: black, value +1).
    Alive = 1,
}

impl Liveness {
    /// Numeric value: −1, 0, or +1.
    pub fn value(self) -> i8 {
        self as i8
    }

    /// Join (least upper bound) — returns the greater of the two states.
    pub fn join(self, other: Liveness) -> Liveness {
        self.max(other)
    }

    /// Parse from i8. Values outside {−1, 0, +1} are clamped.
    pub fn from_i8(v: i8) -> Self {
        use std::cmp::Ordering::*;
        match v.cmp(&0) {
            Less => Liveness::Dead,
            Equal => Liveness::MaybeAlive,
            Greater => Liveness::Alive,
        }
    }

    /// Tri-color name.
    pub fn color_name(self) -> &'static str {
        match self {
            Liveness::Dead => "white",
            Liveness::MaybeAlive => "gray",
            Liveness::Alive => "black",
        }
    }
}

// ---------------------------------------------------------------------------
// LatticeNode
// ---------------------------------------------------------------------------

/// A node in the GPU object graph.
///
/// Each node carries a ternary liveness and a set of outgoing reference edges
/// (indices into the graph's node list). Nodes also have an optional type tag
/// useful for GPU resource classification (texture, buffer, pipeline, etc.).
#[derive(Debug, Clone)]
pub struct LatticeNode {
    /// Current liveness state.
    pub liveness: Liveness,
    /// Outgoing reference edges (target node indices).
    pub edges: Vec<usize>,
    /// Optional type tag (e.g. "texture", "buffer").
    pub tag: Option<String>,
    /// User data payload — an opaque id the caller can use to map back to GPU handles.
    pub gpu_handle: u64,
}

impl LatticeNode {
    /// Create a new node with the given liveness and no edges.
    pub fn new(liveness: Liveness) -> Self {
        Self {
            liveness,
            edges: Vec::new(),
            tag: None,
            gpu_handle: 0,
        }
    }

    /// Create a node with a GPU handle and type tag.
    pub fn with_handle(liveness: Liveness, gpu_handle: u64, tag: impl Into<String>) -> Self {
        Self {
            liveness,
            edges: Vec::new(),
            tag: Some(tag.into()),
            gpu_handle,
        }
    }

    /// Add an outgoing reference edge.
    pub fn add_edge(&mut self, target: usize) {
        if !self.edges.contains(&target) {
            self.edges.push(target);
        }
    }

    /// Remove an edge (if present).
    pub fn remove_edge(&mut self, target: usize) {
        self.edges.retain(|&t| t != target);
    }
}

// ---------------------------------------------------------------------------
// Finalizer callback
// ---------------------------------------------------------------------------

/// Callback invoked for objects that transition to `Dead` during collection.
pub type Finalizer = dyn FnMut(usize, &LatticeNode);

// ---------------------------------------------------------------------------
// LatticeGC — the garbage collector
// ---------------------------------------------------------------------------

/// Lattice-based garbage collector for GPU object graphs.
///
/// Manages a set of `LatticeNode`s indexed by `usize`. Supports incremental
/// tri-color marking, cycle detection, and finalization of dead objects.
pub struct LatticeGC {
    nodes: HashMap<usize, LatticeNode>,
    next_id: usize,
    roots: HashSet<usize>,
    gray_stack: VecDeque<usize>,
    finalizer: Option<Box<Finalizer>>,
    /// Number of nodes propagated in the last `propagate()` call.
    pub last_propagated: usize,
}

impl LatticeGC {
    // -- Construction -------------------------------------------------------

    /// Create an empty GC.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            roots: HashSet::new(),
            gray_stack: VecDeque::new(),
            finalizer: None,
            last_propagated: 0,
        }
    }

    /// Set a finalizer callback invoked for each dead object before reclamation.
    pub fn set_finalizer(&mut self, f: impl FnMut(usize, &LatticeNode) + 'static) {
        self.finalizer = Some(Box::new(f));
    }

    // -- Graph mutation -----------------------------------------------------

    /// Add a node, returning its index.
    pub fn add_node(&mut self, node: LatticeNode) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// Add a directed reference edge from `from` to `to`.
    ///
    /// Panics if either node does not exist.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        assert!(self.nodes.contains_key(&from), "source node {from} not found");
        assert!(self.nodes.contains_key(&to), "target node {to} not found");
        self.nodes.get_mut(&from).unwrap().add_edge(to);
    }

    /// Remove a directed reference edge.
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if let Some(node) = self.nodes.get_mut(&from) {
            node.remove_edge(to);
        }
    }

    /// Mark nodes as permanent roots.
    pub fn set_roots(&mut self, roots: &[usize]) {
        self.roots = roots.iter().copied().collect();
    }

    // -- Tri-color marking --------------------------------------------------

    /// Phase 1: Mark all roots as gray (MaybeAlive) and push them onto the work stack.
    ///
    /// Resets every node to `Dead` first, so the entire graph must be reached from roots.
    pub fn mark_roots(&mut self, roots: &[usize]) {
        // Reset all nodes to Dead
        for node in self.nodes.values_mut() {
            node.liveness = Liveness::Dead;
        }
        self.gray_stack.clear();

        for &id in roots {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.liveness = Liveness::MaybeAlive;
                self.gray_stack.push_back(id);
            }
        }
    }

    /// Phase 2: Propagate liveness through the graph, up to `budget` nodes.
    ///
    /// Each step pops a gray node, marks it black (Alive), and marks all children gray.
    /// Returns the number of nodes processed.
    pub fn propagate(&mut self, budget: usize) -> usize {
        let mut processed = 0;
        while processed < budget {
            let id = match self.gray_stack.pop_front() {
                Some(id) => id,
                None => break,
            };

            // Mark self black
            if let Some(node) = self.nodes.get_mut(&id) {
                node.liveness = Liveness::Alive;
                let edges = node.edges.clone();
                // Mark children gray
                for child_id in edges {
                    if let Some(child) = self.nodes.get_mut(&child_id) {
                        if child.liveness == Liveness::Dead {
                            child.liveness = Liveness::MaybeAlive;
                            self.gray_stack.push_back(child_id);
                        }
                    }
                }
            }
            processed += 1;
        }
        self.last_propagated = processed;
        processed
    }

    /// Returns `true` if `propagate` has no more work (gray stack is empty).
    pub fn is_marking_complete(&self) -> bool {
        self.gray_stack.is_empty()
    }

    /// Convenience: full mark-and-sweep in one call.
    pub fn collect(&mut self, roots: &[usize]) {
        self.mark_roots(roots);
        self.propagate(usize::MAX);
        self.finalize_dead();
    }

    /// Run incremental collection: process up to `budget` nodes, return `(processed, finished)`.
    pub fn collect_incremental(&mut self, budget: usize) -> (usize, bool) {
        let processed = self.propagate(budget);
        let done = self.is_marking_complete();
        if done {
            self.finalize_dead();
        }
        (processed, done)
    }

    // -- Queries ------------------------------------------------------------

    /// Is the node definitely alive?
    pub fn is_alive(&self, id: usize) -> bool {
        self.nodes.get(&id).map_or(false, |n| n.liveness == Liveness::Alive)
    }

    /// Is the node definitely dead?
    pub fn is_dead(&self, id: usize) -> bool {
        self.nodes.get(&id).map_or(true, |n| n.liveness == Liveness::Dead)
    }

    /// Get the liveness of a node.
    pub fn liveness(&self, id: usize) -> Option<Liveness> {
        self.nodes.get(&id).map(|n| n.liveness)
    }

    /// Get a reference to a node.
    pub fn get_node(&self, id: usize) -> Option<&LatticeNode> {
        self.nodes.get(&id)
    }

    /// Return the set of dead node ids (after marking).
    pub fn dead_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.liveness == Liveness::Dead)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return the set of alive node ids.
    pub fn alive_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.liveness == Liveness::Alive)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // -- Cycle detection ----------------------------------------------------

    /// Detect reference cycles using iterative DFS with on-stack tracking.
    ///
    /// Returns a list of cycles, where each cycle is a list of node ids forming the cycle.
    pub fn detect_cycles(&self) -> Vec<Vec<usize>> {
        let mut cycles: Vec<Vec<usize>> = Vec::new();

        // We use a recursive-style DFS simulated with an explicit stack.
        // Each frame tracks: node, parent path, which children we've processed.
        // Instead, use a simpler approach: color-based DFS.
        // color: 0=white(unvisited), 1=gray(on stack), 2=black(done)
        let mut color: HashMap<usize, u8> = HashMap::new();
        let mut path: Vec<usize> = Vec::new();

        for &start in self.nodes.keys() {
            if color.get(&start).copied().unwrap_or(0) != 0 {
                continue;
            }
            // Iterative DFS
            let mut dfs_stack: Vec<(usize, usize)> = vec![(start, 0)]; // (node, child_index)
            while let Some((node, ci)) = dfs_stack.last_mut() {
                let node = *node;
                let c = color.get(&node).copied().unwrap_or(0);
                if c == 0 {
                    // White -> Gray (enter)
                    color.insert(node, 1);
                    path.push(node);
                }
                if c == 2 {
                    // Already black, pop
                    dfs_stack.pop();
                    continue;
                }
                let edges = self.nodes.get(&node).map(|n| n.edges.clone()).unwrap_or_default();
                if *ci < edges.len() {
                    let child = edges[*ci];
                    *ci += 1;
                    let cc = color.get(&child).copied().unwrap_or(0);
                    if cc == 0 {
                        dfs_stack.push((child, 0));
                    } else if cc == 1 {
                        // Back edge -> cycle found
                        if let Some(idx) = path.iter().position(|&x| x == child) {
                            let cycle: Vec<usize> = path[idx..].to_vec();
                            if cycle.len() > 1 {
                                cycles.push(cycle);
                            }
                        }
                    }
                } else {
                    // All children processed -> Gray -> Black (exit)
                    color.insert(node, 2);
                    path.pop();
                    dfs_stack.pop();
                }
            }
        }
        cycles
    }

    // -- Finalization -------------------------------------------------------

    /// Invoke finalizer on all dead nodes and remove them from the graph.
    pub fn finalize_dead(&mut self) {
        let dead: Vec<usize> = self.dead_nodes();
        for id in dead {
            let node = self.nodes.remove(&id);
            if let (Some(node), Some(ref mut f)) = (node, &mut self.finalizer) {
                f(id, &node);
            }
        }
    }
}

impl Default for LatticeGC {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liveness_ordering() {
        assert!(Liveness::Dead < Liveness::MaybeAlive);
        assert!(Liveness::MaybeAlive < Liveness::Alive);
        assert_eq!(Liveness::Dead.join(Liveness::Alive), Liveness::Alive);
        assert_eq!(Liveness::Dead.join(Liveness::MaybeAlive), Liveness::MaybeAlive);
        assert_eq!(Liveness::MaybeAlive.join(Liveness::MaybeAlive), Liveness::MaybeAlive);
    }

    #[test]
    fn test_liveness_from_i8() {
        assert_eq!(Liveness::from_i8(-1), Liveness::Dead);
        assert_eq!(Liveness::from_i8(0), Liveness::MaybeAlive);
        assert_eq!(Liveness::from_i8(1), Liveness::Alive);
        assert_eq!(Liveness::from_i8(-42), Liveness::Dead);
        assert_eq!(Liveness::from_i8(99), Liveness::Alive);
    }

    #[test]
    fn test_tri_color_names() {
        assert_eq!(Liveness::Dead.color_name(), "white");
        assert_eq!(Liveness::MaybeAlive.color_name(), "gray");
        assert_eq!(Liveness::Alive.color_name(), "black");
    }

    #[test]
    fn test_basic_mark_and_sweep() {
        let mut gc = LatticeGC::new();
        let root = gc.add_node(LatticeNode::new(Liveness::Dead));
        let child1 = gc.add_node(LatticeNode::new(Liveness::Dead));
        let child2 = gc.add_node(LatticeNode::new(Liveness::Dead));
        let orphan = gc.add_node(LatticeNode::new(Liveness::Dead));

        gc.add_edge(root, child1);
        gc.add_edge(root, child2);

        gc.collect(&[root]);

        assert!(gc.is_alive(root));
        assert!(gc.is_alive(child1));
        assert!(gc.is_alive(child2));
        assert!(!gc.is_alive(orphan)); // reclaimed
        assert_eq!(gc.node_count(), 3); // orphan was removed
    }

    #[test]
    fn test_incremental_collection() {
        let mut gc = LatticeGC::new();
        let root = gc.add_node(LatticeNode::new(Liveness::Dead));
        let a = gc.add_node(LatticeNode::new(Liveness::Dead));
        let b = gc.add_node(LatticeNode::new(Liveness::Dead));
        let c = gc.add_node(LatticeNode::new(Liveness::Dead));

        gc.add_edge(root, a);
        gc.add_edge(a, b);
        gc.add_edge(b, c);

        gc.mark_roots(&[root]);

        // Process one node at a time
        let (n, done) = gc.collect_incremental(1);
        assert_eq!(n, 1);
        assert!(!done);

        // Continue until done
        let mut total = n;
        while !gc.is_marking_complete() {
            let (n, _) = gc.collect_incremental(1);
            total += n;
        }
        gc.finalize_dead();
        assert_eq!(total, 4); // root, a, b, c
        assert!(gc.is_alive(c));
    }

    #[test]
    fn test_cycle_detection_simple() {
        let mut gc = LatticeGC::new();
        let a = gc.add_node(LatticeNode::new(Liveness::Alive));
        let b = gc.add_node(LatticeNode::new(Liveness::Alive));
        let c = gc.add_node(LatticeNode::new(Liveness::Alive));

        gc.add_edge(a, b);
        gc.add_edge(b, c);
        gc.add_edge(c, a); // cycle: a → b → c → a

        let cycles = gc.detect_cycles();
        assert!(!cycles.is_empty(), "should detect at least one cycle");
    }

    #[test]
    fn test_no_cycle_in_dag() {
        let mut gc = LatticeGC::new();
        let a = gc.add_node(LatticeNode::new(Liveness::Alive));
        let b = gc.add_node(LatticeNode::new(Liveness::Alive));
        let c = gc.add_node(LatticeNode::new(Liveness::Alive));

        gc.add_edge(a, b);
        gc.add_edge(b, c);
        // No back edge — DAG

        let cycles = gc.detect_cycles();
        assert!(cycles.is_empty(), "DAG should have no cycles");
    }

    #[test]
    fn test_finalizer_callback() {
        use std::sync::{Arc, Mutex};
        let finalized = Arc::new(Mutex::new(Vec::new()));
        let finalized_clone = finalized.clone();

        let mut gc = LatticeGC::new();
        gc.set_finalizer(move |id, node| {
            finalized_clone.lock().unwrap().push((id, node.gpu_handle));
        });

        let root = gc.add_node(LatticeNode::with_handle(Liveness::Dead, 100, "texture"));
        let orphan = gc.add_node(LatticeNode::with_handle(Liveness::Dead, 200, "buffer"));

        gc.collect(&[root]);

        let fins = finalized.lock().unwrap();
        assert_eq!(fins.len(), 1);
        assert_eq!(fins[0].0, orphan);
        assert_eq!(fins[0].1, 200);
    }

    #[test]
    fn test_lattice_join_monotonicity() {
        // Demonstrating that liveness only increases (key invariant for GPU GC)
        let mut state = Liveness::Dead;
        state = state.join(Liveness::MaybeAlive);
        assert_eq!(state, Liveness::MaybeAlive);
        state = state.join(Liveness::Alive);
        assert_eq!(state, Liveness::Alive);
        // Joining with lower doesn't regress
        state = state.join(Liveness::Dead);
        assert_eq!(state, Liveness::Alive);
    }

    #[test]
    fn test_deep_graph_propagation() {
        let mut gc = LatticeGC::new();
        let root = gc.add_node(LatticeNode::new(Liveness::Dead));

        // Build a chain of 100 nodes
        let mut prev = root;
        for _ in 0..100 {
            let next = gc.add_node(LatticeNode::new(Liveness::Dead));
            gc.add_edge(prev, next);
            prev = next;
        }

        gc.collect(&[root]);
        assert_eq!(gc.node_count(), 101); // all alive, none reclaimed
        assert!(gc.is_alive(prev)); // last node reachable
    }

    #[test]
    fn test_node_edges() {
        let mut node = LatticeNode::new(Liveness::Alive);
        node.add_edge(1);
        node.add_edge(2);
        node.add_edge(1); // duplicate — ignored
        assert_eq!(node.edges, vec![1, 2]);

        node.remove_edge(1);
        assert_eq!(node.edges, vec![2]);
    }
}
