#[derive(Debug)]
pub(super) struct ConjunctTrie {
    nodes: Vec<ConjunctTrieNode>,
}

#[derive(Debug, Default)]
struct ConjunctTrieNode {
    value: Option<&'static str>,
    edges: Vec<ConjunctTrieEdge>,
}

#[derive(Debug, Clone, Copy)]
struct ConjunctTrieEdge {
    byte: u8,
    node: usize,
}

impl ConjunctTrie {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(ConjunctTrieNode::default());

        Self { nodes }
    }

    pub(super) fn root(&self) -> usize {
        0
    }

    pub(super) fn insert(&mut self, key: &'static str, value: &'static str) {
        let mut node = self.root();

        for byte in key.bytes() {
            node = self.child_or_insert(node, byte);
        }

        assert!(
            self.nodes[node].value.is_none(),
            "duplicate conjunct trie key: {key}"
        );
        self.nodes[node].value = Some(value);
    }

    pub(super) fn sort_edges(&mut self) {
        for node in &mut self.nodes {
            node.edges.sort_unstable_by_key(|edge| edge.byte);
        }
    }

    pub(super) fn advance(&self, node: usize, part: &str) -> Option<usize> {
        let mut current = node;

        for byte in part.bytes() {
            current = self.nodes.get(current)?.child(byte)?;
        }

        Some(current)
    }

    pub(super) fn value(&self, node: usize) -> Option<&'static str> {
        self.nodes.get(node)?.value
    }

    fn child_or_insert(&mut self, node: usize, byte: u8) -> usize {
        if let Some(edge) = self.nodes[node].edges.iter().find(|edge| edge.byte == byte) {
            return edge.node;
        }

        let child = self.nodes.len();
        self.nodes.push(ConjunctTrieNode::default());
        self.nodes[node]
            .edges
            .push(ConjunctTrieEdge { byte, node: child });
        child
    }
}

impl ConjunctTrieNode {
    fn child(&self, byte: u8) -> Option<usize> {
        self.edges
            .binary_search_by_key(&byte, |edge| edge.byte)
            .ok()
            .map(|index| self.edges[index].node)
    }
}
