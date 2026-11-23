# -*- coding: utf-8 -*-
with open('paper.tex', encoding='utf-8') as f:
    content = f.read()

replacements = []

# 1. Improve float placement so tables/figures land closer to the text that discusses them.
before = content.count('\\begin{table}[t]') + content.count('\\begin{figure}[t]')
content = content.replace('\\begin{table}[t]', '\\begin{table}[!ht]')
content = content.replace('\\begin{figure}[t]', '\\begin{figure}[!ht]')
after = content.count('\\begin{table}[!ht]') + content.count('\\begin{figure}[!ht]')
print(f"float placement: {before} floats updated to [!ht]")

# 2. Nodes paragraph -> itemised feature breakdown
old_nodes = r"\textbf{Nodes.} Every amino-acid residue in the protein becomes one node. Each node's feature vector is 1{,}321 numbers long: the residue's 1{,}280-dimensional ESM-2 embedding, a 20-dimensional one-hot flag for its wild-type (unmutated) amino-acid identity, a single flag that is 1 only for the residue being mutated in this particular variant and 0 elsewhere, and a 20-dimensional one-hot encoding of the substituted amino acid at that mutated position (zero everywhere else in the protein)."

new_nodes = r"""\textbf{Nodes.} Every amino-acid residue in the protein becomes one node. Each node's feature vector has 1{,}321 numbers in total, made up of four parts:
\begin{itemize}
\item a 1{,}280-number ESM-2 embedding for that residue, capturing evolutionary and biochemical context;
\item a 20-number one-hot flag marking the residue's own, unmutated amino-acid identity;
\item a single flag, set to 1 only for the residue being mutated in this particular variant and 0 for every other residue;
\item a 20-number one-hot encoding of the amino acid this position is being substituted to, left at zero for every residue except the mutated one.
\end{itemize}"""

assert old_nodes in content, "NODES BLOCK NOT FOUND"
content = content.replace(old_nodes, new_nodes, 1)
print("nodes paragraph converted to list")

# 3. Edges paragraph -> keep the explanation, itemise the two-number breakdown
old_edges = r"\textbf{Edges.} Two residues are connected by an edge if their alpha-carbon atoms, the standard reference atom for a residue's position, sit within 8\r{A} of each other in the folded 3D structure. This is a widely used convention for representing protein structure as a graph \cite{gligorijevic2021deepfri}. It captures a crucial property a flat sequence cannot: two residues can be functionally close in 3D, for example forming part of the same binding pocket, while being far apart in the linear sequence, because the chain folds back on itself. Each edge carries two numbers in the baseline model: the physical C$\alpha$--C$\alpha$ distance, and a \textbf{PAE-derived confidence weight}, which is this paper's central object of study."

new_edges = r"""\textbf{Edges.} Two residues are connected by an edge if their alpha-carbon atoms, the standard reference atom for a residue's position, sit within 8\r{A} of each other in the folded 3D structure. This is a widely used convention for representing protein structure as a graph \cite{gligorijevic2021deepfri}. It captures something a flat sequence cannot: two residues can sit close together in 3D, for example forming part of the same binding pocket, while being far apart in the linear sequence, because the chain folds back on itself. Each edge carries two numbers in the baseline model:
\begin{itemize}
\item the physical distance between the two residues' alpha-carbon atoms;
\item a PAE-derived confidence weight, this paper's central object of study.
\end{itemize}"""

assert old_edges in content, "EDGES BLOCK NOT FOUND"
content = content.replace(old_edges, new_edges, 1)
print("edges paragraph converted to list")

# 4. Insert the architecture figure and an explicit input/output breakdown right after
#    the "Models compared" bullet list, before "Training setup" begins.
old_anchor = r"""\item \textbf{AlphaMissense}: not retrained, but benchmarked head-to-head on the same test variants using its publicly released scores, to answer RQ3 \cite{cheng2023alphamissense}.
\end{itemize}

\subsection{Training setup}"""

new_anchor = r"""\item \textbf{AlphaMissense}: not retrained, but benchmarked head-to-head on the same test variants using its publicly released scores, to answer RQ3 \cite{cheng2023alphamissense}.
\end{itemize}

\begin{figure*}[!ht]
\centering
\includegraphics[width=0.92\textwidth]{figures/architecture_comparison.png}
\caption{How each of the five models turns one protein into a single pathogenicity prediction. All five read out only the mutated residue's own representation before classifying it; what differs between them is what information is allowed to reach that representation. The two boxed panels are this paper's central comparison: identical architectures except for whether the edges carry AlphaFold's confidence signal.}
\label{fig:architecture}
\end{figure*}

\textbf{Input and output, model by model.} Every model produces the same kind of output: a single number between 0 and 1, its estimate of how likely this specific variant is to be pathogenic. What differs between them is the input each one is allowed to see.
\begin{itemize}
\item \textbf{XGBoost} sees only the mutated residue's own flat feature vector, 1{,}320 numbers made up of its ESM-2 embedding, wild-type one-hot, and alternate one-hot. There is no graph and no information from any neighbouring residue.
\item \textbf{GCN and GraphSAGE} see the whole protein graph: every residue as a node carrying the 1{,}321-number feature vector described above, and every contact within 8\r{A} as an edge. Neither model reads any number attached to an edge, only whether it exists.
\item \textbf{GATv2 (unweighted)} also sees the full graph, but, like GCN and GraphSAGE, ignores any number carried on an edge. Its attention weights are learned purely from node features.
\item \textbf{GATv2 (PAE-weighted)} sees the same graph, except each edge additionally carries the two numbers described above, distance and PAE weight, which the attention mechanism can use directly.
\item \textbf{AlphaMissense} is not given any input by this study at all: its own precomputed, published score is looked up for each test variant.
\end{itemize}

\subsection{Training setup}"""

assert old_anchor in content, "ANCHOR BLOCK NOT FOUND"
content = content.replace(old_anchor, new_anchor, 1)
print("figure + input/output list inserted")

with open('paper.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("all edits applied")
