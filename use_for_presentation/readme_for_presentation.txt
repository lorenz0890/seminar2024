 ------------------------------------
|ON THE PLOTS CONTANED IN THIS FOLDER|
 ------------------------------------

CASE STUDY SETUP:
.) hypothesis: expressivity in the sense of the ability of distinguishing non isomorphic graphs by distinguishable
node embeddings is a key requirement to successfully learn molecular property prediction with GNNs

.) 1-wl color clusters computed based on integer encodings derived from initial node features.
each unique feature vector corresponds to an integer (heterogeneous initialization,
opposed to homgeneous init of 1-WL with node features set to one).

.) training was conducted on 80% random train split with 3 layer GIN/GCN and adam optimizer at lr=0.01 for 20 epochs.
embeddings were extreacted at the end of each epoch.

.) intra cluster cosine similarity: cosine similarity between node embeddings of the same WL color of the graph.
inter cluster cosine similarity: cosine similarity between nodes of differen WL color of the graph.
similarities are calculated per graph and averaged over the whole on the whole dataset for plots.

.) datasets are molecular property prediction datasets (binary graph classification)
Important note: in KKI with heterogeneous initialization, every node in every graph
has a unique color after one wl iteration already.
therefore, intra cluster similarity is 0 in that case!

KEY TAKE AWAYS
.) training accuracy is anti correlated with inter cluster similarity.
that is, the more distinguishable embeddings of structurally different nodes are, the
more accurate the prediction during training gets.

.) intra cluster similarity remained almost unchanged during training.
(changes are present but so small they are not visible in the plots.)

CONCLUSION
The better the GNNs evaluated learn to predict the labels of the training data,
the more they tend to produce emebddings that distinguish node embeddings of
nodes with different WL color. Nodes of same WL color barley change their similarity.
Thus, the hypothesis that expressivity in the sense of distinguishing non isomorphic graphs
by distinguishable node embeddings is a key requirement to molecular property prediction with GNNs
holds in the sense that it is not contradicted by the empirical observtions.


