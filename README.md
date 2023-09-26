# RecSys
Recommendation system using collaborative filtering

## Recommendation systems
RecSys is an integral part of industries right from food delivery to sales driven by the recommendations coming from these models. Not only do they help businesses grow, but they also help individuals discover things they like that they may not have encountered before.

Common architecture of recommendation systems:

![Architecture](https://github.com/imBLISP/RecSys/blob/main/images/Architecture.png)

### Retrieval
When we get a query embedding $q$, we search for item embeddings $V_i$ that are close to $q$ in the embedding space. This is a KNN problem.

Similarity measures:
 - DOT: $\langle x, y \rangle = \sum_{i = 1}^d x_i y_i$
 - COSINE: $\frac{a^T b}{\|a\| \cdot \|b\|}$

### Scoring
After retrieval of candidates another model ranks the candidates using additional features.
### Re-ranking
Re-ranking filters the candidates even further to maintain freshness, diversity and fairness.
## Collaborative filtering
 Collaborative filtering is a type of recommendation system that suggests items to users based on the similarities and patterns in the behavior of a group of users. In other words, it recommends items that users with similar preferences have liked or interacted with.

### Matrix factorization
Matrix factorization is a simple embedding model. Given the feedback matrix $\mathrm{A} \in R^{m \times n}$, where $m$ is the number of users (or queries) and $n$ is the number of items, the model learns:
- A user embedding matrix $$U \in \mathbb{R}^{m \times d}$$, where row $i$ is the embedding for user $i$.
- An item embedding matrix $$V \in \mathbb{R}^{n \times d}$$, where row $j$ is the embedding for item $j$. The image below depicts this:

![Matrix factorization](https://github.com/imBLISP/RecSys/blob/main/images/matrix%20factorization.png)

## Optimizers

### Stochastic gradient descent
Stochastic gradient descent (SGD) is a generic method to minimize loss functions.


### Alternating least squares
Weighted Alternating Least Squares (WALS) is specialized to this particular objective. The objective is quadratic in each of the two matrices UU and VV. (Note, however, that the problem is not jointly convex.) WALS works by initializing the embeddings randomly, then alternating between:
   - Fixing $U$ and solving for $V$.
   - Fixing $V$ and solving for $U$.
   
Each stage can be solved exactly (via solution of a linear system) and can be distributed. This technique is guaranteed to converge because each step is guaranteed to decrease the loss.

## Objective functions
Squared distance objective function which calculates the squared distance between two embeddings.

$\min _{U \in \mathbb{R}^{m \times d}, V \in \mathbb{R}^{n \times d}} \sum_{(i, j) \in \mathrm{obs}}\left(A_{i j}-\left\langle U_{i}, V_{j}\right\rangle\right)^{2}$
