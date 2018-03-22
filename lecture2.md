class: middle, center, title-slide

# Advanced Computing Techniques

Lecture 2: Trees and ensembles

---

# Real world data

* non linear problems?
* redundant features?
* categorical features?
* different scales per feature?

Today is about Random forests and gradient boosted trees.

---

class: middle, center

# Decision trees

(Chapter 3)

---

# A decision tree

What should we do today?

.width-100[![](images/decision_tree.png)]


---

# Dataset

.center.width-80[![](images/moon_data.png)]

---

# One level of splits

.center.width-80[![](images/dt_depth1.png)]

---

# One level of splits

.center.width-80[![](images/dt_structure_depth1.svg)]

---

# Two levels of splits

.center.width-80[![](images/dt_depth2.png)]

---

# Two levels of splits

.center.width-80[![](images/dt_structure_depth2.svg)]

---

# Three levels of splits

.center.width-80[![](images/dt_depth3.png)]

---

# Three levels of splits

.center.width-80[![](images/dt_structure_depth3.svg)]

---

# Ten levels of splits

.center.width-80[![](images/dt_depth10.png)]

---

# Ten levels of splits

.center.width-80[![](images/dt_structure_depth10.svg)]

---

# Making predictions

.center.width-80[![](images/dt_structure_depth2.svg)]

Put new sample through our tree. When we arrive at a leaf predict that this
sample belongs to the majority class.

$P(R | X) = \frac{R}{B+R}$ and $P(B | X) = \frac{B}{B+R}$

Trees are very fast to execute at prediction/inference time.

---

# How to grow a tree?

The goal is to create leaves that are pure so we can use majority class as
prediction for new samples.

* finding the optimal partioning of the input space is (in general) not feasible
* grow tree in a greedy (one step at a time) fashion!

Algorithm:
1. Pick a node, check if node is pure
    * if yes mark it as a leaf
    * else:
        1. find split point $S$ for feature $j$ that leads to largest decrease of impurity
        1. split node into two child nodes according to this split
2. go to 1.

---

# Two questions, which one is better?

How full is the train? vs What is the weather like?

.center.width-100[![](images/delayed-trains-tree-splitting.png)]

Trying to predict if a train will be .redbg[delayed] or .bluebg[on time].

---

# Node splitting

For each node find the split point $S$ on feature $j$ that leads to largest
decreases of the impurity.

$$\Delta = I\_T - \frac{n\_L}{N} I\_L - \frac{n\_R}{N} I\_R$$

* $I_T$: impurity of current node
* $I_L$: impurity of left child node
* $I_R$: impurity of right child node

$I_L$ and $I_R$ depend on the feature $j$ and split point $S$ chosen to split
samples in the current node.

---

# Measure impurity of a node

Two common measures for classification tasks:

* cross-entropy: $- \sum\_{k} p\_{mk} \log(p\_{mk})$
* Gini index: $\sum\_k p\_{mk}(1-p\_{mk})$

.center.width-50[![](images/cross-entropy.png)]

---

# Tree stops growing when nodes are pure

.center.width-80[![](images/dt_structure_depth10.svg)]

---

# Tree structure

Growing a fully developed tree generally does not lead to good generalisation.

Either limit tree growth or prune tree after it has been grown.

Implemented in scikit-learn:
* `max_depth`
* `max_leaf_nodes`
* `min_samples_split`
* `min_impurity_decrease`

---

# No limit on tree growth

.center.width-80[![](images/dt_structure_depth10.svg)]

---

# `max_depth = 3`

.center.width-80[![](images/dt_structure_depth3.svg)]

---

# `max_leaf_nodes = 8`

.center.width-50[![](images/dt_structure_max_leaf_nodes8.svg)]

---

# `min_samples_split = 40`

.center.width-60[![](images/dt_structure_min_samples_split40.svg)]

---

# Regression trees

Prediction:

$$ \bar{y}\_m = \frac{1}{N\_m} \sum_{i \in N\_m} y\_i $$

Impurity measures:
  * mean squared error: $$\frac{1}{N\_m} \sum\_{i \in N\_m} (y\_i - \bar{y}\_m)^2 $$
  * mean absolute error: $$\frac{1}{N\_m} \sum\_{i \in N\_m} |y\_i - \bar{y}\_m|$$

---

# Trees have high variance

Two trees fitted on two random subsets of the same data

<div style="display: flex">
 .pull-left.width-100[![](images/dt_unstable1.svg)]
 .pull-right.width-100[![](images/dt_unstable2.svg)]
</div>

---

class: middle, center

# Interlude (decision trees, tune max_depth)
show how accuracy plateaus after a while

---

class: middle, center

# Random Forests

(Chapter 4)

---

# Wisdom of crowds

A crowd of non experts can give you a very good estimate if you ask each
individually.

.center.width-80[![](images/nathan-fertig-64724-unsplash-small.jpg)]

---

# Wisdom of classifier crowds

Combine several uncorrelated models together:

<div style="display: flex">
 .width-100[![](images/voting_lr.png)]
 .width-100[![](images/voting_dt.png)]
 .width-100[![](images/voting_combined.png)]
</div>

Combine predictions from logistic regression and a decision tree.

* Accuracy for LogisticRegression: 0.84
* Accuracy for DecisionTree: 0.80
* Accuracy for combination: 0.88

---

# The key: uncorrelated models

How could we build models that are uncorrelated with each other?

---

# The key: uncorrelated models

How could we build models that are uncorrelated with each other?

* only show a subset of the data to each model
* only consider a subset of the features at each split

Bootstrap (sample with replacement):

.center.width-90[![](images/bootstrap.png)]

Select subset of features:

.center.width-90[![](images/sample-features.png)]

---

# Random forests

Many trees, decorrelate via bootstrap and sampling features at each split.

.center[
<div style="display: flex">
 .width-90[![](images/rf_trees_0.png)]
 .width-90[![](images/rf_trees_1.png)]
 .width-90[![](images/rf_trees_2.png)]
</div>
]

.center[
 .width-30[![](images/rf_all_trees.png)]
]
---

# Feature importances

Weighted mean decrease of impurity.

.center.width-60[![](images/feature_importances.png)]

.footnote[G. Louppe, Understanding Random Forests,
  https://github.com/glouppe/phd-thesis
]

---

# Tuning random forests

* Main parameter: max_features
    * around `sqrt(n_features)` for classification
    * around `n_features` for regression
* `n_estimators` > 100
* Restricting tree growth might help, definitely helps with model size!
    * `max_depth`, `max_leaf_nodes`, `min_samples_split`


---

class: middle, center

# Gradient boosting

(Chapter 4)

---

# Step by step example

$$f_1(x) \approx y$$

$$f_2(x) \approx y - f_1(x)$$

$$f_3(x) \approx y - f_1(x) - f_2(x)$$

At each step you try and fix the mistakes made by the previous model. The
model is fitted to the residuals of the previous step.

---

# Regression example

.center.width-70[![](images/gradient_boost_true.png)]

---

# Stage 0

.center.width-70[![](images/gbr_stage_0.png)]

---

# Stage 0 - residuals

.center.width-70[![](images/gbr_stage_residuals_1.png)]

---

# Stage 1

.center.width-70[![](images/gbr_stage_1.png)]

---

# Stage 1 - residuals

.center.width-70[![](images/gbr_stage_residuals_2.png)]

---

# Stage 2

.center.width-70[![](images/gbr_stage_2.png)]

---

# Stage 2 - residuals

.center.width-70[![](images/gbr_stage_residuals_3.png)]

---

# Stage 3

.center.width-70[![](images/gbr_stage_3.png)]

---

# Full model

.center.width-70[![](images/gbr_data.png)]

---

# Step by step example

$$f_1(x) \approx y$$

$$f_2(x) \approx y - \alpha f_1(x)$$

$$f_3(x) \approx y - \alpha f_1(x) - \alpha f_2(x)$$

$$f_4(x) \approx y - \alpha f_1(x) - \alpha f_2(x) - \alpha f_3(x)$$

$$ ... $$

At each step you try and fix the mistakes made by the previous model. The
model is fitted to the residuals of the previous step.

In practice you want to make small adjustments as you go along, each
model is multiplied by $\alpha$. This is also referred to as "shrinkage"
or "learning rate".

---

class: middle, center

# Interlude 2

???

show that you can overfit if you add more and more trees. Unlike RF.

Pointer to Catboost and dynamic boost, claims to not overfit.

---

# Feature importances

Weighted "mean decrease of impurity".

.center.width-60[![](images/feature_importances.png)]

.footnote[G. Louppe, Understanding Random Forests,
  https://github.com/glouppe/phd-thesis
]

---

# Partial dependence plots

.center.width-70[![](images/pdp-wine.png)]

---

# Partial dependence plots

.center.width-100[![](images/pdp-wine-2d.png)]

---

# Tuning gradient boosting

* `max_features` tends to be small
* smaller learning rate requires more trees
* tune with ~1000 trees (or as big as you have patience)

.center.width-80[![](images/gbrt-recipe.png)]

Once you have good parameters increase `n_estimators` and decrease learning rate.

---

# XGBoost

Fully scikit-learn compatible, but faster!

```
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test))
```

Install it with `conda install -c conda-forge xgboost`.

Used by a lot of people who are "serious" about gradient boosted trees.

---

# LightGBM

Gradient boosting framework develped by Microsoft and it is open-source!
Supports parallel and GPU learning.

```python
import lightgbm as lgb

estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:',
      gbm.best_params_)
```

Set of [benchmarks comparing xgboost and lightgbm](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment).

Looks a bit tricky to install :-/

---

# Viola Jones for face detection

If you want to see the idea of boosting in action in a different context checkout
the Viola-Jones object detection algorithm.

.center.width-100[![](images/viola-jones.jpg)]

---

# Stacking

Why limit yourself to combining trees?

Stacking, or combining different types of models.
```python
voting = VotingClassifier([('logreg',
                            LogisticRegression(C=100)),
                           ('tree',
                            DecisionTreeClassifier(max_depth=5)),
                           ('knn',
                            KNeighborsClassifier(n_neighbors=3))
                          ],
                         voting='soft', flatten_transform=True)
voting.fit(X_train, y_train)
```

---

# Combine models by averaging

<div style="display: flex">
.width-100[![](images/voting_Logistic%20Regression.png)]
.width-100[![](images/voting_KNN.png)]
.width-100[![](images/voting_Decision Tree.png)]
</div>
.center.width-40[![](images/voting_Average.png)]

Can't we learn the weights?

---

# Combine models via LogisticRegression

```python
# `voting` is our original voting classifier,
# when you call `transform()` on it it produces
# class probabilites
stacking = make_pipeline(voting,
                         # only keep probabilites for one class
                         FunctionTransformer(lambda X: X[:, 1::2]),
                         # fit a logistic regression model
                         LogisticRegression())
stacking.fit(X_train, y_train)
print(stacking.score(X_train, y_train))
# -> 0.92
print(stacking.score(X_test, y_test))
# -> 0.85
```

What is the problem now?

---

# Need unbiased predictions

Fit the original models on a subset of the data, predict on the rest. This
way you get unbiased predictions for all of the data.

.width-40[![](images/5fold-cv.png)]
<span style="padding-left: 1em;">
.blackbg.black[pre] = predict and .whitebg.white[fit] = fit.
</span>

```
from sklearn.model_selection import cross_val_predict

first_stage = make_pipeline(voting,
                            FunctionTransformer(
                                lambda X: X[:, 1::2])
                            )
transform_cv = cross_val_predict(first_stage, X_train, y_train,
                                 cv=10, method="transform")
```

---

# Full stacking

```
from sklearn.model_selection import cross_val_predict

first_stage = make_pipeline(voting,
                            FunctionTransformer(
                                lambda X: X[:, 1::2])
                            )
# `transform_cv` will contain unbiased predictions
# for each sample
transform_cv = cross_val_predict(first_stage, X_train, y_train,
                                 cv=5, method="transform")

second_stage = LogisticRegression().fit(transform_cv, y_train)
print(second_stage.coef_)
```

```
print(second_stage.score(transform_cv, y_train))
# -> 0.82
print(second_stage.score(first_stage.transform(X_test), y_test))
# -> 0.85
```
---

# Summary

* non linear problems? No problem!
* redundant features? No problem!
* categorical features? No problem!
* different scales per feature? No problem!

Random forests should be in your baseline. There is essentially no reason
not to use them. Might not produce the absolute best solution.

Gradient boosted trees is the go to solution for most real world
problems. Needs some careful tuning.

Reading:
* Random forests: chapter 15 of "Elements of Statistical Learning"
* Boosted trees: chapter 10.9 of "Elements of Statistical Learning"
