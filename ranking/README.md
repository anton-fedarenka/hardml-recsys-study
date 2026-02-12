# Task description

Creating a recommener system for ad recommendations on a marketplace.\
The goal is to improve the algorithm that will determine the most relevant recommendations 
for each user based on user and product attributes, as well as the history of interactions between them.

Initially, a combination of the ALS algorithm and an additional classification model (CatBoost) 
for re-ranking was tested to address this issue, as well as various special approaches 
to grouping and combining data to improve ranking quality.
However, it turned out that the standard BPR algorithm yielded significantly better results in this case. 