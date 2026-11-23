These C++ files are Wang Jakie's public graph-search strategies, extracted unchanged from the Kaggle notebook
"Santa 2026 | Arithmetic Recipe Pool and MILP" (https://www.kaggle.com/code/wangjakie/santa-2026-arithmetic-recipe-pool-and-milp),
published under the Apache License 2.0. All credit for the search belongs to the original author.
Our only changes: anneal_seeded.cpp and incremental_seeded.cpp add one line that puts the extra input tapes
(near-length historical programs) into the crossover pool — an idea first described by E-Max AI (royalacecat).
Build: g++ -O3 -std=c++17 <strategy>.cpp -o <strategy>; driver: wj_driver.py in this dataset.
