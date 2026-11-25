triwalk.cpp is goodpjw2008's value-set annealer for Santa 2026, copied unchanged from the Kaggle dataset
"Santa 2026 DAG-LNS programs" (https://www.kaggle.com/datasets/goodpjw2008/santa-2026-dag-lns-programs, version of
2026-11-25), published under the Apache License 2.0. All credit for the engine belongs to the original author; their
notebook "Santa 2026 | Fresh-Program Re-Search" describes it and measures it.
Build: g++ -O3 -std=c++17 triwalk.cpp -o triwalk. Our driver is search/runner.py (strategy "triwalk"); the run options
(--cap/--T/--rb/--deep sets) are the defaults of the author's tri_driver.py.
