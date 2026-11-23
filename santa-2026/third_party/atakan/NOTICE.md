These C++ files are Atakan Aldemir's public optimizers, extracted unchanged from the Kaggle notebook
"Santa 2026 | Joint Recipes for Large Puzzles" (https://www.kaggle.com/code/atakanaldemir/santa-2026-joint-recipes-for-large-puzzles),
published on Kaggle under the Apache License 2.0. All credit for these optimizers belongs to the original author.
elf_whole_graph.cpp and elf_global_schedule.cpp come from version 16 of the same notebook (the newest bundle includes them but omits the file).
We made no changes. Build: g++ -O3 -std=c++17 elf_multiplier_frontier.cpp -o frontier ; g++ -O3 -std=c++17 elf_interface_closure.cpp -o closure
Driver: atakan_driver.py in this dataset (it only feeds our programs in and replays every result with our referee).
