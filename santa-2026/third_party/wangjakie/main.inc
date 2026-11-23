
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int count;
    if (!(std::cin >> count)) return 1;
    for (int puzzle_index = 0; puzzle_index < count; ++puzzle_index) {
        Puzzle puzzle;
        int size, seed_count;
        unsigned seed;
        std::cin >> puzzle.limit >> puzzle.registers >> puzzle.rounds >> puzzle.max_neighbors >> seed;
        puzzle.rng.seed(seed);
        std::cin >> size;
        puzzle.starts.resize(size);
        for (auto& value : puzzle.starts) std::cin >> value;
        std::cin >> size;
        puzzle.targets.resize(size);
        for (auto& value : puzzle.targets) std::cin >> value;
        std::cin >> seed_count;
        std::vector<Tape> tapes(seed_count);
        for (auto& tape : tapes) {
            std::cin >> size;
            tape.resize(size);
            for (auto& [op, a, b, dst] : tape) std::cin >> op >> a >> b >> dst;
        }
        if (!std::cin || tapes.empty()) return 2;
        auto best = puzzle.improve(tapes);
        std::cout << best.size();
        for (auto [op, a, b, dst] : best) std::cout << ' ' << op << ' ' << a << ' ' << b << ' ' << dst;
        std::cout << '\n';
    }
}
