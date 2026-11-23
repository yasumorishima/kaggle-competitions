
// Explore equal-length programs with different intermediate values.
    bool diversify(Graph& graph, size_t cap, int wanted) {
        std::vector<Value> order;
        if (!order_for(graph, targets, order)) return false;
        graph = live_only(graph, order);
        std::vector<Value> numbers = starts;
        numbers.insert(numbers.end(), order.begin(), order.end());
        std::shuffle(numbers.begin(), numbers.end(), rng);
        std::unordered_map<Value, std::vector<Recipe>> makers;
        for (size_t i = 0; i < numbers.size(); ++i) {
            for (size_t j = 0; j <= i; ++j) {
                Value a = std::max(numbers[i], numbers[j]), b = std::min(numbers[i], numbers[j]);
                for (char op : {'+', '-', '*', '/'}) {
                    Value v = calculate(op, a, b);
                    if (v && v != a && v != b && !graph.count(v)) makers[v].push_back({op, a, b});
                }
            }
        }
        std::shuffle(order.begin(), order.end(), rng);
        int swaps = 0;
        for (Value v : order) {
            if (!graph.count(v)) continue;
            Recipe old = graph.at(v);
            std::unordered_map<Value,int> used;
            for (auto root : targets) ++used[root];
            for (auto [output, step] : graph) if (step.op) {
                ++used[step.a];
                if (step.a != step.b) ++used[step.b];
            }
            bool has_private = (graph.at(old.a).op && used[old.a] == 1) ||
                               (graph.at(old.b).op && used[old.b] == 1);
            auto replace = [&](Recipe step, Value extra, Recipe build) {
                if (step.a == v || step.b == v) return false;
                if (extra) {
                    if (extra == v || graph.count(extra) || !graph.count(build.a) || !graph.count(build.b)) return false;
                    graph.emplace(extra, build);
                }
                graph[v] = step;
                std::vector<Value> live;
                bool valid = order_for(graph, targets, live) && live.size() <= cap;
                if (valid) graph = live_only(graph, live);
                else {
                    graph[v] = old;
                    if (extra) graph.erase(extra);
                }
                return valid;
            };
            bool changed = false;
            int attempts = 0;
            size_t shift = rng() % numbers.size();
            for (size_t j = 0; j < numbers.size() && !changed; ++j) {
                Value fixed = numbers[(shift+j) % numbers.size()];
                if (fixed == v || !graph.count(fixed)) continue;
                for (auto step : inverses(v, fixed)) {
                    Value need = step.a == fixed ? step.b : step.a;
                    if (need == v || (step.op == old.op &&
                        ((step.a == old.a && step.b == old.b) ||
                         ((step.op=='+' || step.op=='*') && step.a==old.b && step.b==old.a)))) continue;
                    if (graph.count(need)) {
                        if (rng()%4 == 0 && replace(step, 0, {})) {changed=true;break;}
                    } else if (has_private) {
                        auto it = makers.find(need);
                        if (it == makers.end()) continue;
                        auto &opts = it->second;
                        const auto r = opts[rng() % opts.size()];
                        if (r.a==v || r.b==v || (used[old.a]==1 && (r.a==old.a||r.b==old.a||fixed==old.a))
                            || (used[old.b]==1 && (r.a==old.b||r.b==old.b||fixed==old.b))) continue;
                        if (replace(step, need, r)) {changed=true;break;}
                        if (++attempts >= 12) break;
                    }
                }
                if (attempts>=12) break;
            }
            if (changed && ++swaps >= wanted) break;
        }
        return swaps > 0;
    }
