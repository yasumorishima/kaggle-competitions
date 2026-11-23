
#include <set>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Value = int64_t;
struct Recipe { char op; Value a, b; };
struct Inverses {
    std::array<Recipe, 6> entries;
    int size = 0;
    void push_back(Recipe recipe) { entries[size++] = recipe; }
    const Recipe* begin() const { return entries.data(); }
    const Recipe* end() const { return entries.data() + size; }
};
struct Command { char op; int a, b, dst; };
using Tape = std::vector<Command>;
using Graph = std::unordered_map<Value, Recipe>;

struct Puzzle {
    Value limit;
    int registers;
    std::vector<Value> starts, targets;
    int rounds, max_neighbors;
    std::mt19937 rng;

    Value calculate(char op, Value a, Value b) const {
        switch (op) {
        case '+': return a <= limit - b ? a + b : 0;
        case '-': return a > b ? a - b : 0;
        case '*': return a <= limit / b ? a * b : 0;
        case '/': return a % b == 0 ? a / b : 0;
        }
        return 0;
    }

    Graph graph_from(const Tape& tape) const {
        Graph graph;
        std::vector<Value> registers_now(registers);
        for (size_t i = 0; i < starts.size(); ++i) {
            registers_now[i] = starts[i];
            graph.emplace(starts[i], Recipe{0, 0, 0});
        }
        for (auto [op, a, b, dst] : tape) {
            Value left = registers_now[a], right = registers_now[b];
            if (!left || !right) return {};
            Value value = calculate(op, left, right);
            if (!value) return {};
            registers_now[dst] = value;
            graph.emplace(value, Recipe{op, left, right});
        }
        return graph;
    }

    bool visit(Value value, const Graph& graph,
               std::unordered_map<Value, int>& state, std::vector<Value>& order) const {
        auto [mark, fresh] = state.emplace(value, 1);
        if (!fresh) return mark->second == 2;
        auto found = graph.find(value);
        if (found == graph.end()) return false;
        auto [op, a, b] = found->second;
        if (op) {
            if (!visit(a, graph, state, order) || !visit(b, graph, state, order)) return false;
            order.push_back(value);
        }
        state[value] = 2;
        return true;
    }

    bool order_for(const Graph& graph, const std::vector<Value>& roots,
                   std::vector<Value>& order) const {
        std::unordered_map<Value, int> state;
        state.reserve(graph.size());
        order.clear();
        for (Value value : roots)
            if (!visit(value, graph, state, order)) return false;
        return true;
    }

    bool allocate(const Graph& graph, const std::vector<Value>& order, Tape& tape) const {
        std::unordered_map<Value, int> uses, slots;
        std::vector<int> free_slots;
        tape.clear();
        for (Value value : order) {
            auto [op, a, b] = graph.at(value);
            ++uses[a];
            ++uses[b];
        }
        for (int i = registers - 1; i >= 0; --i) {
            if (i >= int(starts.size()) || !uses.count(starts[i]) || slots.count(starts[i]))
                free_slots.push_back(i);
            else
                slots[starts[i]] = i;
        }
        for (Value value : order) {
            auto [op, a, b] = graph.at(value);
            int left = slots.at(a), right = slots.at(b);
            if (--uses[a] == 0) free_slots.push_back(left);
            if (--uses[b] == 0) free_slots.push_back(right);
            if (free_slots.empty()) return false;
            int dst = free_slots.back();
            tape.push_back({op, left, right, dst});
            if (uses.count(value)) {
                slots[value] = dst;
                free_slots.pop_back();
            }
        }
        return true;
    }

    bool pressure_order(const Graph& graph, std::vector<Value>& order) const {
        std::unordered_map<Value, int> pending, remaining;
        std::unordered_map<Value, std::vector<Value>> consumers;
        std::vector<Value> ready, scheduled;
        for (Value value : order) {
            auto [op, a, b] = graph.at(value);
            consumers[a].push_back(value);
            consumers[b].push_back(value);
            pending[value] = int(graph.at(a).op != 0) + int(graph.at(b).op != 0);
            if (!pending[value]) ready.push_back(value);
        }
        for (const auto& [value, uses] : consumers) remaining[value] = int(uses.size());
        while (!ready.empty()) {
            auto delta = [&](Value value) {
                auto [op, a, b] = graph.at(value);
                int releases = a == b ? int(remaining[a] == 2)
                                      : int(remaining[a] == 1) + int(remaining[b] == 1);
                return int(consumers.count(value)) - releases;
            };
            auto chosen = std::min_element(ready.begin(), ready.end(),
                                           [&](Value a, Value b) { return delta(a) < delta(b); });
            Value value = *chosen;
            ready.erase(chosen);
            scheduled.push_back(value);
            auto [op, a, b] = graph.at(value);
            --remaining[a];
            --remaining[b];
            auto found = consumers.find(value);
            if (found != consumers.end())
                for (Value next : found->second)
                    if (--pending[next] == 0) ready.push_back(next);
        }
        if (scheduled.size() != order.size()) return false;
        order.swap(scheduled);
        return true;
    }

    bool compile(const Graph& graph, Tape& tape) {
        auto roots = targets;
        std::vector<Value> order;
        if (!order_for(graph, roots, order)) return false;
        if (allocate(graph, order, tape)) return true;
        if (pressure_order(graph, order) && allocate(graph, order, tape)) return true;
        std::reverse(roots.begin(), roots.end());
        for (int attempt = 0; attempt < 18; ++attempt) {
            if (order_for(graph, roots, order) && allocate(graph, order, tape)) return true;
            std::shuffle(roots.begin(), roots.end(), rng);
        }
        return false;
    }

    Inverses inverses(Value value, Value other) const {
        Inverses recipes;
        if (value > other) recipes.push_back({'+', other, value - other});
        if (value <= limit - other) recipes.push_back({'-', value + other, other});
        if (other > value) recipes.push_back({'-', other, other - value});
        if (value % other == 0) recipes.push_back({'*', other, value / other});
        if (value <= limit / other) recipes.push_back({'/', value * other, other});
        if (other % value == 0) recipes.push_back({'/', other, other / value});
        return recipes;
    }

    Graph live_only(const Graph& graph, const std::vector<Value>& order) const {
        Graph live;
        live.reserve(order.size() + starts.size());
        for (Value value : starts) live.emplace(value, Recipe{0, 0, 0});
        for (Value value : order) live.emplace(value, graph.at(value));
        return live;
    }
