#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Value = std::uint64_t;
constexpr int slots = 48;
using Registers = std::array<Value, slots>;
struct Op { char kind; int a, b, destination; };
struct Goal { Value value; int destination; };
using Program = std::vector<Op>;
using Index = std::unordered_map<Value, int>;
Index index_values(const Registers& regs);
bool one_step(const Index& known, Value value, Value limit, int destination, Op& result);

Value calculate(char op, Value a, Value b, Value limit) {
    if (a == 0 || b == 0) return 0;
    switch (op) {
        case '+': return a <= limit - b ? a + b : 0;
        case '-': return a > b ? a - b : 0;
        case '*': return a <= limit / b ? a * b : 0;
        case '/': return a % b == 0 ? a / b : 0;
        default: return 0;
    }
}

Value execute(Registers& regs, Op op, Value limit) {
    Value value = calculate(op.kind, regs.at(op.a), regs.at(op.b), limit);
    if (!value) throw std::runtime_error("illegal instruction");
    regs.at(op.destination) = value;
    return value;
}

Program elide_helpers(const Registers& initial, const std::vector<Value>& targets,
                      Value limit, Program program, int rounds) {
    std::unordered_set<Value> required(targets.begin(), targets.end());
    for (Value value : initial) required.erase(value);
    for (int round = 0; round < rounds; ++round) {
        int n = static_cast<int>(program.size());
        std::vector<Registers> trace(n + 1, initial);
        std::vector<Value> writes(n);
        std::vector<std::uint64_t> live(n + 1);
        std::unordered_map<Value, int> counts;
        for (int i = 0; i < n; ++i) {
            trace[i + 1] = trace[i];
            writes[i] = execute(trace[i + 1], program[i], limit);
            if (required.count(writes[i])) ++counts[writes[i]];
        }
        for (int i = n - 1; i >= 0; --i) {
            auto op = program[i];
            live[i] = (live[i + 1] & ~(Value{1} << op.destination)) |
                      (Value{1} << op.a) | (Value{1} << op.b);
        }
        bool accepted = false;
        for (int begin = n - 1; begin >= 0; --begin) {
            auto target_count = counts.find(writes[begin]);
            if (target_count != counts.end() && target_count->second <= 1) continue;
            Registers regs = trace[begin];
            std::unordered_map<Value, int> skipped;
            if (target_count != counts.end()) ++skipped[writes[begin]];
            Program replacement;
            std::uint64_t changed = regs[program[begin].destination] == writes[begin]
                                  ? 0 : Value{1} << program[begin].destination;
            int stop = begin + 1;
            bool legal = true;
            for (; stop < n && (changed & live[stop]); ++stop) {
                Op op = program[stop];
                Value result = calculate(op.kind, regs[op.a], regs[op.b], limit);
                if (result != writes[stop] &&
                    !one_step(index_values(regs), writes[stop], limit, op.destination, op)) {
                    if (required.count(writes[stop]) && ++skipped[writes[stop]] >= counts[writes[stop]]) {
                        legal = false;
                        break;
                    }
                } else {
                    execute(regs, op, limit);
                    replacement.push_back(op);
                }
                Value bit = Value{1} << op.destination;
                if (regs[op.destination] == writes[stop]) changed &= ~bit;
                else changed |= bit;
            }
            if (!legal) continue;
            Program next(program.begin(), program.begin() + begin);
            next.insert(next.end(), replacement.begin(), replacement.end());
            next.insert(next.end(), program.begin() + stop, program.end());
            if (next.size() >= program.size()) throw std::logic_error("elision did not shrink program");
            program = std::move(next);
            accepted = true;
            break;
        }
        if (!accepted) break;
    }
    return program;
}

Index index_values(const Registers& regs) {
    Index index;
    index.reserve(slots);
    for (int i = 0; i < slots; ++i)
        if (regs[i]) index.emplace(regs[i], i);
    return index;
}

// Enumerate possible values for x in a final arithmetic operation involving b.
template<class F>
bool inverse(Value t, Value b, Value limit, F match) {
    return (t > b && match(t - b, '+', true)) ||
           (t <= limit - b && match(t + b, '-', true)) ||
           (b > t && match(b - t, '-', false)) ||
           (t % b == 0 && match(t / b, '*', true)) ||
           (t <= limit / b && match(t * b, '/', true)) ||
           (b % t == 0 && match(b / t, '/', false));
}

bool one_step(const Index& known, Value value, Value limit, int destination, Op& result) {
    for (auto [b, ri] : known) {
        if (inverse(value, b, limit, [&](Value x, char c, bool x_first) {
            auto it = known.find(x);
            if (it == known.end()) return false;
            int xi = it->second;
            result = x_first ? Op{c, xi, ri, destination} : Op{c, ri, xi, destination};
            return true;
        })) return true;
    }
    return false;
}

struct Node { Value value; char op; int a, b; };

struct Graph {
    std::vector<Node> nodes;
    std::vector<int> targets;
    std::array<int, slots> initial;
};

Graph recover_graph(Registers regs, Value limit, const std::vector<Value>& targets,
                    const Program& program) {
    Graph graph;
    graph.initial.fill(-1);
    std::unordered_map<Value, int> by_value;
    std::unordered_set<Value> remaining(targets.begin(), targets.end());
    for (int r = 0; r < slots; ++r) if (regs[r]) {
        auto [it, added] = by_value.emplace(regs[r], static_cast<int>(graph.nodes.size()));
        if (added) graph.nodes.push_back({regs[r], 0, -1, -1});
        graph.initial[r] = it->second;
        remaining.erase(regs[r]);
    }
    for (auto op : program) {
        int left = by_value.at(regs[op.a]), right = by_value.at(regs[op.b]);
        Value result = execute(regs, op, limit);
        auto [it, added] = by_value.emplace(result, static_cast<int>(graph.nodes.size()));
        if (added) graph.nodes.push_back({result, op.kind, left, right});
        if (remaining.erase(result)) graph.targets.push_back(it->second);
    }
    return graph;
}

class Scheduler {
    const Graph& graph;
    const int cap;
    const Value limit;
    const int mode;
    const bool flip;
    std::vector<int> plan;
    std::vector<std::vector<int>> uses;
    std::vector<int> location, pinned, cost;
    std::array<int, slots> bank;
    int pos = 0;
    Program code;

    void visit(int id, std::vector<bool>& seen) {
        if (seen[id]) return;
        seen[id] = true;
        const auto& n = graph.nodes[id];
        if (!n.op) return;
        int left = n.a, right = n.b;
        if ((cost[left] < cost[right]) != flip) std::swap(left, right);
        visit(left, seen);
        visit(right, seen);
        plan.push_back(id);
    }

    int next_slot() const {
        int best = -1;
        double lowest = 1e30;
        for (int r = 0; r < slots; ++r) {
            int id = bank[r];
            if (id < 0) return r;
            if (!graph.nodes[id].op || pinned[id]) continue;
            const auto& future = uses[id];
            auto it = std::upper_bound(future.begin(), future.end(), pos);
            if (it == future.end()) return r;
            double distance = *it - pos;
            double expense = cost[id];
            if (mode == 0) expense = 1;
            if (mode == 2) expense *= expense;
            if (mode == 3) distance *= distance;
            double risk = expense / distance;
            if (risk < lowest) { lowest = risk; best = r; }
        }
        return best;
    }

    bool emit(int id, Op op) {
        if (static_cast<int>(code.size()) + 1 >= cap) return false;
        int dest = next_slot();
        if (dest < 0) return false;
        if (bank[dest] >= 0) location[bank[dest]] = -1;
        bank[dest] = id;
        location[id] = dest;
        op.destination = dest;
        code.push_back(op);
        return true;
    }

    bool ensure(int id) {
        if (location[id] >= 0) return true;
        auto [value, kind, a, b] = graph.nodes[id];
        if (!kind) return false;
        if ((location[a] < 0 || location[b] < 0) && mode != 0) {
            Registers regs{};
            for (int r = 0; r < slots; ++r)
                if (bank[r] >= 0) regs[r] = graph.nodes[bank[r]].value;
            Op alternate;
            if (one_step(index_values(regs), value, limit, 0, alternate)) return emit(id, alternate);
        }
        int first = a, second = b;
        if ((cost[first] < cost[second]) != flip) std::swap(first, second);
        if (!ensure(first)) return false;
        ++pinned[first];
        bool ready = ensure(second);
        --pinned[first];
        return ready && emit(id, {kind, location[a], location[b], 0});
    }

public:
    Scheduler(const Graph& graph, int cap, Value limit, int mode, bool flip)
        : graph(graph), cap(cap), limit(limit), mode(mode), flip(flip),
          uses(graph.nodes.size()), location(graph.nodes.size(), -1),
          pinned(graph.nodes.size()), cost(graph.nodes.size()), bank(graph.initial) {
        for (int r = 0; r < slots; ++r) if (bank[r] >= 0) location[bank[r]] = r;
        for (int id = 0; id < static_cast<int>(graph.nodes.size()); ++id) {
            const auto& n = graph.nodes[id];
            if (n.op) cost[id] = std::min(1000, cost[n.a] + (n.a == n.b ? 0 : cost[n.b]) + 1);
        }
    }

    Program run(const std::vector<int>& order) {
        std::vector<bool> seen(graph.nodes.size());
        for (int id : order) visit(id, seen);
        if (static_cast<int>(plan.size()) >= cap && mode == 0) return {};
        for (int i = 0; i < static_cast<int>(plan.size()); ++i) {
            auto& n = graph.nodes[plan[i]];
            uses[n.a].push_back(i);
            if (n.b != n.a) uses[n.b].push_back(i);
        }
        for (; pos < static_cast<int>(plan.size()); ++pos) if (!ensure(plan[pos])) return {};
        return code;
    }
};

Program reschedule(const Registers& initial, const std::vector<Value>& targets,
                   Value limit, Program best, int trials) {
    const Graph graph = recover_graph(initial, limit, targets, best);
    if (graph.targets.size() < 2) return best;
    std::mt19937 rng(static_cast<unsigned>(limit ^ best.size()));
    auto best_order = graph.targets;
    const auto original_order = graph.targets;
    for (int trial = 0; trial < trials; ++trial) {
        auto order = best_order;
        if (trial == 8) std::reverse(order.begin(), order.end());
        if (trial == 9) std::sort(order.begin(), order.end(), [&](int a, int b) {
            return graph.nodes[a].value < graph.nodes[b].value;
        });
        if (trial == 10) std::sort(order.begin(), order.end(), [&](int a, int b) {
            return graph.nodes[a].value > graph.nodes[b].value;
        });
        if (trial > 10) {
            if (trial % 12 == 11) order = original_order;
            const int a = static_cast<int>(rng() % order.size());
            const int b = static_cast<int>(rng() % order.size());
            if (trial % 5 == 0) std::shuffle(order.begin(), order.end(), rng);
            else if (trial % 3 == 0) std::swap(order[a], order[b]);
            else if (a < b) std::rotate(order.begin() + a, order.begin() + a + 1, order.begin() + b + 1);
            else if (a > b) std::rotate(order.begin() + b, order.begin() + a, order.begin() + a + 1);
        }
        auto result = Scheduler(graph, static_cast<int>(best.size()), limit, trial % 4, (trial / 4) % 2).run(order);
        if (!result.empty() && result.size() < best.size()) {
            best = std::move(result);
            best_order = std::move(order);
        }
    }
    return best;
}

Graph topological_graph(const Graph& graph) {
    Graph ordered;
    ordered.initial.fill(-1);
    std::vector<int> mapping(graph.nodes.size(), -1);
    std::function<int(int)> visit = [&](int id) {
        if (mapping[id] >= 0) return mapping[id];
        Node node = graph.nodes[id];
        if (node.op) {
            node.a = visit(node.a);
            node.b = visit(node.b);
        }
        mapping[id] = static_cast<int>(ordered.nodes.size());
        ordered.nodes.push_back(node);
        return mapping[id];
    };
    for (int i = 0; i < slots; ++i)
        if (graph.initial[i] >= 0) ordered.initial[i] = visit(graph.initial[i]);
    for (int target : graph.targets) ordered.targets.push_back(visit(target));
    return ordered;
}

struct Replacement {
    int id = -1, saving = 0;
    Node next{}, helper{}, other{};
    std::uint32_t priority = 0;
};

class GraphRewriter {
    Graph& graph;
    const Value limit;
    std::vector<int> refs;
    int active = 0;
    bool allow_equal = false;
    std::mt19937 random;

    void update(int id, int delta) {
        int before = refs[id];
        refs[id] += delta;
        const auto& node = graph.nodes[id];
        if (node.op && (before == 0 || refs[id] == 0)) {
            active += delta;
            update(node.a, delta);
            update(node.b, delta);
        }
    }

    void measure(Replacement& best, int id, Node recipe, Node helper = {}, Node other = {}) {
        const int previous = active;
        const auto old = graph.nodes[id];
        if (!helper.op && !other.op && old.op == recipe.op &&
            ((old.a == recipe.a && old.b == recipe.b) ||
             ((old.op == '+' || old.op == '*') && old.a == recipe.b && old.b == recipe.a))) return;
        if (helper.op) {
            graph.nodes.push_back(helper);
            refs.push_back(0);
        }
        if (other.op) {
            graph.nodes.push_back(other);
            refs.push_back(0);
        }
        update(recipe.a, 1);
        update(recipe.b, 1);
        update(old.a, -1);
        update(old.b, -1);
        const int saving = previous - active;
        std::uint32_t priority = allow_equal && saving >= 0 ? random() : 0;
        if (saving > best.saving ||
            (allow_equal && saving == best.saving && priority > best.priority))
            best = {id, saving, recipe, helper, other, priority};
        update(old.b, 1);
        update(old.a, 1);
        update(recipe.b, -1);
        update(recipe.a, -1);
        for (int i = 0; i < static_cast<int>(bool(helper.op)) + static_cast<int>(bool(other.op)); ++i) {
            refs.pop_back();
            graph.nodes.pop_back();
        }
        if (previous != active) throw std::logic_error("inconsistent reference counts");
    }

public:
    GraphRewriter(Graph& graph, Value limit) : graph(graph), limit(limit), refs(graph.nodes.size()),
                                              random(static_cast<unsigned>(limit ^ graph.nodes.size())) {
        for (int root : graph.targets) update(root, 1);
    }

    bool share_helper() {
        const int n = static_cast<int>(graph.nodes.size());
        std::vector<int> active_ids;
        std::unordered_set<Value> existing;
        for (int id = 0; id < n; ++id) {
            if (refs[id] || !graph.nodes[id].op) {
                active_ids.push_back(id);
                existing.insert(graph.nodes[id].value);
            }
        }
        std::vector<std::vector<unsigned char>> ancestors(n, std::vector<unsigned char>(n));
        for (int id : active_ids) {
            std::function<void(int)> visit = [&](int node) {
                if (ancestors[id][node]) return;
                ancestors[id][node] = 1;
                const Node& here = graph.nodes[node];
                if (here.op) {
                    visit(here.a);
                    visit(here.b);
                }
            };
            visit(id);
        }
        std::unordered_map<Value, std::vector<Node>> recipes;
        for (int a : active_ids) for (int b : active_ids) {
            if (graph.nodes[a].value < graph.nodes[b].value ||
                (graph.nodes[a].value == graph.nodes[b].value && a < b)) continue;
            for (char op : {'+', '-', '*', '/'}) {
                Value v = calculate(op, graph.nodes[a].value, graph.nodes[b].value, limit);
                if (v && !existing.count(v)) {
                    auto& options = recipes[v];
                    if (options.size() < 6) options.push_back({v, op, a, b});
                }
            }
        }
        struct Link { int id, operand; char op; bool helper_first; };
        std::unordered_map<Value, std::vector<Link>> links;
        for (int id : active_ids) {
            const auto& root = graph.nodes[id];
            if (!root.op) continue;
            int available_uses = root.a == root.b ? 4 : 2;
            auto may_remove = [&](int operand) {
                return graph.nodes[operand].op && refs[operand] <= available_uses;
            };
            if (!may_remove(root.a) && !may_remove(root.b)) continue;
            for (int b : active_ids) {
                if (ancestors[b][id]) continue;
                inverse(root.value, graph.nodes[b].value, limit, [&](Value v, char op, bool first) {
                    if (recipes.count(v)) {
                        auto& list = links[v];
                        if (list.size() < 20) list.push_back({id, b, op, first});
                    }
                    return false;
                });
            }
        }
        int best = 0;
        Node helper{};
        std::array<Link, 2> chosen{};
        for (const auto& [v, candidates] : links) {
            if (candidates.size() < 2) continue;
            for (const auto& recipe : recipes.at(v)) {
                for (std::size_t i = 0; i < candidates.size(); ++i) {
                    auto first = candidates[i];
                    int u = first.id;
                    if (ancestors[recipe.a][u] || ancestors[recipe.b][u]) continue;
                    for (std::size_t j = i + 1; j < candidates.size(); ++j) {
                        auto second = candidates[j];
                        int w = second.id;
                        if (ancestors[u][w] || ancestors[w][u] ||
                            ancestors[recipe.a][w] || ancestors[recipe.b][w] ||
                            ancestors[first.operand][w] || ancestors[second.operand][u]) continue;
                        auto left = graph.nodes[u], right = graph.nodes[w];
                        int before = active;
                        // Charge exactly one new instruction for the shared value.
                        update(recipe.a, 1);
                        update(recipe.b, 1);
                        update(first.operand, 1);
                        update(second.operand, 1);
                        update(left.a, -1);
                        update(left.b, -1);
                        update(right.a, -1);
                        update(right.b, -1);
                        int saving = before - active - 1;
                        if (saving > best) {
                            best = saving;
                            helper = recipe;
                            chosen = {first, second};
                        }
                        update(right.b, 1);
                        update(right.a, 1);
                        update(left.b, 1);
                        update(left.a, 1);
                        update(second.operand, -1);
                        update(first.operand, -1);
                        update(recipe.b, -1);
                        update(recipe.a, -1);
                        if (before != active) throw std::logic_error("shared helper accounting drift");
                    }
                }
            }
        }
        if (best == 0) return false;
        graph.nodes.push_back(helper);
        refs.push_back(0);
        for (auto link : chosen) {
            auto old = graph.nodes[link.id];
            graph.nodes[link.id] = {old.value, link.op, link.helper_first ? n : link.operand,
                                   link.helper_first ? link.operand : n};
            update(n, 1);
            update(link.operand, 1);
            update(old.a, -1);
            update(old.b, -1);
        }
        return true;
    }

    bool improve(int max_helpers, bool take_equal = false) {
        allow_equal = take_equal;
        const int n = static_cast<int>(graph.nodes.size());
        std::vector<int> candidates;
        Index known;
        std::vector<std::vector<int>> children(n);
        for (int i = 0; i < n; ++i) {
            auto node = graph.nodes[i];
            if (refs[i] || !node.op) {
                candidates.push_back(i);
                known.emplace(node.value, i);
                if (node.op) {
                    children[node.a].push_back(i);
                    children[node.b].push_back(i);
                }
            }
        }
        std::unordered_map<Value, std::vector<Op>> helpers;
        if (max_helpers) {
            for (int a : candidates) for (int b : candidates) {
                if (a < b) continue;
                Value x = graph.nodes[a].value, y = graph.nodes[b].value;
                if (x < y) { std::swap(x, y); }
                int i = graph.nodes[a].value >= graph.nodes[b].value ? a : b;
                int j = i == a ? b : a;
                for (char op : {'+', '-', '*', '/'}) {
                    Value value = calculate(op, x, y, limit);
                    if (value && !known.count(value)) {
                        auto& routes = helpers[value];
                        if (routes.size() < 6) routes.push_back({op, i, j, 0});
                    }
                }
            }
        }
        Replacement best;
        for (int id : candidates) {
            const auto target = graph.nodes[id];
            if (!target.op) continue;
            // With no exclusively used operand there is no work to eliminate.
            const int occurrences = target.a == target.b ? 2 : 1;
            const bool can_release = refs[target.a] <= occurrences || refs[target.b] <= occurrences;
            if (!take_equal && !can_release) continue;
            std::vector<bool> forbidden(n);
            std::function<void(int)> block = [&](int node) {
                if (forbidden[node]) return;
                forbidden[node] = true;
                for (int child : children[node]) block(child);
            };
            block(id);
            for (int b : candidates) {
                if (forbidden[b]) continue;
                inverse(target.value, graph.nodes[b].value, limit, [&](Value x, char op, bool first) {
                    auto found = known.find(x);
                    if (found != known.end()) {
                        int a = found->second;
                        if (!forbidden[a])
                            measure(best, id, {target.value, op, first ? a : b, first ? b : a});
                    } else if (max_helpers && can_release) {
                        auto options = helpers.find(x);
                        if (options != helpers.end()) for (auto route : options->second) {
                            if (forbidden[route.a] || forbidden[route.b]) continue;
                            measure(best, id, {target.value, op, first ? n : b, first ? b : n},
                                    {x, route.kind, route.a, route.b});
                        }
                    }
                    return false;
                });
            }
            if (max_helpers < 2) continue;
            // Bound the best possible saving before considering two new nodes.
            const int before = active;
            update(target.a, -1);
            update(target.b, -1);
            const int removable = before - active;
            update(target.b, 1);
            update(target.a, 1);
            if (removable - 2 <= best.saving) continue;
            auto allowed = [&](Op op) { return !forbidden[op.a] && !forbidden[op.b]; };
            if (max_helpers >= 3) {
                for (int b : candidates) {
                    if (forbidden[b]) continue;
                    inverse(target.value, graph.nodes[b].value, limit, [&](Value x, char op, bool first) {
                        if (known.count(x) || helpers.count(x)) return false;
                        for (int c : candidates) {
                            if (forbidden[c]) continue;
                            inverse(x, graph.nodes[c].value, limit, [&](Value y, char middle, bool inner) {
                                auto hits = helpers.find(y);
                                if (hits == helpers.end()) return false;
                                for (auto route : hits->second) if (allowed(route)) {
                                    measure(best, id,
                                            {target.value, op, first ? n + 1 : b, first ? b : n + 1},
                                            {y, route.kind, route.a, route.b},
                                            {x, middle, inner ? n : c, inner ? c : n});
                                }
                                return removable - 2 <= best.saving;
                            });
                            if (removable - 2 <= best.saving) return true;
                        }
                        return false;
                    });
                    if (removable - 2 <= best.saving) break;
                }
            }
            if (removable - 2 <= best.saving) continue;
            for (const auto& [y, rpaths] : helpers) {
                const auto right = std::find_if(rpaths.begin(), rpaths.end(), allowed);
                if (right == rpaths.end()) continue;
                inverse(target.value, y, limit, [&](Value x, char op, bool first) {
                    if (x == y) return false;
                    auto lpaths = helpers.find(x);
                    if (lpaths == helpers.end()) return false;
                    for (auto left : lpaths->second) if (allowed(left)) {
                        measure(best, id, {target.value, op, first ? n : n + 1, first ? n + 1 : n},
                                {x, left.kind, left.a, left.b}, {y, right->kind, right->a, right->b});
                    }
                    return removable - 2 <= best.saving;
                });
                if (removable - 2 <= best.saving) break;
            }
        }
        if (best.id < 0) return false;
        if (best.helper.op) {
            graph.nodes.push_back(best.helper);
            refs.push_back(0);
        }
        if (best.other.op) {
            graph.nodes.push_back(best.other);
            refs.push_back(0);
        }
        auto old = graph.nodes[best.id];
        graph.nodes[best.id] = best.next;
        update(best.next.a, 1);
        update(best.next.b, 1);
        update(old.a, -1);
        update(old.b, -1);
        return true;
    }
};

Program rewrite_graph(const Registers& initial, const std::vector<Value>& targets,
                      Value limit, Program best, int rounds, int trials, int max_helpers,
                      int perturb = 0, bool share = false) {
    Graph graph = recover_graph(initial, limit, targets, best);
    GraphRewriter rewriter(graph, limit);
    int improvements = 0;
    for (int i = 0; i < rounds; ++i) {
        if ((!share || !rewriter.share_helper()) &&
            !rewriter.improve(0) && !rewriter.improve(max_helpers) &&
            (!perturb || !rewriter.improve(perturb - 1, true))) break;
        ++improvements;
    }
    if (!improvements) return best;
    graph = topological_graph(graph);
    std::mt19937 rng(static_cast<unsigned>(limit ^ best.size()));
    auto best_order = graph.targets;
    for (int i = 0; i < trials; ++i) {
        auto order = best_order;
        if (i == 8) std::reverse(order.begin(), order.end());
        if (i >= 9 && order.size() >= 2) {
            if (i % 9 == 0) std::shuffle(order.begin(), order.end(), rng);
            else std::swap(order[rng() % order.size()], order[rng() % order.size()]);
        }
        auto result = Scheduler(graph, static_cast<int>(best.size()), limit, i % 4, (i / 4) % 2).run(order);
        if (!result.empty() && result.size() < best.size()) {
            best = std::move(result);
            best_order = std::move(order);
        }
    }
    return best;
}

using RecipeTable = std::unordered_map<Value, Op>;
void enumerate_one_step(const Index& known, Value limit, RecipeTable& recipes) {
    recipes.reserve(known.size() * known.size() * 2);
    for (auto [a, i] : known) for (auto [b, j] : known) {
        if (a < b) continue;
        for (char op : {'+', '-', '*', '/'}) {
            auto x = calculate(op, a, b, limit);
            if (x) recipes.emplace(x, Op{op, i, j, 0});
        }
    }
}

bool two_steps(const Index& known, Value target, Value limit, int destination,
               std::uint64_t live, RecipeTable& recipes, Program& result) {
    if (recipes.empty()) enumerate_one_step(known, limit, recipes);
    for (auto [base, ri] : known) {
        if (inverse(target, base, limit, [&](Value intermediate, char op, bool temp_first) {
            auto recipe = recipes.find(intermediate);
            if (recipe == recipes.end()) return false;
            int temp = destination;
            if (temp == ri) {
                for (int i = 0; i < slots; ++i)
                    if (i != ri && !(live & (Value{1} << i))) { temp = i; break; }
            }
            if (temp == ri) return false;
            Op first = recipe->second;
            first.destination = temp;
            Op last = temp_first ? Op{op, temp, ri, destination} : Op{op, ri, temp, destination};
            result = {first, last};
            return true;
        })) return true;
    }
    if (target % 2 == 0) {
        auto half = recipes.find(target / 2);
        if (half != recipes.end()) {
            Op first = half->second;
            first.destination = destination;
            result = {first, Op{'+', destination, destination, destination}};
            return true;
        }
    }
    return false;
}

struct Symbolic {
    char kind;
    int a, b;
};

bool place_fragment(const std::vector<Symbolic>& fragment, int destination, std::uint64_t live,
                    Program& result) {
    const int final_id = static_cast<int>(fragment.size()) - 1;
    std::array<int, 3> allocated{};
    for (int first = 0; first < slots; ++first) {
        if (first != destination && live & (Value{1} << first)) continue;
        allocated[0] = first;
        for (int second = 0; second < slots; ++second) {
            if (second != destination && live & (Value{1} << second)) continue;
            allocated[1] = second;
            allocated[final_id] = destination;
            std::array<int, slots> physical;
            for (int r = 0; r < slots; ++r) physical[r] = r;
            Program proposal;
            for (int i = 0; i <= final_id; ++i) {
                auto op = fragment[i];
                int a = op.a >= slots ? allocated[op.a - slots] : op.a;
                int b = op.b >= slots ? allocated[op.b - slots] : op.b;
                if (physical[a] != op.a || physical[b] != op.b) break;
                proposal.push_back({op.kind, a, b, allocated[i]});
                physical[allocated[i]] = slots + i;
            }
            if (proposal.size() == fragment.size()) {
                result = std::move(proposal);
                return true;
            }
            if (final_id == 1) break;
        }
    }
    return false;
}

Symbolic symbolic(Op op) { return {op.kind, op.a, op.b}; }
Symbolic combine(char kind, bool first, int a, int b) {
    return first ? Symbolic{kind, a, b} : Symbolic{kind, b, a};
}

bool three_steps(const Index& known, Value target, Value limit, int destination,
                 std::uint64_t live, RecipeTable& recipes, Program& result) {
    if (recipes.empty()) enumerate_one_step(known, limit, recipes);
    for (auto [base, rb] : known) {
        if (inverse(target, base, limit, [&](Value x, char last, bool last_first) {
            if (known.count(x) || recipes.count(x)) return false;
            for (auto [c, rc] : known) {
                if (inverse(x, c, limit, [&](Value helper, char middle, bool middle_first) {
                    auto it = recipes.find(helper);
                    return it != recipes.end() &&
                        place_fragment({symbolic(it->second), combine(middle, middle_first, slots, rc),
                                        combine(last, last_first, slots + 1, rb)}, destination, live, result);
                })) return true;
            }
            // Squaring and doubling the sole intermediate are also legal.
            auto sqrt_x = static_cast<Value>(std::sqrt(static_cast<long double>(x)));
            if ((sqrt_x + 1) * (sqrt_x + 1) == x) ++sqrt_x;
            for (auto [root, op] : {std::pair<Value, char>{sqrt_x, '*'}, {x / 2, '+'}}) {
                if (calculate(op, root, root, limit) != x) continue;
                auto it = recipes.find(root);
                if (it != recipes.end() &&
                    place_fragment({symbolic(it->second), {op, slots, slots},
                                    combine(last, last_first, slots + 1, rb)}, destination, live, result))
                    return true;
            }
            return false;
        })) return true;
    }
    // Two independently generated intermediates may give a shorter final combination.
    for (const auto& [y, right] : recipes) {
        if (known.count(y)) continue;
        if (inverse(target, y, limit, [&](Value x, char op, bool x_first) {
            if (known.count(x) || x == y) return false;
            auto left = recipes.find(x);
            return left != recipes.end() &&
                (place_fragment({symbolic(left->second), symbolic(right),
                                 combine(op, x_first, slots, slots + 1)}, destination, live, result) ||
                 place_fragment({symbolic(right), symbolic(left->second),
                                 combine(op, x_first, slots + 1, slots)}, destination, live, result));
        })) return true;
    }
    return false;
}

bool two_goals(const Registers& regs, const Index& known, std::vector<Goal> goals,
               Value limit, std::uint64_t live, Program& result) {
    for (int order = 0; order < 2; ++order) {
        auto [first_value, first_dest] = goals[order];
        auto [last_value, last_dest] = goals[1 - order];
        for (int i = 0; i < slots; ++i) {
            if (first_dest != -1 && i != first_dest) continue;
            if (first_dest == -1 && (live & (Value{1} << i))) continue;
            Op first;
            if (!one_step(known, first_value, limit, i, first)) continue;
            auto next = regs;
            next[i] = first_value;
            Op last;
            int d = last_dest == -1 ? i : last_dest;
            // If the last goal is target-only, it may reuse the first scratch
            // but must not overwrite a boundary register we just restored.
            if (last_dest == -1 && first_dest != -1) {
                d = 0;
                while (d < slots && (live & (Value{1} << d))) ++d;
                if (d == slots) continue;
            }
            if (one_step(index_values(next), last_value, limit, d, last)) {
                result = {first, last};
                return true;
            }
        }
    }
    return false;
}

Program sweep(const Registers& initial, const std::vector<Value>& targets,
              Value limit, const Program& original, int width) {
    bool allow_three = width >= 100;
    if (allow_three) width -= 100;
    int n = static_cast<int>(original.size());
    std::vector<Registers> trace(n + 1);
    std::vector<Value> written(n);
    std::unordered_set<Value> target_set(targets.begin(), targets.end());
    std::unordered_map<Value, int> last_target_write;
    std::vector<std::uint64_t> live(n + 1);
    trace[0] = initial;
    for (int i = 0; i < n; ++i) {
        trace[i + 1] = trace[i];
        written[i] = execute(trace[i + 1], original[i], limit);
        if (target_set.count(written[i])) last_target_write[written[i]] = i;
    }
    for (int i = n - 1; i >= 0; --i) {
        auto [kind, a, b, d] = original[i];
        live[i] = (live[i + 1] & ~(Value{1} << d)) | (Value{1} << a) | (Value{1} << b);
    }

    Registers regs = initial;
    std::unordered_set<Value> covered;
    for (Value x : initial) if (target_set.count(x)) covered.insert(x);
    Program output;
    for (int begin = 0; begin < n;) {
        Program best;
        int end = begin;
        Index known;
        RecipeTable recipes;
        for (int stop = std::min(n, begin + width); stop > begin; --stop) {
            std::vector<Goal> goals;
            for (int r = 0; r < slots; ++r) {
                if ((live[stop] & (Value{1} << r)) && regs[r] != trace[stop][r])
                    goals.push_back({trace[stop][r], r});
            }
            if (goals.size() > 2) continue;
            for (int i = begin; i < stop; ++i) {
                Value v = written[i];
                auto last = last_target_write.find(v);
                if (last == last_target_write.end() || last->second != i || covered.count(v)) continue;
                if (std::none_of(goals.begin(), goals.end(), [v](Goal g) { return g.value == v; }))
                    goals.push_back({v, -1});
            }
            int length = stop - begin;
            if (goals.size() > 2 || goals.size() >= static_cast<std::size_t>(length)) continue;
            Program proposal;
            if (!goals.empty() && known.empty()) known = index_values(regs);
            if (goals.size() == 1) {
                auto [value, dest] = goals[0];
                if (dest == -1) {
                    dest = 0;
                    while (dest < slots && (live[stop] & (Value{1} << dest))) ++dest;
                    if (dest == slots) continue;
                }
                Op one;
                if (one_step(known, value, limit, dest, one)) proposal = {one};
                else {
                    if (length < 3) continue;
                    if (!two_steps(known, value, limit, dest, live[stop], recipes, proposal) &&
                        (!allow_three || length < 4 ||
                         !three_steps(known, value, limit, dest, live[stop], recipes, proposal))) continue;
                }
            } else if (goals.size() == 2 && !two_goals(regs, known, goals, limit, live[stop], proposal)) {
                continue;
            }
            if (stop - static_cast<int>(proposal.size()) > end - static_cast<int>(best.size())) {
                best = std::move(proposal);
                end = stop;
            }
            // A shorter window cannot beat this combination.
            if (end - static_cast<int>(best.size()) >= stop - 1) break;
        }
        if (end == begin) {
            best = {original[begin]};
            end = begin + 1;
        }
        for (auto op : best) {
            auto v = execute(regs, op, limit);
            if (target_set.count(v)) covered.insert(v);
            output.push_back(op);
        }
        for (int r = 0; r < slots; ++r)
            if ((live[end] & (Value{1} << r)) && regs[r] != trace[end][r])
                throw std::runtime_error("rewrite changed a live register");
        begin = end;
    }
    if (covered.size() != target_set.size()) throw std::runtime_error("rewrite lost a target");
    return output;
}

#ifndef ELF_LIBRARY
int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        const int width = argc > 1 ? std::stoi(argv[1]) : 10;
        const int passes = argc > 2 ? std::stoi(argv[2]) : 8;
        std::string id;
        Value limit;
        int n_starts, n_targets, n_ops;
        while (std::cin >> id >> limit >> n_starts >> n_targets >> n_ops) {
            Registers initial{};
            std::vector<Value> targets(n_targets);
            Program program(n_ops);
            for (int i = 0; i < n_starts; ++i) std::cin >> initial.at(i);
            for (auto& target : targets) std::cin >> target;
            for (auto& op : program) std::cin >> op.kind >> op.a >> op.b >> op.destination;
            if (width == 0) program = elide_helpers(initial, targets, limit, std::move(program), passes);
            if (width <= -1000) {
                int helpers = (-width) / 1000;
                program = rewrite_graph(initial, targets, limit, std::move(program),
                                        passes, -width - 1000 * helpers,
                                        helpers >= 4 ? 2 : helpers,
                                        helpers == 6 ? 0 : std::max(0, helpers - 3), helpers == 6);
            }
            else if (width < 0)
                program = reschedule(initial, targets, limit, std::move(program), -width);
            for (int i = 0; width > 0 && i < passes && !program.empty(); ++i) {
                auto next = sweep(initial, targets, limit, program, width);
                if (next.size() >= program.size()) break;
                program = std::move(next);
            }
            std::cout << id << ' ' << program.size() << '\n';
            for (auto [kind, a, b, d] : program)
                std::cout << kind << ' ' << a << ' ' << b << ' ' << d << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
