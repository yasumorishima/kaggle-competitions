// From-scratch joint synthesis: discover latent hubs by inverse arithmetic.
// Reuses the local verified register scheduler and the credited beam primitive.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>
#include "santa_factor64.h"

#define ELF_LIBRARY
#include "elf_window.cpp"

namespace local_beam {
#define main unused_beam_main
#include "elf_congruence_generated.cpp"
#undef main
}

using SInt = local_beam::Int;
struct Step { Value x; char op; Value a, b; };
struct HubRoute { int target; Value factor, offset; char op; int weight; };
struct Hub { Value value; std::vector<HubRoute> children; double score; };
struct Proposal { std::vector<Step> steps; int pivot = -1; double order = 0; };
struct FactorRoute { Value a, b, rest; char op; };

struct Construction {
    const Value limit;
    const std::vector<Value> targets;
    Graph g;
    Index index;
    std::vector<Value> palette;
    std::vector<Value> starts;
    std::vector<int> degrees;
    int paid_hubs = 0, hub_goals = 0;

    Construction(Value cut, std::vector<Value> s, std::vector<Value> t, bool bootstrap_enabled=true)
        : limit(cut), targets(std::move(t)), starts(std::move(s)), degrees(targets.size()) {
        g.initial.fill(-1);
        for (int i = 0; i < (int)starts.size(); ++i) {
            auto [entry, added] = index.emplace(starts[i], (int)g.nodes.size());
            if (added) g.nodes.push_back({starts[i], 0, -1, -1});
            g.initial.at(i) = entry->second;
        }
        if(!bootstrap_enabled) {
            palette=starts;
            return;
        }
        local_beam::Puzzle bootstrap((SInt)limit, 1);
        bootstrap.starts.assign(starts.begin(), starts.end());
        bootstrap.setup();
        for (SInt v : bootstrap.order) {
            auto [a, b] = bootstrap.args.at(v);
            append({(Value)v, bootstrap.defs.at(v).op, (Value)a, (Value)b});
        }
        palette.assign(bootstrap.palette.begin(), bootstrap.palette.end());
        palette.insert(palette.end(), starts.begin(), starts.end());
    }

    void append(Step op) {
        if (index.count(op.x)) return;
        if (calculate(op.op, op.a, op.b, limit) != op.x)
            throw std::runtime_error("Invalid graph node in synthesis");
        int a = index.at(op.a), b = index.at(op.b);
        index.emplace(op.x, (int)g.nodes.size());
        g.nodes.push_back({op.x, op.op, a, b});
    }

    std::vector<SInt> leaves(int budget) const {
        std::unordered_set<Value> seen;
        std::vector<SInt> values;
        auto add = [&](Value v) {
            if ((int)values.size() < budget && seen.insert(v).second)
                values.push_back((SInt)v);
        };
        for (Value v : starts) add(v);
        for (Value v : palette) add(v);
        for (Value v : targets) if (index.count(v)) add(v);
        for (auto it = g.nodes.rbegin(); it != g.nodes.rend(); ++it) add(it->value);
        return values;
    }
};

static int bit_count(Value v) {
    int bits = 0;
    for (; v; v >>= 1) ++bits;
    return bits;
}

// Keep one exact short recipe per target and hub. A demand can be met by
// q*f, q*f+r, q*f-r, q+r, q-r, or r-q. No prior program is inspected.
std::vector<Hub> discover_hubs(Construction& c, const local_beam::Dictionary& d) {
    std::unordered_map<Value, std::vector<HubRoute>> waiting;
    const int n = std::min(d.n1, 420);
    std::unordered_set<Value> goals(c.targets.begin(), c.targets.end());
    std::vector<Hub> result;
    for (int i = 0; i < (int)c.targets.size(); ++i) {
        Value t = c.targets[i];
        if (c.index.count(t)) continue;
        auto offer = [&](Value q, Value f, Value r, char op, int cost) {
            if (q < 2 || q == t || q > c.limit) return;
            if (c.index.count(q) && !goals.count(q)) return;
            if (goals.count(q) && q != t && cost <= 3)
                ++c.degrees[std::find(c.targets.begin(), c.targets.end(), q) - c.targets.begin()];
            auto& routes = waiting[q];
            HubRoute item{i, f, r, op, cost};
            if (routes.empty() || routes.back().target != i) routes.push_back(item);
            else if (routes.back().weight > cost) routes.back() = item;
        };
        for (int j = 0; j < n; ++j) {
            Value x = (Value)d.e[j].v;
            int cost = d.e[j].cost;
            // Offsets may themselves be computed cheaply from the seed palette.
            if (t > x) offer(t - x, 0, x, '+', cost + 1);
            if (x > t) offer(x - t, 0, x, 'r', cost + 1);
            if (t <= c.limit - x) offer(t + x, 0, x, '-', cost + 1);
            if (x < 2 || x >= t) continue;
            Value q = t / x, rem = t % x;
            if (!rem) offer(q, x, 0, '*', cost + 1);
            else {
                int plus = d.index.get((SInt)rem);
                if (plus >= 0) offer(q, x, rem, '+', cost + d.e[plus].cost + 2);
                int minus = d.index.get((SInt)(x - rem));
                if (minus >= 0 && (q + 1) <= c.limit / x)
                    offer(q + 1, x, x - rem, '-', cost + d.e[minus].cost + 2);
            }
        }
    }
    for (auto& [q, group] : waiting) {
        if (group.size() < 2 && !goals.count(q)) continue;
        double expected = 0;
        for (const auto& route : group) {
            expected += std::max(0.0, 2.9 + .049 * bit_count(c.targets[route.target]) - route.weight);
        }
        expected -= goals.count(q) ? 0.0 : 1.8 + .06 * bit_count(q);
        if (expected > -0.5) result.push_back({q, std::move(group), expected});
    }
    std::sort(result.begin(), result.end(), [](const Hub& a, const Hub& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.value < b.value;
    });
    return result;
}

std::vector<Hub> divisor_hubs(const Construction& c, const local_beam::Dictionary& d) {
    // Hypothesize a common factor (or a common arithmetic-progression modulus)
    // directly from several objective values. No family label is supplied.
    std::vector<Value> offsets = {0, 1, 2, 3, 5};
    offsets.insert(offsets.end(), c.starts.begin(), c.starts.end());
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    std::vector<std::vector<Value>> shifted;
    shifted.reserve(c.targets.size());
    for (Value t : c.targets) {
        std::vector<Value> possible;
        for (Value v : offsets) {
            if (t > v) possible.push_back(t - v);
            if (v && v <= c.limit - t) possible.push_back(t + v);
        }
        shifted.push_back(std::move(possible));
    }
    std::unordered_map<Value, uint64_t> support;
    for (int i = 0; i < (int)c.targets.size(); ++i) {
        for (int j = i + 1; j < (int)c.targets.size(); ++j) {
            const auto record = [&](Value g) {
                if (g >= 96 && g < std::min(c.targets[i], c.targets[j]))
                    support[g] |= (1ULL << i) | (1ULL << j);
            };
            for (Value l : shifted[i])
                for (Value r : shifted[j]) record(std::gcd(l, r));
            for (int k = j + 1; k < (int)c.targets.size(); ++k)
                record(std::gcd(c.targets[j] - c.targets[i], c.targets[k] - c.targets[i]));
        }
    }
    auto complexity = [&](Value v) -> int {
        if (c.index.count(v)) return 0;
        int id = d.index.get((SInt)v);
        if (id >= 0) return d.e[id].cost;
        return 3 + bit_count(v) / 18;
    };
    std::vector<Hub> result;
    for (auto [g, related] : support) {
        if (bit_count(g) < 8 && !c.index.count(g) && d.index.get((SInt)g) < 0) continue;
        std::vector<HubRoute> group;
        double total = -complexity(g);
        for (int i = 0; i < (int)c.targets.size(); ++i) {
            Value t = c.targets[i], q = t / g, r = t % g;
            if (c.index.count(t) || q < 2) continue;
            int add = 100, sub = 100;
            if (!r || d.index.get((SInt)r) >= 0 || (related & (1ULL << i)))
                add = complexity(q) + 1 + (r ? 1 + complexity(r) : 0);
            if (r && q + 1 <= c.limit / g && (d.index.get((SInt)(g - r)) >= 0 || (related & (1ULL << i))))
                sub = complexity(q + 1) + 2 + complexity(g - r);
            int weight = std::min(add, sub);
            double improvement = 3.55 + .057 * bit_count(t) - weight;
            if (improvement < .30) continue;
            if (add <= sub) group.push_back({i, q, r, '+', weight});
            else group.push_back({i, q + 1, g - r, '-', weight});
            total += improvement;
        }
        if (group.size() >= 2 && total > -1.25) result.push_back({g, std::move(group), total});
    }
    return result;
}

struct Planner {
    const Construction& c;
    local_beam::Dictionary& d;
    std::vector<Step> plan;
    std::unordered_set<Value> made;
    void push(Value v, char op, Value a, Value b) {
        if (!c.index.count(v) && made.insert(v).second) {
            if (calculate(op, a, b, c.limit) != v) throw std::runtime_error("Invalid expression");
            plan.push_back({v, op, a, b});
        }
    }
    void place(int id) {
        const auto& e = d.e[id];
        Value v = (Value)e.v;
        if (c.index.count(v) || made.count(v)) return;
        if (!e.op) throw std::runtime_error("Unknown leaf");
        place(e.a); place(e.b);
        push(v, e.op, (Value)d.e[e.a].v, (Value)d.e[e.b].v);
    }
    bool attach(Value v) {
        if (c.index.count(v) || made.count(v)) return true;
        int id = d.index.get((SInt)v);
        if (id < 0) return false;
        place(id);
        return true;
    }
    void binary(Value v) {
        if (attach(v)) return;
        binary(v / 2);
        Value half = v / 2;
        push(half * 2, '+', half, half);
        if (v & 1) push(v, '+', half * 2, 1);
    }
};

std::vector<Step> synthesize_new(const Construction& c, local_beam::Dictionary& d,
                                 Value target, int width, bool allow_beam) {
    Planner p{c, d};
    if (p.attach(target)) return p.plan;
    int best = 5;
    int a_best = -1, b_best = -1;
    char shape = 0;
    // All one-step operands are used on one side, with two-step forms
    // permitted on the other. Parent overlap is credited explicitly.
    for (int j = 0; j < d.n1 && best > 2; ++j) {
        inverse(target, (Value)d.e[j].v, c.limit, [&](Value x, char op, bool first) {
            int i = d.index.get((SInt)x);
            if (i < 0) return false;
            const auto& l = d.e[i]; const auto& r = d.e[j];
            int cost = 1 + l.cost + r.cost;
            if (i == j) cost -= r.cost;
            else if (l.cost == 2 && l.a == j) --cost;
            if (cost >= best) return false;
            best = cost;
            a_best = first ? i : j;
            b_best = first ? j : i;
            shape = op;
            return best <= 2;
        });
    }
    if (a_best >= 0) {
        p.place(a_best); p.place(b_best);
        p.push(target, shape, (Value)d.e[a_best].v, (Value)d.e[b_best].v);
        return p.plan;
    }
    if (!allow_beam) return {};
    local_beam::Search beam(d, width);
    beam.run((SInt)target);
    if (beam.end < 0) {
        p.binary(target);
        return p.plan;
    }
    p.place(beam.terminal);
    Value value = (Value)d.e[beam.terminal].v;
    for (int k = beam.end; beam.history[k].parent >= 0; k = beam.history[k].parent) {
        auto [factor, atom, op] = beam.history[k].link;
        if (factor >= 0) {
            p.place(factor);
            Value f = (Value)d.e[factor].v, x = calculate('*', value, f, c.limit);
            p.push(x, '*', value, f); value = x;
        }
        if (atom >= 0) {
            p.place(atom);
            Value b = (Value)d.e[atom].v;
            Value x = op == 'r' ? calculate('-', b, value, c.limit) : calculate(op, value, b, c.limit);
            if (op == 'r') p.push(x, '-', b, value);
            else p.push(x, op, value, b);
            value = x;
        }
    }
    if (value != target) throw std::runtime_error("Beam did not reach planned root");
    return p.plan;
}

std::vector<FactorRoute> factor_routes(const Construction& c, Value t, bool deep) {
    std::vector<FactorRoute> out;
    std::vector<Value> shifts = {0, 1};
    shifts.insert(shifts.end(), c.starts.begin(), c.starts.end());
    if (deep) shifts.insert(shifts.end(), {2, 3, 5, 7, 15, 31, 63, 1023});
    std::sort(shifts.begin(), shifts.end());
    shifts.erase(std::unique(shifts.begin(), shifts.end()), shifts.end());
    for (Value v : shifts) {
        for (char op : {'+', '-'}) {
            if (op == '+' && v >= t) continue;
            if (op == '-' && (v == 0 || v > c.limit - t)) continue;
            Value shifted = op == '+' ? t - v : t + v;
            if (shifted < 4) continue;
            auto ds = small_factor::divisors(shifted, deep ? 9000 : 4500);
            for (Value a : ds) {
                if (a > shifted / a) break;
                if (a < 2) continue;
                out.push_back({a, shifted / a, v, op});
            }
        }
    }
    return out;
}

std::vector<Step> factor_plans(const Construction& c, local_beam::Dictionary& d,
        Value t, const std::vector<FactorRoute>& routes, std::vector<Step> best, int width,
        bool deep) {
    auto estimate = [&](Value x) {
        int ix = d.index.get((SInt)x);
        if (ix >= 0) return (double)d.e[ix].cost;
        if (c.index.count(x)) return 0.0;
        return std::max(2.5, .114 * bit_count(x) + .45);
    };
    std::vector<std::pair<double, int>> ranked;
    for (int j = 0; j < (int)routes.size(); ++j) {
        auto [a, b, r, op] = routes[j];
        double cost = 1.0 + estimate(a) + (a != b ? estimate(b) : 0.0);
        if (r) cost += 1.0 + estimate(r);
        if (cost < best.size() + 0.5) ranked.emplace_back(cost, j);
    }
    int candidates = std::min((int)ranked.size(), deep ? 16 : 7);
    std::partial_sort(ranked.begin(), ranked.begin() + candidates, ranked.end());
    std::unordered_map<Value, std::vector<Step>> memo;
    for (int i = 0; i < candidates; ++i) {
        auto [a, b, r, op] = routes[ranked[i].second];
        Planner p{c, d};
        auto synth = [&](Value x) {
            if (p.attach(x)) return;
            auto [entry, insert] = memo.try_emplace(x);
            if (insert) entry->second = synthesize_new(c, d, x, std::min(6, width), true);
            for (auto next : entry->second) p.push(next.x, next.op, next.a, next.b);
        };
        // Synthesize the small factors first to reveal sharing in their DAGs.
        synth(a);
        if (p.plan.size() + 1 >= best.size()) continue;
        synth(b);
        if (p.plan.size() + 1 >= best.size()) continue;
        Value x = calculate('*', a, b, c.limit);
        p.push(x, '*', a, b);
        if (r) {
            synth(r);
            p.push(t, op, x, r);
        }
        if (p.plan.size() < best.size()) best = std::move(p.plan);
    }
    return best;
}

void exact_batches(Construction& c, int leaf_budget, int width, int hub_budget,
                   int pivot_width, int order_mode) {
    local_beam::Dictionary d((SInt)c.limit);
    d.build(c.leaves(leaf_budget));
    auto hubs = discover_hubs(c, d);
    const bool common_factor_search = order_mode >= 9;
    if (common_factor_search) {
        auto extras = divisor_hubs(c, d);
        hubs.insert(hubs.end(), extras.begin(), extras.end());
        std::sort(hubs.begin(), hubs.end(), [](const Hub& a, const Hub& b) {
            return std::make_pair(-a.score, a.value) < std::make_pair(-b.score, b.value);
        });
    }
    for (int evaluation = 0; evaluation < hub_budget && evaluation < (int)hubs.size(); ++evaluation) {
        auto& h = hubs[evaluation];
        Planner p{c, d};
        std::vector<int> children;
        const auto supply = [&](Value v) {
            if (p.attach(v)) return true;
            if (!common_factor_search) return false;
            auto plan = synthesize_new(c, d, v, width, true);
            if (plan.size() > 6) return false;
            for (auto op : plan) p.push(op.x, op.op, op.a, op.b);
            return p.made.count(v) != 0;
        };
        for (const auto& route : h.children) {
            Value t = c.targets[route.target];
            if (c.index.count(t) || p.made.count(t)) continue;
            if (route.factor && !supply(route.factor)) continue;
            if (route.offset && !supply(route.offset)) continue;
            // After adding support operands, avoid committing a goal merely
            // because it already has a cheap independent dictionary recipe.
            if (d.index.get((SInt)t) >= 0) continue;
            children.push_back(route.target);
        }
        if (children.size() < 2 && !(children.size() && std::find(c.targets.begin(), c.targets.end(), h.value) != c.targets.end()))
            continue;
        for (auto op : synthesize_new(c, d, h.value, width, true)) p.push(op.x, op.op, op.a, op.b);
        if (!c.index.count(h.value) && !p.made.count(h.value)) continue;
        std::unordered_set<int> chosen(children.begin(), children.end());
        for (const auto& route : h.children) if (chosen.count(route.target)) {
            Value t = c.targets[route.target], v = h.value;
            if (route.factor) {
                Value x = calculate('*', v, route.factor, c.limit);
                p.push(x, '*', v, route.factor); v = x;
            }
            if (route.offset) {
                if (route.op == 'r') p.push(t, '-', route.offset, v);
                else p.push(t, route.op, v, route.offset);
            }
        }
        // Only buy a hub if the exact program is reasonably cheap per
        // covered goal. This is a strict guard, not a topology guarantee.
        double separate = 0;
        int realized = 0;
        for (int i : children) {
            if (!p.made.count(c.targets[i])) continue;
            ++realized;
            separate += synthesize_new(c, d, c.targets[i], width, true).size();
        }
        if (realized && p.plan.size() + 0.5 < separate) {
            for (auto op : p.plan) c.append(op);
            ++c.paid_hubs;
            c.hub_goals += realized;
            d.build(c.leaves(leaf_budget));
        }
    }

    std::vector<std::vector<FactorRoute>> factored(c.targets.size());
    std::vector<bool> factor_ready(c.targets.size());
    const bool factor_search = (order_mode >= 3 && order_mode < 9) || order_mode >= 12;
    const bool deep_factor = order_mode >= 6 && order_mode < 9;
    const bool anticipate = order_mode >= 15;
    order_mode %= 3;
    for (int round = 0; round <= (int)c.targets.size(); ++round) {
        std::vector<int> pending;
        for (int i = 0; i < (int)c.targets.size(); ++i)
            if (!c.index.count(c.targets[i])) pending.push_back(i);
        if (pending.empty()) return;
        d.build(c.leaves(leaf_budget));
        std::vector<Proposal> plans;
        for (int i : pending) {
            auto steps = synthesize_new(c, d, c.targets[i], width, false);
            if (!steps.empty()) plans.push_back({std::move(steps), i});
        }
        if (plans.empty()) {
            std::sort(pending.begin(), pending.end(), [&](int i, int j) {
                double direction = (order_mode == 1 ? -1.0 : 1.0);
                double x = direction * bit_count(c.targets[i]) - (order_mode == 2 ? 1.5 : 4.0) * std::min(5, c.degrees[i]);
                double y = direction * bit_count(c.targets[j]) - (order_mode == 2 ? 1.5 : 4.0) * std::min(5, c.degrees[j]);
                return std::make_pair(x, c.targets[i]) < std::make_pair(y, c.targets[j]);
            });
            if ((int)pending.size() > pivot_width) pending.resize(pivot_width);
            for (int i : pending) {
                Value t = c.targets[i];
                auto base = synthesize_new(c, d, t, width, true);
                if (factor_search && base.size() >= 5) {
                    if (!factor_ready[i]) {
                        factored[i] = factor_routes(c, t, deep_factor);
                        factor_ready[i] = true;
                    }
                    base = factor_plans(c, d, t, factored[i], std::move(base), width, deep_factor);
                }
                plans.push_back({std::move(base), i});
            }
        }
        for (auto& candidate : plans) {
            int new_goals = 0;
            std::unordered_set<Value> materialized;
            for (const auto& step : candidate.steps)
                new_goals += std::binary_search(c.targets.begin(), c.targets.end(), step.x);
            int degree = std::min(6, c.degrees[candidate.pivot]);
            candidate.order = candidate.steps.size() / (double)std::max(1, new_goals)
                              - 0.24 * degree;
            if (anticipate && plans.size() > 1) {
                for (auto node : candidate.steps) materialized.insert(node.x);
                double benefit = 0;
                for (int i : pending) {
                    if (i == candidate.pivot || materialized.count(c.targets[i])) continue;
                    Value t = c.targets[i];
                    int known = d.index.get((SInt)t);
                    int today = known < 0 ? 5 : d.e[known].cost;
                    for (auto& alt : plans)
                        if (alt.pivot == i) today = std::min(today, (int)alt.steps.size());
                    int tomorrow = today;
                    for (const auto& step : candidate.steps) {
                        inverse(t, step.x, c.limit, [&](Value x, char, bool) {
                            int hit = d.index.get((SInt)x);
                            if (c.index.count(x) || materialized.count(x)) tomorrow = std::min(tomorrow, 1);
                            else if (hit >= 0) tomorrow = std::min(tomorrow, 1 + d.e[hit].cost);
                            return tomorrow == 1;
                        });
                        if (tomorrow == 1) break;
                    }
                    benefit += std::min(2, today - tomorrow);
                }
                candidate.order -= std::min(2.0, 0.28 * benefit);
            }
        }
        auto& chosen = *std::min_element(plans.begin(), plans.end(), [](const auto& a, const auto& b) {
            return std::make_pair(a.order, a.pivot) < std::make_pair(b.order, b.pivot);
        });
        if (chosen.steps.empty()) throw std::runtime_error("No graph progress");
        for (const auto& op : chosen.steps) c.append(op);
    }
    throw std::runtime_error("Incomplete construction");
}

std::vector<int> live_frontier_order(const Graph& g, int start_index, int strength, int& logical) {
    const int n = (int)g.nodes.size(), t = (int)g.targets.size();
    std::vector<int> pending(n), opened(n), done(t);
    std::vector<std::vector<int>> dependencies(t);
    for (int i = 0; i < t; ++i) {
        std::vector<char> visit(n);
        std::function<void(int)> dfs = [&](int v) {
            if (visit[v] || !g.nodes[v].op) return;
            visit[v] = 1;
            ++pending[v];
            dependencies[i].push_back(v);
            dfs(g.nodes[v].a); dfs(g.nodes[v].b);
        };
        dfs(g.targets[i]);
    }
    logical = (int)std::count_if(pending.begin(), pending.end(), [](int count) { return count > 0; });
    std::vector<int> order;
    for (int i = 0; i < t; ++i) {
        int choose = -1, score = 1000000;
        for (int j = 0; j < t; ++j) {
            if (done[j] || (i == 0 && start_index >= 0 && j != start_index)) continue;
            int cost = 0;
            for (int v : dependencies[j]) {
                if (!opened[v] && pending[v] > 1) cost += strength;
                if (opened[v] && pending[v] == 1) cost -= strength;
                if (!opened[v]) ++cost;
            }
            if (cost < score) { choose = j; score = cost; }
        }
        done[choose] = 1;
        for (int v : dependencies[choose]) { opened[v] = 1; --pending[v]; }
        order.push_back(g.targets[choose]);
    }
    return order;
}

#ifndef ELF_SHARED_LIBRARY
int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
        int hubs = argc > 1 ? std::stoi(argv[1]) : 10;
        int width = argc > 2 ? std::stoi(argv[2]) : 5;
        int leaves = argc > 3 ? std::stoi(argv[3]) : 54;
        int pivots = argc > 4 ? std::stoi(argv[4]) : 2;
        int order_mode = argc > 5 ? std::stoi(argv[5]) : 0;
        local_beam::INDEX_BITS = 18;
        local_beam::ALPHA = argc > 6 ? std::stod(argv[6]) : .18;
        std::string key;
        Value limit;
        int ns, nt;
        while (std::cin >> key >> limit >> ns >> nt) {
            if (ns < 1 || ns > slots || nt < 1 || nt > 63 || limit == 0 || limit >= (Value{1} << 62))
                throw std::runtime_error("Puzzle outside the supported input bounds");
            std::vector<Value> seeds(ns), goals(nt);
            for (auto& x : seeds) std::cin >> x;
            for (auto& x : goals) std::cin >> x;
            std::sort(goals.begin(), goals.end());
            Construction c(limit, seeds, goals);
            exact_batches(c, leaves, width, hubs, pivots, order_mode);
            for (Value t : goals) c.g.targets.push_back(c.index.at(t));
            auto target_order = c.g.targets;
            std::mt19937 gen((uint32_t)limit);
            Program best;
            int logical = 0;
            for (int i = 0; i < 20; ++i) {
                if (i == 4) std::reverse(target_order.begin(), target_order.end());
                if (i == 8) std::sort(target_order.begin(), target_order.end());
                if (i == 12) target_order = live_frontier_order(c.g, -1, 3, logical);
                if (i >= 16) target_order = live_frontier_order(c.g, (int)(gen() % nt), 2 + i % 4, logical);
                auto program = Scheduler(c.g, best.empty() ? 4097 : (int)best.size(),
                                         limit, i % 4, (i / 2) % 2).run(target_order);
                if (!program.empty() && (best.empty() || program.size() < best.size())) best = std::move(program);
                if (logical && best.size() == (size_t)logical) break;
            }
            std::cout << key << " [";
            for (size_t i = 0; i < best.size(); ++i) {
                if (i) std::cout << ',';
                auto [op, a, b, dest] = best[i];
                std::cout << "[\"" << op << "\"," << a << ',' << b << ',' << dest << ']';
            }
            std::cout << "]\n";
            std::cerr << key << ' ' << c.paid_hubs << ' ' << c.hub_goals << ' ' << logical << '\n';
        }
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
#endif
