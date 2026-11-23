// Multi-output straight-line blocks. Generate and evaluate complete short
// programs for several roots in the same search, with no preselected hub.
#define ELF_SHARED_LIBRARY
#include "elf_shared_synthesis.cpp"

using Script = std::vector<Step>;

struct Blocks {
    Value limit;
    const std::vector<Value> goals;
    const std::vector<SInt> ready;
    const int width, depth, cap;
    local_beam::Dictionary d;
    std::vector<std::vector<Script>> options;
    std::unordered_set<Value> leaves;
    std::vector<int> original_position;

    Blocks(Value limit, std::vector<Value> goals, std::vector<SInt> ready,
           int width, int depth, int cap)
        : limit(limit), goals(std::move(goals)), ready(std::move(ready)), width(width),
          depth(depth), cap(cap), d((SInt)limit), options(this->goals.size()) {
        leaves.insert(this->ready.begin(), this->ready.end());
        d.build(this->ready);
    }

    void add(const Step& node, Script& out, std::unordered_set<Value>& exists) const {
        if (exists.insert(node.x).second) out.push_back(node);
    }

    void fetch(int id, Script& out, std::unordered_set<Value>& exists) const {
        const auto& e = d.e[id];
        if (exists.count(e.v)) return;
        if (!e.op) throw std::logic_error("Missing leaf in arithmetic block");
        fetch(e.a, out, exists);
        fetch(e.b, out, exists);
        add({(Value)e.v, e.op, (Value)d.e[e.a].v, (Value)d.e[e.b].v}, out, exists);
    }

    std::pair<int, std::uint64_t> metrics(const Script& code) const {
        // A block can hit some goals incidentally, including while forming
        // an operand. Track those writes, not just the expected final node.
        std::uint64_t actual = 0;
        for (int i = 0; i < (int)goals.size(); ++i) {
            if (leaves.count(goals[i]) ||
                std::any_of(code.begin(), code.end(), [&](const auto& s){ return s.x == goals[i]; }))
                actual |= 1ULL << i;
        }
        int covered = 0;
        for (auto bits = actual; bits; bits &= bits - 1) ++covered;
        return {covered, actual};
    }

    static std::uint64_t signature(const Script& code) {
        std::uint64_t hash = 0;
        // Deduplicate states by the exact set of values approximately.
        // Hash collisions only discard alternatives; they cannot accept bad code.
        for (const auto& step : code) {
            U64hash(hash, step.x);
        }
        return hash;
    }

    static void U64hash(std::uint64_t& hash, Value x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        hash ^= x ^ (x >> 31);
    }

    void enumerate(int goal) {
        Value t = goals[goal];
        std::unordered_set<std::uint64_t> seen;
        std::vector<Script> found;
        const int maximum = std::min(depth, cap - (int)goals.size() + 1);
        if (maximum < 1 || leaves.count(t)) return;
        auto take = [&](Script plan) {
            if ((int)plan.size() <= maximum && !plan.empty() && seen.insert(signature(plan)).second)
                found.push_back(std::move(plan));
        };
        const int exact = d.index.get((SInt)t);
        if (exact >= 0) {
            Script plan;
            auto known = leaves;
            fetch(exact, plan, known);
            take(std::move(plan));
        }
        const int stop = depth >= 5 ? (int)d.e.size() : d.n1;
        // Join bounded expressions on either side of the final instruction.
        // This includes products of two new values and nonmonotone inverses.
        for (int j = 0; j < stop; ++j) {
            Value b = (Value)d.e[j].v;
            inverse(t, b, limit, [&](Value a, char op, bool fst) {
                int i = d.index.get((SInt)a);
                if (i < 0 || a == t || b == t) return false;
                const auto& x = d.e[i]; const auto& y = d.e[j];
                if (x.cost + y.cost > maximum && i != j) return false;
                if (j >= d.n1 && i <= j) return false; // this split visited from the cheaper side
                Script code;
                auto known = leaves;
                fetch(i, code, known);
                fetch(j, code, known);
                add({t, op, fst ? a : b, fst ? b : a}, code, known);
                take(std::move(code));
                return false;
            });
        }
        // Preserve distinct ways to start a group, even when one costs one
        // more in isolation. Different internal writes can serve other goals.
        auto rank = [&](const Script& code) {
            int cover = metrics(code).first;
            int discounts = 0;
            for (auto& step : code) {
                for (int i = 0; i < (int)goals.size(); ++i) {
                    if (i == goal || step.x == goals[i]) continue;
                    bool direct = inverse(goals[i], step.x, limit, [&](Value required, char, bool) {
                        return d.index.get((SInt)required) >= 0;
                    });
                    discounts += direct;
                }
            }
            return (int)code.size() * 8 - cover * 7 - std::min(5, discounts);
        };
        std::vector<std::pair<int, unsigned>> order;
        order.reserve(found.size());
        for (unsigned i = 0; i < found.size(); ++i) order.emplace_back(rank(found[i]), i);
        auto keep = std::min<std::size_t>(order.size(), std::max(4, width * 2));
        std::partial_sort(order.begin(), order.begin() + keep, order.end());
        for (unsigned i = 0; i < keep; ++i)
            options[goal].push_back(std::move(found[order[i].second]));
    }

    // Pay only for values missing in this prefix. Continuations can read any
    // prior write, including an already completed goal, on either operand.
    void extend(const Script& prefix, int goal, int budget,
                std::vector<Script>& moves) const {
        auto known = leaves;
        for (auto& step : prefix) known.insert(step.x);
        const Value t = goals[goal];
        const auto offer = [&](Script moves_only) {
            if (!moves_only.empty() && (int)moves_only.size() <= budget)
                moves.push_back(std::move(moves_only));
        };
        for (const auto& recipe : options[goal]) {
            Script extra;
            auto active = known;
            for (auto step : recipe) add(step, extra, active);
            offer(std::move(extra));
        }
        if (prefix.empty()) return;
        for (auto& step : prefix) {
            Value b = step.x;
            if (b == t) continue;
            inverse(t, b, limit, [&](Value a, char shape, bool first) {
                if (a == t) return false;
                const Step finish{t, shape, first ? a : b, first ? b : a};
                if (known.count(a)) offer(Script{finish});
                else {
                    int at = d.index.get((SInt)a);
                    if (budget > 1 && at >= 0 && d.e[at].cost < budget) {
                        Script extra;
                        auto active = known;
                        fetch(at, extra, active);
                        add(finish, extra, active);
                        offer(std::move(extra));
                    }
                    if (budget >= 2 && depth >= 4) {
                        // Short backward join with one new value as an operand
                        // in both levels. No shared helper was given in advance.
                        for (auto& other : prefix) {
                            if (other.x == a) continue;
                            inverse(a, other.x, limit, [&](Value x, char op, bool fst) {
                                if (!known.count(x) || x == a) return false;
                                offer(Script{{a, op, fst ? x : other.x, fst ? other.x : x}, finish});
                                return true;
                            });
                        }
                    }
                }
                return false;
            });
        }
    }

    Script search(std::vector<Script> fallback = {}, std::vector<Script>* archive = nullptr,
                  int retained = 6) {
        for (int i = 0; i < (int)goals.size(); ++i) {
            enumerate(i);
            if (!fallback.empty() && !fallback[i].empty())
                options[i].push_back(std::move(fallback[i]));
        }
        struct State { Script code; std::uint64_t covered; int rank; };
        auto [done, bits] = metrics({});
        std::vector<State> frontier{{{}, bits, 0}};
        Script best;
        std::unordered_set<std::uint64_t> saved;
        auto record = [&](const Script& code) {
            if (!archive) return;
            std::uint64_t key=0;
            for (const auto& op : code) {
                std::uint64_t word=0;
                U64hash(word,op.x);
                U64hash(word,op.a ^ 0xb492b66fbe98f273ULL);
                U64hash(word,op.b ^ (std::uint64_t)op.op*157627);
                U64hash(key,word);
            }
            if (!saved.insert(key).second) return;
            // Retain varied equal-cost scripts. The shortlist is ranked by
            // standalone work; acceptance is deferred to the global model.
            if ((int)archive->size() < retained) {
                archive->push_back(code);
                return;
            }
            auto worst=std::max_element(archive->begin(),archive->end(),[](auto& a,auto& b) {
                return a.size()<b.size();
            });
            if (code.size()<worst->size()) *worst=code;
        };
        int goal_count = (int)goals.size();
        for (int level = 0; level < goal_count && !frontier.empty(); ++level) {
            std::vector<State> next;
            std::unordered_set<std::uint64_t> seen;
            for (const auto& state : frontier) {
                int ceiling = (archive || best.empty()) ? cap : (int)best.size() - 1;
                for (int i = 0; i < goal_count; ++i) {
                    if (state.covered & (1ULL << i)) continue;
                    std::vector<Script> moves;
                    extend(state.code, i, ceiling - (int)state.code.size(), moves);
                    for (const auto& move : moves) {
                        Script candidate = state.code;
                        candidate.insert(candidate.end(), move.begin(), move.end());
                        auto [count, mask] = metrics(candidate);
                        int remaining = goal_count - count;
                        if ((int)candidate.size() + remaining > ceiling) continue;
                        if (!remaining) {
                            record(candidate);
                            if (best.empty() || candidate.size() < best.size()) {
                                best = std::move(candidate);
                                if (!archive) ceiling = (int)best.size() - 1;
                            }
                            continue;
                        }
                        if (!seen.insert(signature(candidate)).second) continue;
                        int future = 0;
                        for (int j = 0; j < goal_count; ++j) if (!(mask & (1ULL << j))) {
                            int so_far = std::min(4, cap);
                            for (const auto& option : options[j]) {
                                int cost = 0;
                                for (const Step& s : option) {
                                    bool live = std::any_of(candidate.begin(), candidate.end(),
                                        [&](const auto& q) { return s.x == q.x; });
                                    cost += !live;
                                }
                                so_far = std::min(so_far, cost);
                            }
                            for (const auto& s : candidate) {
                                inverse(goals[j], s.x, limit, [&](Value a, char, bool) {
                                    auto n = d.index.get((SInt)a);
                                    if (n < 0) return false;
                                    so_far = std::min(so_far, 1 + d.e[n].cost);
                                    return so_far <= 1;
                                });
                            }
                            future += so_far;
                        }
                        int rank = 4 * (int)candidate.size() + 3 * future + remaining;
                        next.push_back({std::move(candidate), mask, rank});
                    }
                }
            }
            std::stable_sort(next.begin(), next.end(), [](const auto& a, const auto& b) {
                return std::make_pair(a.rank,a.code.size()) < std::make_pair(b.rank,b.code.size());
            });
            if ((int)next.size() > width) next.resize(width);
            frontier = std::move(next);
        }
        return best;
    }

    // A bounded backward search whose state is a SET of unresolved values.
    // Intermediate values can enter and leave this set on either output's
    // behalf. Unlike forward prefix search, no output has to finish first.
    Script backward(int max_width, const Script& incumbent = {}) const {
        struct Need {
            std::vector<Value> pending;
            Script reverse;
            double rank;
        };
        auto infer_cost = [&](Value x) {
            int id = d.index.get((SInt)x);
            if (id >= 0) return (double)d.e[id].cost;
            return 2.5 + .035 * bit_count(x);
        };
        std::vector<Need> queue{{goals,{},0.0}};
        Script best = incumbent;
        int ceiling = incumbent.empty() ? cap : (int)incumbent.size()-1;
        auto normalized = [](std::vector<Value>& values) {
            std::sort(values.begin(),values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        };
        std::vector<Value> atoms;
        atoms.insert(atoms.end(), ready.begin(), ready.end());
        // Add a small sample of cheapest exact forms. Bias towards one-step
        // magnitudes that are likely to give well-balanced inverse operands.
        for (int i=0; i<std::min(d.n1-d.n0,96); ++i)
            atoms.push_back((Value)d.e[d.n0 + (d.n1-d.n0) * i / 96].v);
        for (int level=0; level<ceiling && !queue.empty(); ++level) {
            std::vector<Need> next;
            std::unordered_set<std::uint64_t> signatures;
            for (const auto& state : queue) {
                if ((int)state.reverse.size()+(int)state.pending.size()>ceiling) continue;
                for (Value x : state.pending) {
                    std::unordered_set<Value> defined;
                    for (const auto& op : state.reverse) defined.insert(op.x);
                    auto candidate = [&](Value a, Value b, char op) {
                        if (!a || !b || a==x || b==x || defined.count(a) || defined.count(b)) return;
                        if (calculate(op,a,b,limit)!=x) return;
                        Need n = state;
                        n.reverse.push_back({x,op,a,b});
                        n.pending.erase(std::find(n.pending.begin(),n.pending.end(),x));
                        if (!leaves.count(a)) n.pending.push_back(a);
                        if (!leaves.count(b)) n.pending.push_back(b);
                        normalized(n.pending);
                        int work = (int)n.reverse.size()+(int)n.pending.size();
                        if (work > ceiling) return;
                        if (n.pending.empty()) {
                            ceiling = (int)n.reverse.size()-1;
                            best.assign(n.reverse.rbegin(), n.reverse.rend());
                            return;
                        }
                        double future=0;
                        for (Value v : n.pending) future+=infer_cost(v);
                        n.rank = n.reverse.size()+.78*future
                               + 0.19 * (int)n.pending.size();
                        std::uint64_t key = signature(n.reverse);
                        for (Value v : n.pending) U64hash(key,v ^ 0xb492b66fbe98f273ULL);
                        if (!signatures.insert(key).second) return;
                        next.push_back(std::move(n));
                    };
                    int fixed = d.index.get((SInt)x);
                    if (fixed >= 0) {
                        const auto& e = d.e[fixed];
                        if (e.op) candidate((Value)d.e[e.a].v,(Value)d.e[e.b].v,e.op);
                    }
                    auto root = std::find(goals.begin(),goals.end(),x);
                    if (root != goals.end()) for (const auto& option : options[root-goals.begin()]) {
                        if (!option.empty() && option.back().x == x) {
                            auto& last = option.back(); candidate(last.a,last.b,last.op);
                        }
                    }
                    const auto expand = [&](Value b) {
                        if (b == x || defined.count(b)) return;
                        inverse(x,b,limit,[&](Value a,char action,bool first) {
                            // Leave a difficult operand open only if the
                            // reduction is substantial, or connects two goals.
                            bool shared = std::find(state.pending.begin(),state.pending.end(),a) != state.pending.end();
                            if (a < x || shared || d.index.get((SInt)a) >= 0)
                                candidate(first?a:b,first?b:a,action);
                            return false;
                        });
                    };
                    for (Value b : state.pending) expand(b);
                    for (Value b : atoms) expand(b);
                }
            }
            std::size_t keep = std::min<std::size_t>(next.size(),max_width);
            std::partial_sort(next.begin(),next.begin()+keep,next.end(),[](const auto& a,const auto& b) {
                return std::make_pair(a.rank,a.pending.size()) < std::make_pair(b.rank,b.pending.size());
            });
            next.resize(keep);
            queue = std::move(next);
        }
        return best;
    }
};

struct Cut {
    std::vector<int> roots;
    std::vector<char> fixed, terminal;
    int budget, priority;
};

Cut make_cut(const Graph& graph, const std::vector<int>& roots, int minimum = 1,
             int retained = -1) {
    const int n = (int)graph.nodes.size();
    std::vector<char> fixed(n), blocked(n), removed(n);
    for (int id : roots) removed[id] = 1;
    std::function<void(int)> visit = [&](int id) {
        if (fixed[id] || removed[id]) return;
        fixed[id] = 1;
        const auto& node = graph.nodes[id];
        if (node.op) { visit(node.a); visit(node.b); }
    };
    for (int t : graph.targets) visit(t);
    for (int id : graph.initial) if (id >= 0) visit(id);
    if (retained >= 0) visit(retained);
    int budget = 0;
    for (int i = 0; i < n; ++i) {
        const auto& node = graph.nodes[i];
        blocked[i] = removed[i] || (node.op && (blocked[node.a] || blocked[node.b]));
        budget += !fixed[i] && node.op;
    }
    if (budget <= (int)roots.size() && budget < minimum) budget = 0;
    std::vector<char> terminal = fixed;
    for (int i = 0; i < n; ++i) terminal[i] &= !blocked[i];
    return {roots, std::move(fixed), std::move(terminal), budget, 0};
}

std::vector<SInt> terminal_values(const Graph& graph, const Cut& cut, int limit,
                                  int variety = 0) {
    std::vector<std::pair<double,int>> ranks;
    std::vector<SInt> ready;
    ranks.reserve(graph.nodes.size());
    const auto lg = [](Value v) { return std::log2(1.0 + (double)v); };
    for (int i = 0; i < (int)graph.nodes.size(); ++i) {
        if (!cut.terminal[i]) continue;
        const auto& node = graph.nodes[i];
        Value v = node.value;
        double weight = node.op ? 20.0 : -30.0;
        for (int root : cut.roots) {
            Value t = graph.nodes[root].value;
            Value delta = v < t ? t - v : v - t;
            double favor = 2.1 + 0.21 * lg(delta);
            if (t > v) {
                Value r = t % v;
                double q = lg(t / v), rem = lg(std::min(r, v-r));
                favor = std::min(favor, 1.0 + (variety ? .24 : .16) * q + .28 * rem);
            }
            weight = std::min(weight, favor);
        }
        if (v < 40) weight -= 5.0;
        if (v && !(v & (v-1))) weight -= 4.0;
        ranks.emplace_back(weight,i);
    }
    auto keep = std::min<std::size_t>(ranks.size(), limit);
    std::partial_sort(ranks.begin(),ranks.begin()+keep,ranks.end());
    for (std::size_t i = 0; i < keep; ++i) ready.push_back((SInt)graph.nodes[ranks[i].second].value);
    return ready;
}

Program emit_graph(const Graph& graph, Value limit, int cap, int trials) {
    int work = 0;
    for (const auto& n : graph.nodes) work += n.op != 0;
    if (work >= cap) return {};
    Program best;
    auto order = graph.targets;
    std::mt19937 random((std::uint32_t)(limit ^ work));
    for (int i = 0; i < trials; ++i) {
        if (i == 4) std::reverse(order.begin(), order.end());
        if (i == 8) order = live_frontier_order(graph, -1, 3, work);
        if (i >= 12) std::shuffle(order.begin(), order.end(), random);
        auto code = Scheduler(graph, cap, limit, i % 4, (i / 2) % 2).run(order);
        if (!code.empty() && (int)code.size() < cap) {
            cap = (int)code.size();
            best = std::move(code);
            if (cap == work) break;
        }
    }
    return best;
}

Program replace_block(const Graph& graph, const Cut& cut, const Script& block,
                       Value limit, int cap, int trials) {
    Graph draft = graph;
    Index known, outputs;
    for (int i = 0; i < (int)graph.nodes.size(); ++i)
        if (cut.terminal[i]) known.emplace(graph.nodes[i].value, i);
    for (int root : cut.roots) outputs.emplace(graph.nodes[root].value, root);
    for (const auto& op : block) {
        if (known.count(op.x)) continue;
        if (calculate(op.op, op.a, op.b, limit) != op.x ||
            !known.count(op.a) || !known.count(op.b))
            throw std::logic_error("Invalid synthesised block");
        Node node{op.x, op.op, known.at(op.a), known.at(op.b)};
        if (auto have = outputs.find(op.x); have != outputs.end()) {
            known.emplace(op.x, have->second);
            draft.nodes[have->second] = node;
        } else {
            known.emplace(op.x, (int)draft.nodes.size());
            draft.nodes.push_back(node);
        }
    }
    for (auto [value, id] : outputs)
        if (!known.count(value)) throw std::logic_error("Incomplete synthesised block");
    return emit_graph(topological_graph(draft), limit, cap, trials);
}

std::vector<std::pair<int,std::vector<int>>> choose_cuts(const Graph& g, int group,
                                                        int jobs, int variant) {
    std::vector<int> targets;
    for (int root : g.targets) if (g.nodes[root].op) targets.push_back(root);
    const int n = (int)targets.size();
    if (n < group || n > 63) return {};
    std::vector<std::uint64_t> used(g.nodes.size(),0);
    for (int i = 0; i < n; ++i) {
        std::function<void(int)> visit = [&](int id) {
            auto& visits = used[id];
            if (visits & (1ULL << i)) return;
            visits |= 1ULL << i;
            if (g.nodes[id].op) { visit(g.nodes[id].a); visit(g.nodes[id].b); }
        };
        visit(targets[i]);
    }
    std::vector<std::pair<std::uint64_t,int>> support;
    std::map<std::uint64_t,int> counts;
    for (int i = 0; i < (int)g.nodes.size(); ++i)
        if (used[i] && g.nodes[i].op) ++counts[used[i]];
    for (auto [mask,f] : counts)
        if (g.targets.size() < 64) support.emplace_back(mask,f);
    std::vector<std::pair<int,std::vector<int>>> candidate;
    std::vector<int> indexes;
    std::function<void(int,std::uint64_t)> walk = [&](int begin, std::uint64_t mask) {
        if ((int)indexes.size() == group) {
            int budget = 0, shared = 0;
            for (auto [m, amount] : support) if (!(m & ~mask)) {
                budget += amount;
                if (m & (m - 1)) shared += amount;
            }
            std::vector<int> roots;
            for (int i : indexes) roots.push_back(targets[i]);
            // A target can be an input to a retained target. Its entire
            // dependency cone looks shared to the footprint approximation,
            // but only its result has to survive. Recompute that cut exactly.
            if (std::any_of(roots.begin(),roots.end(),[&](int v){ return (used[v]&~mask)!=0; }))
                budget=make_cut(g,roots).budget;
            if (budget < group + 2) return;
            std::uint64_t hash = 0;
            for (int i : roots) Blocks::U64hash(hash,(Value)i);
            int rank = budget*15 + shared*8;
            for (int i = 0; i < group; ++i) for (int j = i+1; j < group; ++j) {
                Value a = g.nodes[roots[i]].value, b = g.nodes[roots[j]].value;
                Value difference = a < b ? b-a : a-b;
                int near = bit_count(std::max(a,b))-bit_count(difference);
                rank += (variant == 0 ? 4 : 9)*near;
            }
            rank += variant >= 2 ? (int)(hash % 55) : (int)(hash % 7);
            candidate.emplace_back(-rank, std::move(roots));
            return;
        }
        for (int i = begin; i < n; ++i) {
            indexes.push_back(i);
            walk(i+1,mask | (1ULL << i));
            indexes.pop_back();
        }
    };
    walk(0,0);
    auto keep=std::min<std::size_t>(candidate.size(),jobs);
    if (variant >= 5) {
        // Search several disjoint parts of the program; otherwise the same
        // expensive target can dominate nearly every proposed group.
        std::vector<std::pair<int,std::vector<int>>> selected;
        std::vector<int> frequency(g.nodes.size());
        for (std::size_t i=0;i<keep;++i) {
            auto penalty = [&](const auto& task) {
                int reuse=0;
                for (int v : task.second) reuse+=frequency[v];
                return task.first + 29*reuse;
            };
            auto it=std::min_element(candidate.begin()+i,candidate.end(),[&](const auto& a,const auto& b) {
                return penalty(a)<penalty(b);
            });
            std::iter_swap(candidate.begin()+i,it);
            for (int v : candidate[i].second) ++frequency[v];
            selected.push_back(std::move(candidate[i]));
        }
        return selected;
    }
    std::partial_sort(candidate.begin(), candidate.begin()+keep, candidate.end());
    candidate.resize(keep);
    return candidate;
}

Program contextual(const Registers& regs, const std::vector<Value>& targets, Value limit,
                   Program best, int group, int jobs, int breadth, int depth,
                   int leaves, int variant, int& wins) {
    Program active = best;
    const int rounds = variant>=6 && variant<8 ? 2 : 1;
    auto same = [](const Program& a,const Program& b) {
        if (a.size()!=b.size()) return false;
        for (std::size_t i=0;i<a.size();++i)
            if (std::make_tuple(a[i].kind,a[i].a,a[i].b,a[i].destination) !=
                std::make_tuple(b[i].kind,b[i].a,b[i].b,b[i].destination)) return false;
        return true;
    };
    for (int turn=0;turn<rounds;++turn) {
        const auto graph = topological_graph(recover_graph(regs,limit,targets,active));
        auto jobs_to_try = choose_cuts(graph,group,jobs,variant+turn);
        const int logical = (int)std::count_if(graph.nodes.begin(),graph.nodes.end(),[](const auto& x){return x.op!=0;});
        bool shorter = false;
        Program neutral;
        for (const auto& [rank, roots] : jobs_to_try) {
            auto cut = make_cut(graph,roots,group+2);
            if (variant>=8 && cut.budget>=group+3) {
                std::vector<std::pair<int,int>> preserve;
                for (int root : roots) {
                    auto& node=graph.nodes[root];
                    for (int vertex:{node.a,node.b}) {
                        if (vertex<0 || cut.fixed[vertex] || !graph.nodes[vertex].op) continue;
                        int rank=bit_count(graph.nodes[vertex].value);
                        preserve.emplace_back(-rank,vertex);
                    }
                }
                std::sort(preserve.begin(),preserve.end());
                for (auto [rank, vertex] : preserve) {
                    auto shallower=make_cut(graph,roots,group+2,vertex);
                    if (shallower.terminal[vertex] && shallower.budget>=group+1) {
                        cut=std::move(shallower);break;
                    }
                }
            }
            const bool mutate = variant>=6 && turn+1<rounds && neutral.empty() && !shorter;
            // Only traverse a neutral plateau in the first round. The final
            // portfolio never selects a longer program.
            int measured_cap = (int)best.size() + (mutate ? 1 : 0);
            const int budget = std::min(19,cut.budget - 1 + measured_cap-logical);
            if (budget < group) continue;
            auto terminal = terminal_values(graph,cut,leaves,variant);
            std::vector<Value> values;
            for (int root : roots) values.push_back(graph.nodes[root].value);
            Blocks model(limit,values,terminal,breadth,depth,budget);
            auto found = model.search();
            if (variant >= 3 && variant != 5 && variant != 8)
                found = model.backward(std::min(breadth,8),found);
            if (found.empty()) continue;
            auto next = replace_block(graph,cut,found,limit,measured_cap,16);
            if (!next.empty() && next.size() < best.size()) {
                best = std::move(next);
                ++wins;
                shorter = true;
            } else if (mutate && !next.empty() && next.size()==best.size() && !same(next,active)) {
                neutral = std::move(next);
            }
        }
        if (shorter) active=best;
        else if (!neutral.empty()) active=std::move(neutral);
        else break;
    }
    return best;
}

Program blind(const std::vector<Value>& starts, const std::vector<Value>& targets, Value limit,
              int group, int jobs, int breadth, int depth, int leaves, int variant, int& wins) {
    Construction c(limit,starts,targets);
    local_beam::Dictionary dictionary((SInt)limit);
    for (;;) {
        std::vector<Value> open;
        for (Value t : c.targets) if (!c.index.count(t)) open.push_back(t);
        if (open.empty()) break;
        const auto ready = c.leaves(leaves);
        dictionary.build(ready);
        const auto pivot = variant == 0 ? open.front() : open.back();
        std::sort(open.begin(),open.end(),[&](Value a, Value b) {
            auto close = [&](Value t) {
                if (t == pivot) return -999.0;
                Value diff = t < pivot ? pivot-t : t-pivot;
                return (double)bit_count(diff)
                       - 0.65 * bit_count(std::gcd(t,pivot));
            };
            return std::make_pair(close(a),a) < std::make_pair(close(b),b);
        });
        open.resize(std::min<std::size_t>(open.size(),group));
        std::vector<Script> fallback;
        int separate = 0;
        for (auto target : open) {
            auto candidate = synthesize_new(c,dictionary,target,std::max(4,breadth/2),true);
            separate += (int)candidate.size();
            fallback.push_back(std::move(candidate));
        }
        Script block;
        if (open.size() >= 2) {
            Blocks model(limit,open,ready,breadth,depth,separate);
            block = model.search(fallback);
            if (variant >= 3) block = model.backward(std::min(breadth,8),block);
        }
        if (block.empty()) block = std::move(fallback.front());
        else if (block.size() < (size_t)separate) ++wins;
        if (block.empty()) throw std::runtime_error("Blind solver made no progress");
        for (auto step : block) c.append(step);
    }
    for (Value t : targets) c.g.targets.push_back(c.index.at(t));
    return emit_graph(topological_graph(c.g),limit,4097,20);
}

#ifndef ELF_BLOCK_LIBRARY
int main(int argc,char** argv) {
    try {
        std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
        const int group = argc > 1 ? std::stoi(argv[1]) : 2;
        const int jobs = argc > 2 ? std::stoi(argv[2]) : 12;
        const int breadth = argc > 3 ? std::stoi(argv[3]) : 8;
        const int depth = argc > 4 ? std::stoi(argv[4]) : 4;
        const int leaves = argc > 5 ? std::stoi(argv[5]) : 64;
        const int variant = argc > 6 ? std::stoi(argv[6]) : 0;
        if (group < 1 || group > 4 || breadth < 1 || breadth > 30 || leaves < 6 || leaves > 128)
            throw std::invalid_argument("Unsupported search profile");
        local_beam::INDEX_BITS = leaves > 86 ? 19 : 18;
        local_beam::ALPHA = .26;
        std::string key;
        Value limit;
        int ns, nt, n;
        while (std::cin >> key >> limit >> ns >> nt >> n) {
            if (ns < 1 || ns > slots || nt < 1 || nt > 63 || limit == 0 || limit >= (Value{1} << 62))
                throw std::runtime_error("Puzzle outside the supported input bounds");
            std::vector<Value> starts(ns), targets(nt);
            Registers regs{};
            for (int i=0;i<ns;++i) std::cin >> starts[i], regs[i]=starts[i];
            for (auto& t : targets) std::cin >> t;
            int accepted = 0;
            Program result;
            if (n < 0) {
                std::sort(targets.begin(),targets.end());
                result = blind(starts,targets,limit,group,jobs,breadth,depth,leaves,variant,accepted);
            } else {
                Program old(n);
                for (auto& action : old) std::cin >> action.kind >> action.a >> action.b >> action.destination;
                result = contextual(regs,targets,limit,old,group,jobs,breadth,depth,leaves,variant,accepted);
            }
            std::cout << key << ' ' << result.size() << ' ' << accepted << '\n';
            for (const auto& op : result)
                std::cout << op.kind << ' ' << op.a << ' ' << op.b << ' ' << op.destination << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
