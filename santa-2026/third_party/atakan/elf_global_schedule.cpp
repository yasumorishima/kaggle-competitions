#define ELF_LIBRARY
#include "elf_window.cpp"

// A no-recomputation register scheduler. A result that dies immediately
// only needs a destination for its single write.
Program pressure_schedule(const Graph& graph, int variant, std::uint32_t seed, int cap) {
    const int n=(int)graph.nodes.size();
    std::vector<int> remain(n),indegree(n),where(n,-1),height(n);
    std::vector<std::vector<int>> users(n);
    std::vector<int> ready;
    int need=0;
    for (int i=0;i<slots;++i) if (graph.initial[i]>=0) where[graph.initial[i]]=i;
    for (int i=0;i<n;++i) {
        const auto& v=graph.nodes[i];
        if (!v.op) continue;
        for (int u:{v.a,v.a==v.b?-1:v.b}) if (u>=0) {
            ++remain[u];
            users[u].push_back(i);
            indegree[i]+=graph.nodes[u].op!=0;
        }
        if (!indegree[i]) ready.push_back(i);
        ++need;
    }
    if (need>=cap) return {};
    for (int i=n-1;i>=0;--i) for (int u:users[i]) height[i]=std::max(height[i],height[u]+1);
    std::array<bool,slots> taken{};
    for (int i=0;i<n;++i) if (where[i]>=0 && remain[i]) taken[where[i]]=true;
    std::mt19937 random(seed);
    std::vector<int> priorities(n);
    for (int& x:priorities) x=int(random()%10000u);
    Program program;
    while(!ready.empty()) {
        auto rank=[&](int id) {
            const auto& v=graph.nodes[id];
            int freed=(remain[v.a]==1)+(v.a!=v.b && remain[v.b]==1);
            int pressure=int(remain[id]>0)-freed;
            int bridge=height[id];
            int tie=variant<3?id:priorities[id];
            return std::make_tuple(pressure*(variant%3==2?3:8) +
                                   (variant%3==1?bridge:-bridge), tie);
        };
        auto pick=std::min_element(ready.begin(),ready.end(),[&](int a,int b){return rank(a)<rank(b);});
        int id=*pick;
        *pick=ready.back();ready.pop_back();
        auto v=graph.nodes[id];
        int a=where[v.a],b=where[v.b];
        if (a<0 || b<0) return {};
        if (!--remain[v.a]) taken[a]=false;
        if (v.a!=v.b && !--remain[v.b]) taken[b]=false;
        auto free=std::find(taken.begin(),taken.end(),false);
        if (free==taken.end()) return {};
        int dest=int(free-taken.begin());
        program.push_back({v.op,a,b,dest});
        if (remain[id]) {taken[dest]=true;where[id]=dest;}
        for (int next:users[id]) if (!--indegree[next]) ready.push_back(next);
    }
    return (int)program.size()==need ? program : Program{};
}

#ifndef ELF_GLOBAL_LIBRARY
int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        int attempts = argc > 1 ? std::stoi(argv[1]) : 24;
        std::string id;
        Value limit;
        int n_starts, n_targets, n_recipes, cap;
        while (std::cin >> id >> limit >> n_starts >> n_targets >> n_recipes >> cap) {
            Graph graph;
            graph.initial.fill(-1);
            Index index;
            for (int slot = 0; slot < n_starts; ++slot) {
                Value number;
                std::cin >> number;
                auto [entry, added] = index.emplace(number, static_cast<int>(graph.nodes.size()));
                if (added) graph.nodes.push_back({number, 0, -1, -1});
                graph.initial.at(slot) = entry->second;
            }
            std::vector<Value> targets(n_targets);
            for (auto& target : targets) std::cin >> target;
            for (int j = 0; j < n_recipes; ++j) {
                Value x, a, b;
                char op;
                std::cin >> x >> op >> a >> b;
                if (calculate(op, a, b, limit) != x || index.count(x))
                    throw std::runtime_error("Invalid or duplicate planned value");
                index.emplace(x, static_cast<int>(graph.nodes.size()));
                graph.nodes.push_back({x, op, index.at(a), index.at(b)});
            }
            for (Value target : targets) graph.targets.push_back(index.at(target));
            std::mt19937 random(static_cast<std::uint32_t>(limit ^ graph.nodes.size()));
            Program best;
            auto order = graph.targets;
            for (int k=0;k<std::min(attempts,16);++k) {
                best=pressure_schedule(graph,k,random(),cap);
                if (!best.empty()) break;
            }
            bool compact=!best.empty();
            for (int trial = 0; trial < attempts && !compact; ++trial) {
                if (trial == 8) std::reverse(order.begin(), order.end());
                if (trial > 8 && !order.empty()) {
                    if (trial % 7 == 0) order = graph.targets;
                    std::swap(order[random() % order.size()], order[random() % order.size()]);
                }
                auto output = Scheduler(graph, cap, limit, trial % 4, (trial / 4) % 2).run(order);
                if (!output.empty() && static_cast<int>(output.size()) < cap) {
                    cap = static_cast<int>(output.size());
                    best = std::move(output);
                }
            }
            std::cout << id << ' ' << best.size() << '\n';
            for (auto op : best)
                std::cout << op.kind << ' ' << op.a << ' ' << op.b << ' ' << op.destination << '\n';
        }
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
#endif
