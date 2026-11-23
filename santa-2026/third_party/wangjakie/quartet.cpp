
#include <cmath>
#include "common.inc"
#include "compact.inc"

    struct Gate { int output, other; Recipe step; };
    struct Alternative { int left, right, count; uint32_t tie; };
    using OwnerMask = uint64_t;

    bool search_quartets(Graph& graph, Tape& best,
                         const std::vector<std::pair<Value,Value>>& swaps) {
        std::vector<Value> ordered;
        if (!order_for(graph, targets, ordered)) return false;
        std::vector<Value> values;
        std::unordered_map<Value,int> index;
        auto add=[&](Value x) {
            if(index.emplace(x,int(values.size())).second)values.push_back(x);
        };
        for(Value x:starts) add(x);
        const int initial=int(values.size());
        for(Value x:ordered) add(x);
        const int incumbent=int(values.size());
        std::unordered_map<Value,int> support;
        for(auto [original,substitute]:swaps)++support[substitute];
        std::vector<std::pair<uint64_t,Value>> ranked;
        for(auto [value,count]:support)
            ranked.emplace_back((uint64_t(count)<<32)|rng(),value);
        std::sort(ranked.rbegin(),ranked.rend());
        for(size_t i=0;i<std::min<size_t>(ranked.size(),105);++i)add(ranked[i].second);
        // A bridge can participate in two different arithmetic identities before it occurs
        // in any feasible program. Include a limited number of those untested integers.
        std::unordered_map<Value,Alternative> bridges;
        const int known=int(values.size());
        for(int i=0;i<known;++i)for(int j=0;j<=i;++j) {
            Value a=std::max(values[i],values[j]),b=std::min(values[i],values[j]);
            for(char op:{'+','-','*','/'}) {
                Value x=calculate(op,a,b);
                if(!x || index.count(x))continue;
                auto [it,fresh]=bridges.try_emplace(x,Alternative{i,j,1,0});
                if(!fresh && (it->second.left!=i || it->second.right!=j)) {
                    ++it->second.count;
                    if(it->second.count==2)it->second.tie=rng();
                }
            }
        }
        ranked.clear();
        for(auto [v,a]:bridges) if(a.count>=2)
            ranked.emplace_back((uint64_t(std::min(12,a.count))<<32)|a.tie,v);
        std::sort(ranked.rbegin(),ranked.rend());
        for(size_t i=0;i<std::min<size_t>(ranked.size(),130);++i)add(ranked[i].second);
        int n=int(values.size());
        std::vector<std::vector<Gate>> next(n);
        auto connect=[&](int x,int a,int b,Recipe r) {
            next[a].push_back({x,b,r});
            if(a!=b)next[b].push_back({x,a,r});
        };
        for(int i=0;i<n;i++)for(int j=0;j<=i;j++){
            int a=values[i]>=values[j]?i:j,b=i+j-a;
            for(char op:{'+','-','*','/'}) {
                auto it=index.find(calculate(op,values[a],values[b]));
                if(it!=index.end() && it->second>=initial && it->second!=a && it->second!=b)
                    connect(it->second,a,b,{op,values[a],values[b]});
            }
        }
        std::vector<OwnerMask> owners(n),goal(n);
        int needed=0;
        for(int i=0;i<int(targets.size());++i) {
            int t=index.at(targets[i]);goal[t]=OwnerMask(1)<<i;
            if(t>=initial) ++needed;
        }
        if(!needed)return false;
        for(auto it=ordered.rbegin();it!=ordered.rend();++it) {
            int v=index.at(*it);auto r=graph.at(*it);
            OwnerMask mask=goal[v]?goal[v]:owners[v];
            owners[index.at(r.a)]|=mask; owners[index.at(r.b)]|=mask;
        }
        std::map<OwnerMask,int> footprints;
        for(int v=initial;v<incumbent;++v)
            if(!goal[v] && owners[v] && __builtin_popcountll(owners[v])<=4)++footprints[owners[v]];
        std::vector<OwnerMask> clusters;
        for(auto entry:footprints) if(__builtin_popcountll(entry.first)>=2) {
            OwnerMask mask=entry.first;
            while(__builtin_popcountll(mask)<std::min<int>(4,targets.size())) {
                int best_weight=-1;OwnerMask chosen=0;
                for(int i=0;i<int(targets.size());i++) {
                    OwnerMask bit=OwnerMask(1)<<i;
                    if(mask&bit)continue;
                    int weight=0;
                    for(auto [m,k]:footprints) if((m&bit)&&!(m&~(mask|bit)))weight+=k;
                    if(weight>best_weight){best_weight=weight;chosen=bit;}
                }
                if(!chosen)break;
                mask|=chosen;
            }
            clusters.push_back(mask);
        }
        if(clusters.empty())clusters.push_back(~OwnerMask(0));
        std::sort(clusters.begin(),clusters.end());
        clusters.erase(std::unique(clusters.begin(),clusters.end()),clusters.end());
        std::vector<int> queue(n),seen(n);
        std::vector<char> banned(n),witness(n);
        std::vector<Recipe> recipes(n);
        int stamp=0;
        auto close=[&](bool capture) {
            ++stamp;
            int head=0,tail=0,remaining=needed;
            for(int i=0;i<initial;++i){queue[tail++]=i;seen[i]=stamp;}
            while(head<tail && remaining) {
                int a=queue[head++];
                for(const auto& e:next[a]) {
                    if(seen[e.output]==stamp||banned[e.output]||seen[e.other]!=stamp)continue;
                    seen[e.output]=stamp;queue[tail++]=e.output;
                    if(capture)recipes[e.output]=e.step;
                    if(goal[e.output])--remaining;
                }
            }
            return remaining==0;
        };
        auto capture=[&]() {
            if(!close(true))return false;
            std::fill(witness.begin(),witness.end(),0);
            auto mark=[&](auto&& self,int v)->void {
                if(v<initial||witness[v])return;
                witness[v]=1;auto r=recipes[v];
                self(self,index.at(r.a));self(self,index.at(r.b));
            };
            for(Value t:targets)mark(mark,index.at(t));
            int size=int(std::count(witness.begin(),witness.end(),char(1)));
            if(size>=int(best.size()))return false;
            Graph trial;for(int i=0;i<initial;++i)trial[values[i]]={0,0,0};
            for(int i=initial;i<n;++i)if(witness[i])trial[values[i]]=recipes[i];
            Tape tape;
            if(!compile(trial,tape)||tape.size()>=best.size())return false;
            best=std::move(tape);graph=std::move(trial);return true;
        };
        for(int attempt=0;attempt<rounds;attempt++) {
            OwnerMask scope=clusters[attempt%clusters.size()];
            if(attempt%4==0)std::shuffle(clusters.begin(),clusters.end(),rng);
            if(attempt%7==0)scope=~OwnerMask(0);
            std::vector<std::pair<double,int>> schedule;
            std::fill(banned.begin(),banned.end(),0);
            for(int i=initial;i<n;i++)if(!goal[i]){
                if(i<incumbent && !(owners[i]&scope))continue;
                double rank=double(rng())/std::mt19937::max();
                if(i>=incumbent)rank+=0.12*(attempt%5);
                else if(!(owners[i]&~scope))rank-=0.19;
                schedule.emplace_back(rank,i);
            }
            std::sort(schedule.begin(),schedule.end());
            if(capture())return true;
            for(auto [rank,v]:schedule) {
                banned[v]=1;
                if(!witness[v])continue;
                if(!close(false)){banned[v]=0;continue;}
                if(capture())return true;
            }
        }
        return false;
    }
    Tape improve(const std::vector<Tape>& tapes) {
        Tape best=tapes.front();Graph graph=graph_from(best);
        for(int pass=0;pass<3;++pass) {
            std::vector<std::pair<Value,Value>> swaps;
            if(knockout(graph,best,&swaps))continue;
            if(!search_quartets(graph,best,swaps))break;
        }
        return best;
    }
};
#include "main.inc"
