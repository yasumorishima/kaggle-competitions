
bool crossover(const Graph& base, const std::vector<Graph>& archives,
                   Graph& winning, Tape& best, const std::vector<Value>& extras={}, const std::unordered_set<Value>& holes={}) {
        std::vector<Value> values=starts;
        std::sort(values.begin(),values.end());
        values.erase(std::unique(values.begin(),values.end()),values.end());
        int initial=int(values.size());
        std::unordered_map<Value,int> ids;
        for(int i=0;i<initial;i++)ids[values[i]]=i;
        auto append=[&](const Graph& g){
            std::vector<Value> seq;
            if(!order_for(g,targets,seq))return;
            for(Value x:seq)if(!ids.count(x)){ids[x]=int(values.size());values.push_back(x);}
        };
        append(base);
        const int orig=int(values.size());
        for(auto& g:archives)append(g);
        for(Value x:extras) if(!ids.count(x)){ids[x]=int(values.size());values.push_back(x);}
        int n=int(values.size());
        std::vector<std::vector<PairRule>> recipes(n);
        for(int i=initial;i<orig;i++){auto r=base.at(values[i]);recipes[i].push_back({ids.at(r.a),ids.at(r.b),r.op});}
        for(auto& g:archives)for(auto [v,r]:g)if(r.op&&ids.count(v)&&ids.count(r.a)&&ids.count(r.b))
            recipes[ids.at(v)].push_back({ids.at(r.a),ids.at(r.b),r.op});
        for(int i=0;i<n;i++)for(int j=0;j<=i;j++) {
            int a=values[i]>=values[j]?i:j,b=i+j-a;
            for(char op:{'+','-','*','/'}){
                Value x=calculate(op,values[a],values[b]);
                auto it=ids.find(x);
                if(it!=ids.end()&&it->second>=initial&&x!=values[a]&&x!=values[b])recipes[it->second].push_back({a,b,op});
            }
        }
        std::vector<PairRule> edges;
        std::vector<int> cuts(n+1),needed(n);
        int goal_count=0;
        for(auto v:targets)if(ids.at(v)>=initial){needed[ids.at(v)]=1;++goal_count;}
        for(int i=initial;i<n;i++){cuts[i]=int(edges.size());edges.insert(edges.end(),recipes[i].begin(),recipes[i].end());}
        cuts[n]=int(edges.size());
        std::vector<unsigned char> banned(n),seen(n);
        std::vector<int> pick(n);
        auto close=[&]() {
            int remaining=goal_count;
            std::fill(seen.begin(),seen.end(),0);
            std::fill_n(seen.begin(),initial,1);
            bool progress=true;
            while(progress&&remaining) {
                progress=false;
                for(int v=initial;v<n;v++)if(!seen[v]&&!banned[v]){
                    for(int j=cuts[v];j<cuts[v+1];j++){
                        auto r=edges[j];if(!seen[r.a]||!seen[r.b])continue;
                        seen[v]=1;pick[v]=j;remaining-=needed[v];progress=true;break;
                    }
                }
            }
            return !remaining;
        };
        std::vector<int> support(n);
        for(auto& g:archives)for(auto [v,r]:g)if(ids.count(v))++support[ids.at(v)];
        std::vector<int> optional;
        for(int i=initial;i<n;i++)if(!needed[i])optional.push_back(i);
        for(int attempt=0;attempt<7;attempt++){
            std::vector<std::pair<double,int>> ranking;
            for(int i:optional){
                double x=double(rng())/std::mt19937::max();
                if(attempt%3!=0)x+=(attempt%3==1?0.22:0.09)*support[i];
                if(!extras.empty() && attempt%3!=0) {
                    if(i<orig) x+=0.32;
                    if(holes.count(values[i])) x-=0.18+0.11*attempt;
                }
                ranking.emplace_back(x,i);
            }
            std::sort(ranking.begin(),ranking.end());
            std::fill(banned.begin(),banned.end(),0);
            for(auto [priority,i]:ranking){banned[i]=1;if(!close())banned[i]=0;}
            if(!close())continue;
            Graph trial;
            for(int i=0;i<initial;i++)trial[values[i]]={0,0,0};
            for(int i=initial;i<n;i++)if(seen[i]){auto r=edges[pick[i]];trial[values[i]]={r.op,values[r.a],values[r.b]};}
            std::vector<Value> sequence;
            Tape tape;
            if(order_for(trial,targets,sequence)&&sequence.size()<best.size()&&compile(trial,tape)){
                winning=live_only(trial,sequence);best=std::move(tape);return true;
            }
        }
        return false;
    }
