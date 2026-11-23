
struct PairRule {int a,b;char op;};
    struct Use {int output, other; Recipe step;};
    struct Middle {Value value; std::vector<PairRule> make; std::vector<Use> use;};
    using Bits=std::array<uint64_t,8>;

    bool knockout(Graph& graph,Tape& best, std::vector<std::pair<Value,Value>>* swaps=nullptr, std::vector<Graph>* neighbors=nullptr, int width=0) {
        std::vector<Value> order;
        if(!order_for(graph,targets,order)) return false;
        graph=live_only(graph,order);
        std::vector<Value> val=starts;
        std::sort(val.begin(),val.end());
        val.erase(std::unique(val.begin(),val.end()),val.end());
        int first=int(val.size());
        val.insert(val.end(),order.begin(),order.end());
        int n=int(val.size()),words=(n+63)/64;
        if(words>8) return false;
        std::unordered_map<Value,int> id,mid;
        for(int i=0;i<n;i++) id[val[i]]=i;
        std::vector<Middle> lib;
        std::vector<std::vector<PairRule>> direct(n);
        std::vector<int> roots;
        std::vector<bool> required(n);
        for(Value t:targets){int i=id.at(t);if(i>=first)roots.push_back(i);required[i]=true;}
        for(int i=first;i<n;i++){auto r=graph.at(val[i]);direct[i].push_back({id.at(r.a),id.at(r.b),r.op});}
        for(int i=0;i<n;i++) for(int j=0;j<=i;j++){
            int a=val[i]>=val[j]?i:j,b=i+j-a;
            for(char op:{'+','-','*','/'}){
                Value x=calculate(op,val[a],val[b]);
                if(!x || x==val[a] || x==val[b])continue;
                auto hit=id.find(x);
                if(hit!=id.end()){if(hit->second>=first)direct[hit->second].push_back({a,b,op});}
                else {
                    auto [it,fresh]=mid.emplace(x,int(lib.size()));
                    if(fresh)lib.push_back({x,{},{}});
                    lib[it->second].make.push_back({a,b,op});
                }
            }
        }
        for(int v=first;v<n;v++) for(int f=0;f<n;f++) if(f!=v) {
            for(auto r:inverses(val[v],val[f])) {
                Value h=r.a==val[f]?r.b:r.a;
                auto it=mid.find(h);
                if(it!=mid.end())lib[it->second].use.push_back({v,f,r});
            }
        }
        for(auto& h:lib)for(char op:{'+','*','/'}) {
            Value x=calculate(op,h.value,h.value);
            auto it=id.find(x);
            if(it!=id.end() && it->second>=first)h.use.push_back({it->second,-1,{op,h.value,h.value}});
        }
        std::vector<Bits> allow(n),base(n);
        Bits selectable{};
        for(int i=first;i<n;i++)if(!required[i])selectable[i/64]|=uint64_t(1)<<(i%64);
        for(int i=0;i<n;i++) {
            allow[i]=selectable;allow[i][i/64]&=~(uint64_t(1)<<(i%64));
            if(i<first)base[i]=selectable;
        }
        auto close_word=[&](std::vector<Bits>& rows,int w,const Middle* h,uint64_t hmask){
            bool changed=true;
            while(changed){
                changed=false;
                for(int v=first;v<n;v++){
                    uint64_t previous=rows[v][w],open=allow[v][w]&~previous;
                    if(!open)continue;
                    uint64_t gain=0;
                    for(auto e:direct[v])gain|=rows[e.a][w]&rows[e.b][w];
                    gain&=open;
                    if(gain){rows[v][w]|=gain;changed=true;}
                }
                if(h)for(auto &u:h->use){
                    uint64_t gain=hmask&allow[u.output][w]&~rows[u.output][w];
                    if(u.other>=0)gain&=rows[u.other][w];
                    if(gain){rows[u.output][w]|=gain;changed=true;}
                }
            }
        };
        auto reconstruct=[&](const Middle* h,int ban1,int ban2,Graph* neutral=nullptr) {
            std::vector<unsigned char> got(n);
            std::fill_n(got.begin(),first,1);
            std::vector<Recipe> chosen(n);
            Recipe helper{};
            bool progress=true;
            while(progress){
                progress=false;
                for(int v=first;v<n;v++)if(v!=ban1&&v!=ban2&&!got[v])for(auto e:direct[v]){
                    if(got[e.a]&&got[e.b]){got[v]=1;chosen[v]={e.op,val[e.a],val[e.b]};progress=true;break;}
                }
                if(h&&!helper.op)for(auto e:h->make)if(got[e.a]&&got[e.b]){
                    helper={e.op,val[e.a],val[e.b]};progress=true;break;
                }
                if(h&&helper.op)for(auto &u:h->use)if(!got[u.output]&&u.output!=ban1&&u.output!=ban2&&(u.other<0||got[u.other])){
                    got[u.output]=1;chosen[u.output]=u.step;progress=true;
                }
            }
            for(int v:roots)if(!got[v])return false;
            Graph trial;
            for(int i=0;i<n;i++)if(got[i])trial.emplace(val[i],chosen[i]);
            if(h&&helper.op)trial.emplace(h->value,helper);
            std::vector<Value> live;
            Tape tape;
            if(!order_for(trial,targets,live))return false;
            if(neutral) {
                if(live.size()!=best.size() || !helper.op || !h) return false;
                *neutral=live_only(trial,live);
                return neutral->count(h->value)>0;
            }
            if(live.size()>=best.size()||!compile(trial,tape))return false;
            graph=live_only(trial,live);best=std::move(tape);return true;
        };
        Bits base_ok{};
        for(int w=0;w<words;w++){
            close_word(base,w,nullptr,0);
            base_ok[w]=selectable[w];for(int t:roots)base_ok[w]&=base[t][w];
        }
        for(int i=first;i<n;i++)if((base_ok[i/64]>>(i%64))&1)if(reconstruct(nullptr,i,-1))return true;
        std::shuffle(lib.begin(),lib.end(),rng);
        std::map<int,std::vector<int>> exchange_for;
        for(auto& h:lib) {
            if(h.use.empty())continue;
            Bits viable=base_ok;
            std::vector<Bits> rows;
            for(int w=0;w<words;w++){
                uint64_t m=0;
                for(auto e:h.make)m|=base[e.a][w]&base[e.b][w];
                if(!m)continue;
                bool useful=false;
                for(auto &u:h.use){
                    uint64_t delta=m&allow[u.output][w]&~base[u.output][w];
                    if(u.other>=0)delta&=base[u.other][w];
                    if(delta){useful=true;break;}
                }
                if(!useful)continue;
                if(rows.empty())rows=base;
                close_word(rows,w,&h,m);
                uint64_t ok=selectable[w];
                for(int v:roots)ok&=rows[v][w];
                viable[w]=ok;
            }
            if(rows.empty())continue;
            std::vector<int> drops;
            for(int w=0;w<words;w++){
                uint64_t bits=viable[w];
                while(bits){drops.push_back(64*w+__builtin_ctzll(bits));bits&=bits-1;}
            }
            if(neighbors) for(int x:drops) if(!((base_ok[x/64]>>(x%64))&1))
                exchange_for[x].push_back(int(&h-lib.data()));
            if(swaps) for(int x:drops) if(!((base_ok[x/64]>>(x%64))&1))
                swaps->push_back({val[x],h.value});
            if(drops.size()<2)continue;
            for(size_t i=0;i<drops.size();i++){
                if(reconstruct(&h,drops[i],-1))return true;
                for(size_t j=0;j<i;j++)if(reconstruct(&h,drops[i],drops[j]))return true;
            }
        }
        if(neighbors && width>0) {
            std::vector<int> pivots;
            for(auto& [pivot,choices]:exchange_for) {
                pivots.push_back(pivot);std::shuffle(choices.begin(),choices.end(),rng);
            }
            bool more=true;
            while(more && int(neighbors->size())<width){
                more=false;
                std::shuffle(pivots.begin(),pivots.end(),rng);
                for(int pivot:pivots) {
                    auto& choices=exchange_for[pivot];
                    if(choices.empty())continue;
                    more=true;
                    int h=choices.back();choices.pop_back();Graph result;
                    if(reconstruct(&lib[h],pivot,-1,&result))neighbors->push_back(std::move(result));
                    if(int(neighbors->size())==width)break;
                }
            }
        }
        return false;
    }
