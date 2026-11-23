// Forward speculative work is charged against the reduction in actual
// witnessed route lengths for *all* still-open targets. Inspired by H_cub,
// but every multiply/divide is charged and every candidate is register replayed.
#define ELF_RESIDUAL_INCLUDE
#include "elf_block_rebuild.cpp"

namespace cumulative {
using Clock=std::chrono::steady_clock;
using Deadline=Clock::time_point;
using Signature=std::uint64_t;
struct Stats {int cuts=0,wins=0,states=0,joint=0,logical=0,rounds=0;};
struct Plan {Script steps;int size=0;};

Signature mix(Signature x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

// When a candidate intermediate is already obtained through another route,
// its original dependency chain becomes unnecessary, not merely one write.
Script needed(const Script& recipe,Value root,const std::unordered_set<Value>& free) {
    if(free.count(root))return {};
    std::unordered_set<Value> pending{root};
    Script proof;
    for(auto it=recipe.rbegin();it!=recipe.rend();++it){
        auto s=*it;
        if(!pending.erase(s.x))continue;
        proof.push_back(s);
        if(!free.count(s.a))pending.insert(s.a);
        if(!free.count(s.b))pending.insert(s.b);
    }
    if(!pending.empty())throw std::logic_error("Ungrounded candidate in cumulative planner");
    std::reverse(proof.begin(),proof.end());
    return proof;
}

Script below(const Construction& c,Value root,const std::unordered_set<Value>& free) {
    std::unordered_set<Value> marked;
    Script code;
    auto visit=[&](auto& self,Value x)->void {
        if(free.count(x)||!marked.insert(x).second)return;
        const auto n=c.g.nodes.at(c.index.at(x));
        if(!n.op)throw std::logic_error("Free seed missing");
        Value a=c.g.nodes[n.a].value,b=c.g.nodes[n.b].value;
        self(self,a);self(self,b);code.push_back({x,n.op,a,b});
    };
    visit(visit,root);
    return code;
}

std::vector<SInt> leaves(const Construction& c,const std::vector<Value>& goals,int budget,Value salt) {
    std::vector<SInt> out;
    std::vector<std::pair<double,Value>> ranked;
    std::unordered_set<Value> used;
    for(Value s:c.starts)if(used.insert(s).second)out.push_back((SInt)s);
    for(auto n:c.g.nodes)if(n.value<64&&(n.value<=5||n.value==8||n.value==16||n.value==32))
        if(used.insert(n.value).second)out.push_back((SInt)n.value);
    for(const auto& n:c.g.nodes)if(!used.count(n.value)) {
        const Value a=n.value;
        double score=200;
        for(Value t:goals) {
            if(t==a)continue;
            Value d=t>a?t-a:a-t;
            double room=std::log2(double(d));
            if(t>a&&a>2) {
                Value r=t%a;
                room=std::min(room,1+.63*std::log2(double(t/a))+
                    .81*std::log2(double(1+std::min(r,a-r))));
            }
            score=std::min(room,score);
        }
        score+=(a<64?-5.:0.)+.05*(mix(a^salt)%131);
        ranked.emplace_back(score,a);
    }
    std::sort(ranked.begin(),ranked.end());
    for(auto [r,a]:ranked) {
        if((int)out.size()>=budget)break;
        out.push_back((SInt)a);
    }
    return out;
}

std::vector<Script> bank_routes(const Construction& c,local_beam::Dictionary& d,Value target,
                               int options,Deadline end,Value salt) {
    struct Hint {double price;Value a,b,rest;char op,sign;};
    std::vector<Hint> raw;
    auto cheap=[&](Value v){
        if(c.index.count(v))return 0;
        int i=d.index.get((SInt)v);
        return i<0?100:d.e[i].cost;
    };
    auto offer=[&](Value a,Value b,char op,Value r=0,char sign='+'){
        if(a==target||b==target||r==target||!a||!b)return;
        if((op=='+'||op=='*')&&b<a)std::swap(a,b);
        int price=1+cheap(a)+(a==b?0:cheap(b));
        if(r)price+=1+(r==a||r==b?0:cheap(r));
        if(price>6)return;
        if(r && calculate(op,a,b,c.limit)==r)return;
        double priority=price+.001*(mix(a^mix(b)^mix(r)^salt)%177);
        raw.push_back({priority,a,b,r,op,sign});
    };
    for(int j=0;j<d.n1;++j){
        if((j%512)==0&&Clock::now()>=end)break;
        const Value x=(Value)d.e[j].v;
        if(x==target)continue;
        inverse(target,x,c.limit,[&](Value other,char op,bool first){
            offer(first?other:x,first?x:other,op);return false;
        });
        if(x>1&&x<target){
            const Value q=target/x,r=target%x;
            if(r&&q>1) {
                if(cheap(r)<=2)offer(x,q,'*',r,'+');
                if(x-r<=c.limit-target&&cheap(x-r)<=2)offer(x,q+1,'*',x-r,'-');
            }
        }
        // Near squares only need the arithmetic witness for their root.
        // The square itself is calculated and checked with integers.
        if((j<d.n0||x<4096)&&x<target) {
            const Value y=target-x;
            const auto q=(Value)std::sqrt((long double)y);
            for(Value z:{q,q+1})if(z>=2&&z<=y/z&&z*z==y)offer(z,z,'*',x,'+');
        }
    }
    std::stable_sort(raw.begin(),raw.end(),[](const auto& a,const auto& b){return a.price<b.price;});
    std::vector<Script> code;
    std::set<std::tuple<Value,Value,Value,char,char>> distinct;
    for(auto x:raw){
        if((int)code.size()>=options)break;
        if(!distinct.emplace(x.a,x.b,x.rest,x.op,x.sign).second)continue;
        Planner p{c,d};
        if(!p.attach(x.a)||!p.attach(x.b))continue;
        Value v=calculate(x.op,x.a,x.b,c.limit);
        if(!v||v==x.a||v==x.b)continue;
        p.push(v,x.op,x.a,x.b);
        if(x.rest){
            if(!p.attach(x.rest))continue;
            p.push(target,x.sign,v,x.rest);
        }
        if(v!=target&&!x.rest)throw std::logic_error("Inverse route arithmetic error");
        if(!p.plan.empty())code.push_back(std::move(p.plan));
    }
    return code;
}

std::unordered_set<Value> snapshot(const Construction& c) {
    std::unordered_set<Value> out;
    for(const auto& n:c.g.nodes)out.insert(n.value);
    return out;
}

struct Arm {Script code;Value terminal;double score=0;int supports=0;};
struct Election {Arm move;int visited=0;};
Election elect(const Construction& c,const std::vector<Value>& remaining,
               const std::vector<std::vector<Script>>& options,bool cumulative,Value salt) {
    const auto ready=snapshot(c);
    std::unordered_map<Value,Arm> sites;
    Script cheapest;
    Value chosen=0;
    std::vector<int> before(remaining.size(),10000);
    std::unordered_map<Value,std::vector<int>> members;
    for(int t=0;t<(int)remaining.size();++t){
        for(const auto& recipe:options[t])if(!recipe.empty()) {
            const int bill=(int)recipe.size();
            before[t]=std::min(bill,before[t]);
            if(cheapest.empty()||bill<(int)cheapest.size()||
                (bill==(int)cheapest.size()&&mix(remaining[t]^salt)<mix(chosen^salt)))cheapest=recipe,chosen=remaining[t];
        }
        for(const auto& recipe:options[t])if((int)recipe.size()<=before[t]+1){
            for(auto step:recipe){
                auto [p,new_site]=sites.try_emplace(step.x,Arm{{},step.x});
                auto& reach=members[step.x];
                if(reach.empty()||reach.back()!=t)reach.push_back(t);
                auto sub=needed(recipe,step.x,ready);
                if(p->second.code.empty()||sub.size()<p->second.code.size())p->second.code=std::move(sub);
            }
        }
    }
    Arm best{std::move(cheapest),chosen};
    if(best.code.empty())return {};
    // Progress on the most affordable actual target is the baseline move.
    // A preparatory investment must buy strictly more witnessed savings
    // than its own cost. No arbitrary bit-distance bonus admits a helper.
    for(auto& [v,site]:sites) {
        if(site.code.empty())continue;
        if(!cumulative&&members[v].size()<2)continue;
        if(site.code.size()>6)continue;
        auto future=ready;
        for(auto step:site.code)future.insert(step.x);
        int total=0,maximum=0,supports=0;
        for(int t:members[v]){
            int cut=0;
            for(const auto& r:options[t])if(!r.empty())
                cut=std::max(cut,before[t]-(int)needed(r,remaining[t],future).size());
            total+=cut;maximum=std::max(maximum,cut);supports+=cut>0;
        }
        if(supports<2)continue;
        int merit=(cumulative?total:maximum)-(int)site.code.size();
        if(merit<1)continue;
        site.score=merit-.013*site.code.size()+.001*(mix(v^salt)%41);
        site.supports=supports;
        if(site.score>best.score)best=std::move(site);
    }
    return {std::move(best),(int)sites.size()};
}

Script build(Construction& c,const std::vector<Value>& goals,const std::unordered_set<Value>& stable,
             int op_cap,int bank_size,int beam,int opt_budget,bool joint,Value salt,Deadline until,Stats& stats,
             int stop_after=0) {
    Value seed=c.starts.front();
    if(!c.index.count(1))c.append({1,'/',seed,seed});
    // Do not precompute a radix palette. Even a prunable palette can make
    // the online ranking mistake speculative constants for paid inputs.
    for(int round=0;round<op_cap*2 && Clock::now()<until;++round) {
        std::vector<Value> active;
        for(Value t:goals)if(!c.index.count(t))active.push_back(t);
        if(stop_after>0&&(int)(goals.size()-active.size())>=stop_after&&!active.empty())return {};
        if(active.empty()){
            Script answer;
            auto paid=stable;
            for(Value t:goals) {
                auto work=below(c,t,paid);
                for(auto x:work)answer.push_back(x),paid.insert(x.x);
            }
            return (int)answer.size()<=op_cap?answer:Script{};
        }
        local_beam::Dictionary d((SInt)c.limit);
        d.build(leaves(c,active,bank_size,salt+round));
        std::vector<std::vector<Script>> options;
        bool ordinary=false;
        for(Value t:active){
            options.push_back(bank_routes(c,d,t,opt_budget,until,salt));
            ordinary|=!options.back().empty();
        }
        int repaired=0;
        // Sparse starts are not covered by short, exact witness enumeration.
        // For them propose real reverse-beam paths to several pivots.
        for(int i=0;i<(int)active.size()&&repaired<beam&&Clock::now()<until;++i) {
            int idx=(i+round)%(int)active.size();
            if(!options[idx].empty()|| (ordinary&&repaired>=2))continue;
            const auto code=synthesize_new(c,d,active[idx],std::min(9,beam+2),true);
            if(!code.empty())options[idx].push_back(code);
            ++repaired;
        }
        auto selected=elect(c,active,options,joint,salt+round);
        stats.states+=selected.visited;
        ++stats.rounds;
        if(selected.move.code.empty())break;
        if(selected.move.supports>=2)++stats.joint;
        for(auto x:selected.move.code)c.append(x);
        // If already achieved outputs require too many unique writes, no
        // later write can make these frozen routes cheaper. Finish early.
        auto paid=stable;
        int cost=0;
        for(Value t:goals)if(c.index.count(t)) {
            for(auto x:below(c,t,paid))if(paid.insert(x.x).second)++cost;
        }
        int outstanding=0;
        for(Value t:goals)outstanding+=!c.index.count(t);
        if(cost+outstanding>op_cap)break;
    }
    return {};
}

Program run(const Registers& regs,const std::vector<Value>& seeds,const std::vector<Value>& goals,
            Value limit,Program tape,int width,int leaves,int jobs,int options,bool joint,Value salt,
            int millis,Stats& stats) {
    auto g=topological_graph(recover_graph(regs,limit,goals,tape));
    const auto wall=Clock::now()+std::chrono::milliseconds(millis);
    auto regions=rebuild16::regions(g,jobs,salt);
    for(int i=0;i<(int)regions.size()&&Clock::now()<wall;++i){
        const auto cut=make_cut(g,regions[i].roots);
        std::vector<Value> selected;
        for(int v:cut.roots)selected.push_back(g.nodes[v].value);
        Construction c(limit,seeds,selected,false);
        for(int j=0;j<(int)g.nodes.size();++j)if(cut.terminal[j]) {
            const auto n=g.nodes[j];if(n.op)c.append({n.value,n.op,g.nodes[n.a].value,g.nodes[n.b].value});
        }
        const auto ready=snapshot(c);
        const auto start=Clock::now();
        const auto end=std::min(wall,start+(wall-start)/((int)regions.size()-i));
        ++stats.cuts;
        auto p=build(c,selected,ready,cut.budget-1,leaves,width,options,joint,salt^regions[i].mask,end,stats);
        if(p.empty())continue;
        ++stats.logical;
        auto out=replace_block(g,cut,p,limit,(int)tape.size(),12);
        if(!out.empty())tape=std::move(out),++stats.wins;
    }
    return tape;
}
}

#ifndef ELF_CUMULATIVE_INCLUDE
int main(int argc,char** argv) {
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=8)throw std::invalid_argument("reverse-pivots leaves cuts alternatives coupled salt millis");
        int width=std::stoi(argv[1]),leaves=std::stoi(argv[2]),cuts=std::stoi(argv[3]),opt=std::stoi(argv[4]);
        int joint=std::stoi(argv[5]),millis=std::stoi(argv[7]);const auto salt=std::stoull(argv[6]);
        if(width<1||width>12||leaves<12||leaves>86||cuts<1||opt<1||opt>48||joint<0||joint>1||millis<1)
            throw std::invalid_argument("Invalid cumulative profile");
        local_beam::INDEX_BITS=leaves>60?19:18;
        std::string key;Value limit;int ns,nt,n;
        while(std::cin>>key>>limit>>ns>>nt>>n) {
            if(ns<1||ns>slots||nt<1||nt>63||!limit||limit>=(Value(1)<<62)||n<0||n>4096)
                throw std::invalid_argument("Invalid puzzle");
            Registers regs{};std::vector<Value> seeds(ns),goals(nt);Program tape(n);
            for(int i=0;i<ns;++i){std::cin>>seeds[i];regs[i]=seeds[i];}
            for(Value& x:goals)std::cin>>x;
            for(auto& s:tape)std::cin>>s.kind>>s.a>>s.b>>s.destination;
            cumulative::Stats m;
            auto result=cumulative::run(regs,seeds,goals,limit,std::move(tape),width,leaves,cuts,opt,joint!=0,salt,millis,m);
            std::unordered_set<Value> seen(seeds.begin(),seeds.end());
            for(auto s:result)seen.insert(execute(regs,s,limit));
            for(Value t:goals)if(!seen.count(t))throw std::runtime_error("Lost target");
            std::cout<<key<<' '<<result.size()<<' '<<m.cuts<<' '<<m.wins<<' '<<m.states<<' '<<m.joint
                <<' '<<m.logical<<' '<<m.rounds<<'\n';
            for(auto s:result)std::cout<<s.kind<<' '<<s.a<<' '<<s.b<<' '<<s.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
