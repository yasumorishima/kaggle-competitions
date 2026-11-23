// Choose an opening by the number of writes in its complete future graph.
// Unlike grafting an opening to a fixed suffix, generate the remaining routes
// against the actual new prefix and keep the old graph only as a fallback.
#define ELF_CUMULATIVE_INCLUDE
#include "elf_cumulative_sets.cpp"

namespace rollout {
using Clock=std::chrono::steady_clock;
using Deadline=Clock::time_point;
struct Stats {int openings=0,wins=0,evaluated=0,improved_hint=0,logical=0,rounds=0;};

struct Model {
    const Graph& reference;
    std::unordered_map<Value,Step> old;
    std::unordered_map<Value,int> id;
    std::vector<int> references;
    explicit Model(const Graph& g):reference(g),references(g.nodes.size()) {
        for(int i=0;i<(int)g.nodes.size();++i){
            const auto n=g.nodes[i];
            id.emplace(n.value,i);
            if(!n.op)continue;
            ++references[n.a];if(n.b!=n.a)++references[n.b];
            old.emplace(n.value,Step{n.value,n.op,g.nodes[n.a].value,g.nodes[n.b].value});
        }
    }
    Step operation(const Construction& c,Value x) const {
        if(auto i=c.index.find(x);i!=c.index.end()){
            const auto n=c.g.nodes[i->second];
            if(!n.op)return {};
            return {x,n.op,c.g.nodes[n.a].value,c.g.nodes[n.b].value};
        }
        const auto k=old.find(x);
        if(k==old.end())throw std::logic_error("Ungrounded fallback value");
        return k->second;
    }

    // The override is an already grounded prefix relative to c. Hence an
    // override can never create a cycle through unfinished fallback values.
    Script complete(const Construction& c,const std::vector<Value>& goals,
                    const Script& override,bool relative) const {
        std::unordered_map<Value,Step> draft;
        for(auto x:override)if(!c.index.count(x.x))draft.emplace(x.x,x);
        Script out;
        std::unordered_set<Value> done;
        auto visit=[&](auto& self,Value x)->void {
            if(!done.insert(x).second)return;
            if(relative&&c.index.count(x))return;
            auto m=draft.find(x);
            Step s=m==draft.end()?operation(c,x):m->second;
            if(!s.x)return;
            self(self,s.a);self(self,s.b);out.push_back(s);
        };
        for(Value x:goals)visit(visit,x);
        return out;
    }

    std::vector<Value> preferred(const Construction& c,const std::vector<Value>& active,Value salt,int howmany) const {
        auto incoming=complete(c,active,{},true);
        std::unordered_map<Value,int> demand;
        for(auto z:incoming){++demand[z.a];if(z.a!=z.b)++demand[z.b];}
        std::vector<std::pair<double,Value>> order;
        for(auto [x,n]:demand)if(n>=2&&!c.index.count(x)&&old.count(x)) {
            double priority=3.0*std::min(n,7)+.05*bit_count(x)+.001*(cumulative::mix(x^salt)%1349);
            order.emplace_back(-priority,x);
        }
        std::sort(order.begin(),order.end());
        std::vector<Value> next;
        for(int i=0;i<std::min(howmany,(int)order.size());++i)next.push_back(order[i].second);
        return next;
    }
};

struct Move {Value goal;Script code;int direct=0,price=0,delta=0;bool new_route=false;double rank=0;};

Script follow(const Construction& c,local_beam::Dictionary& dict,const Script& prefix,Value t) {
    if(c.index.count(t)||std::any_of(prefix.begin(),prefix.end(),[&](auto x){return x.x==t;}))return {};
    Script best;
    auto proof=[&](Step terminal,Step product={}) {
        Planner p{c,dict};p.plan=prefix;
        for(auto x:prefix)p.made.insert(x.x);
        auto cheap=[&](Value x) {
            if(c.index.count(x)||p.made.count(x))return true;
            int k=dict.index.get((SInt)x);
            return k>=0&&dict.e[k].cost<=1&&p.attach(x);
        };
        if(product.x){
            if(!cheap(product.a)||!cheap(product.b))return;
            p.push(product.x,product.op,product.a,product.b);
        }
        if(!cheap(terminal.a)||!cheap(terminal.b))return;
        p.push(terminal.x,terminal.op,terminal.a,terminal.b);
        if(p.plan.size()>prefix.size()+3||(!best.empty()&&p.plan.size()>=best.size()))return;
        best=std::move(p.plan);
    };
    for(auto x:prefix) {
        if(x.x==t)continue;
        inverse(t,x.x,c.limit,[&](Value v,char op,bool first){
            if(v!=t)proof({t,op,first?v:x.x,first?x.x:v});return false;
        });
        if(x.x<2||x.x>=t)continue;
        Value q=t/x.x,res=t%x.x;
        if(res&&q>=2) {
            proof({t,'+',q*x.x,res},{q*x.x,'*',q,x.x});
            if(q+1<=c.limit/x.x)proof({t,'-',(q+1)*x.x,x.x-res},{(q+1)*x.x,'*',q+1,x.x});
        }
    }
    return best;
}

std::vector<Value> nominations(const Construction& c,const Model& old,const std::vector<Value>& active,
                               int breadth,Value salt) {
    std::vector<std::pair<double,Value>> ready;
    for(Value t:active) {
        auto s=old.complete(c,{t},{},true);
        const int reach=cumulative::mix(t^salt)%33;
        double rank=s.size() + .011*reach;
        // Mix inexpensive progress and branch endings that are expensive
        // enough for a different prefix to replace substantial work.
        if(salt%3==1)rank=-rank;
        if(salt%3==2)rank=.012*bit_count(t)+(double)reach/5;
        ready.emplace_back(rank,t);
    }
    std::sort(ready.begin(),ready.end());
    std::vector<Value> out;
    for(int i=0;i<std::min(breadth,(int)ready.size());++i)out.push_back(ready[i].second);
    return out;
}

std::vector<Move> expand(const Construction& c,const Model& old,const std::vector<Value>& all,
                        const std::vector<Value>& active,int leaves,int beam,int width,bool future,
                        bool allow_ghost,Value salt,Deadline until,Stats& stats) {
    auto watch=nominations(c,old,active,width,salt);
    local_beam::Dictionary d((SInt)c.limit);
    d.build(cumulative::leaves(c,watch,leaves,salt));
    const int baseline=(int)old.complete(c,all,{},false).size();
    std::vector<Move> moves;
    std::unordered_set<std::uint64_t> seen;
    auto prop=[&](Value t,Script proof,int cost,bool fresh) {
        if(proof.empty())return;
        std::uint64_t sig=0;
        for(auto z:proof)sig^=cumulative::mix(z.x^cumulative::mix(z.a)^cumulative::mix(z.b+973));
        if(!seen.insert(sig).second)return;
        int ball=(int)old.complete(c,all,proof,false).size();
        ++stats.evaluated;
        double noise=.0001*(cumulative::mix(t^salt^sig)%821);
        double rank=future ? 7.*ball+.25*proof.size()+.12*(proof.size()>=cost)+noise
                           : 7.*proof.size()+noise;
        moves.push_back({t,std::move(proof),cost,ball,ball-baseline,fresh,rank});
    };
    for(Value t:watch) {
        if(Clock::now()>=until)break;
        auto pass=old.complete(c,{t},{},true);
        const int bill=(int)pass.size();
        prop(t,std::move(pass),bill,false);
        auto early=cumulative::bank_routes(c,d,t,9,until,salt);
        for(auto& s:early)if((int)s.size()<=bill+3)prop(t,std::move(s),bill,true);
        if(Clock::now()>=until)break;
        auto wide=synthesize_new(c,d,t,beam,true);
        if((int)wide.size()<=bill+3)prop(t,std::move(wide),bill,true);
    }
    if(allow_ghost&&Clock::now()<until) {
        auto soft=old.preferred(c,active,salt,5);
        Construction ghost=c;
        auto ready=cumulative::snapshot(c);
        Script staged;
        for(Value x:soft) {
            auto old_recipe=old.complete(ghost,{x},{},true);
            if(old_recipe.size()>7)continue;
            for(auto z:old_recipe)ghost.append(z),staged.push_back(z);
        }
        if(!staged.empty()) {
            local_beam::Dictionary new_dict((SInt)c.limit);
            new_dict.build(cumulative::leaves(ghost,watch,leaves,salt+317));
            for(Value t:watch) {
                if(Clock::now()>=until)break;
                auto s=ghost.index.count(t)?Script{}:synthesize_new(ghost,new_dict,t,beam,true);
                auto proof=staged;proof.insert(proof.end(),s.begin(),s.end());
                auto next=cumulative::needed(proof,t,ready);
                const int old_len=(int)old.complete(c,{t},{},true).size();
                if((int)next.size()<=old_len+3)prop(t,std::move(next),old_len,true);
            }
        }
    }
    std::sort(moves.begin(),moves.end(),[](const Move& a,const Move& b){return a.rank<b.rank;});
#ifdef TAIL_JOIN_ENABLED
    if(future){
        std::vector<Move> branch;
        int tried=0;
        for(const auto& move:moves) {
            if(!move.new_route)continue;
            if(tried++>=6||Clock::now()>=until)break;
            if(move.price>baseline+2)continue;
            for(Value t:active) {
                if(Clock::now()>=until)break;
                if(t==move.goal)continue;
                auto paired=follow(c,d,move.code,t);
                if(paired.empty())continue;
                int bill=(int)old.complete(c,all,paired,false).size();
                ++stats.evaluated;
                if(bill>move.price)continue;
                const double rating=7.*bill+.25*paired.size()-.55;
                branch.push_back({t,std::move(paired),move.direct,bill,bill-baseline,true,rating});
            }
        }
        for(auto& a:branch)moves.push_back(std::move(a));
        std::stable_sort(moves.begin(),moves.end(),[](const Move& a,const Move& b){return a.rank<b.rank;});
    }
#endif
    return moves;
}

void install(Construction& c,const Script& steps){for(auto x:steps)c.append(x);}

struct Outcome {Program best;int raw=100000;};

Outcome run(const Registers& regs,const std::vector<Value>& starts,const std::vector<Value>& goals,Value limit,
        Program baseline,int width,int leaves,int profiles,int pivots,bool future,Value salt,int millis,Stats& stats) {
    auto g=topological_graph(recover_graph(regs,limit,goals,baseline));
    Model old(g);
    Outcome result{baseline};
    auto finish=Clock::now()+std::chrono::milliseconds(millis);
    for(int test=0;test<profiles&&Clock::now()<finish;++test) {
        auto from=Clock::now();auto until=from+(finish-from)/(profiles-test);
        Construction c(limit,starts,goals,false);
        auto seed_set=cumulative::snapshot(c);
        if(test%3!=0){
            const auto open_stop=from+(until-from)/4;
            cumulative::Stats head;
            cumulative::build(c,goals,seed_set,4096,leaves,std::min(6,pivots),10,test%3==1,
                              salt+131*test,open_stop,head,test%3==1?5:3);
        }
        ++stats.openings;
        auto price=[&](){return old.complete(c,goals,{},false);};
        Script best=price();
        const Value one_source=starts.front();
        if(!c.index.count(1))c.append({1,'/',one_source,one_source});
        for(int turn=0;turn<(int)goals.size()&&Clock::now()<until;++turn) {
            std::vector<Value> active;
            for(Value t:goals)if(!c.index.count(t))active.push_back(t);
            if(active.empty())break;
            auto now=Clock::now();
            auto quantum=now+(until-now)/std::min(4,(int)active.size());
            auto options=expand(c,old,goals,active,leaves,width,pivots,future,turn%3!=2,
                                salt+test*197+turn*131,quantum,stats);
            if(options.empty())break;
            auto at=options.begin();
            // A mild change in the exploration arm allows a price-neutral
            // preparation to be followed by an actual improvement.
            if(future&&test%3==2&&options.size()>1&&options[1].price<=options[0].price+1&&turn%5==3)++at;
            install(c,at->code);
            ++stats.rounds;
            stats.improved_hint+=at->delta<0;
            auto after=price();
            if(after.size()<best.size())best=std::move(after);
        }
        result.raw=std::min(result.raw,(int)best.size());
        if(best.size()>=result.best.size())continue;
        ++stats.logical;
        Construction finished(limit,starts,goals,false);
        install(finished,best);
        for(Value t:goals)finished.g.targets.push_back(finished.index.at(t));
        auto registers=emit_graph(topological_graph(finished.g),limit,(int)result.best.size(),16);
        if(!registers.empty())result.best=std::move(registers),++stats.wins;
    }
    return result;
}
}

#ifndef ELF_TAIL_ROLLOUT_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=8)throw std::invalid_argument("beam leaves openings pivots future salt ms");
        const int width=std::stoi(argv[1]),leaves=std::stoi(argv[2]),profiles=std::stoi(argv[3]);
        const int pivots=std::stoi(argv[4]),future=std::stoi(argv[5]),ms=std::stoi(argv[7]);
        const auto salt=std::stoull(argv[6]);
        if(width<1||leaves<12||leaves>86||profiles<1||pivots<1||future<0||future>1||ms<10)
            throw std::invalid_argument("Invalid rollout profile");
        local_beam::INDEX_BITS=leaves>60?19:18;
        Value limit;int ns,nt,n;std::string key;
        while(std::cin>>key>>limit>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||n<0||n>4096||!limit||limit>=(Value(1)<<62))
                throw std::invalid_argument("Invalid puzzle");
            Registers regs{};std::vector<Value> starts(ns),goals(nt);Program old(n);
            for(int i=0;i<ns;++i){std::cin>>starts[i];regs[i]=starts[i];}
            for(auto& x:goals)std::cin>>x;
            for(auto& s:old)std::cin>>s.kind>>s.a>>s.b>>s.destination;
            rollout::Stats s;
            auto result=rollout::run(regs,starts,goals,limit,std::move(old),width,leaves,profiles,pivots,future!=0,salt,ms,s);
            std::unordered_set<Value> check(starts.begin(),starts.end());
            for(auto x:result.best)check.insert(execute(regs,x,limit));
            for(Value v:goals)if(!check.count(v))throw std::logic_error("Target not in compiled tape");
            std::cout<<key<<' '<<result.best.size()<<' '<<s.openings<<' '<<s.wins<<' '<<s.evaluated
                <<' '<<s.improved_hint<<' '<<s.logical<<' '<<s.rounds<<'\n';
            for(auto x:result.best)std::cout<<x.kind<<' '<<x.a<<' '<<x.b<<' '<<x.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
