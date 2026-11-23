// A shared-interface beam with an exact continuation at every state.
// Unresolved goals may keep their old definitions, but their dependencies
// are optimized together with every newly built number and charged once.
#define ELF_JOINT_INTERFACES_LIBRARY
#include "elf_joint_interfaces.cpp"

namespace relay_interface {
using interface_search::Clock;
using interface_search::Deadline;
using interface_search::State;
using interface_search::hash;
using interface_search::One;
struct Stats {int cuts=0,wins=0,estimates=0,retired=0,internal=0,packets=0;};

struct Completion {
    const Graph& g;
    const Cut& c;
    Index values;
    std::vector<int> private_roots;
    std::unordered_set<Value> outside;
    Completion(const Graph& graph,const Cut& cut):g(graph),c(cut){
        values.reserve(g.nodes.size());
        std::vector<int> users(g.nodes.size());
        for(int i=0;i<(int)g.nodes.size();++i){
            auto n=g.nodes[i];values.emplace(n.value,i);
            if(c.terminal[i])outside.insert(n.value);
            if(n.op){++users[n.a];if(n.a!=n.b)++users[n.b];}
        }
        std::vector<std::pair<int,int>> scored;
        std::vector<int> cost(g.nodes.size());
        for(int i=0;i<(int)g.nodes.size();++i)if(auto n=g.nodes[i];n.op&&!c.terminal[i]){
            cost[i]=std::min(32,1+cost[n.a]+(n.a==n.b?0:cost[n.b]));
            if(c.fixed[i]||cost[i]<2||std::binary_search(c.roots.begin(),c.roots.end(),i))continue;
            scored.emplace_back(-std::min(14,cost[i])*8-std::min(8,users[i])*7,i);
        }
        const int top=std::min((int)scored.size(),45);
        std::partial_sort(scored.begin(),scored.begin()+top,scored.end());
        for(int j=0;j<top;++j)private_roots.push_back(scored[j].second);
    }
    Script close(const Script& tape,int& retired)const{
        std::unordered_map<Value,const Step*> override;
        override.reserve(tape.size());
        for(const auto& s:tape)override.emplace(s.x,&s);
        Script proof;
        std::unordered_set<Value> made=outside,active;
        auto emit=[&](auto& self,Value v)->bool{
            if(made.count(v))return true;
            if(!active.insert(v).second)return false;
            Step s{};
            auto n=values.find(v);
            if(auto it=override.find(v);it!=override.end())s=*it->second;
            else if(n!=values.end()){
                const auto old=g.nodes[n->second];
                if(old.op)s={v,old.op,g.nodes[old.a].value,g.nodes[old.b].value};
            }
            if(!s.x||!self(self,s.a)||!self(self,s.b))return false;
            proof.push_back(s);active.erase(v);made.insert(v);return true;
        };
        for(int i:c.roots)if(!emit(emit,g.nodes[i].value))return {};
        retired=0;
        for(int i=0;i<(int)g.nodes.size();++i)if(!c.fixed[i]&&g.nodes[i].op&&!made.count(g.nodes[i].value))++retired;
        return proof;
    }
};

struct Proposal {Step first,second;int priority;};
std::vector<Proposal> extras(const Completion& c,const State& s,const One& base,std::size_t initial,
                          Value limit,bool packets,int width,Value salt,Deadline until,Stats& m){
    const auto fresh=interface_search::one_step_bank(s,limit,until,initial);
    auto fetch=[&](Value v)->Step {
        if(!v||s.held.count(v))return {};
        if(auto it=base.find(v);it!=base.end())return it->second;
        if(auto it=fresh.find(v);it!=fresh.end())return it->second;
        return {};
    };
    std::vector<Proposal> list;
    for(int i:c.private_roots){
        if(Clock::now()>=until)break;
        const auto t=c.g.nodes[i].value;
        if(s.held.count(t))continue;
        if(auto direct=fetch(t);direct.x){
            list.push_back({direct,{},12+(int)list.size()});++m.internal;continue;
        }
        if(!packets)continue;
        int kept=0;
        std::unordered_set<Value> firsts;
        auto take=[&](Value v,char op,Value a,Value b){
            if(v==t||!firsts.insert(v).second)return;
            auto first=fetch(v);if(!first.x)return;
            list.push_back({first,{t,op,a,b},40+(int)(hash(t^hash(v)^salt)%43)});
            ++kept;++m.packets;
        };
        for(std::size_t j=0;j<s.order.size()&&kept<5;++j){
            const auto b=s.order[j];
            inverse(t,b,limit,[&](Value a,char op,bool lhs){
                take(a,op,lhs?a:b,lhs?b:a);return kept>=5;
            });
        }
        if((t&1)==0)take(t/2,'+',t/2,t/2);
        const auto u=(Value)std::sqrt((long double)t);
        for(Value k:{u-1,u,u+1})if(k&&k==t/k&&t%k==0)take(k,'*',k,k);
    }
    const int cap=std::min((int)list.size(),std::max(8,width*3));
    std::partial_sort(list.begin(),list.begin()+cap,list.end(),[](const auto& a,const auto& b){
        return std::tie(a.priority,a.first.x,a.second.x)<std::tie(b.priority,b.first.x,b.second.x);
    });
    list.resize(cap);return list;
}

struct StateRank {State state;int price;Value tie;};
Script solve(const Graph& g,const Cut& cut,Value bound,int width,int mode,Value salt,Deadline until,Stats& m){
    Completion calculator(g,cut);
    std::vector<Value> exits,ready;
    for(int i:cut.roots)exits.push_back(g.nodes[i].value);
    for(int j=0;j<(int)g.nodes.size();++j)if(cut.terminal[j])ready.push_back(g.nodes[j].value);
    const int budget=cut.budget-(int)exits.size()-1;
    if(budget<0||exits.empty()||exits.size()>=64)return {};
    State initial(ready,(int)exits.size());
    interface_search::closure(initial,exits,bound,0);
    const auto start_size=initial.order.size();
    const auto bank=interface_search::one_step_bank(initial,bound,until);
    const int max_helpers=std::min(10,budget);
    std::vector<std::vector<StateRank>> layer(max_helpers+1);
    std::unordered_set<Value> visited;
    Script best;
    int best_margin=0;
    auto evaluate=[&](State s){
        if(!visited.insert(s.fingerprint).second)return;
        int retired=0;
        auto p=calculator.close(s.script,retired);
        if(p.empty())return;
        ++m.estimates;
        const int price=(int)p.size();
        if(price<cut.budget&&(best.empty()||price<(int)best.size())){
            best=std::move(p);best_margin=retired;
        }
        if(!s.remaining||s.helpers>=max_helpers||price>cut.budget+2)return;
        const auto tie=hash(s.fingerprint^salt);
        layer[s.helpers].push_back({std::move(s),price,tie});
    };
    evaluate(initial);
    for(int depth=0;depth<max_helpers&&Clock::now()<until;++depth){
        auto& rows=layer[depth];
        auto score=[](const StateRank& a){return std::tuple{a.price,rebuild16::pop(a.state.remaining),a.tie};};
        const int k=std::min(width,(int)rows.size());
        std::partial_sort(rows.begin(),rows.begin()+k,rows.end(),[&](const auto& a,const auto& b){return score(a)<score(b);});
        for(int j=0;j<k&&Clock::now()<until;++j){
            const auto& current=rows[j].state;
            interface_search::Meter ignored;
            const auto immediate=interface_search::moves(current,exits,bound,std::max(10,width*2),false,
                salt^current.fingerprint,until,ignored,&bank,start_size,true);
            for(auto step:immediate){
                auto next=current;
                next.put(step,true);
                interface_search::closure(next,exits,bound,current.order.size());
                evaluate(std::move(next));
            }
            if(mode&&Clock::now()<until)for(auto job:extras(calculator,current,bank,start_size,bound,
                        mode==2&&depth+1<max_helpers,width,salt^depth,until,m)){
                auto next=current;
                next.put(job.first,true);
                if(job.second.x)next.put(job.second,true);
                if(next.helpers>max_helpers)continue;
                interface_search::closure(next,exits,bound,current.order.size());
                evaluate(std::move(next));
            }
        }
    }
    m.retired+=best_margin;
    return best;
}

Program run(const Registers& starts,const std::vector<Value>& goals,Value bound,Program best,
            int cuts,int width,int mode,Value salt,int ms,Stats& meter){
    const auto begin=Clock::now(),stop=begin+std::chrono::milliseconds(ms);
    auto graph=topological_graph(recover_graph(starts,bound,goals,best));
    const auto joint=combine_interfaces::jobs(combine_interfaces::Context(graph),bound,cuts,salt,begin+(stop-begin)/6);
    const auto single=interface_search::scopes(graph,bound,cuts,salt^12271,true);
    for(int i=0;i<cuts&&Clock::now()<stop;++i){
        const int j=i/2;const bool pair=j<(int)joint.size()&&(i%2==0||j>=(int)single.size());
        if(!pair&&j>=(int)single.size())continue;
        const auto& c=pair?joint[j].joined:single[j].cut;
        ++meter.cuts;
        const auto now=Clock::now();
        auto draft=solve(graph,c,bound,width,mode,salt+i*673,now+(stop-now)/std::min(6,cuts-i),meter);
        if(draft.empty()||draft.size()>=(size_t)c.budget)continue;
        auto tape=numeric_splice::apply(graph,draft,bound,(int)best.size(),20);
        if(tape.empty())tape=replace_block(graph,c,draft,bound,(int)best.size(),16);
        if(!tape.empty()){++meter.wins;best=std::move(tape);}
    }
    return best;
}
}

#ifndef ELF_INTERFACE_RELAY_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=6)throw std::invalid_argument("expected cuts width mode salt millis");
        int cuts=std::stoi(argv[1]),width=std::stoi(argv[2]),mode=std::stoi(argv[3]),ms=std::stoi(argv[5]);
        auto salt=std::stoull(argv[4]);
        if(cuts<1||width<1||width>40||mode<0||mode>2||ms<50)throw std::invalid_argument("bad relay profile");
        std::string id;Value cap;int ns,nt,n;
        while(std::cin>>id>>cap>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||!cap||cap>=(Value(1)<<62)||n<0||n>4096)throw std::invalid_argument("bad case");
            Registers r{};std::vector<Value> goals(nt);Program p(n);
            for(int i=0;i<ns;++i)std::cin>>r[i];for(auto& t:goals)std::cin>>t;
            for(auto& z:p)std::cin>>z.kind>>z.a>>z.b>>z.destination;
            relay_interface::Stats m;
            auto tape=relay_interface::run(r,goals,cap,std::move(p),cuts,width,mode,salt,ms,m);
            std::unordered_set<Value> seen(r.begin(),r.end());for(auto z:tape)seen.insert(execute(r,z,cap));
            for(Value t:goals)if(!seen.count(t))throw std::logic_error("missing scored output");
            std::cout<<id<<' '<<tape.size()<<' '<<m.cuts<<' '<<m.wins<<' '<<m.estimates<<' '
                     <<m.retired<<' '<<m.internal<<' '<<m.packets<<'\n';
            for(const auto& z:tape)std::cout<<z.kind<<' '<<z.a<<' '<<z.b<<' '<<z.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
