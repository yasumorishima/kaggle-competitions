// Rebuild a blocked outside descendant first. This can shorten the joint
// inside/outside graph even when it increases the isolated interface cost.
#define ELF_INTERFACE_CLOSURE_LIBRARY
#include "elf_interface_closure.cpp"

namespace outside_alias {
using interface_search::Clock;
using interface_search::Deadline;
using interface_search::State;
using interface_search::One;
using interface_search::hash;
struct Stats {int cuts=0,wins=0,aliases=0,joined=0,checked=0,compound=0;};
struct Offer {Value v;Script steps;int rank;};

std::vector<Offer> candidates(const Graph& g,const Cut& cut,const State& before,const One& one,
                             const std::vector<Value>& exits,Value bound,int mode,Value salt,Deadline until){
    std::vector<int> users(g.nodes.size());
    for(const auto& n:g.nodes)if(n.op){++users[n.a];if(n.a!=n.b)++users[n.b];}
    std::vector<Offer> list;
    for(int i=0;i<(int)g.nodes.size()&&Clock::now()<until;++i){
        if(!g.nodes[i].op||!cut.fixed[i]||cut.terminal[i])continue;
        const Value v=g.nodes[i].value;
        auto propose=[&](Script steps){
            int support=0;
            for(Value t:exits)if(interface_search::from(t,v,before.held,bound).x)++support;
            const int rank=36*support+13*std::min(users[i],5)-25*(int)steps.size()+int(hash(v^salt)%19);
            list.push_back({v,std::move(steps),rank});
        };
        if(auto it=one.find(v);it!=one.end()){propose({it->second});continue;}
        if(mode<2)continue;
        int todo=0;
        for(Value x:before.order){
            if(x==v||todo>=3)continue;
            inverse(v,x,bound,[&](Value other,char op,bool left){
                const auto q=one.find(other);
                if(q==one.end()||q->second.x==v)return false;
                propose({q->second,{v,op,left?other:x,left?x:other}});++todo;return true;
            });
        }
        if(todo<3&&v%2==0)if(auto p=one.find(v/2);p!=one.end())propose({p->second,{v,'+',v/2,v/2}});
    }
    std::sort(list.begin(),list.end(),[](const auto& a,const auto& b){return std::tie(a.rank,a.v)>std::tie(b.rank,b.v);});
    // Distinct ready values rather than many equivalent encodings.
    std::vector<Offer> keep;
    std::unordered_set<Value> used;
    for(auto& q:list)if(used.insert(q.v).second){
        keep.push_back(std::move(q));if(keep.size()>=18)break;
    }
    return keep;
}

void apply(const Graph& g,const Cut& cut,Value bound,int width,int mode,Value salt,
           Deadline stop,Program& best,Stats& met){
    std::vector<Value> ready,exits;
    for(int i=0;i<(int)g.nodes.size();++i)if(cut.terminal[i])ready.push_back(g.nodes[i].value);
    for(int i:cut.roots)exits.push_back(g.nodes[i].value);
    State clean(ready,(int)exits.size());
    const auto one=interface_search::one_step_bank(clean,bound,stop);
    const auto options=candidates(g,cut,clean,one,exits,bound,mode,salt,stop);
    met.aliases+=(int)options.size();
    auto accept=[&](const Script& patch){
        ++met.checked;
        auto pass=numeric_splice::apply(g,patch,bound,(int)best.size(),20);
        if(pass.empty())return;
        best=std::move(pass);++met.wins;
    };
    auto build=[&](const Offer& a,const Offer* b,int i,Deadline until){
        State state=clean;
        auto attach=[&](const Offer& z){
            for(auto s:z.steps){
                if(state.held.count(s.x))continue;
                if(!state.held.count(s.a)||!state.held.count(s.b)||calculate(s.op,s.a,s.b,bound)!=s.x)return false;
                state.put(s,true);
            }
            return true;
        };
        if(!attach(a)||(b&&!attach(*b)))return;
        accept(state.script);
        if(!mode||Clock::now()>=until)return;
        int absent=0;
        for(Value t:exits)absent+=!state.held.count(t);
        // Reusing a blocked outer value may cancel an outside write.
        // Allow that credit only at whole-program verification.
        const int delay=mode==2?2:1;
        const int budget=std::min(12,cut.budget+delay-(int)state.script.size()-absent);
        if(budget<0||!absent)return;
        interface_search::Meter ignored;
        auto result=interface_search::synth(state.order,exits,bound,budget,width,5,salt^Value(i*3023),until,ignored);
        if(result.empty())return;
        ++met.joined;
        state.script.insert(state.script.end(),result.begin(),result.end());
        accept(state.script);
    };
    for(int i=0;i<(int)options.size()&&Clock::now()<stop;++i){
        const auto now=Clock::now();
        build(options[i],nullptr,i,now+(stop-now)/std::min(8,(int)options.size()-i));
        met.compound+=options[i].steps.size()>1;
    }
    if(mode==2)for(int i=0;i<std::min((int)options.size(),5)&&Clock::now()<stop;++i){
        for(int j=i+1;j<std::min((int)options.size(),6)&&Clock::now()<stop;++j){
            const auto now=Clock::now();
            build(options[i],&options[j],i*9+j,now+(stop-now)/6);
        }
    }
}

Program run(const Registers& r,const std::vector<Value>& targets,Value bound,Program incumbent,
            int cuts,int width,int mode,Value salt,int millis,Stats& m){
    const auto begin=Clock::now(),end=begin+std::chrono::milliseconds(millis);
    auto g=topological_graph(recover_graph(r,bound,targets,incumbent));
    auto scopes=interface_search::scopes(g,bound,std::max(20,cuts*2),salt,true);
    int count=0;
    for(int i=0;i<(int)scopes.size()&&count<cuts&&Clock::now()<end;++i){
        const auto& q=scopes[i].cut;
        bool absent=false;
        for(int v=0;v<(int)g.nodes.size();++v)absent|=q.fixed[v]&&!q.terminal[v]&&g.nodes[v].op;
        if(!absent)continue;
        ++count;++m.cuts;
        const auto now=Clock::now(),until=now+(end-now)/std::min(7,cuts-count+1);
        apply(g,q,bound,width,mode,salt+i*1999,until,incumbent,m);
    }
    return incumbent;
}
}

#ifndef ELF_OUTSIDE_ALIAS_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=6)throw std::invalid_argument("expected cuts width mode salt millis");
        int cuts=std::stoi(argv[1]),width=std::stoi(argv[2]),mode=std::stoi(argv[3]),ms=std::stoi(argv[5]);
        auto salt=std::stoull(argv[4]);
        if(cuts<1||width<1||width>35||mode<0||mode>2||ms<50)throw std::invalid_argument("bad alias profile");
        std::string id;Value cap;int ns,nt,n;
        while(std::cin>>id>>cap>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||!cap||cap>=(Value(1)<<62)||n<0||n>4096)throw std::invalid_argument("bad case");
            Registers r{};std::vector<Value> goals(nt);Program p(n);
            for(int i=0;i<ns;++i)std::cin>>r[i];for(auto& x:goals)std::cin>>x;
            for(auto& z:p)std::cin>>z.kind>>z.a>>z.b>>z.destination;
            outside_alias::Stats m;
            auto best=outside_alias::run(r,goals,cap,std::move(p),cuts,width,mode,salt,ms,m);
            std::unordered_set<Value> seen(r.begin(),r.end());for(auto z:best)seen.insert(execute(r,z,cap));
            for(Value t:goals)if(!seen.count(t))throw std::logic_error("missing target");
            std::cout<<id<<' '<<best.size()<<' '<<m.cuts<<' '<<m.wins<<' '<<m.aliases<<' '
                     <<m.joined<<' '<<m.checked<<' '<<m.compound<<'\n';
            for(const auto& z:best)std::cout<<z.kind<<' '<<z.a<<' '<<z.b<<' '<<z.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
