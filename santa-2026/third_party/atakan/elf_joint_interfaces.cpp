// Co-design two overlapping arithmetic interfaces, demoting any old port
// without a real consumer outside the combined region.
#define ELF_INTERFACE_CLOSURE_LIBRARY
#include "elf_interface_closure.cpp"

namespace combine_interfaces {
using interface_search::Clock;
using interface_search::Deadline;
using interface_search::hash;
struct Meter {int jobs=0,wins=0,demoted=0,joint_only=0,closures=0,expanded=0;};
struct Context {
    const Graph& g;
    std::vector<std::vector<int>> users;
    std::vector<char> output;
    explicit Context(const Graph& graph):g(graph),users(g.nodes.size()),output(g.nodes.size()){
        for(int v:g.targets)output[v]=1;
        for(int i=0;i<(int)g.nodes.size();++i)if(auto z=g.nodes[i];z.op){
            users[z.a].push_back(i);if(z.b!=z.a)users[z.b].push_back(i);
        }
    }
    std::vector<int> real_ports(const Cut& a,const Cut& b)const{
        std::vector<int> exposed;
        for(int i=0;i<(int)g.nodes.size();++i)if(g.nodes[i].op&&(!a.fixed[i]||!b.fixed[i])) {
            if(output[i]||std::any_of(users[i].begin(),users[i].end(),[&](int v){return a.fixed[v]&&b.fixed[v];}))
                exposed.push_back(i);
        }
        return exposed;
    }
    std::vector<int> literal_ports(const Cut& a,const Cut& b)const{
        std::vector<int> result;
        std::set_union(a.roots.begin(),a.roots.end(),b.roots.begin(),b.roots.end(),std::back_inserter(result));
        return result;
    }
};

struct Job {Cut literal,joined;int released,demoted,rank,left,right;};

std::vector<Job> jobs(const Context& ctx,Value limit,int count,Value salt,Deadline stop){
    const auto& g=ctx.g;
    auto basic=interface_search::scopes(g,limit,std::max(18,std::min(95,count*2)),salt,true);
    std::vector<Job> ideas;
    std::set<std::pair<std::vector<int>,std::vector<int>>> included;
    for(int i=0;i<(int)basic.size()&&Clock::now()<stop;++i){
        const auto& a=basic[i].cut;
        for(int j=i+1;j<(int)basic.size();++j){
            if((j&15)==0&&Clock::now()>=stop)break;
            const auto& b=basic[j].cut;
            int overlap=0,unique_a=0,unique_b=0;
            for(int k=0;k<(int)g.nodes.size();++k)if(g.nodes[k].op){
                overlap+=!a.fixed[k]&&!b.fixed[k];
                unique_a+=!a.fixed[k]&&b.fixed[k];
                unique_b+=a.fixed[k]&&!b.fixed[k];
            }
            if(unique_a<2||unique_b<2)continue;
            auto plain=ctx.literal_ports(a,b);
            if(plain.size()<3||plain.size()>61)continue;
            auto adjusted=ctx.real_ports(a,b);
            if(adjusted.size()<2||adjusted.size()>52||adjusted.size()>plain.size())continue;
            auto old=make_cut(g,plain), joint=make_cut(g,adjusted);
            const int extra=joint.budget-(int)adjusted.size();
            if(extra<1||extra>42||joint.budget<8||joint.budget>85)continue;
            int released=0;
            for(int k=0;k<(int)g.nodes.size();++k)if(g.nodes[k].op)
                released+=!joint.fixed[k]&&a.fixed[k]&&b.fixed[k];
            const int demoted=(int)plain.size()-(int)adjusted.size();
            if(released==0&&demoted==0)continue;
            if(!included.emplace(plain,adjusted).second)continue;
            int rank=28*std::min(10,released)+17*std::min(8,demoted)+9*std::min(13,extra) +
                4*std::min(10,std::min(unique_a,unique_b)) + 2*std::min(12,overlap) -
                3*std::max((int)adjusted.size()-17,0)-5*std::max(extra-16,0)+
                int(hash(salt^g.nodes[adjusted.front()].value^hash(g.nodes[adjusted.back()].value))%23);
            ideas.push_back({std::move(old),std::move(joint),released,demoted,rank,i,j});
        }
    }
    std::sort(ideas.begin(),ideas.end(),[](const auto& a,const auto& b){
        return std::tie(a.rank,a.released,a.demoted)>std::tie(b.rank,b.released,b.demoted);
    });
    std::vector<int> uses(basic.size());
    std::vector<Job> out;
    while(!ideas.empty()&&(int)out.size()<count){
        // Avoid spending the entire search on one almost identical set of
        // ports. Check a small queue of ranked candidates at every pick.
        const auto grade=[&](const auto& a){
            int common=0;
            for(const auto& v:out){
                int n=0;
                for(int x:a.joined.roots)n+=std::binary_search(v.joined.roots.begin(),v.joined.roots.end(),x);
                common=std::max(common,100*n/(int)(a.joined.roots.size()+v.joined.roots.size()-n));
            }
            return a.rank-24*(uses[a.left]+uses[a.right])-common;
        };
        auto at=std::max_element(ideas.begin(),ideas.begin()+std::min(75,(int)ideas.size()),
                                 [&](const auto& a,const auto& b){return grade(a)<grade(b);});
        ++uses[at->left];++uses[at->right];out.push_back(std::move(*at));ideas.erase(at);
    }
    return out;
}

void try_cut(const Graph& g,const Cut& c,Value limit,int width,int mode,Value salt,Deadline until,
             Meter& met,Program& best){
    std::vector<Value> ready,goals;
    const int available=c.budget-(int)c.roots.size()-1;
    if(available<0)return;
    for(int k=0;k<(int)g.nodes.size();++k)if(c.terminal[k])ready.push_back(g.nodes[k].value);
    for(int k:c.roots)goals.push_back(g.nodes[k].value);
    interface_search::Meter inner;
    interface_search::Unions alternatives(g,c);
    auto proof=interface_search::synth(ready,goals,limit,std::min(available,22),width,mode==2?7:5,salt,until,
                                       inner,mode==2?&alternatives:nullptr);
    auto accept=[&](const Cut& cut,const Script& code){
        if(code.empty()||code.size()>=(size_t)cut.budget)return;
        ++met.closures;
        auto tape=replace_block(g,cut,code,limit,(int)best.size(),16);
        if(!tape.empty()){best=std::move(tape);++met.wins;}
    };
    accept(c,proof);
    for(const auto& option:alternatives.plans)accept(option.cut,option.recipe);
    met.expanded+=inner.expanded;
}

Program run(const Registers& start,const std::vector<Value>& goals,Value limit,Program best,
            int count,int width,int mode,Value salt,int ms,Meter& stats){
    const auto start_time=Clock::now(),stop=start_time+std::chrono::milliseconds(ms);
    auto graph=topological_graph(recover_graph(start,limit,goals,best));
    Context ctx(graph);
    auto group=jobs(ctx,limit,count,salt,start_time+(stop-start_time)/5);
    for(int i=0;i<(int)group.size()&&Clock::now()<stop;++i){
        const auto& x=group[i];
        const auto now=Clock::now(),end=now+(stop-now)/std::min(6,(int)group.size()-i);
        ++stats.jobs;stats.demoted+=x.demoted;stats.joint_only+=x.released;
        try_cut(graph,mode?x.joined:x.literal,limit,width,mode,salt+i*871,end,stats,best);
    }
    return best;
}
}

#ifndef ELF_JOINT_INTERFACES_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=6)throw std::invalid_argument("expected cuts width mode salt millis");
        int cuts=std::stoi(argv[1]),width=std::stoi(argv[2]),mode=std::stoi(argv[3]),ms=std::stoi(argv[5]);
        auto salt=std::stoull(argv[4]);
        if(cuts<1||width<1||width>40||mode<0||mode>2||ms<50)throw std::invalid_argument("bad joint profile");
        std::string id;Value cap;int ns,nt,n;
        while(std::cin>>id>>cap>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||!cap||cap>=(Value(1)<<62)||n<0||n>4096)throw std::invalid_argument("bad case");
            Registers r{};std::vector<Value> goals(nt);Program p(n);
            for(int i=0;i<ns;++i)std::cin>>r[i];for(auto& t:goals)std::cin>>t;
            for(auto& z:p)std::cin>>z.kind>>z.a>>z.b>>z.destination;
            combine_interfaces::Meter m;
            auto tape=combine_interfaces::run(r,goals,cap,std::move(p),cuts,width,mode,salt,ms,m);
            std::unordered_set<Value> seen(r.begin(),r.end());for(auto z:tape)seen.insert(execute(r,z,cap));
            for(Value t:goals)if(!seen.count(t))throw std::logic_error("missing scored output");
            std::cout<<id<<' '<<tape.size()<<' '<<m.jobs<<' '<<m.wins<<' '<<m.demoted<<' '
                     <<m.joint_only<<' '<<m.closures<<' '<<m.expanded<<'\n';
            for(const auto& z:tape)std::cout<<z.kind<<' '<<z.a<<' '<<z.b<<' '<<z.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
