// Re-root a bridge at a numerically cheaper descendant. The old bridge
// remains available to every other consumer, but the new graph first
// constructs the descendant and recovers the bridge by legal inversion.
#define ELF_PRODUCT_DISTANCE_INCLUDE
#include "elf_product_distance.cpp"

namespace reverse_bridges {
using Clock=std::chrono::steady_clock;
struct Job {std::vector<int> path;std::vector<Step> inverse;Cut cut;int price;};
struct Stats {int opens=0,wins=0,candidates=0,reroots=0,logical=0,arith=0;};

std::vector<Job> propose(const Graph& g,Value limit,Value salt,int jobs) {
    const int n=(int)g.nodes.size();
    std::vector<std::vector<int>> follow(n);
    for(int z=0;z<n;++z)if(g.nodes[z].op&&g.nodes[z].a!=g.nodes[z].b){
        follow[g.nodes[z].a].push_back(z);follow[g.nodes[z].b].push_back(z);
    }
    std::vector<Job> out;
    std::unordered_set<int> scoring(g.targets.begin(),g.targets.end());
    int walked=0;
    for(int i=0;i<n&&walked<22000;++i) {
        if(!g.nodes[i].op||follow[i].empty())continue;
        const auto first=make_cut(g,{i});
        if(first.budget<3)continue;
        std::vector<int> route{i};
        auto explore=[&](auto& self)->void {
            const int length=(int)route.size();
            if(++walked>22000)return;
            if(length>=2){
                auto cut=make_cut(g,route);
                if(cut.budget>=length+2){
                    std::vector<Step> back;
                    for(int j=length-1;j>0;--j){
                        int old=route[j-1],newer=route[j];
                        Node n=g.nodes[newer];
                        const bool a=old==n.a;
                        int other=a?n.b:n.a;
                        if(!cut.terminal[other]&&std::find(route.begin()+j,route.end(),other)==route.end())break;
                        const Value x=g.nodes[old].value,y=n.value,z=g.nodes[other].value;
                        Step inverse{x,'-',y,z};
                        switch(n.op) {
                            case '+':break;
                            case '-':inverse=a?Step{x,'+',y,z}:Step{x,'-',z,y};break;
                            case '*':inverse={x,'/',y,z};break;
                            case '/':inverse=a?Step{x,'*',y,z}:Step{x,'/',z,y};break;
                            default:throw std::logic_error("Unknown arithmetic operation");
                        }
                        if(calculate(inverse.op,inverse.a,inverse.b,limit)!=x)break;
                        back.push_back(inverse);
                    }
                    if((int)back.size()==length-1){
                        int retained=(int)follow[i].size()-1;
                        int price=25*(cut.budget-length)+13*std::min(retained,4)+5*(scoring.count(route.back())!=0)
                                  -3*(length-2)+int(cumulative::mix(g.nodes[route.back()].value^salt)%17);
                        out.push_back({route,std::move(back),std::move(cut),price});
                    }
                }
            }
            if(length>=7)return;
            for(int v:follow[route.back()]) {
                route.push_back(v);self(self);route.pop_back();
            }
        };
        explore(explore);
    }
    std::stable_sort(out.begin(),out.end(),[](const Job& a,const Job& b){return a.price>b.price;});
    std::vector<Job> final;
    std::unordered_map<int,int> begin,end;
    for(auto& z:out)if(begin[z.path.front()]++<4&&end[z.path.back()]++<4){
        final.push_back(std::move(z));if((int)final.size()>=jobs)break;
    }
    return final;
}

Program run(const Registers& r,const std::vector<Value>& seeds,const std::vector<Value>& goals,Value limit,
            Program best,int beam,int leaves,int jobs,int factors,bool reverse,Value salt,int millis,Stats& s) {
    const auto start=Clock::now();
    const auto finish=start+std::chrono::milliseconds(millis);
    product_distance::Stats precedent;
    int direct_ms=reverse?std::max(1,millis/3):millis;
    best=product_distance::run(r,seeds,goals,limit,std::move(best),beam,leaves,jobs,factors,true,salt,direct_ms,precedent);
    s.opens=precedent.cuts;s.wins=precedent.wins;s.candidates=precedent.states;s.logical=precedent.logical;s.arith=precedent.deep;
    if(!reverse||Clock::now()>=finish)return best;
    auto g=topological_graph(recover_graph(r,limit,goals,best));
    auto order=propose(g,limit,salt,jobs);
    for(int i=0;i<(int)order.size()&&Clock::now()<finish;++i){
        auto& j=order[i];
        Value t=g.nodes[j.path.back()].value;
        int allowed=j.cut.budget-(int)j.inverse.size()-1;
        if(allowed<1)continue;
        Construction c(limit,seeds,{t},false);
        for(int k=0;k<(int)g.nodes.size();++k)if(j.cut.terminal[k]&&g.nodes[k].op){
            auto n=g.nodes[k];c.append({n.value,n.op,g.nodes[n.a].value,g.nodes[n.b].value});
        }
        auto stable=cumulative::snapshot(c);
        if(!stable.count(1))c.append({1,'/',seeds.front(),seeds.front()});
        const auto now=Clock::now();
        const auto end=now+(finish-now)/std::min(8,(int)order.size()-i);
        product_distance::Stats extra;
        auto proof=product_distance::solve(c,stable,t,allowed,leaves,factors,beam,true,salt+i*479,now+(end-now)*3/4,extra);
        ++s.opens;s.candidates+=extra.states;s.arith+=extra.deep;
        if(proof.empty()&&Clock::now()<end){
            local_beam::Dictionary d((SInt)limit);
            d.build(cumulative::leaves(c,{t},leaves,salt+i*41));
            auto fast=synthesize_new(c,d,t,std::max(beam,8),true);
            if(!stable.count(1))fast.insert(fast.begin(),{1,'/',seeds.front(),seeds.front()});
            proof=cumulative::needed(fast,t,stable);
        }
        if(proof.empty()||(int)proof.size()>allowed)continue;
        auto refs=stable;
        for(Step z:proof)refs.insert(z.x);
        for(Step z:j.inverse)if(!refs.count(z.x)){
            if(!refs.count(z.a)||!refs.count(z.b))throw std::logic_error("Inconsistent reverse bridge");
            proof.push_back(z);refs.insert(z.x);
        }
        ++s.logical;
        auto program=replace_block(g,j.cut,proof,limit,(int)best.size(),14);
        if(!program.empty())best=std::move(program),++s.reroots,++s.wins;
    }
    return best;
}
}

int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=8)throw std::invalid_argument("beam leaves jobs factors reverse salt millis");
        const int width=std::stoi(argv[1]),leaves=std::stoi(argv[2]),jobs=std::stoi(argv[3]),
                  factors=std::stoi(argv[4]),mode=std::stoi(argv[5]),millis=std::stoi(argv[7]);
        const auto salt=std::stoull(argv[6]);
        if(width<1||leaves<12||leaves>86||jobs<1||factors<1||mode<0||mode>1||millis<10)
            throw std::invalid_argument("Invalid reverse bridge profile");
        local_beam::INDEX_BITS=leaves>62?19:18;
        Value bound;int ns,nt,n;std::string key;
        while(std::cin>>key>>bound>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||!bound||bound>=(Value(1)<<62)||n<0||n>4096)
                throw std::invalid_argument("Invalid puzzle");
            Registers regs{};std::vector<Value> starts(ns),goals(nt);Program base(n);
            for(int j=0;j<ns;++j){std::cin>>starts[j];regs[j]=starts[j];}
            for(auto& x:goals)std::cin>>x;
            for(auto& s:base)std::cin>>s.kind>>s.a>>s.b>>s.destination;
            reverse_bridges::Stats stats;
            auto p=reverse_bridges::run(regs,starts,goals,bound,std::move(base),width,leaves,jobs,factors,mode!=0,salt,millis,stats);
            std::unordered_set<Value> seen(starts.begin(),starts.end());for(auto s:p)seen.insert(execute(regs,s,bound));
            for(Value v:goals)if(!seen.count(v))throw std::logic_error("Missing target");
            std::cout<<key<<' '<<p.size()<<' '<<stats.opens<<' '<<stats.wins<<' '<<stats.candidates
                <<' '<<stats.reroots<<' '<<stats.logical<<' '<<stats.arith<<'\n';
            for(auto s:p)std::cout<<s.kind<<' '<<s.a<<' '<<s.b<<' '<<s.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
