// Nearest known products, rather than nearest integer quotient. Search
// around unequal-cost operands and evaluate exact arithmetic corrections.
#define ELF_CUMULATIVE_INCLUDE
#include "elf_cumulative_sets.cpp"

namespace product_distance {
using Clock=std::chrono::steady_clock;
using Deadline=Clock::time_point;
struct Stats {int cuts=0,wins=0,states=0,split=0,logical=0,deep=0;};
struct Form {Value a,b,r,d,prod;char sign;float score;int cost,bin;};

Script solve(const Construction& c,const std::unordered_set<Value>& free,Value t,int limit_cost,
             int leaves,int stride_count,int tail_beam,bool use_new,Value salt,Deadline end,Stats& st) {
    local_beam::Dictionary d((SInt)c.limit);
    d.build(cumulative::leaves(c,{t},leaves,salt));
    auto cost=[&](Value v) {
        if(c.index.count(v))return 0;
        int here=d.index.get((SInt)v);return here<0?99:d.e[here].cost;
    };
    std::vector<Form> choices;
    auto offer=[&](Value a,Value b,Value den) {
        if(!a||!b||a==t||b==t||a>b||b>c.limit/a||a==1||b==1)return;
        Value n=t*den,m=a*b;
        if(den>1&&m==t)return;
        Value r=n>m?n-m:m-n;
        if(r==t||r==m)return;
        char op=n>=m?'+':'-';
        int k=cost(a)+(a==b?0:cost(b))+int(r!=0)+(den!=1?1+cost(den):0)+1;
        int fixed=k+(r?cost(r):0);
        if(k>limit_cost+1)return;
        float price=(float)k;
        if(r){
            if(fixed<95)price=(float)fixed;
            else {
                int bits=bit_count(r);
                if(!use_new||bits>=bit_count(n)-5||k+2>limit_cost+1)return;
                price+=2.12f+.037f*bits;
            }
        }
        if(price>limit_cost+1.65f)return;
        int bucket=std::min(27,bit_count(a)/3)+28*(r!=0&&fixed>=95);
        price+=.003f*(r?bit_count(r):0)+.00009f*(cumulative::mix(a^r^salt)%199);
        choices.push_back({a,b,r,den,m,op,price,fixed,bucket});
        ++st.states;
    };
    std::vector<int> factors;
    for(int j=0;j<d.n0;++j)factors.push_back(j);
    if(use_new) {
        std::vector<std::pair<int,int>> extra;
        for(int j=d.n0;j<d.n1;++j) {
            Value x=(Value)d.e[j].v;
            if(x<12||x>c.limit/x||x>t/2)continue;
            // We retain factors across the whole numeric range, not only
            // the earliest dictionary rows or the nearest square.
            int z=(int)(cumulative::mix(x^salt)%991);
            if(x>t/x/4096&&x<t/x*16)z-=300;
            extra.emplace_back(z,j);
        }
        int keep=std::min(stride_count,(int)extra.size());
        std::partial_sort(extra.begin(),extra.begin()+keep,extra.end());
        for(int j=0;j<keep;++j)factors.push_back(extra[j].second);
    }
    std::vector<Value> divisors{1};
    if(use_new)for(Value z:{2,3,5,7,11,16,23})if(z<=c.limit/t&&cost(z)==0)divisors.push_back(z);
    for(Value den:divisors) {
        if(Clock::now()>=end)break;
        Value n=t*den;
        for(int pos=0;pos<(int)factors.size();++pos) {
            if((pos%128)==0&&Clock::now()>=end)break;
            int a_index=factors[pos];
            Value a=(Value)d.e[a_index].v;
            if(a<2||a>c.limit/a||a>n/2)continue;
            if(den>1&&pos>d.n0+stride_count/3)continue;
            Value q=n/a,r=n%a;
            if(cost(q)<=2&&(!r||cost(r)<=2))offer(a,q,den);
            if(r&&cost(q+1)<=2&&cost(a-r)<=2)offer(a,q+1,den);
            if(!use_new)continue;
            for(int depth=0;depth<3;++depth) {
                const auto& line=d.sorted[depth];
                auto it=std::lower_bound(line.begin(),line.end(),std::make_pair((SInt)q,-1));
                const auto centre=it-line.begin();
                for(int gap:{-3,-2,-1,0,1,2}) {
                    const auto at=centre+gap;
                    if(at<0||at>=(int)line.size())continue;
                    offer(a,(Value)line[at].first,den);
                }
            }
        }
    }
    std::sort(choices.begin(),choices.end(),[](auto a,auto b){
        return std::tie(a.score,a.a,a.b,a.d)<std::tie(b.score,b.a,b.b,b.d);
    });
    std::set<std::tuple<Value,Value,Value>> distinct;
    std::vector<int> bands(56);
    std::unordered_map<Value,Script> remainders;
    Script best;
    int visits=0;
    for(const auto& f:choices) {
        if(Clock::now()>=end||visits>=48)break;
        if(!distinct.emplace(f.a,f.b,f.d).second||bands[f.bin]++>=9)continue;
        Planner build{c,d};
        if(!build.attach(f.a)||!build.attach(f.b)||!build.attach(f.d))continue;
        build.push(f.prod,'*',f.a,f.b);
        if(f.r&&!build.attach(f.r)) {
            if(!use_new||tail_beam<1)continue;
            auto [where,missing]=remainders.try_emplace(f.r);
            if(missing){
                // First try an exact one-instruction relation to one of the
                // newly computed limbs (free only inside this proposed proof).
                bool done=false;
                for(Value limb:{f.a,f.b,f.prod})if(!done) {
                    inverse(f.r,limb,c.limit,[&](Value other,char op,bool first){
                        if(other==f.r||!build.attach(other))return false;
                        build.push(f.r,op,first?other:limb,first?limb:other);
                        done=true;return true;
                    });
                }
                // A cached residual must not depend on a different prefix.
                if(!done)where->second=synthesize_new(c,d,f.r,tail_beam,true);
                else remainders.erase(where);
                ++st.deep;
                if(done)++st.split;
            }
            if(!build.made.count(f.r)) {
                const auto& s=remainders.at(f.r);
                for(auto x:s)build.push(x.x,x.op,x.a,x.b);
            }
        }
        Value n=t*f.d;
        if(f.r)build.push(n,f.sign,f.prod,f.r);
        if(f.d>1)build.push(t,'/',n,f.d);
        Script full;
        if(!free.count(1)) {
            Value x=c.starts.front();full.push_back({1,'/',x,x});
        }
        full.insert(full.end(),build.plan.begin(),build.plan.end());
        auto proof=cumulative::needed(full,t,free);
        ++visits;
        if((int)proof.size()<=limit_cost&&(best.empty()||proof.size()<best.size())) {
            best=std::move(proof);limit_cost=(int)best.size()-1;
            if(limit_cost<2)break;
        }
    }
    return best;
}

Program run(const Registers& regs,const std::vector<Value>& starts,const std::vector<Value>& targets,
            Value limit,Program tape,int beam,int leaves,int jobs,int factors,bool newer,
            Value salt,int ms,Stats& st) {
    auto g=topological_graph(recover_graph(regs,limit,targets,tape));
    const auto finish=Clock::now()+std::chrono::milliseconds(ms);
    std::vector<std::pair<int,int>> entries;
    for(int i=0;i<(int)g.nodes.size();++i)if(g.nodes[i].op) {
        const auto a=make_cut(g,{i});
        if(a.budget>=4)entries.emplace_back(-a.budget*24+(int)(cumulative::mix(g.nodes[i].value^salt)%23),i);
    }
    std::sort(entries.begin(),entries.end());
    const int n=std::min(jobs,(int)entries.size());
    for(int j=0;j<n&&Clock::now()<finish;++j) {
        int root=entries[j].second;
        auto cut=make_cut(g,{root});
        const Value t=g.nodes[root].value;
        Construction c(limit,starts,{t},false);
        for(int k=0;k<(int)g.nodes.size();++k)if(cut.terminal[k]&&g.nodes[k].op){
            auto a=g.nodes[k];c.append({a.value,a.op,g.nodes[a.a].value,g.nodes[a.b].value});
        }
        auto free=cumulative::snapshot(c);
        if(!free.count(1))c.append({1,'/',starts.front(),starts.front()});
        auto at=Clock::now();auto until=at+(finish-at)/std::min(8,n-j);
        ++st.cuts;
        auto proof=solve(c,free,t,cut.budget-1,leaves,factors,beam,newer,salt+j*173,until,st);
        if(proof.empty())continue;
        ++st.logical;
        auto exact=replace_block(g,cut,proof,limit,(int)tape.size(),12);
        if(!exact.empty())tape=std::move(exact),++st.wins;
    }
    return tape;
}
}

#ifndef ELF_PRODUCT_DISTANCE_INCLUDE
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=8)throw std::invalid_argument("beam leaves cuts factors near salt millis");
        int beam=std::stoi(argv[1]),leaves=std::stoi(argv[2]),cuts=std::stoi(argv[3]),factors=std::stoi(argv[4]);
        int noisy=std::stoi(argv[5]),ms=std::stoi(argv[7]);const auto salt=std::stoull(argv[6]);
        if(beam<1||leaves<12||leaves>86||cuts<1||factors<1||noisy<0||noisy>1||ms<10)
            throw std::invalid_argument("Invalid product-distance profile");
        local_beam::INDEX_BITS=leaves>62?19:18;
        Value limit;int ns,nt,n;std::string key;
        while(std::cin>>key>>limit>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||n<0||n>4096||!limit||limit>=(Value(1)<<62))
                throw std::invalid_argument("Invalid puzzle");
            Registers regs{};std::vector<Value> seeds(ns),targets(nt);Program baseline(n);
            for(int j=0;j<ns;++j){std::cin>>seeds[j];regs[j]=seeds[j];}
            for(auto& v:targets)std::cin>>v;
            for(auto& s:baseline)std::cin>>s.kind>>s.a>>s.b>>s.destination;
            product_distance::Stats st;
            auto exact=product_distance::run(regs,seeds,targets,limit,std::move(baseline),beam,leaves,cuts,factors,noisy!=0,salt,ms,st);
            std::unordered_set<Value> seen(seeds.begin(),seeds.end());for(auto s:exact)seen.insert(execute(regs,s,limit));
            for(Value t:targets)if(!seen.count(t))throw std::logic_error("Missing target");
            std::cout<<key<<' '<<exact.size()<<' '<<st.cuts<<' '<<st.wins<<' '<<st.states<<' '<<st.split<<' '<<st.logical<<' '<<st.deep<<'\n';
            for(auto s:exact)std::cout<<s.kind<<' '<<s.a<<' '<<s.b<<' '<<s.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
