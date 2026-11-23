// Synthesize several mandatory interface values while charging the count
// of extra helper writes once. Existing descendants are not free operands.
#define ELF_RESIDUAL_INCLUDE
#include "elf_block_rebuild.cpp"
#include "elf_numeric_splice.h"

namespace interface_search {
using rebuild16::Clock;
using rebuild16::Deadline;
using rebuild16::hash;

struct Meter {int cuts=0,wins=0,exits=0,expanded=0,closures=0,dormant=0;};
struct Scope {Cut cut;int kind,quality;};

std::vector<Scope> scopes(const Graph& g,Value limit,int count,Value salt,bool expanded=false){
    const int n=(int)g.nodes.size();
    std::vector<int> level(n);
    std::vector<char> required(n);
    for(int i:g.targets)required[i]=1;
    for(int i=0;i<n;++i)if(auto v=g.nodes[i];v.op){
        level[i]=1+std::max(level[v.a],level[v.b]);
    }
    std::set<std::vector<int>> seen;
    std::vector<Scope> all;
    auto add=[&](std::vector<char> inside,int kind){
        std::vector<char> port(n);
        for(int i=0;i<n;++i){
            if(inside[i]&&required[i])port[i]=1;
            if(!inside[i]&&g.nodes[i].op){
                auto v=g.nodes[i];
                if(inside[v.a])port[v.a]=1;
                if(inside[v.b])port[v.b]=1;
            }
        }
        std::vector<int> roots;
        for(int i=0;i<n;++i)if(port[i]&&g.nodes[i].op)roots.push_back(i);
        if(roots.size()<2||roots.size()>(expanded?52u:33u)||!seen.insert(roots).second)return;
        const auto c=make_cut(g,roots);
        const int hidden=c.budget-(int)roots.size();
        if(hidden<1||hidden>27)return;
        int players=0;
        for(auto i:roots)players+=required[i];
        int rank=12*std::min(hidden,9)+2*std::min(20,(int)roots.size())-5*std::max(0,hidden-11)
                +3*players+int(hash(g.nodes[roots.front()].value^salt)%17);
        all.push_back({std::move(c),kind,rank});
    };
    for(int k=5;k<=(expanded?45:39);k+=expanded?1:2){
        const Value cap=std::min(limit,(Value(1)<<k)-1);
        std::vector<char> low(n);
        for(int i=0;i<n;++i)low[i]=g.nodes[i].value<=cap;
        add(std::move(low),0);
        if(cap==limit)break;
    }
    const int h=*std::max_element(level.begin(),level.end());
    for(int d=2;d<=std::min(h-1,30);++d){
        std::vector<char> early(n);
        for(int i=0;i<n;++i)early[i]=level[i]<=d;
        add(std::move(early),1);
    }
    for(int f=1;f<=8;++f){
        std::vector<char> early(n);
        const int enter=f*n/10;
        for(int i=0;i<enter;++i)early[i]=1;
        add(std::move(early),2);
    }
    if(expanded){
        // These frontiers can expose low and high values together. Treat
        // the consumers outside the chosen region as a fixed contract.
        for(int k=8;k<=40;k+=4)for(int d:{4,7,10,13,17,21}){
            const Value cap=std::min(limit,(Value(1)<<k)-1);
            std::vector<char> mix(n);
            for(int i=0;i<n;++i)mix[i]=g.nodes[i].value<=cap&&level[i]<=d;
            add(std::move(mix),1);
        }
        std::vector<std::vector<int>> users(n);
        for(int i=0;i<n;++i)if(auto v=g.nodes[i];v.op){
            users[v.a].push_back(i);if(v.b!=v.a)users[v.b].push_back(i);
        }
        for(int i=0;i<n;++i)if(g.nodes[i].op&&users[i].size()>1){
            std::vector<char> keep(n);
            std::vector<int> todo{i};
            keep[i]=1;
            for(int step=0;step<3&&!todo.empty();++step){
                std::vector<int> later;
                for(int v:todo)for(int x:users[v])if(!keep[x]){
                    later.push_back(x);keep[x]=1;
                }
                if(step>=1)add(keep,2);
                todo=std::move(later);
            }
        }
    }
    std::vector<Scope> out;
    std::array<int,3> shapes{};
    while(!all.empty()&&(int)out.size()<count){
        auto rank=[&](const Scope& s){
            int overlap=0;
            for(const auto& p:out){
                int shared=0;
                for(auto i:s.cut.roots)shared+=std::binary_search(p.cut.roots.begin(),p.cut.roots.end(),i);
                overlap=std::max(overlap,100*shared/(int)(p.cut.roots.size()+s.cut.roots.size()-shared));
            }
            return s.quality-overlap-shapes[s.kind]*6;
        };
        auto at=std::max_element(all.begin(),all.end(),[&](const auto& a,const auto& b){return rank(a)<rank(b);});
        ++shapes[at->kind];out.push_back(std::move(*at));all.erase(at);
    }
    return out;
}

struct State {
    std::vector<Value> order;
    std::unordered_set<Value> held;
    Script script;
    Value remaining=0;
    Value fingerprint=0;
    int helpers=0;
    explicit State(const std::vector<Value>& v,int count):order(v),held(v.begin(),v.end()),
                                       remaining((Value(1)<<count)-1){held.reserve(v.size()+64);}
    void put(Step step,bool helper){
        if(!held.insert(step.x).second)return;
        order.push_back(step.x);script.push_back(step);
        fingerprint^=hash(step.x+7919);helpers+=helper;
    }
};

struct Pocket {Cut cut;Script recipe;Value mask;};
struct Unions {
    const Graph& full;
    const Cut& template_cut;
    std::unordered_map<Value,int> margins;
    std::vector<Pocket> plans;
    explicit Unions(const Graph& f,const Cut& c):full(f),template_cut(c){margins.reserve(128);}
    static int worth(const Pocket& p){return p.cut.budget-(int)p.recipe.size();}
    void observe(const State& now){
        if(!now.remaining||now.script.empty())return;
        const auto set=((Value(1)<<template_cut.roots.size())-1)^now.remaining;
        const auto completed=rebuild16::pop(set);
        if(completed<2)return;
        auto it=margins.find(set);
        if(it==margins.end()&&margins.size()>=128)return;
        if(it!=margins.end()&&it->second<=completed)return;
        std::vector<int> roots;
        for(int i=0;i<(int)template_cut.roots.size();++i)if(set&(Value(1)<<i))roots.push_back(template_cut.roots[i]);
        auto hit=std::find_if(plans.begin(),plans.end(),[&](const auto& q){return q.mask==set;});
        if(hit!=plans.end()&&(int)hit->recipe.size()==completed)return;
        const int credit=it!=margins.end()?it->second:margins.emplace(set,make_cut(full,roots).budget).first->second;
        if(credit<=completed)return;
        // Search histories can contain a helper that is not an ancestor
        // of the retained exits. Charge the actual needed slice.
        std::unordered_set<Value> needed;
        for(int r:roots)needed.insert(full.nodes[r].value);
        Script kept;
        for(auto z=now.script.rbegin();z!=now.script.rend();++z){
            if(!needed.count(z->x))continue;
            kept.push_back(*z);needed.insert(z->a);needed.insert(z->b);
        }
        std::reverse(kept.begin(),kept.end());
        if((int)kept.size()>=credit)return;
        if(hit!=plans.end()){
            if(kept.size()<hit->recipe.size())hit->recipe=std::move(kept);
            return;
        }
        Pocket p{make_cut(full,roots),std::move(kept),set};
        if(plans.size()>=12){
            auto last=std::min_element(plans.begin(),plans.end(),[&](const auto& a,const auto& b){return worth(a)<worth(b);});
            if(worth(p)>worth(*last))*last=std::move(p);
        }else plans.push_back(std::move(p));
    }
};

Step from(Value t,Value a,const std::unordered_set<Value>& state,Value limit){
    Step out{};
    inverse(t,a,limit,[&](Value b,char op,bool left){
        if(!state.count(b)||b==t||a==t)return false;
        out={t,op,left?b:a,left?a:b};return true;
    });
    return out;
}

void closure(State& a,const std::vector<Value>& targets,Value bound,std::size_t since){
    // The old delta was already closed, so only new values can trigger
    // another exit. Adding an exit costs a write, but not a helper.
    for(std::size_t j=since;j<a.order.size()&&a.remaining;++j){
        const Value x=a.order[j],prior=a.remaining;
        for(int i=0;i<(int)targets.size();++i)if(prior&(Value(1)<<i)){
            const auto t=targets[i];
            if(a.held.count(t)){
                a.remaining&=~(Value(1)<<i);continue;
            }
            if(auto s=from(t,x,a.held,bound);s.x){
                a.put(s,false);a.remaining&=~(Value(1)<<i);
            }
        }
    }
}

using One=std::unordered_map<Value,Step>;
One one_step_bank(const State& a,Value limit,Deadline until,std::size_t offset=0){
    One out;out.reserve(a.order.size()*(a.order.size()-offset));
    for(int i=(int)offset;i<(int)a.order.size();++i){
        if((i&7)==0&&Clock::now()>=until)break;
        for(int j=0;j<=i;++j){
            Value x=a.order[i],y=a.order[j];if(x<y)std::swap(x,y);
            for(char op:{'+','-','*','/'}){
                Value z=calculate(op,x,y,limit);
                if(z&&!a.held.count(z))out.try_emplace(z,Step{z,op,x,y});
            }
        }
    }
    return out;
}

template<class F> void same_input(Value t,const F& offer){
    if((t&1)==0)offer(t/2);
    const Value n=(Value)std::sqrt((long double)t);
    for(Value q:{n-1,n,n+1})if(q>1&&t/q==q&&t%q==0)offer(q);
}

struct Vote {Value direct=0,later=0;};
std::vector<Step> moves(const State& a,const std::vector<Value>& targets,Value bound,int width,
                        bool deeper,Value salt,Deadline until,Meter& met,
                        const One* common=nullptr,std::size_t common_size=0,bool delta=false,bool exact=false){
    auto extra=common?(delta?one_step_bank(a,bound,until,common_size):One{}):one_step_bank(a,bound,until);
    const auto& bank=common?*common:extra;
    auto locate=[&](Value num)->Step{
        if(auto there=bank.find(num);there!=bank.end())return there->second;
        if(common){
            if(delta){
                if(auto where=extra.find(num);where!=extra.end())return where->second;
            }else for(std::size_t j=common_size;j<a.order.size();++j){
                if(auto proof=from(num,a.order[j],a.held,bound);proof.x)return proof;
            }
        }
        return {};
    };
    std::unordered_map<Value,Vote> proposals;
    std::unordered_map<Value,Value> unrealized;
    for(int i=0;i<(int)targets.size()&&Clock::now()<until;++i){
        if(!(a.remaining&(Value(1)<<i)))continue;
        const Value t=targets[i],bit=Value(1)<<i;
        auto register_need=[&](Value x){
            if(!x||x==t||a.held.count(x))return;
            if(locate(x).x)proposals[x].direct|=bit;
            else if(deeper)unrealized[x]|=bit;
        };
        for(Value known:a.order)inverse(t,known,bound,[&](Value x,char,bool){register_need(x);return false;});
        same_input(t,register_need);
    }
    if(deeper&&!unrealized.empty()){
        std::vector<std::pair<int,Value>> possible;
        for(auto [v,coverage]:unrealized)possible.emplace_back(-60*rebuild16::pop(coverage)+bit_count(v)+(int)(hash(v^salt)%17),v);
        const int n=exact?(int)possible.size():std::min((int)possible.size(),std::max(40,width*5));
        std::partial_sort(possible.begin(),possible.begin()+n,possible.end());
        for(int i=0;i<n&&Clock::now()<until;++i){
            const Value later=possible[i].second,bits=unrealized.at(later);
            auto nominate=[&](Value h){
                if(h==later||!bank.count(h))return;
                auto& v=proposals[h];v.later|=bits;
            };
            for(Value k:a.order)inverse(later,k,bound,[&](Value h,char,bool){nominate(h);return false;});
            same_input(later,nominate);
        }
        // The second helper can be paired with the first in the same
        // output. Check this different shape with already-constructible
        // one-write partners, or partners using the first helper.
        std::vector<std::pair<Value,Value>> sampled;
        for(auto& [x,z]:bank)sampled.emplace_back(hash(x^salt),x);
        const int n2=exact?(int)sampled.size():std::min((int)sampled.size(),std::max(90,width*12));
        std::partial_sort(sampled.begin(),sampled.begin()+n2,sampled.end());
        for(int y=0;y<n2&&Clock::now()<until;++y){
            const Value h=sampled[y].second;
            for(int i=0;i<(int)targets.size();++i)if(a.remaining&(Value(1)<<i)){
                const Value t=targets[i];
                inverse(t,h,bound,[&](Value partner,char,bool){
                    if(partner==h||partner==t||a.held.count(partner))return false;
                    if(bank.count(partner)||from(partner,h,a.held,bound).x||
                       calculate('+',h,h,bound)==partner||calculate('*',h,h,bound)==partner)
                        proposals[h].later|=Value(1)<<i;
                    return false;
                });
            }
        }
    }
    std::vector<std::pair<int,Value>> ranks;
    for(auto& [v,h]:proposals){
        if(!h.direct&&h.later)++met.dormant;
        ranks.emplace_back(-100*rebuild16::pop(h.direct)-47*rebuild16::pop(h.later&~h.direct)
                                    +bit_count(v)/3+(int)(hash(v^salt)%29),v);
    }
    int take=exact?(int)ranks.size():std::min(width,(int)ranks.size());
    std::partial_sort(ranks.begin(),ranks.begin()+take,ranks.end());
    std::vector<Step> out;
    for(int k=0;k<take;++k)out.push_back(locate(ranks[k].second));
    return out;
}

Script tight(const std::vector<Value>& ready,const std::vector<Value>& exits,Value limit,
             int helpers,Value salt,Deadline until,Meter& met){
    if(helpers<0||helpers>2||exits.size()>=64||exits.empty())return {};
    State base(ready,(int)exits.size());
    closure(base,exits,limit,0);
    if(!base.remaining)return base.script;
    if(!helpers)return {};
    ++met.expanded;
    const auto h1=moves(base,exits,limit,32,helpers==2,salt,until,met,nullptr,0,false,true);
    for(auto s:h1){
        if(Clock::now()>=until)break;
        auto next=base;
        next.put(s,true);
        closure(next,exits,limit,base.order.size());
        if(!next.remaining)return next.script;
        if(helpers<2)continue;
        ++met.expanded;
        const auto h2=moves(next,exits,limit,32,false,salt^s.x,until,met,nullptr,0,false,true);
        for(auto t:h2){
            if(Clock::now()>=until)break;
            auto drop=next;
            drop.put(t,true);
            closure(drop,exits,limit,next.order.size());
            if(!drop.remaining)return drop.script;
        }
    }
    return {};
}

Script synth(const std::vector<Value>& ready,const std::vector<Value>& exits,Value limit,int helper_cap,
             int width,int mode,Value salt,Deadline until,Meter& met,Unions* stops=nullptr){
    if(helper_cap<0||exits.empty()||exits.size()>=64)return {};
    if(mode==6)return tight(ready,exits,limit,helper_cap,salt,until,met);
    State first(ready,(int)exits.size());
    closure(first,exits,limit,0);
    if(!first.remaining)return first.script;
    if(stops)stops->observe(first);
    if(!helper_cap)return {};
    const bool cached=mode>=3;
    const std::size_t ready_count=first.order.size();
    const auto shared=cached?one_step_bank(first,limit,until):One{};
    std::vector<State> wave{std::move(first)};
    std::unordered_set<Value> states;
    for(int stage=0;stage<helper_cap&&!wave.empty()&&Clock::now()<until;++stage){
        std::vector<State> next;
        for(const auto& s:wave){
            if(Clock::now()>=until)break;
            ++met.expanded;
            const auto choices=moves(s,exits,limit,width*(stage<2?4:2),mode>0&&mode<3&&stage+1<helper_cap,
                                     salt^s.fingerprint,until,met,cached?&shared:nullptr,ready_count,mode==5||mode>=7);
            for(auto candidate:choices){
                auto n=s;
                const auto before=n.order.size();
                n.put(candidate,true);
                closure(n,exits,limit,before);
                if(!n.remaining)return n.script;
                if(stops)stops->observe(n);
                auto signature=n.fingerprint;
                if(states.insert(signature).second)next.push_back(std::move(n));
            }
        }
        auto order=[&](const State& x){return std::tuple{rebuild16::pop(x.remaining),x.helpers,hash(x.fingerprint^salt)};};
        const int n=std::min(width,(int)next.size());
        std::partial_sort(next.begin(),next.begin()+n,next.end(),[&](auto& a,auto& b){return order(a)<order(b);});
        if((int)next.size()>n)next.erase(next.begin()+n,next.end());
        wave=std::move(next);
    }
    return {};
}

Program run(const Registers& regs,const std::vector<Value>& goals,Value bound,Program baseline,
            int count,int width,int mode,Value salt,int ms,Meter& met){
    const auto end=Clock::now()+std::chrono::milliseconds(ms);
    auto g=topological_graph(recover_graph(regs,bound,goals,baseline));
    auto jobs=scopes(g,bound,count,salt,mode>=4);
    for(int i=0;i<(int)jobs.size()&&Clock::now()<end;++i){
        const auto& c=jobs[i].cut;
        if(mode==6&&c.budget-(int)c.roots.size()>3)continue;
        std::vector<Value> ready,exits;
        for(int j=0;j<(int)g.nodes.size();++j)if(c.terminal[j])ready.push_back(g.nodes[j].value);
        for(auto j:c.roots)exits.push_back(g.nodes[j].value);
        ++met.cuts;met.exits+=(int)exits.size();
        const auto now=Clock::now(),stop=now+(end-now)/std::min((int)jobs.size()-i,5);
        Unions pockets(g,c);
        const int slack=mode==9?2:0;
        auto block=synth(ready,exits,bound,c.budget-1+slack-(int)exits.size(),width,mode,salt+i*757,stop,met,
                         mode>=7?&pockets:nullptr);
        auto accept=[&](const Cut& region,const Script& plan){
            if(plan.empty()||plan.size()>=(std::size_t)(region.budget+slack))return;
            ++met.closures;
            auto trial=mode>=8?numeric_splice::apply(g,plan,bound,(int)baseline.size(),16):Program{};
            if(trial.empty()&&plan.size()<(std::size_t)region.budget)
                trial=replace_block(g,region,plan,bound,(int)baseline.size(),16);
            if(trial.empty())return;
            ++met.wins;baseline=std::move(trial);
        };
        accept(c,block);
        if(mode>=7)for(const auto& q:pockets.plans){
            if(Clock::now()>=end)break;
            ++met.dormant;
            accept(q.cut,q.recipe);
        }
    }
    return baseline;
}
}

#ifndef ELF_INTERFACE_CLOSURE_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=6)throw std::invalid_argument("cuts width mode salt ms");
        int cuts=std::stoi(argv[1]),width=std::stoi(argv[2]),mode=std::stoi(argv[3]),ms=std::stoi(argv[5]);
        const auto salt=std::stoull(argv[4]);
        if(cuts<1||width<1||width>35||mode<0||mode>9||ms<10)throw std::invalid_argument("Invalid closure profile");
        Value limit;int ns,nt,n;std::string key;
        while(std::cin>>key>>limit>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||!limit||limit>=(Value(1)<<62)||n<0||n>4096)throw std::invalid_argument("Bad case");
            Registers r{};std::vector<Value> goals(nt);Program tape(n);
            for(int i=0;i<ns;++i)std::cin>>r[i];
            for(auto& t:goals)std::cin>>t;
            for(auto& q:tape)std::cin>>q.kind>>q.a>>q.b>>q.destination;
            interface_search::Meter d;
            auto best=interface_search::run(r,goals,limit,std::move(tape),cuts,width,mode,salt,ms,d);
            std::unordered_set<Value> hit(r.begin(),r.end());
            for(auto q:best)hit.insert(execute(r,q,limit));
            for(auto t:goals)if(!hit.count(t))throw std::logic_error("Missing target");
            std::cout<<key<<' '<<best.size()<<' '<<d.cuts<<' '<<d.wins<<' '<<d.exits<<' '<<d.expanded<<' '
                <<d.closures<<' '<<d.dormant<<'\n';
            for(auto q:best)std::cout<<q.kind<<' '<<q.a<<' '<<q.b<<' '<<q.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
