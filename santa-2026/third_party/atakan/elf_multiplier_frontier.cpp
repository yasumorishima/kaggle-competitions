// Explore several physically equal-cost arithmetic graphs in parallel.
// V14 is the only source of prior programs; no additional program bank.
#define ELF_MULTIPLIER_LIBRARY
#include "elf_multiplier_lanes.cpp"
#include "santa_factor64.h"
#include <chrono>

namespace frontier {
using lanes::Edits;

Value scramble(Value x){
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

struct Signature {Value first=0,second=0,shape=0;};
Signature fingerprint(const Graph& g){
    Signature out;
    for(auto n:g.nodes)if(n.op) {
        auto a=g.nodes[n.a].value,b=g.nodes[n.b].value;
        if((n.op=='+'||n.op=='*')&&a>b)std::swap(a,b);
        const auto h=scramble(n.value)^scramble(a+16685271)^scramble(b+89123543)^scramble(n.op);
        out.first+=scramble(h);
        out.second^=scramble(h^0xc2b2ae3d27d4eb4fULL);
        if(n.op=='*'||n.op=='/')out.shape^=1ULL<<(h%64);
    }
    return out;
}

using Key=std::pair<Value,Value>;
Key key(Signature x){return {x.first,x.second};}
struct SearchNode {
    Program code;Signature sig;Value last_factor;int quality,root_count,parent;bool joined;
    std::vector<Value> nucleus;
};
struct EditChoice {Edits edits;int cost,roots,quality;Value factor,tag;bool joined;};
struct Stats {int expanded=0,improved=0,equal=0,multi=0,peak=0,fused=0,uphill=0,escaped=0,coalitions=0;};

int bits(Value x) {int n=0;for(;x;x&=x-1)++n;return n;}
Value edit_signature(const Edits& e){
    Value h=0;
    for(auto [i,j]:e)h^=scramble(Value(i)*1009+Value(j)*131071+0x335d);
    return h;
}

std::vector<Value> numeric_nucleus(const Graph& graph){
    std::vector<unsigned char> use(graph.nodes.size());
    std::vector<char> target(graph.nodes.size());
    for(auto i:graph.targets)target[i]=1;
    for(auto v:graph.nodes)if(v.op){
        if(use[v.a]<2)++use[v.a];
        if(v.a!=v.b&&use[v.b]<2)++use[v.b];
    }
    std::vector<Value> result;
    for(std::size_t i=0;i<graph.nodes.size();++i){
        const auto v=graph.nodes[i];
        if(v.op&&!target[i]){
            result.push_back(v.value<<1);
            if(use[i]>=2)result.push_back((v.value<<1)|1);
        }
    }
    std::sort(result.begin(),result.end());
    return result;
}

int symmetric_distance(const std::vector<Value>& a,const std::vector<Value>& b){
    std::size_t i=0,j=0;
    int common=0;
    while(i<a.size()&&j<b.size()){
        if(a[i]<b[j])++i;
        else if(a[i]>b[j])++j;
        else ++i,++j,++common;
    }
    return (int)(a.size()+b.size())-2*common;
}

Program physical(Graph& g,Value limit,int cap) {
    for(int k=0;k<8;++k){
        auto tape=pressure_schedule(g,k,(std::uint32_t)(limit/7+k*1493),cap);
        if(!tape.empty())return tape;
    }
    for(int k=0;k<5;++k){
        auto tape=Scheduler(g,cap,limit,k%4,k/3).run(g.targets);
        if(!tape.empty())return tape;
    }
    return {};
}

bool combine(Edits& base,const Edits& add,int conflict_mode) {
    for(auto [id,recipe]:add){
        auto where=std::find_if(base.begin(),base.end(),[&](auto e){return e.first==id;});
        if(where==base.end())base.emplace_back(id,recipe);
        else if(where->second!=recipe){
            if(!conflict_mode)return false;
            if(conflict_mode==2)where->second=recipe;
        }
    }
    return true;
}

void translated_products(const lanes::Atlas& atlas,const std::vector<Value>& scales,
                         const std::vector<int>& roots,lanes::Buckets& output,int mode){
    const auto& p=atlas.p;
    const int keep_scales=std::min((int)scales.size(),mode==7?67:42);
    const int per_root=mode==7?12:7;
    using Bucket=std::unordered_map<Value,std::vector<int>>;
    for(int ix=0;ix<keep_scales;++ix) {
        const Value factor=scales[ix];
        if(factor<=2||factor>p.limit/3)continue;
        Bucket same;
        same.reserve(atlas.n);
        for(int i=0;i<atlas.n;++i){
            auto x=p.nums[i];
            if(x>=256)same[x%factor].push_back(i);
        }
        for(int root:roots){
            const Value t=p.nums[root];
            if(t<512||t<factor*2 || atlas.private_size[root]<2)continue;
            struct Shift {Value a,q,d;char sign;int merit;Value denominator;};
            std::vector<Shift> choices;
            auto look=[&](Value query,Value den){
                auto found=same.find(query%factor);if(found==same.end())return;
                for(int i:found->second) {
                    const Value a=p.nums[i];
                    if(a==query||a==t||atlas.parent[i][root])continue;
                    Value d=query>a?query-a:a-query;
                    if(d<=factor||d==a||d==t)continue;
                    Value q=d/factor;
                    const bool paid=atlas.ready(d,root);
                    const bool direct=atlas.known(q);
                    if(!paid&&!direct&&(mode!=7||atlas.private_size[root]<4))continue;
                    const int merit=(paid?0:atlas.ready(q,root)?3:direct?5:10)
                                   +(den>1?3:0)+(d>query/3?2:0);
                    choices.push_back({a,q,d,query>a?'+':'-',merit,den});
                }
            };
            look(t,1);
            if(mode==7&&atlas.private_size[root]>=4)
                for(Value den:{2,3,5}){
                    if(t>p.limit/den||!atlas.ready(den,root))continue;
                    look(t*den,den);
                }
            std::sort(choices.begin(),choices.end(),[](auto& a,auto& b) {
                return std::tie(a.merit,a.d,a.a)<std::tie(b.merit,b.d,b.a);
            });
            int tried=0,kept=0;
            for(auto z:choices){
                if(kept>=per_root||tried++>=per_root*2)break;
                const Value numerator=t*z.denominator;
                lanes::Proof plan;
                if(z.merit>1&&!atlas.ready(z.d,root)) {
                    if(!atlas.take(factor,root,plan)||
                       !atlas.take(z.q,root,plan,mode==7?2:1)||
                       !atlas.add(plan,z.d,'*',factor,z.q))continue;
                }
                if(!atlas.add(plan,numerator,z.sign,z.a,z.d))continue;
                if(z.denominator>1 && !atlas.add(plan,t,'/',numerator,z.denominator))continue;
                // Index separately by the scaled difference: that exact
                // new value can pay off across more than one destination.
                if(!atlas.ready(z.d,root))
                    lanes::keep(atlas,output,root,z.d,plan,3);
                lanes::keep(atlas,output,root,factor,std::move(plan),4);
                ++kept;
            }
        }
    }
}

using DivisorCache=std::unordered_map<Value,std::vector<Value>>;
void latent_factors(const lanes::Atlas& a,const std::vector<int>& roots,
                    lanes::Buckets& results,DivisorCache& cache,int mode,Value salt) {
    struct Peel {int root;Value n,offset;char op;};
    std::vector<Peel> changes;
    const auto& p=a.p;
    std::unordered_map<Value,int> hits;
    std::unordered_map<Value,int> last;
    for(int id:roots) {
        const auto t=p.nums[id];
        const auto r=p.choices[id][0];
        if(t<15000||a.private_size[id]<2)continue;
        auto add=[&](Value v,Value q,char op) {
            if(v<10000||v>p.limit)return;
            if(std::none_of(changes.rbegin(),changes.rend(),[&](auto e){
                return e.root==id&&e.n==v&&e.offset==q&&e.op==op;
            }))changes.push_back({id,v,q,op});
        };
        add(t,0,'+');
        if(a.private_size[id]<4)continue;
        Value edit=std::min(p.nums[r.a],p.nums[r.b]);
        if(r.op=='-')edit=p.nums[r.b];
        if((r.op=='+'||r.op=='-')&&edit<t/8&&a.ready(edit,id)) {
            if(t>edit)add(t-edit,edit,'+');
            if(edit<=p.limit-t)add(t+edit,edit,'-');
        }
        // Near products are considered only with genuinely live corrections.
        const int budget=mode>=9?4:2;
        int hints=0;
        for(Value k:{1,2,3,4,5,7,8,11,16,31,32,64}) {
            if(!a.ready(k,id) || (scramble(k^salt^t)%4)==1)continue;
            if(hints++>=budget)break;
            add(t-k,k,'+');if(k<=p.limit-t)add(t+k,k,'-');
        }
    }
    for(auto job:changes){
        auto [it,inserted]=cache.try_emplace(job.n);
        if(inserted)it->second=small_factor::divisors(job.n,mode>=9?2200:1280);
        for(Value f:it->second) {
            if(f<=3||f>=job.n||f>1000000000000ULL)continue;
            auto remember=last.find(f);
            if(remember==last.end()||remember->second!=job.root) {
                ++hits[f];last[f]=job.root;
            }
        }
    }
    for(auto job:changes){
        std::vector<std::tuple<int,Value,Value>> offered;
        for(auto f:cache.at(job.n)) {
            if(f<=1||f>=job.n||f>1000000000000ULL)continue;
            const Value q=job.n/f;
            const bool f0=a.ready(f,job.root), q0=a.ready(q,job.root);
            const bool f1=a.known(f),q1=a.known(q);
            if((!f1&&!q1)||(!f1&&hits[f]<2)||(!q1&&a.private_size[job.root]<3))continue;
            const int rank=(f0?0:f1?4:10)+(q0?0:q1?4:10)+int(job.offset!=0)*3
                          -std::min(hits[f],5)*2;
            offered.emplace_back(rank,f,q);
        }
        std::sort(offered.begin(),offered.end());
        int tried=0;
        for(auto [_,f,q]:offered) {
            if(tried++>=(mode>=9?7:4))break;
            lanes::Proof proof;
            if(!a.take(f,job.root,proof,2)||!a.take(q,job.root,proof,2))continue;
            if(!job.offset || !a.ready(job.n,job.root)) {
                if(!a.add(proof,job.n,'*',f,q))continue;
            }
            if(job.offset&&!a.add(proof,p.nums[job.root],job.op,job.n,job.offset))continue;
            lanes::keep(a,results,job.root,f,std::move(proof),5);
        }
    }
}

struct Expansion {
    const Registers& regs;
    const std::vector<Value>& targets;
    const SearchNode& node;
    const Value limit;
    const int family_width,depth,style,count,epoch;
    const Value seed;
    PlanPool pool;
    Evaluator *check=nullptr;
    State base, active;
    std::vector<EditChoice> nominees;
    std::vector<EditChoice> bridge;
    std::vector<EditChoice> escape;
    std::unordered_set<Value> registered;
    const int max_cost;
#ifdef FRONTIER_COALITION
    bool synchronize=false;
    int complete_coalitions=0;
    std::vector<EditChoice> atomic;
    std::unordered_set<Value> recorded;
#endif

    Expansion(const Registers& r,const std::vector<Value>& t,const SearchNode& n,Value l,
              int width,int d,int policy,int want,int e,Value salt)
        :regs(r),targets(t),node(n),limit(l),family_width(width),depth(d),style(policy),count(want),
         epoch(e),seed(salt),pool(r,t,l,n.code),max_cost((int)n.code.size()) {}

    int eval(const Edits& edits) {
        active=base;
        for(auto [i,k]:edits)active[i]=k;
        return check->eval(active);
    }

    // Prefer genuinely different reusable products; equal-cost edits are
    // not selected solely by the size of their edit list.
    int potential(const Edits& edits,int roots,Value family,Value tag) const {
        int useful=0,n=0;
        for(auto [i,k]:edits) {
            if(!check->mark[i])continue;
            auto recipe=pool.choices[i][k];
            const bool mult=recipe.op=='*'||recipe.op=='/';
            if(i>=pool.previous_nodes) {
                const Value x=pool.nums[i];
                int weight=6+4*mult+3*std::min(5,check->uses[i]);
                if(style!=2 && style!=7)weight+=4*(x>=128 && x<100000000ULL);
                else weight+=6*(x>=100000000ULL);
                useful+=weight;
                ++n;
            } else if(mult&&k) {
                useful+=1+std::min(4,check->uses[i]);
            }
        }
        const int diversity=(int)(scramble(tag^scramble(family)^seed)% (style==3?36:14));
        return useful*2+std::min(5,n)+roots*4+diversity;
    }

    void observe(const Edits& edits,int cost,int roots,Value family,bool fused=false){
        if(cost>max_cost+1||edits.empty())return;
        const Value tag=edit_signature(edits);
        if(registered.count(tag))return;
        const int q=potential(edits,roots,family,tag);
        auto eject=[&](auto& table,int cap,bool keep_family) {
            int worst=-1,same=0;
            for(int i=0;i<(int)table.size();++i) {
                if(table[i].factor==family)++same;
                if(worst<0||appeal(table[worst])>appeal(table[i]))worst=i;
            }
            if(keep_family&&same>=std::max(3,count)){
                worst=-1;
                for(int i=0;i<(int)table.size();++i)
                    if(table[i].factor==family&&(worst<0||appeal(table[worst])>appeal(table[i])))worst=i;
            } else if((int)table.size()<cap)worst=-1;
            EditChoice proposal{edits,cost,roots,q,family,tag,fused};
            if(worst>=0) {
                if(appeal(table[worst])>=appeal(proposal))return false;
                table[worst]=std::move(proposal);
            } else table.emplace_back(std::move(proposal));
            return true;
        };
        if(cost<=max_cost && eject(nominees,std::max(22,count*9),true))registered.insert(tag);
        if(style>=10&&cost==max_cost+1&&eject(escape,std::max(8,count*2),false))registered.insert(tag);
        if(style>0 && !fused && (cost>=max_cost-1) &&
            (roots>=2||edits.size()>=2) && eject(bridge,64,false))registered.insert(tag);
    }

    static int appeal(const EditChoice& c){return -c.cost*132+c.quality;}

    int promising_links(const EditChoice& edit,const lanes::Atlas& atlas,
                        const std::vector<int>& demand) {
        int gain=0;
        for(auto [id,recipe]:edit.edits) {
            if(id<pool.previous_nodes||!check->mark[id])continue;
            const Value h=pool.nums[id];
            for(int r:demand) {
                if(atlas.private_size[r]<3||h==pool.nums[r])continue;
                const Value t=pool.nums[r];
                int chance=0;
                inverse(t,h,limit,[&](Value peer,char op,bool) {
                    if(!peer||peer==t||peer==h||!atlas.known(peer))return false;
                    chance=std::max(chance,atlas.ready(peer,r)?3:1);
                    return chance>=3;
                });
                if(h>1&&h<t/2) {
                    const Value q=t/h,x=t%h;
                    if(atlas.known(q)&&(!x||atlas.known(x)))chance=std::max(chance,x?1:3);
                    if(x && q+1<=limit/h && atlas.known(q+1)&&atlas.known(h-x))
                        chance=std::max(chance,1);
                }
                // A recipe that was already possible without the new
                // value is not credited as a useful second-generation link.
                gain+=chance;
            }
        }
        return std::min(120,gain*3);
    }

    void combine_lanes() {
        if(style==0||bridge.size()<2)return;
        // Two individually neutral edits around different factors can
        // release the same old shared chain when made together.
        std::stable_sort(bridge.begin(),bridge.end(),[](auto& a,auto& b){return appeal(a)>appeal(b);});
        if(bridge.size()>48)bridge.resize(48);
        for(int i=0;i<(int)bridge.size();++i)for(int j=i+1;j<(int)bridge.size();++j){
            const auto& a=bridge[i];const auto& b=bridge[j];
            if(a.factor==b.factor)continue;
            bool complementary=false;
            for(auto [id,recipe]:a.edits)for(auto [ib,kb]:b.edits) {
                if(id==ib && (id>=pool.previous_nodes||recipe==kb))complementary=true;
                if(id<pool.previous_nodes&&ib<pool.previous_nodes&&id!=ib &&
                    !pool.choices[id].empty()&&!pool.choices[ib].empty()){
                    auto x=pool.choices[id][0],y=pool.choices[ib][0];
                    if(x.a==y.a||x.a==y.b||x.b==y.a||x.b==y.b)complementary=true;
                }
            }
                    if(!complementary && style!=3 && style!=5 && style!=7)continue;
            for(int alt=0;alt<(style>=2?3:1);++alt) {
                Edits joined=a.edits;
                if(!combine(joined,b.edits,alt)||joined.size()<=a.edits.size()||joined.size()<=b.edits.size())continue;
                const int score=eval(joined);
                observe(joined,score,a.roots+b.roots,scramble(a.factor^scramble(b.factor)),true);
            }
        }
    }

#ifdef FRONTIER_COALITION
    // Retire all direct consumers of one old scaffold value as a bundle.
    // An individual change may temporarily cost more than it saves because
    // another consumer still keeps that scaffold alive. Price every union
    // exactly; the coverage mask is a search priority, not a waiver of cost.
    void join_shared(const lanes::Atlas& atlas){
        if(atomic.size()<2)return;
        struct Concern {int pivot;std::vector<int> users;int weight;};
        std::vector<std::vector<int>> users(pool.previous_nodes);
        std::unordered_set<int> scoring(pool.targets.begin(),pool.targets.end());
        for(int v=0;v<pool.previous_nodes;++v)if(!pool.choices[v].empty()){
            auto r=pool.choices[v][0];users[r.a].push_back(v);if(r.b!=r.a)users[r.b].push_back(v);
        }
        std::vector<Concern> watches;
        for(int v=0;v<pool.previous_nodes;++v){
            if(users[v].size()<2||users[v].size()>10||pool.choices[v].empty()||scoring.count(v))continue;
            int scorers=0;
            for(int g:pool.targets)scorers+=!!atlas.parent[g][v];
            if(scorers<2)continue;
            int weight=9*std::min(7,atlas.private_size[v])+3*std::min(12,scorers)
                       +int(scramble(pool.nums[v]^seed)%13);
            watches.push_back({v,std::move(users[v]),weight});
        }
        std::sort(watches.begin(),watches.end(),[](auto& a,auto& b){return a.weight>b.weight;});
        if(watches.size()>16)watches.resize(16);
        if(watches.empty())return;

        auto impact=[&](const Concern& c){
            unsigned mask=0;
            for(int i=0;i<(int)c.users.size();++i){
                int v=c.users[i];
                auto z=pool.choices[v][active[v]];
                if(!check->mark[v]||(z.a!=c.pivot&&z.b!=c.pivot))mask|=1u<<i;
            }
            return mask;
        };
        struct Support {int id;unsigned mask;int grade;};
        std::vector<std::vector<Support>> support(watches.size());
        for(int i=0;i<(int)atomic.size();++i){
            auto& atom=atomic[i];
            auto c=eval(atom.edits);
            if(c>max_cost+3)continue;
            for(int j=0;j<(int)watches.size();++j){
                const auto changed=impact(watches[j]);
                if(changed)support[j].push_back({i,changed,92*(c-max_cost)-11*bits(changed)
                          +int(scramble(atom.tag^seed)%17)});
            }
        }
        int all_visits=0;
        for(int j=0;j<(int)watches.size()&&all_visits<1100;++j){
            auto& group=support[j];
            const auto& c=watches[j];
            if(group.size()<2)continue;
            std::sort(group.begin(),group.end(),[](auto a,auto b){return a.grade<b.grade;});
            const unsigned done=(1u<<c.users.size())-1;
            std::vector<std::vector<Support>> choices(c.users.size());
            for(auto s:group)for(int i=0;i<(int)c.users.size();++i)if((s.mask&(1u<<i))&&choices[i].size()<10)
                choices[i].push_back(s);
            if(std::any_of(choices.begin(),choices.end(),[](auto& x){return x.empty();}))continue;
            struct Plan {Edits edits;unsigned coverage;int bill,parts;Value tag;};
            std::vector<Plan> open{{{},0u,max_cost,0,0}};
            int visits=0;
            for(int step=0;step<6&&!open.empty()&&visits<160&&all_visits<1100;++step){
                std::vector<Plan> next;
                std::unordered_set<Value> attempted;
                for(const auto& prior:open){
                    int first=-1;
                    for(int k=0;k<(int)c.users.size();++k)if(!(prior.coverage&(1u<<k))&&
                       (first<0||choices[k].size()<choices[first].size()))first=k;
                    if(first<0)continue;
                    for(auto selected:choices[first]){
                        if(visits>=160||all_visits>=1100)break;
                        Edits joined=prior.edits;
                        if(!combine(joined,atomic[selected.id].edits,0)||joined.size()==prior.edits.size())continue;
                        Value tag=edit_signature(joined);
                        if(!attempted.insert(tag).second)continue;
                        ++visits;++all_visits;
                        const int bill=eval(joined);
                        if(bill>max_cost+4)continue;
                        const auto mask=impact(c);
                        if(bits(mask)<=bits(prior.coverage))continue;
                        if(mask==done){
                            if(prior.parts>=1&&bill<=max_cost+1&&!check->mark[c.pivot]){
                                ++complete_coalitions;
                                observe(joined,bill,prior.parts+1,pool.nums[c.pivot],true);
                            }
                        }else next.push_back({std::move(joined),mask,bill,prior.parts+1,tag});
                    }
                }
                std::sort(next.begin(),next.end(),[](const Plan& a,const Plan& b){
                    return std::make_tuple(100*a.bill-91*bits(a.coverage),a.parts,a.tag)
                         <std::make_tuple(100*b.bill-91*bits(b.coverage),b.parts,b.tag);
                });
                if(next.size()>14)next.resize(14);
                open=std::move(next);
            }
        }
    }
#endif

    std::vector<SearchNode> run(const std::set<Key>& expanded,
                                int factors,int goals,int parent,DivisorCache& cache){
        lanes::Atlas atlas(pool);
        auto [fac,demand]=lanes::choose(atlas,factors,goals,depth);
        auto generated=lanes::generate(atlas,fac,demand,style==3?5:4,depth,
#ifdef FRONTIER_FAST_RESIDUES
                                       true
#else
                                       false
#endif
        );
        if(style>=6)translated_products(atlas,fac,demand,generated,style>=7?7:6);
        if(style>=8)latent_factors(atlas,demand,generated,cache,style,seed);
        int nproofs=0;
        auto laneset=lanes::register_choices(pool,generated,nproofs);
        generated.clear();
        base.assign(pool.nums.size(),0);
        Evaluator context(pool);
        check=&context;
        for(auto& family:laneset){
            std::vector<std::pair<int,int>> order;
            for(const auto& [root,variants]:family.groups) {
                int best=100000;
                for(const auto& pick:variants) {
                    int price=eval(pick.edits);
                    best=std::min(best,price);
                    observe(pick.edits,price,1,family.factor);
#ifdef FRONTIER_COALITION
                    if(synchronize&&price<=max_cost+3&&atomic.size()<2200){
                        const auto tag=edit_signature(pick.edits);
                        if(recorded.insert(tag).second)atomic.push_back({pick.edits,price,1,0,family.factor,tag,false});
                    }
#endif
                }
                if(best<=max_cost+3)order.push_back({best,root});
            }
            if(order.size()<2||family_width==0)continue;
            std::sort(order.begin(),order.end());
            if(order.size()>16)order.resize(16);
            std::vector<EditChoice> accum;
            for(auto [_,root]:order) {
                auto next=accum;
                for(const auto& p:family.groups.at(root)){
                    const int bill=eval(p.edits);
                    if(bill>max_cost+3)continue;
                    next.push_back({p.edits,bill,1,0,family.factor,edit_signature(p.edits),false});
                    for(auto& have:accum) {
                        if(have.roots>=6)continue;
                        Edits merged=have.edits;
                        if(!combine(merged,p.edits,0)||merged.size()==have.edits.size())continue;
                        const int cost=eval(merged);
                        if(cost>max_cost+3)continue;
                        if(cost<=max_cost+1)observe(merged,cost,have.roots+1,family.factor);
                        next.push_back({std::move(merged),cost,have.roots+1,0,family.factor,0,false});
                        next.back().tag=edit_signature(next.back().edits);
                    }
                }
                std::sort(next.begin(),next.end(),[](auto& a,auto& b){
                    return std::make_tuple(a.cost*6-a.roots,a.edits.size(),a.tag)
                          <std::make_tuple(b.cost*6-b.roots,b.edits.size(),b.tag);
                });
                accum.clear();
                std::unordered_set<Value> unique;
                for(auto& option:next)if(unique.insert(option.tag).second){
                    accum.push_back(std::move(option));
                    if((int)accum.size()>=family_width)break;
                }
            }
        }
        combine_lanes();
#ifdef FRONTIER_COALITION
        if(synchronize)join_shared(atlas);
#endif
        if(style>=4)for(auto& option:nominees) {
            if(eval(option.edits)>max_cost)continue;
            option.quality+=promising_links(option,atlas,demand);
        }
        if(style>=10){
            for(auto& option:escape){
                if(eval(option.edits)!=max_cost+1)continue;
                option.quality+=promising_links(option,atlas,demand);
            }
            std::sort(escape.begin(),escape.end(),[](auto& a,auto& b){
                return std::make_pair(-appeal(a),a.tag)<std::make_pair(-appeal(b),b.tag);
            });
        }
        std::sort(nominees.begin(),nominees.end(),[](auto& a,auto& b){
            return std::make_pair(-appeal(a),a.tag)<std::make_pair(-appeal(b),b.tag);
        });
        std::set<Key> unique;
        std::map<Value,int> covers;
        std::vector<SearchNode> result;
        for(auto& edit:nominees){
            if((int)result.size()>=count)break;
            if(count>1&&covers[edit.factor]>=std::max(1,count/3))continue;
            const int bill=eval(edit.edits);
            if(bill>max_cost)continue;
            auto graph=check->freeze(active);
            auto sig=fingerprint(graph);
            if(expanded.count(key(sig))||!unique.insert(key(sig)).second)continue;
            auto tape=physical(graph,limit,max_cost+1);
            if(tape.empty()||tape.size()>node.code.size())continue;
            if(bill==max_cost&&tape.size()!=node.code.size())continue;
            ++covers[edit.factor];
            result.push_back({std::move(tape),sig,edit.factor,edit.quality,edit.roots,parent,edit.joined,
                              style==13?numeric_nucleus(graph):std::vector<Value>{}});
        }
        if(style>=10){
            int retained=0;
            for(auto& edit:escape){
                if(retained>=std::max(1,count/3))break;
                if(eval(edit.edits)!=max_cost+1)continue;
                auto graph=check->freeze(active);
                auto sig=fingerprint(graph);
                if(expanded.count(key(sig))||!unique.insert(key(sig)).second)continue;
                auto tape=physical(graph,limit,max_cost+2);
                if(tape.empty()||tape.size()>node.code.size()+1)continue;
                ++retained;
                result.push_back({std::move(tape),sig,edit.factor,edit.quality,edit.roots,parent,edit.joined,
                                  style==13?numeric_nucleus(graph):std::vector<Value>{}});
            }
        }
        check=nullptr;
        return result;
    }
};

// Splice exact backward slices from different surviving branches while
// testing all unchanged targets against the same base program.
std::vector<SearchNode> graft(const Registers& regs,const std::vector<Value>& t,Value limit,
                              const SearchNode& home,const std::vector<SearchNode>& family,
                              const std::set<Key>& expanded,int parent_bound,Value salt,int style) {
    if(family.size()<2)return {};
    PlanPool pool(regs,t,limit,home.code);
    struct Graft {int donor,root;Edits edits;std::unordered_set<int> shared;int price;};
    std::vector<Graft> poollet;
    for(int k=0;k<(int)family.size();++k) {
        const auto& source=family[k];
        if(key(source.sig)==key(home.sig))continue;
        auto graph=topological_graph(recover_graph(regs,limit,t,source.code));
        std::vector<int> bridge;
        std::vector<int> recipe_number;
        for(auto s:graph.nodes) {
            const int id=pool.intern(s.value);
            bridge.push_back(id);
            int option=0;
            if(s.op){
                int a=bridge[s.a],b=bridge[s.b];
                if((s.op=='*'||s.op=='+')&&a>b)std::swap(a,b);
                pool.add(id,s.op,a,b,90);
                const auto& choices=pool.choices[id];
                auto existing=std::find_if(choices.begin(),choices.end(),[&](auto e){
                    if(e.op!=s.op)return false;
                    if(e.a==a&&e.b==b)return true;
                    return (e.op=='+'||e.op=='*')&&e.a==b&&e.b==a;
                });
                if(existing==choices.end())throw std::logic_error("Unavailable donor recipe");
                option=(int)(existing-choices.begin());
            }
            recipe_number.push_back(option);
        }
        for(int r:graph.targets) {
            for(int deep=0;deep<2;++deep){
                Edits edits;
                std::vector<unsigned char> visited(graph.nodes.size());
                auto visit=[&](auto& self,int j)->void{
                    if(visited[j])return;
                    visited[j]=1;
                    const auto& value=graph.nodes[j];if(!value.op)return;
                    int id=bridge[j],ix=recipe_number[j];
                    const bool differs=id>=pool.previous_nodes||ix!=0;
                    if(!differs&&!deep)return;
                    self(self,value.a);self(self,value.b);
                    if(differs)edits.emplace_back(id,ix);
                };
                visit(visit,r);
                if(edits.empty()||edits.size()>36)continue;
                std::unordered_set<int> shared;
                for(auto [i,j]:edits){
                    const auto used=pool.choices[i][j];
                    if(used.op=='*'||used.op=='/'){
                        shared.insert(i);shared.insert(used.a);shared.insert(used.b);
                    }
                    if(i<pool.previous_nodes&&!pool.choices[i].empty()){
                        auto old=pool.choices[i][0];
                        shared.insert(old.a);shared.insert(old.b);
                    }
                }
                poollet.push_back({k,bridge[r],std::move(edits),std::move(shared),0});
            }
        }
    }
    std::vector<SearchNode> out;
    if(poollet.empty())return out;
    Evaluator evaluator(pool);
    State state(pool.nums.size());
    std::set<Key> accepted;
    std::unordered_set<Value> seen;
    auto worth=[&](const Edits& e){
        std::fill(state.begin(),state.end(),0);
        for(auto [i,j]:e)state[i]=j;
        return evaluator.eval(state);
    };
    auto test=[&](const Edits& e,int roots,int donor,int bill) {
        if(bill>parent_bound || bill<0 || !seen.insert(edit_signature(e)).second)return;
        auto g=evaluator.freeze(state);
        const auto sig=fingerprint(g);
        if(expanded.count(key(sig))||!accepted.insert(key(sig)).second)return;
        auto tape=physical(g,limit,parent_bound+1);
        if(tape.empty())return;
        Value product=0;
        int novelty=0;
        for(auto [v,r]:e)if(evaluator.mark[v]&&pool.choices[v][r].op=='*') {
            product^=scramble(pool.nums[v]);
            novelty+=3+evaluator.uses[v];
        }
        int q=std::min(124,novelty*3+(int)(scramble(salt^product)%31));
        out.push_back({std::move(tape),sig,product,q,roots,donor,true,
                       style==13?numeric_nucleus(g):std::vector<Value>{}});
    };
    for(auto& patch:poollet) {
        patch.price=worth(patch.edits);
        if(patch.price<parent_bound)test(patch.edits,1,patch.donor,patch.price);
    }
    std::stable_sort(poollet.begin(),poollet.end(),[](auto& a,auto& b){
        return std::make_pair(a.price,a.edits.size())<std::make_pair(b.price,b.edits.size());
    });
    while(!poollet.empty()&&poollet.back().price>parent_bound+2)poollet.pop_back();
    if(poollet.size()>82)poollet.resize(82);
    int count=0;
    for(int i=0;i<(int)poollet.size();++i)for(int j=i+1;j<(int)poollet.size();++j){
        auto& a=poollet[i];auto& b=poollet[j];
        if(a.donor==b.donor||a.root==b.root)continue;
        bool related=false;
        for(int x:a.shared)if(b.shared.count(x)){related=true;break;}
        if(!related)continue;
        if(++count>1000)break;
        Edits together=a.edits;
        if(!combine(together,b.edits,(style==5||style==7)?1:0)||together.size()<=std::max(a.edits.size(),b.edits.size()))continue;
        int bill=worth(together);
        if(bill>parent_bound|| (bill==parent_bound&&out.size()>=6))continue;
        test(together,2,a.donor,bill);
    }
    return out;
}

std::vector<SearchNode> diverse(std::vector<SearchNode>& proposals,int width,int best,int style){
    std::vector<SearchNode> result;
    std::set<Key> signatures;
    std::map<int,int> family;
    while(!proposals.empty()&&(int)result.size()<width) {
        int top=-1,score=-100000000;
        const bool allow_longer=style>=10&&width>=3;
        const int slack=style>=11?2:1;
        const int quota=allow_longer?std::max(1,width/4):0;
        const int uphill=(int)std::count_if(result.begin(),result.end(),[&](auto& v){return (int)v.code.size()>best;});
        const bool spend=allow_longer&&uphill<quota&&(int)result.size()>=width-quota;
        for(int i=0;i<(int)proposals.size();++i){
            auto& q=proposals[i];
            int excess=(int)q.code.size()-best;
            if(q.code.empty()||signatures.count(key(q.sig))||excess>(width>1?slack:0))continue;
            if(allow_longer&&excess>0&&uphill>=quota)continue;
            int distance=64,reused=0,numeric=40;
            for(const auto& x:result) {
                distance=std::min(distance,bits(x.sig.shape^q.sig.shape));
                if(style==13)numeric=std::min(numeric,symmetric_distance(x.nucleus,q.nucleus));
                reused+=x.last_factor==q.last_factor;
            }
            const int variety=style==13?distance+4*numeric:distance*(style==0?1:3);
            int value=-140*(int)q.code.size()+q.quality/2+variety
                       -family[q.parent]*13-reused*17;
            if(spend&&excess>0)value+=140*excess+95;
            if(value>score)score=value,top=i;
        }
        if(top<0)break;
        auto& q=proposals[top];
        signatures.insert(key(q.sig));
        ++family[q.parent];
        result.push_back(std::move(q));
        q.code.clear();
    }
    return result;
}

std::pair<Program,Stats> solve(const Registers& regs,const std::vector<Value>& t,Value limit,Program old,
                               int width,int budget,int packet,int depth,int style,Value salt,
                               const std::function<bool()>& finished={}){
    std::set<Key> expanded;
#ifdef FRONTIER_COALITION
    const bool coalition=style==12;
    if(coalition)style=9;
#endif
    DivisorCache divisors;
    auto seed=topological_graph(recover_graph(regs,limit,t,old));
    std::vector<SearchNode> level{{old,fingerprint(seed),0,0,0,-1,false,
                                   style==13?numeric_nucleus(seed):std::vector<Value>{}}};
    Stats stats;
    Program best=old;
    int epoch=0;
    while(!level.empty()&&stats.expanded<budget&&(!finished||!finished())){
        stats.peak=std::max(stats.peak,(int)level.size());
        std::vector<SearchNode> next;
        if(style>=4&&epoch%2==1&&level.size()>1){
            auto shorter=std::min_element(level.begin(),level.end(),[](auto& x,auto& y) {
                return x.code.size()<y.code.size();
            });
            auto fused=graft(regs,t,limit,*shorter,level,expanded,(int)best.size(),salt^epoch,style);
            for(auto& q:fused){
                if(finished&&finished())break;
                if(q.code.size()<best.size())best=q.code,++stats.improved;
                if(q.root_count>1)++stats.multi;
                ++stats.fused;
                next.push_back(std::move(q));
            }
        }
        for(int i=0;i<(int)level.size()&&stats.expanded<budget;++i) {
            if(finished&&finished())break;
            auto& home=level[i];
            if(!expanded.insert(key(home.sig)).second)continue;
            ++stats.expanded;
            const bool rising=home.code.size()>best.size();
            stats.uphill+=rising;
            const int count=width==1?1:std::min(12,3+width*2);
            Expansion trial(regs,t,home,limit,packet,depth,style,count,epoch,
                            scramble(salt^home.sig.second^(epoch*97+13)));
#ifdef FRONTIER_COALITION
            trial.synchronize=coalition;
#endif
            auto children=trial.run(expanded,depth>=9?96:82,depth>=9?145:128,i,divisors);
#ifdef FRONTIER_COALITION
            stats.coalitions+=trial.complete_coalitions;
#endif
            for(auto& q:children) {
                if(finished&&finished())break;
                if(style>=10 && q.code.size()>best.size()+(style>=11?2u:1u))continue;
                if(q.code.size()==home.code.size())++stats.equal;
                if(q.root_count>1)++stats.multi;
                if(q.joined)++stats.fused;
                if(q.code.size()<best.size())best=q.code,++stats.improved,stats.escaped+=rising;
                next.push_back(std::move(q));
            }
        }
        level=diverse(next,width,(int)best.size(),style);
        ++epoch;
    }
    return {std::move(best),stats};
}
} // namespace frontier

#ifndef ELF_FRONTIER_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=7&&argc!=8)throw std::invalid_argument("frontier nodes packet depth style salt [milliseconds]");
        int width=std::stoi(argv[1]),budget=std::stoi(argv[2]),inner=std::stoi(argv[3]),
            depth=std::stoi(argv[4]),style=std::stoi(argv[5]);
#ifdef FRONTIER_UPHILL_PAIR
        if(style!=0&&style!=1)throw std::invalid_argument("Paired arm must be 0 or 1");
        style=style?10:9;
#endif
        auto salt=std::stoull(argv[6]);
        const auto time=std::chrono::milliseconds(argc==8?std::stoi(argv[7]):0);
        if(width<1||budget<1||inner<0||depth<1||style<0||style>13
#ifndef FRONTIER_COALITION
            ||style==12
#endif
        )
            throw std::invalid_argument("Invalid frontier profile");
        std::string id;Value limit;int ns,nt,n;
        while(std::cin>>id>>limit>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||n<0||n>4096||!limit||limit>=(Value(1)<<62))
                throw std::invalid_argument("Invalid puzzle dimensions");
            Registers regs{};std::vector<Value> t(nt);Program old(n);
            for(int i=0;i<ns;++i)std::cin>>regs[i];
            for(auto& x:t)std::cin>>x;
            for(auto& s:old)std::cin>>s.kind>>s.a>>s.b>>s.destination;
            const auto started=std::chrono::steady_clock::now();
            std::function<bool()> finished;
            if(time.count()>0)finished=[&]{
                return std::chrono::steady_clock::now()>=started+time;
            };
            auto [result,s]=frontier::solve(regs,t,limit,std::move(old),width,budget,inner,depth,style,salt,finished);
            std::unordered_set<Value> seen(regs.begin(),regs.end());
            for(auto op:result)seen.insert(execute(regs,op,limit));
            for(Value x:t)if(!seen.count(x))throw std::logic_error("Missing final target");
            std::cout<<id<<' '<<result.size()<<' '<<s.expanded<<' '<<s.improved<<' '<<s.equal<<' '
#ifdef FRONTIER_UPHILL_PAIR
                     <<s.uphill<<' '<<s.escaped<<' '<<s.fused<<'\n';
#elif defined(FRONTIER_COALITION_STATS)
                     <<s.multi<<' '<<s.peak<<' '<<s.coalitions<<'\n';
#else
                     <<s.multi<<' '<<s.peak<<' '<<s.fused<<'\n';
#endif
            for(auto op:result)std::cout<<op.kind<<' '<<op.a<<' '<<op.b<<' '<<op.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
