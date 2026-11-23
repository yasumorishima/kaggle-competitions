// Rebuild complete dependency cuts with 4--8 actual scoring outputs.
#define ELF_BLOCK_LIBRARY
#include "elf_target_blocks.cpp"
#include <chrono>

namespace rebuild16 {
using Word=std::uint64_t;
using Clock=std::chrono::steady_clock;
using Deadline=Clock::time_point;
int pop(Word x){int n=0;for(;x;x&=x-1)++n;return n;}
Word hash(Word x){Word out=0;Blocks::U64hash(out,x);return out;}
bool over(Deadline d){return Clock::now()>=d;}

struct Region {Word mask;std::vector<int> roots;int rank,spent;};
std::vector<Region> regions(const Graph& g,int wanted,Word salt) {
    std::vector<int> goals;
    for(int v:g.targets)if(g.nodes[v].op)goals.push_back(v);
    int n=(int)goals.size();
    if(n<4 || n>63)return {};
    std::vector<Word> footprint(g.nodes.size(),0);
    for(int i=0;i<n;++i) {
        const Word bit=Word(1)<<i;
        auto visit=[&](auto& self,int v)->void{
            if(footprint[v]&bit)return;
            footprint[v]|=bit;
            auto x=g.nodes[v];
            if(x.op)self(self,x.a),self(self,x.b);
        };
        visit(visit,goals[i]);
    }
    std::map<Word,int> weights;
    for(std::size_t j=0;j<footprint.size();++j)
        if(g.nodes[j].op&&footprint[j])++weights[footprint[j]];
    int affinity[63][63]={};
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
        const auto x=g.nodes[goals[i]].value,y=g.nodes[goals[j]].value;
        int proximity=bit_count(std::max(x,y))-bit_count(x>y?x-y:y-x);
        const Word both=(Word(1)<<i)|(Word(1)<<j);
        int support=0;
        for(auto [mask,c]:weights)if((mask&both)==both&&pop(mask)<=12)
            support+=c*std::max(1,9-pop(mask));
        affinity[i][j]=affinity[j][i]=std::min(22,2*proximity+support/2);
    }
    std::vector<Word> seeds;
    for(auto [w,c]:weights)if(pop(w)>=2&&pop(w)<=8)seeds.push_back(w);
    for(int i=0;i<n;++i){
        std::vector<std::pair<int,int>> near;
        for(int j=0;j<n;++j)if(j!=i)near.push_back({-affinity[i][j]-(int)(hash(salt^i^j)%5),j});
        std::sort(near.begin(),near.end());
        for(int j=0;j<std::min(6,(int)near.size());++j)
            seeds.push_back((Word(1)<<i)|(Word(1)<<near[j].second));
    }
    std::sort(seeds.begin(),seeds.end());seeds.erase(std::unique(seeds.begin(),seeds.end()),seeds.end());
    std::vector<Region> picks;
    std::unordered_set<Word> visited;
    for(Word m:seeds)for(int size=pop(m);size<=8;++size) {
        if(size>=4&&visited.insert(m).second) {
            std::vector<int> roots;
            int shared=0,near=0;
            for(int i=0;i<n;++i)if(m&(Word(1)<<i)) {
                roots.push_back(goals[i]);
                for(int j=0;j<i;++j)if(m&(Word(1)<<j))near+=affinity[i][j];
            }
            const auto cut=make_cut(g,roots);
            for(auto [w,c]:weights)if(pop(w)>1&&!(w&~m))shared+=c;
            if(cut.budget>=size+2&&cut.budget<=51) {
                // Prefer a genuine common branch, without filling all
                // attempts with almost identical expensive large cuts.
                int rank=10*(cut.budget-size)+14*shared+near/std::max(2,size)
                    -6*std::max(0,cut.budget-25)-2*(size-4)+(int)(hash(m^salt)%14);
                picks.push_back({m,std::move(roots),rank,cut.budget});
            }
        }
        int best=-100000,next=-1;
        for(int i=0;i<n;++i)if(!(m&(Word(1)<<i))) {
            const Word bit=Word(1)<<i, change=m|bit;
            int gain=0,near=0;
            for(auto [w,c]:weights)if((w&bit)&&!(w&~change))gain+=c*(pop(w)>1?7:3);
            for(int j=0;j<n;++j)if(m&(Word(1)<<j))near+=affinity[i][j];
            int score=gain+near/std::max(1,size)+(int)(hash(change^salt)%17);
            if(score>best)best=score,next=i;
        }
        if(next<0)break;
        m|=Word(1)<<next;
    }
    std::vector<Region> ranked;
    int usage[63]={}, by_size[9]={};
    for(int j=0;j<std::min(wanted,(int)picks.size());++j) {
        auto merit=[&](const Region& x){
            int penalty=0;
            for(int i=0;i<n;++i)if(x.mask&(Word(1)<<i))penalty+=usage[i];
            int collision=0;
            for(auto& y:ranked)collision=std::max(collision,pop(x.mask&y.mask)*16/pop(x.mask|y.mask));
            return x.rank-7*penalty-collision*3-by_size[x.roots.size()]*15;
        };
        auto at=std::max_element(picks.begin()+j,picks.end(),[&](auto& a,auto& b){return merit(a)<merit(b);});
        std::iter_swap(picks.begin()+j,at);
        for(int i=0;i<n;++i)if(picks[j].mask&(Word(1)<<i))++usage[i];
        ++by_size[picks[j].roots.size()];
        ranked.push_back(std::move(picks[j]));
    }
    return ranked;
}

struct Recipe {char op;int a,b;};
struct Library {
    Value limit;Deadline end;
    Blocks atlas;
    std::unordered_set<Value> free;
    std::vector<Value> values;
    Index ids;
    std::vector<std::vector<Recipe>> edges;
    std::vector<int> need,depth;
    std::vector<int> reference;
    Library(const Graph& g,const Cut& cut,Value mx,Deadline e,int leaves,int policy)
        :limit(mx),end(e),atlas(mx,outputs(g,cut),
          terminal_values(g,cut,leaves,policy),5,6,cut.budget) {
        for(int i=0;i<(int)g.nodes.size();++i)if(cut.terminal[i]) {
            auto v=g.nodes[i].value;free.insert(v);intern(v);
        }
        for(int i:cut.roots)need.push_back(intern(g.nodes[i].value));
        std::unordered_set<int> memo;
        auto retain=[&](auto& self,int i)->void {
            auto x=g.nodes[i];
            if(free.count(x.value)||!x.op||!memo.insert(i).second)return;
            self(self,x.a);self(self,x.b);
            add({x.value,x.op,g.nodes[x.a].value,g.nodes[x.b].value});
            reference.push_back(intern(x.value));
        };
        for(int i:cut.roots)retain(retain,i);
    }
    static std::vector<Value> outputs(const Graph& g,const Cut& c) {
        std::vector<Value> out;for(int i:c.roots)out.push_back(g.nodes[i].value);return out;
    }
    int intern(Value x) {
        auto [it,put]=ids.emplace(x,(int)values.size());
        if(put){values.push_back(x);edges.emplace_back();}
        return it->second;
    }
    void add(Step s) {
        if(!s.x||s.x==s.a||s.x==s.b||free.count(s.x)||calculate(s.op,s.a,s.b,limit)!=s.x)return;
        int x=intern(s.x),a=intern(s.a),b=intern(s.b);
        if((s.op=='+'||s.op=='*')&&a>b)std::swap(a,b);
        auto& e=edges[x];
        if(e.size()>=22)return;
        if(std::none_of(e.begin(),e.end(),[&](auto u){return u.op==s.op&&u.a==a&&u.b==b;}))
            e.push_back({s.op,a,b});
    }
    void add(const Script& p){for(auto s:p)add(s);}
    bool find(Value v,Script& out,std::unordered_set<Value>& known) {
        if(known.count(v))return true;
        int ix=atlas.d.index.get((SInt)v);
        if(ix<0)return false;
        atlas.fetch(ix,out,known);
        return known.count(v)!=0;
    }
    void beam(Value target,int width) {
        if(over(end))return;
        local_beam::Search s(atlas.d,width);
        s.run((SInt)target);
        if(s.end<0)return;
        Script out;auto seen=free;Value v=(Value)atlas.d.e[s.terminal].v;
        if(!find(v,out,seen))return;
        for(int k=s.end;s.history[k].parent>=0;k=s.history[k].parent) {
            const auto [factor,atom,op]=s.history[k].link;
            if(factor>=0) {
                Value f=(Value)atlas.d.e[factor].v;
                if(!find(f,out,seen))return;
                auto x=calculate('*',v,f,limit);if(!x)return;
                atlas.add({x,'*',v,f},out,seen);v=x;
            }
            if(atom>=0) {
                Value r=(Value)atlas.d.e[atom].v;
                if(!find(r,out,seen))return;
                Value x=op=='r'?calculate('-',r,v,limit):calculate(op,v,r,limit);if(!x)return;
                if(op=='r')atlas.add({x,'-',r,v},out,seen);else atlas.add({x,op,v,r},out,seen);
                v=x;
            }
        }
        if(v==target&&out.size()<=14)add(out);
    }
    void complements(Value t,bool root,int policy) {
        struct Pick {int cost;Value a,b;char op;};
        std::vector<Pick> avail;
        auto cost=[&](Value v) {
            if(free.count(v))return 0;
            int j=atlas.d.index.get((SInt)v);
            return j<0?20:atlas.d.e[j].cost;
        };
        const auto offer=[&](Value a,Value b,char op) {
            if(!a||!b||a==t||b==t)return;
            if((op=='+'||op=='*')&&a>b)std::swap(a,b);
            const int c=1+cost(a)+(b==a?0:cost(b));
            if(c>(root?5:3))return;
            avail.push_back({c,a,b,op});
        };
        const int last=(root&&policy>=1?(int)atlas.d.e.size():atlas.d.n1);
        const int stride=policy>=3?std::max(1,(last-atlas.d.n1)/8300):1;
        for(int i=0;i<last;i+=i>=atlas.d.n1?stride:1) {
            if((i&4095)==0&&over(end))break;
            Value b=atlas.d.e[i].v;
            inverse(t,b,limit,[&](Value a,char op,bool first) {
                if(i>=atlas.d.n1 && cost(a)>=2 && a<b)return false;
                offer(first?a:b,first?b:a,op);return false;
            });
        }
        if(avail.empty())return;
        auto rating=[&](const Pick& x) {
            int simple=(x.op=='*'?0:1);
            return std::make_tuple(x.cost,simple,hash(x.a^hash(x.b)^Value(policy*37)),x.a,x.b);
        };
        std::sort(avail.begin(),avail.end(),[&](auto& a,auto& b){return rating(a)<rating(b);});
        int kept=0;
        std::set<std::tuple<Value,Value,char>> seen;
        for(auto z:avail) {
            if(kept>=(root?13:5)||z.cost>avail[0].cost+2)break;
            if(!seen.emplace(z.a,z.b,z.op).second)continue;
            Script code;auto ready=free;
            if(!find(z.a,code,ready)||!find(z.b,code,ready))continue;
            atlas.add({t,z.op,z.a,z.b},code,ready);
            add(code);++kept;
        }
    }
    void weave(int policy) {
        // The old interior is NOT free: it can only serve as a dependency
        // if its new block pays for it. Re-use it across root boundaries
        // rather than insisting that each target have a seed-only recipe.
        auto points=reference;
        std::sort(points.begin(),points.end());
        points.erase(std::unique(points.begin(),points.end()),points.end());
        const auto all_numeric=values;
        std::vector<Value> factors;
        for(Value x:all_numeric)if(x>=2&&x<10000000000ULL)factors.push_back(x);
        std::sort(factors.begin(),factors.end());
        factors.erase(std::unique(factors.begin(),factors.end()),factors.end());
        for(auto i:points){
            if(over(end))return;
            const Value t=values[i];
            struct Proposal { Value a,b;char op;int cost; };
            std::vector<Proposal> proposals;
            auto cost=[&](Value x){
                if(free.count(x))return 0;
                if(ids.count(x))return 2;
                const auto j=atlas.d.index.get((SInt)x);
                return j>=0?1+atlas.d.e[j].cost:99;
            };
            for(int j:points)if(i!=j) {
                const Value peer=values[j];
                inverse(t,peer,limit,[&](Value v,char op,bool first){
                    if(v==t||!v)return false;
                    const int c=cost(v);
                    if(c<=3)proposals.push_back({first?v:peer,first?peer:v,op,c});
                    return false;
                });
            }
            std::stable_sort(proposals.begin(),proposals.end(),[](auto a,auto b){return a.cost<b.cost;});
            int tried=0;
            for(auto z:proposals){
                if(tried++>=12)break;
                Script code;auto avail=free;
                if((!ids.count(z.a)&&!find(z.a,code,avail))||
                   (!ids.count(z.b)&&!find(z.b,code,avail)))continue;
                add(code);add({t,z.op,z.a,z.b});
            }
            if(policy<4 || std::find(need.begin(),need.end(),i)==need.end())continue;
            struct Affine {Value factor,q,res;char sign;int cost;};
            std::vector<Affine> cases;
            for(Value f:factors) {
                if(f>=t/2)break;
                const Value q=t/f,r=t%f;
                auto support=[&](Value x,Value res,char sign){
                    if(x<=1||x>limit/f||x==t||res==t)return;
                    int c=cost(f)+cost(x)+(res?cost(res)+1:0);
                    if(c<7)cases.push_back({f,x,res,sign,c});
                };
                support(q,r,'+');
                if(r)support(q+1,f-r,'-');
            }
            std::sort(cases.begin(),cases.end(),[](auto a,auto b){
                return std::tie(a.cost,a.factor,a.q)<std::tie(b.cost,b.factor,b.q);
            });
            tried=0;
            for(auto z:cases) {
                if(tried++>=18)break;
                if(z.res>=limit||z.res==z.factor*z.q)continue;
                Script code;auto avail=free;
                bool ok=true;
                for(Value v:{z.factor,z.q,z.res})
                    if(v && !ids.count(v)&&!find(v,code,avail))ok=false;
                if(!ok)continue;
                add(code);
                Value core=z.factor*z.q;
                add({core,'*',z.factor,z.q});
                if(z.res)add({t,z.sign,core,z.res});
            }
        }
    }
    void bridges(int policy){
        // Demands from several outputs vote for the same new hub. In a
        // proposed recipe the hub is not assumed ready: it costs whatever
        // its witness costs exactly once for the whole block.
        struct Option {int root;Value atom,hub;char op;bool first;int cost;};
        struct Share {unsigned mask=0;std::vector<Option> use;};
        std::unordered_map<Value,Share> joint;
        for(int k=0;k<(int)need.size();++k) {
            const auto t=values[need[k]];
            for(int j=0;j<std::min(atlas.d.n1,atlas.d.n0+80);++j) {
                const Value a=(Value)atlas.d.e[j].v;
                inverse(t,a,limit,[&](Value b,char op,bool first) {
                    if(!b||b==t||free.count(b))return false;
                    const int ix=atlas.d.index.get((SInt)b);
                    if(ix<0&&!ids.count(b)&&policy<2)return false;
                    auto& entry=joint[b];
                    if((entry.mask&(1u<<k))==0||entry.use.size()<need.size()+2) {
                        entry.mask|=1u<<k;
                        entry.use.push_back({need[k],a,b,op,first,atlas.d.e[j].cost});
                    }
                    return false;
                });
            }
        }
        std::vector<std::tuple<int,int,Value>> ranked;
        for(auto& [v,entry]:joint) {
            if(pop(entry.mask)<2)continue;
            const int ix=atlas.d.index.get((SInt)v);
            if(ix<0&&!ids.count(v))continue;
            ranked.emplace_back(-pop(entry.mask),ix>=0?atlas.d.e[ix].cost:3,v);
        }
        std::sort(ranked.begin(),ranked.end());
        int count=0;
        for(auto [_,__,v]:ranked) {
            if(count++>=60||over(end))break;
            for(auto z:joint.at(v).use){
                Script code;auto ready=free;
                if(!find(z.atom,code,ready))continue;
                if(!find(z.hub,code,ready)&&!ids.count(z.hub))continue;
                add(code);
                Value t=values[z.root];
                add({t,z.op,z.first?z.hub:z.atom,z.first?z.atom:z.hub});
            }
        }
    }
    void linked(const std::vector<int>& goals) {
        for(int i:goals) {
            const auto t=values[i];
            int accepted=0;
            for(int j:goals)if(i!=j) {
                const auto a=values[j];
                inverse(t,a,limit,[&](Value b,char op,bool first) {
                    if(accepted>=12||b==t)return false;
                    Script code;auto seeds=free;
                    if(!find(b,code,seeds))return false;
                    add(code);add({t,op,first?b:a,first?a:b});++accepted;return false;
                });
            }
        }
    }
    void finalize(int policy) {
        if(policy>=3)weave(policy);
        for(int i:need) {
            if(over(end))return;
            const auto v=values[i];complements(v,true,policy);
            if(policy<=2||policy==5)beam(v,policy>=2?12:7);
        }
        for(int i:reference) {
            if(over(end))return;
            if(std::find(need.begin(),need.end(),i)==need.end())complements(values[i],false,policy);
        }
        linked(need);bridges(policy);
        depth.resize(values.size(),99);
        for(int i=0;i<(int)values.size();++i)if(free.count(values[i]))depth[i]=0;
        bool update=true;
        for(int phase=0;phase<24&&update;++phase) {
            update=false;
            for(int i=0;i<(int)values.size();++i)for(auto r:edges[i]){
                const int d=1+std::max(depth[r.a],depth[r.b]);
                if(d<depth[i])depth[i]=d,update=true;
            }
        }
    }
    Script repair(int cap,int width,int policy,int& explored,Word salt) {
        if(depth.empty()||over(end))return {};
        const int n=(int)values.size();
        std::vector<int> base(n,0),chosen(n),selected(n);
        std::vector<unsigned char> marks(n);
        std::vector<int> order;
        auto evaluate=[&](const std::vector<int>& opts) {
            order.clear();std::fill(marks.begin(),marks.end(),0);
            auto visit=[&](auto& self,int id)->bool {
                if(depth[id]==0||marks[id]==2)return true;
                if(marks[id]==1||edges[id].empty()||opts[id]>=(int)edges[id].size())return false;
                marks[id]=1;
                auto e=edges[id][opts[id]];
                if(!self(self,e.a)||!self(self,e.b))return false;
                order.push_back(id);marks[id]=2;return true;
            };
            for(int root:need)if(!visit(visit,root))return 10000;
            return (int)order.size();
        };
        auto materialize=[&](auto& s) {
            Script out;
            for(int i:order){auto z=edges[i][s[i]];out.push_back({values[i],z.op,values[z.a],values[z.b]});}
            return out;
        };
        int home=evaluate(base);
        if(home==10000)return {};
        int best=home;
        chosen=base;selected=base;
        Script proof;
        if(home<=cap)proof=materialize(base);
        std::mt19937 rng((unsigned)(limit^salt^values[need.front()]));
        std::vector<int> old_order=order;
        std::vector<std::vector<std::pair<int,int>>> alternatives(n);
        for(int v=0;v<n;++v)for(int k=0;k<(int)edges[v].size();++k){
            auto r=edges[v][k];
            alternatives[r.a].push_back({v,k});
            if(r.a!=r.b)alternatives[r.b].push_back({v,k});
        }
        int cost=home;
        const int steps=std::min(9000,std::max(600,120*width));
        for(int step=0;step<steps;++step){
            if((step%24)==0&&over(end))break;
            if(step%320==0){
                chosen=selected;cost=evaluate(chosen);old_order=order;
            }
            if(old_order.empty())break;
            auto opts=chosen;
            if(step%7==0) {
                int retired=old_order[rng()%old_order.size()], edits=0;
                for(int v:old_order) {
                    auto e=edges[v][opts[v]];
                    if(e.a!=retired&&e.b!=retired)continue;
                    int q=-1,val=500;
                    for(int k=0;k<(int)edges[v].size();++k){
                        auto a=edges[v][k];
                        if(a.a==retired||a.b==retired)continue;
                        int w=3*(depth[a.a]+depth[a.b])+int(rng()%8);
                        if(w<val)val=w,q=k;
                    }
                    if(q>=0)opts[v]=q,++edits;
                }
                if(!edits)continue;
            } else if(step%7==1) {
                int x=(int)(rng()%n);
                const auto& in=alternatives[x];
                if(in.size()<2)continue;
                int changed=0;
                for(int k=0;k<12&&changed<4;++k) {
                    auto [a,r]=in[rng()%in.size()];
                    if(std::find(old_order.begin(),old_order.end(),a)==old_order.end())continue;
                    if(opts[a]!=r)opts[a]=r,++changed;
                }
                if(changed<2)continue;
            } else {
                int v=-1;
                for(int tries=0;tries<7;++tries) {
                    int r=old_order[rng()%old_order.size()];
                    if(edges[r].size()>1){v=r;break;}
                }
                if(v<0)continue;
                int k=(opts[v]+1+(int)(rng()%(edges[v].size()-1)))%(int)edges[v].size();
                opts[v]=k;
            }
            const int bill=evaluate(opts);
            ++explored;
            if(bill<best) {
                best=bill;selected=opts;
                if(bill<=cap)proof=materialize(opts);
            }
            if(bill>cost){
                if(bill>home+2 || policy==3 || rng()%100>=unsigned(step%320<160?13:4))
                    continue;
            }
            chosen=std::move(opts);cost=bill;old_order=order;
        }
        return proof;
    }
    struct State {
        std::vector<std::pair<int,int>> assigned;
        std::vector<int> open;
        Word sig;int score;
    };
    Script search(int cap,int width,int policy,int& explored){
        if(depth.empty()||over(end))return {};
        std::vector<State> states{{{},need,0,0}};
        Script result;
        for(int level=0;level<cap&&!states.empty()&&!over(end);++level){
            std::vector<State> next;
            std::unordered_set<Word> unique;
            for(const auto& state:states) {
                if(over(end)||(int)state.assigned.size()+(int)state.open.size()>cap)continue;
                std::vector<int> active(values.size(),-1);
                for(auto [a,k]:state.assigned)active[a]=k;
                auto tension=[&](int id) {
                    return std::make_tuple(3*depth[id]-(int)edges[id].size()/3,bit_count(values[id]),id);
                };
                const int x=*std::max_element(state.open.begin(),state.open.end(),[&](int a,int b){return tension(a)<tension(b);});
                for(int k=0;k<(int)edges[x].size();++k){
                    auto r=edges[x][k];
                    if(depth[r.a]>=99||depth[r.b]>=99)continue;
                    auto cyclic=[&](auto& self,int q)->bool {
                        if(q==x)return true;
                        if(active[q]<0)return false;
                        auto a=edges[q][active[q]];
                        return self(self,a.a)||self(self,a.b);
                    };
                    if(cyclic(cyclic,r.a)||cyclic(cyclic,r.b))continue;
                    State n=state;
                    n.assigned.push_back({x,k});
                    n.open.erase(std::find(n.open.begin(),n.open.end(),x));
                    for(int a:{r.a,r.b})
                        if(depth[a]&&active[a]<0&&std::find(n.open.begin(),n.open.end(),a)==n.open.end())
                            n.open.push_back(a);
                    ++explored;
                    int committed=(int)n.assigned.size()+(int)n.open.size();
                    if(committed>cap)continue;
                    if(n.open.empty()){
                        active[x]=k;
                        std::vector<unsigned char> seen(values.size(),0);
                        Script candidate;
                        auto each=[&](auto& self,int i)->void {
                            if(seen[i]||depth[i]==0)return;
                            seen[i]=1;
                            auto s=edges[i][active[i]];
                            self(self,s.a);self(self,s.b);
                            candidate.push_back({values[i],s.op,values[s.a],values[s.b]});
                        };
                        for(int id:need)each(each,id);
                        active[x]=-1;
                        if(result.empty()||candidate.size()<result.size()){
                            result=std::move(candidate);
                            cap=(int)result.size()-1;
                        }
                        continue;
                    }
                    n.sig^=hash(Word(x)*173619+Word(k)*1009);
                    if(!unique.insert(n.sig).second)continue;
                    int h=0,worst=0;
                    for(int a:n.open){h+=depth[a]-1;worst=std::max(worst,depth[a]-1);}
                    n.score=24*committed+4*h+3*worst+
                            (policy==2?2:4)*(int)n.open.size()+(int)(hash(n.sig^policy)%5);
                    next.push_back(std::move(n));
                }
            }
            auto keep=std::min((int)next.size(),width);
            std::partial_sort(next.begin(),next.begin()+keep,next.end(),
                              [](auto& a,auto& b){return std::tie(a.score,a.sig)<std::tie(b.score,b.sig);});
            next.resize(keep);
            states=std::move(next);
        }
        return result;
    }
};

struct Stats {int blocks=0,wins=0,sizes=0,largest=0,states=0,neutral=0;};
std::pair<Program,Stats> run(const Registers& r,const std::vector<Value>& targets,Value limit,
                 Program best,Deadline stop,int width,int leaves,int regions_per_pass,int mode,Word salt){
    Stats stats;
    Program work=best;
    for(int round=0;round<5&&!over(stop);++round){
        auto g=topological_graph(recover_graph(r,limit,targets,work));
        auto queue=regions(g,regions_per_pass,hash(salt^round));
        const int logical=(int)std::count_if(g.nodes.begin(),g.nodes.end(),[](auto x){return x.op!=0;});
        bool updated=false;
        int at=0;
        for(auto& region:queue){
            if(over(stop))break;
            auto cut=make_cut(g,region.roots);
            auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(stop-Clock::now());
            auto quantum=std::chrono::milliseconds(std::min<long long>(remaining.count(),
                          std::max<long long>(25,remaining.count()/std::min(5,(int)queue.size()-at))));
            auto deadline=std::min(stop,Clock::now()+quantum);
            ++at;
            ++stats.blocks;stats.largest=std::max(stats.largest,(int)cut.roots.size());
            stats.sizes|=1<<cut.roots.size();
            int cap=(int)best.size()-1-logical+cut.budget;
            if(cap<(int)cut.roots.size())continue;
            Library pool(g,cut,limit,deadline,leaves,mode);
            pool.finalize(mode);
            auto plan=mode>=3?pool.repair(cap,width,mode,stats.states,salt^region.mask):Script{};
            if(plan.empty()&&mode!=4)plan=pool.search(cap,width,mode,stats.states);
            if(plan.empty()||over(stop))continue;
            auto tape=replace_block(g,cut,plan,limit,(int)best.size(),17);
            if(!tape.empty()&&!over(stop)){
                best=std::move(tape);
                work=best;++stats.wins;updated=true;
                break;
            }
        }
        if(!updated && (over(stop)||round>=2))break;
    }
    return {std::move(best),stats};
}
} // namespace rebuild16

#ifndef ELF_RESIDUAL_INCLUDE
int main(int argc,char** argv) {
    try{
        using namespace rebuild16;
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=7)throw std::invalid_argument("width leaves regions policy salt milliseconds");
        const int width=std::stoi(argv[1]),leaves=std::stoi(argv[2]),
                  jobs=std::stoi(argv[3]),mode=std::stoi(argv[4]),ms=std::stoi(argv[6]);
        auto salt=std::stoull(argv[5]);
        if(width<1||leaves<8||leaves>128||ms<1)throw std::invalid_argument("Invalid block profile");
        local_beam::INDEX_BITS=leaves>84?19:18;local_beam::ALPHA=.24;
        Value limit;std::string key;int ns,nt,n;
        while(std::cin>>key>>limit>>ns>>nt>>n) {
            if(ns<1||ns>slots||nt<1||nt>63||n<0||n>4096||!limit||limit>=(Value(1)<<62))
                throw std::invalid_argument("Invalid puzzle dimensions");
            Registers regs{};std::vector<Value> t(nt);Program old(n);
            for(int i=0;i<ns;++i)std::cin>>regs[i];
            for(auto& v:t)std::cin>>v;
            for(auto& z:old)std::cin>>z.kind>>z.a>>z.b>>z.destination;
            auto [answer,s]=run(regs,t,limit,std::move(old),Clock::now()+std::chrono::milliseconds(ms),
                                width,leaves,jobs,mode,salt);
            std::unordered_set<Value> seen(regs.begin(),regs.end());
            for(auto z:answer)seen.insert(execute(regs,z,limit));
            for(Value v:t)if(!seen.count(v))throw std::logic_error("The cut changed a necessary target");
            std::cout<<key<<' '<<answer.size()<<' '<<s.blocks<<' '<<s.wins<<' '<<s.sizes<<' '<<s.largest
                     <<' '<<s.states<<' '<<s.neutral<<'\n';
            for(auto z:answer)std::cout<<z.kind<<' '<<z.a<<' '<<z.b<<' '<<z.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
