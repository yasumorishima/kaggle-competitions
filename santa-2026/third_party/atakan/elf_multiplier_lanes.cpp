// Rewrite several target chains around a single, explicitly paid multiplier.
// The incumbent is an input to this optimizer, not a source of score claims.
#include <set>
#define main dormant_global_optimizer
#include "elf_whole_graph.cpp"
#undef main

namespace lanes {
struct Edge { Value x; char op; Value a,b; };
using Proof=std::vector<Edge>;
struct Trial { int root; Value family; Proof proof; int rank; };
struct RecipePair { int a,b; char op; };
#ifdef FRONTIER_INLINE_ATLAS
// The original atlas already caps this list at four witnesses. Keeping
// them inline avoids an allocation for each new numerical lookup key.
struct RecipeList {
    std::array<RecipePair,4> options;
    unsigned count=0;
    std::size_t size()const{return count;}
    void push_back(RecipePair item){options[count++]=item;}
    const RecipePair* begin()const{return options.data();}
    const RecipePair* end()const{return options.data()+count;}
};
#else
using RecipeList=std::vector<RecipePair>;
#endif

struct Atlas {
    const PlanPool& p;
    const int n;
    std::unordered_map<Value,RecipeList> one;
    std::vector<std::vector<unsigned char>> parent;
    std::vector<int> refs,private_size;
    explicit Atlas(const PlanPool& pool):p(pool),n(pool.previous_nodes),
             parent(n,std::vector<unsigned char>(n)),refs(n),private_size(n) {
        one.reserve(n*n*2);
        for(int goal:p.targets)++refs[goal];
        for(int i=0;i<n;++i) {
            parent[i][i]=1;
            if(p.choices[i].empty())continue;
            auto r=p.choices[i][0];
            ++refs[r.a];if(r.a!=r.b)++refs[r.b];
            for(int j=0;j<i;++j)parent[i][j]=parent[r.a][j]||parent[r.b][j];
        }
        for(int i=0;i<n;++i)if(!p.choices[i].empty()){
            auto r=p.choices[i][0];
            private_size[i]=1+(refs[r.a]==1?private_size[r.a]:0)
                             +(r.b!=r.a&&refs[r.b]==1?private_size[r.b]:0);
        }
        for(int i=0;i<n;++i)for(int j=0;j<=i;++j) {
            int a=i,b=j;if(p.nums[a]<p.nums[b])std::swap(a,b);
            for(char op:{'+','-','*','/'}) {
                Value x=calculate(op,p.nums[a],p.nums[b],p.limit);
                if(x==p.nums[a]||x==p.nums[b]||!x)continue;
                auto& list=one[x];
                if(list.size()<4)list.push_back({a,b,op});
            }
        }
    }
    bool add(Proof& out,Value x,char op,Value a,Value b)const {
        if(!x||x==a||x==b||calculate(op,a,b,p.limit)!=x)return false;
        if((op=='+'||op=='*')&&a>b)std::swap(a,b);
        if(std::none_of(out.begin(),out.end(),[&](auto s){return s.x==x;}))
            out.push_back({x,op,a,b});
        return true;
    }
    bool ready(Value a,int root)const {
        auto at=p.ids.find(a);
        return at!=p.ids.end() && at->second<n && !parent[at->second][root];
    }
    bool take(Value x,int root,Proof& out,int depth=1)const {
        if(!x||x==p.nums[root])return false;
        if(ready(x,root)||std::any_of(out.begin(),out.end(),[&](auto s){return s.x==x;}))return true;
        auto one_step=[&](Value y,Proof& trial){
            if(y==p.nums[root])return false;
            if(ready(y,root))return true;
            auto at=one.find(y);
            if(at==one.end())return false;
            for(auto e:at->second)
                if(!parent[e.a][root]&&!parent[e.b][root])
                    return add(trial,y,e.op,p.nums[e.a],p.nums[e.b]);
            return false;
        };
        if(one_step(x,out))return true;
        if(depth<=1)return false;
        for(int i=0;i<n;++i){
            if(parent[i][root])continue;
            const auto a=p.nums[i];
            if(a==x)continue;
            bool ok=inverse(x,a,p.limit,[&](Value b,char op,bool first){
                if(b==x||b==p.nums[root])return false;
                Proof attempt=out;
                if(!one_step(b,attempt)||!add(attempt,x,op,first?b:a,first?a:b))return false;
                out=std::move(attempt);
                return true;
            });
            if(ok)return true;
        }
        return false;
    }
    bool known(Value x)const{return p.ids.count(x)||one.count(x);}
};

using Buckets=std::map<Value,std::map<int,std::vector<Trial>>>;

void keep(const Atlas& a,Buckets& all,int root,Value m,Proof proof,int cap) {
    if(proof.empty()||proof.back().x!=a.p.nums[root])return;
    const auto last=proof.back();
    auto old=a.p.choices[root][0];
    Value l=a.p.nums[old.a],r=a.p.nums[old.b];
    if((old.op=='+'||old.op=='*')&&l>r)std::swap(l,r);
    if(last.op==old.op&&last.a==l&&last.b==r)return;
    if(proof.size()>(std::size_t)std::max(4,a.private_size[root]+2))return;
    auto& list=all[m][root];
    auto rank=8*(int)proof.size()-3*a.private_size[root];
    // Slightly favor reusing a maintained multiplier over a speculative one
    // when both have the same exact write count.
    rank+=a.ready(m,root)?-2:0;
    for(const auto& entry:list) {
        if(entry.proof.size()!=proof.size())continue;
        bool same=true;
        for(std::size_t j=0;j<proof.size();++j) {
            const auto x=entry.proof[j],y=proof[j];
            if(x.x!=y.x||x.op!=y.op||x.a!=y.a||x.b!=y.b){same=false;break;}
        }
        if(same)return;
    }
    if((int)list.size()>=cap&&rank>=list.back().rank)return;
    list.push_back({root,m,std::move(proof),rank});
    std::stable_sort(list.begin(),list.end(),[](auto& l,auto& r){return l.rank<r.rank;});
    if((int)list.size()>cap)list.resize(cap);
}

void affine(const Atlas& a,Buckets& all,int root,Value m,Value q,Value r,
            char sign,int cap,int deep=0,Value divisor=1,bool center=false) {
    const auto t=a.p.nums[root];
    if(m<=1||q<=1||q==t||r==t||m==t||q>a.p.limit/m||t>a.p.limit/divisor)return;
    const Value n=t*divisor;
    const Value pr=m*q;
    if((r && (r==pr||pr==n))||(pr==t && divisor>1))return;
    Proof out;
    if(r&&!a.take(r,root,out))return;
    if(!a.take(m,root,out)||!a.take(q,root,out,deep?2:1))return;
    if(r==0||!a.ready(pr,root)) {
        if(!a.add(out,pr,'*',m,q))return;
    }
    if(r&&!a.add(out,n,sign=='r'?'-':sign,sign=='r'?r:pr,sign=='r'?pr:r))return;
    if(divisor>1&&(!a.take(divisor,root,out)||!a.add(out,t,'/',n,divisor)))return;
    if(center&&r && pr>=128&&!a.ready(pr,root))keep(a,all,root,pr,out,cap);
    keep(a,all,root,m,std::move(out),cap);
}

void fold(const Atlas& a,Buckets& all,int root,const std::vector<Value>& factors,int cap) {
    const auto base=a.p.choices[root][0];
    const Value t=a.p.nums[root], x=a.p.nums[base.a], y=a.p.nums[base.b];
    if(base.op!='+'&&base.op!='-'&&base.op!='*')return;
    for(Value f:factors) {
        if(f<=1||f==t)continue;
        if(base.op=='*') {
            for(auto [u,v]:{std::pair<Value,Value>{x,y},{y,x}}){
                if(u==f||u%f||u/f>a.p.limit/v)continue;
                Value q=(u/f)*v;
                if(q==1||q==t)continue;
                affine(a,all,root,f,q,0,'+',cap,0);
            }
            continue;
        }
        if(x%f||y%f)continue;
        const Value u=x/f,v=y/f,d=t/f;
        if(d==1||v==d||u==d)continue;
        Proof out;
        if(!a.take(f,root,out))continue;
        if(!a.take(d,root,out)) {
            if(!a.take(u,root,out)||!a.take(v,root,out)||!a.add(out,d,base.op,u,v))continue;
        }
        if(a.add(out,t,'*',f,d))keep(a,all,root,f,std::move(out),cap);
    }
}

// Lift the regrouping out of one local rotation. For up to six factors,
// price all binary partitions, allowing the resulting products to become
// family keys for other roots. A division may be cancelled across leaves.
void reassociate(const Atlas& a,Buckets& all,int root,int cap,bool wider) {
    const auto b=a.p.choices[root][0];
    if(b.op!='*'&&b.op!='/')return;
    using Parts=std::vector<int>;
    std::vector<Parts> front;
    if(b.op=='*')front.push_back({b.a,b.b});else front.push_back({b.a});
    std::set<Parts> seen;
    const int effort=wider?18:9;
    for(int j=0;j<(int)front.size()&&j<effort;++j) {
        auto f=front[j];
        if(f.size()<6)for(int k=0;k<(int)f.size();++k) {
            const int x=f[k];
            if(a.p.choices[x].empty())continue;
            auto e=a.p.choices[x][0];
            if(e.op!='*')continue;
            Parts next=f;
            next[k]=e.a;next.push_back(e.b);
            std::sort(next.begin(),next.end());
            if(seen.insert(next).second)front.push_back(std::move(next));
        }
        if(f.size()<3 && b.op=='*')continue;
        if(f.size()<2 && b.op=='/')continue;
        // Test different cancellation orders without inventing negative,
        // fractional, or out-of-range arithmetic states.
        int orders=b.op=='*'?1:(wider?std::min(4,(int)f.size()):2);
        for(int shift=0;shift<orders;++shift){
            auto adjusted=f;
            std::rotate(adjusted.begin(),adjusted.begin()+shift,adjusted.end());
            std::vector<Value> leaves;
            Proof seed;
            Value den=b.op=='/'?a.p.nums[b.b]:1;
            for(int x:adjusted) {
                const Value y=a.p.nums[x],d=std::gcd(y,den);
                den/=d;
                if(y/d==1)continue;
                if(!a.take(y/d,root,seed,1)){den=0;break;}
                leaves.push_back(y/d);
            }
            if(den!=1||leaves.size()<2)continue;
            const int goal=(1<<(int)leaves.size())-1;
            std::array<Value,64> product{};
            std::array<Proof,64> plans;
            std::array<bool,64> feasible{};
            product[0]=1;
            for(int mask=1;mask<=goal;++mask) {
                const int low=mask&-mask;
                int leaf=0;for(int x=low;x>1;x>>=1)++leaf;
                const Value before=product[mask-low],new_part=leaves[leaf];
                if(!before||before>a.p.limit/new_part)continue;
                const Value value=before*new_part;
                product[mask]=value;
                if(low==mask) {feasible[mask]=true;continue;}
                if(mask!=goal) {
                    Proof keyed=seed;
                    if(a.take(value,root,keyed)) {
                        plans[mask].assign(keyed.begin()+seed.size(),keyed.end());
                        feasible[mask]=true;
                        if(plans[mask].empty())continue;
                    }
                }
                for(int l=(mask-1)&mask;l;l=(l-1)&mask){
                    const int r=mask^l;
                    if(l>r||!feasible[l]||!feasible[r])continue;
                    Proof candidate=plans[l];
                    for(const auto e:plans[r])
                        if(std::none_of(candidate.begin(),candidate.end(),[&](auto s){return s.x==e.x;}))
                            candidate.push_back(e);
                    if(!a.add(candidate,value,'*',product[l],product[r]))continue;
                    if(mask==goal){
                        Proof clean=seed;
                        for(auto s:candidate)
                            if(std::none_of(clean.begin(),clean.end(),[&](auto q){return s.x==q.x;}))
                                clean.push_back(s);
                        if(value!=a.p.nums[root])continue;
                        keep(a,all,root,product[l],clean,cap);
                        keep(a,all,root,product[r],std::move(clean),cap);
                    } else if(!feasible[mask]||candidate.size()<plans[mask].size()){
                        feasible[mask]=true;
                        plans[mask]=std::move(candidate);
                    }
                }
            }
        }
    }
}

std::pair<std::vector<Value>,std::vector<int>> choose(const Atlas& a,int n_factor,int n_roots,int depth) {
    const auto& p=a.p;
    std::vector<int> demand;
    std::vector<unsigned char> target(a.n);
    for(int v:p.targets)target[v]=1;
    for(int i=0;i<a.n;++i)
        if(!p.choices[i].empty()&&(target[i]||a.private_size[i]>=2||p.choices[i][0].op=='*'))
            demand.push_back(i);
    std::stable_sort(demand.begin(),demand.end(),[&](int x,int y) {
        return 3*a.private_size[x]+2*target[x]+std::min(4,a.refs[x])>
               3*a.private_size[y]+2*target[y]+std::min(4,a.refs[y]);
    });
    if((int)demand.size()>n_roots)demand.resize(n_roots);
    std::unordered_map<Value,int> frequency;
    for(int i=0;i<a.n;++i)if(!p.choices[i].empty()) {
        const auto r=p.choices[i][0];
        if(r.op=='*') {
            ++frequency[p.nums[r.a]];
            if(r.a!=r.b)++frequency[p.nums[r.b]];
        }
    }
    for(int i=0;i<a.n;++i)
        if(p.nums[i]>=2&&(a.refs[i]>1||p.nums[i]<=32))frequency.try_emplace(p.nums[i],0);
    // One-write roots of shared rational multipliers, found from this
    // puzzle's live coefficients, not a hard-coded per-case catalogue.
    std::vector<Value> regular;
    for(auto [f,w]:frequency)if(w>=2)regular.push_back(f);
    std::sort(regular.begin(),regular.end());
    for(std::size_t i=0;i<regular.size();++i)
        for(std::size_t j=i+1;j<regular.size();++j){
            Value g=std::gcd(regular[i],regular[j]);
            if(g>=4 && a.known(g))++frequency[g];
        }
    for(Value x:regular)for(Value d:{2,3,4,8,16}) {
        if(x%d==0&&x/d>=2&&a.known(x/d))frequency.try_emplace(x/d,1);
        if(x<=p.limit/d&&a.known(x*d))frequency.try_emplace(x*d,0);
    }
    std::vector<std::pair<int,Value>> order;
    for(auto [f,uses]:frequency) {
        if(f<2||f>p.limit/2)continue;
        int strength=4*std::min(uses,12);
        for(int ax:demand){
            Value t=p.nums[ax];
            if(t<=f)continue;
            Value q=t/f,r=t%f;
            if(q>1 && !r && a.known(q))strength+=6;
            else if(q>1 && a.known(q) && a.known(r))strength+=3;
            if(r && q+1<=p.limit/f && a.known(q+1)&&a.known(f-r))strength+=3;
        }
        if(strength>0)order.emplace_back(-strength,f);
    }
    std::sort(order.begin(),order.end());
    std::vector<Value> factors;
    for(auto [_,f]:order) {
        if((int)factors.size()>=n_factor)break;
        factors.push_back(f);
    }
    if(depth>=5) {
        // A latent coefficient must have feasible quotient and remainder
        // witnesses for multiple roots; reward their investment together.
        std::vector<std::pair<int,Value>> candidates;
        for(const auto& [m,_]:a.one) {
            if(m<51||m>=p.limit/3||p.ids.count(m)||frequency.count(m))continue;
            int count=0,strength=0;
            for(int i:demand) {
                const Value t=p.nums[i];
                if(t<m*2)continue;
                const Value q=t/m,r=t%m;
                int gain=0;
                if((!r || a.known(r))&&a.known(q))
                    gain=std::max(gain,p.ids.count(q) ? (r?3:6) : (r?1:3));
                if(r&&q+1<=p.limit/m&&a.known(q+1)&&a.known(m-r))
                    gain=std::max(gain,p.ids.count(q+1)?3:1);
                if(gain){++count;strength+=gain;}
            }
            if(count>=2&&strength>=4)candidates.emplace_back(-strength,m);
        }
        const auto cap=std::min<std::size_t>(depth>=7?105:42,candidates.size());
        std::partial_sort(candidates.begin(),candidates.begin()+cap,candidates.end());
        for(std::size_t i=0;i<cap;++i)factors.push_back(candidates[i].second);
    }
    return {factors,demand};
}

struct ResidueIndex {
    using Entries=std::vector<std::pair<Value,int>>;
    using Iterator=Entries::const_iterator;
    const PlanPool& p;
    std::map<Value,Entries> table;
    std::pair<Iterator,Iterator> get(Value modulus,Value residue){
        auto [at,added]=table.try_emplace(modulus);
        auto& row=at->second;
        if(added){
            row.reserve(p.nums.size());
            for(int i=0;i<(int)p.nums.size();++i)row.emplace_back(p.nums[i]%modulus,i);
            std::sort(row.begin(),row.end());
        }
        return {std::lower_bound(row.cbegin(),row.cend(),std::pair<Value,int>{residue,-1}),
                std::upper_bound(row.cbegin(),row.cend(),std::pair<Value,int>{residue,(int)p.nums.size()})};
    }
};

Buckets generate(const Atlas& a,const std::vector<Value>& factors,const std::vector<int>& goals,
                 int quantity,int depth,bool indexed=false) {
    Buckets all;
    const Value L=a.p.limit;
    const bool centers=depth>=5;
    ResidueIndex residues{a.p,{}};
    for(int root:goals){
        const auto t=a.p.nums[root];
        fold(a,all,root,factors,quantity);
        if(depth>=8)reassociate(a,all,root,quantity,depth>=9);
        for(Value m:factors) {
            if(m>=t||m==0)continue;
            const Value q=t/m,r=t%m;
            for(int k=-std::min(depth,2);k<=std::min(depth,2)+1;++k) {
                if(k<0&&q<=Value(1-k))continue;
                const Value v=k<0?q-Value(-k):q+Value(k);
                if(v> L/m || v<=1)continue;
                const Value u=m*v,w=u<t?t-u:u-t;
                if((!w||a.known(w))&&a.known(v))
                    affine(a,all,root,m,v,w,u<=t?'+':'-',quantity,0,1,centers);
            }
            // Check noncanonical remainders shared elsewhere in the graph.
            // The same computed offset can be hundreds of residues away.
            const bool extended=depth>=3 && a.private_size[root]>=3;
            int promoted=0;
            auto shifted=[&](int i,bool same,bool opposite) {
                if(a.parent[i][root])return;
                const Value s=a.p.nums[i];
                if(s==t)return;
                if(same) {
                    const Value z=s>t?s-t:t-s,v=z/m;
                    if(a.known(v)||(extended&&promoted<2)) {
                        affine(a,all,root,m,v,s,s>t?'r':'+',quantity,!a.known(v),1,centers);
                        promoted+=!a.known(v);
                    }
                }
                if(s<=L-t && opposite) {
                    const Value v=(t+s)/m;
                    if(a.known(v)||(extended&&promoted<2)) {
                        affine(a,all,root,m,v,s,'-',quantity,!a.known(v),1,centers);
                        promoted+=!a.known(v);
                    }
                }
            };
            if(indexed){
                auto [e,e_end]=residues.get(m,r);
                auto [f,f_end]=residues.get(m,r?m-r:0);
                // Visit the same graph indices in the original order. Even
                // a bounded promotion chosen by order is preserved exactly.
                while(e!=e_end||f!=f_end){
                    int i=e!=e_end?e->second:a.n,j=f!=f_end?f->second:a.n;
                    shifted(std::min(i,j),i<=j,j<=i);
                    if(i<=j)++e;
                    if(j<=i)++f;
                }
            }else for(int i=0;i<a.n;++i){
                const auto s=a.p.nums[i];
                if(a.parent[i][root]||s==t)continue;
                shifted(i,s%m==r,(s+r)%m==0);
            }
            if(depth<6|| a.private_size[root]<2)continue;
            for(Value d:{2,3,4,5,7,8,11,16}) {
                if(t>L/d||!a.ready(d,root)||m%d==0)continue;
                const Value n=t*d,q=n/m,r=n%m;
                if(q>1&&a.known(q)&&(!r||a.known(r)))
                    affine(a,all,root,m,q,r,'+',quantity,0,d,true);
                if(r&&q+1<=L/m&&a.known(q+1)&&a.known(m-r))
                    affine(a,all,root,m,q+1,m-r,'-',quantity,0,d,true);
                if(depth<7||a.private_size[root]<4)continue;
                // Try one inexpensive shared additive correction outside
                // the canonical quotient interval.
                int tries=0;
                auto correction=[&](int i){
                    auto b=a.p.nums[i];
                    if(a.parent[i][root]||b<1||b>=n)return;
                    Value v=(n-b)/m;
                    if(!a.known(v))return;
                    ++tries;
                    affine(a,all,root,m,v,b,'+',quantity,0,d,true);
                };
                if(indexed){
                    auto [pos,end]=residues.get(m,r);
                    for(;pos!=end&&tries<4;++pos)correction(pos->second);
                }else for(int i=0;i<a.n&&tries<4;++i){
                    const auto b=a.p.nums[i];
                    if(!a.parent[i][root]&&b<n&&b%m==r)correction(i);
                }
            }
        }
    }
    return all;
}

using Edits=std::vector<std::pair<int,int>>;
struct Change {int root;Edits edits;int rank;};
struct Lane {Value factor;std::map<int,std::vector<Change>> groups;int rank=0;};

std::vector<Lane> register_choices(PlanPool& pool,const Buckets& options,int& proofs) {
    std::vector<Lane> result;
    for(const auto& [f,map]:options){
        Lane lane{f,{}};
        for(const auto& [root,list]:map)for(const auto& item:list) {
            Edits edits;
            bool fitted=true;
            for(auto s:item.proof) {
                const int id=pool.intern(s.x),l=pool.ids.at(s.a),r=pool.ids.at(s.b);
                pool.add(id,s.op,l,r,90);
                const auto& opts=pool.choices[id];
                auto equivalent=[&](auto e){
                    return e.op==s.op&&((e.a==l&&e.b==r)||
                          ((s.op=='+'||s.op=='*')&&e.a==r&&e.b==l));
                };
                const auto at=std::find_if(opts.begin(),opts.end(),equivalent);
                if(at==opts.end()) {fitted=false;break;}
                edits.push_back({id,(int)(at-opts.begin())});
            }
            if(!fitted)continue;
            ++proofs;
            lane.groups[root].push_back({root,std::move(edits),item.rank});
        }
        for(const auto& [_,v]:lane.groups)lane.rank+=std::max(0,22-v[0].rank);
        if(!lane.groups.empty())result.push_back(std::move(lane));
    }
    std::stable_sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.rank>b.rank;});
    return result;
}

bool mix(Edits& united,const Edits& alt){
    for(auto [idx,j]:alt){
        auto pos=std::find_if(united.begin(),united.end(),[&](auto s){return s.first==idx;});
        if(pos==united.end())united.emplace_back(idx,j);
        else if(pos->second!=j)return false;
    }
    return true;
}

struct Outcome {Program code;int lanes=0,proofs=0,single=0,joint=0,deepest=0,neutral=0;};

Outcome solve(const Registers& regs,Value bound,const std::vector<Value>& targets,Program base,
              int factors,int goals,int width,int passes,int depth,int mode) {
    Outcome out{base};
    Program working=base;
    std::set<Edits> visited_morphs;
    for(int pass=0;pass<passes;++pass) {
        PlanPool pool(regs,targets,bound,working);
        Atlas support(pool);
        auto [fac,demand]=choose(support,factors,goals,depth);
        auto preset=generate(support,fac,demand,depth<2?2:4,depth);
        auto lanes=register_choices(pool,preset,out.proofs);
        out.lanes+=(int)lanes.size();
        State current(pool.nums.size(),0);
        Evaluator check(pool);
        auto cost=[&](const Edits& edits){
            State s=current;
            for(auto [i,j]:edits)s[i]=j;
            return check.eval(s);
        };
        auto commit=[&](const Edits& edits,int roots){
            State s=current;
            for(auto [i,j]:edits)s[i]=j;
            const int bill=check.eval(s),cap=(int)out.code.size();
            if(bill>=cap)return false;
            auto fresh=check.freeze(s);
            for(int k=0;k<19;++k) {
                auto code=pressure_schedule(fresh,k,(std::uint32_t)(bound/7+k*257),cap);
                if(code.empty())code=Scheduler(fresh,cap,bound,k%4,(k/4)%2).run(fresh.targets);
                if(!code.empty()){
                    out.code=std::move(code);
                    working=out.code;
                    if(roots>1)++out.joint, out.deepest=std::max(out.deepest,roots);
                    else ++out.single;
                    current=std::move(s);
                    return true;
                }
            }
            return false;
        };
        bool improved=false;
        Edits drift;
        int fitness=0;
        for(int sweep=0;sweep<3;++sweep){
            const int base_cost=check.eval(current);
            int best=(int)out.code.size(),roots=0;
            Edits best_edit;
            auto scout=[&](const Edits& e,int score,int roots){
                if(mode<3||pass+1>=passes||score!=(int)out.code.size()||e.empty())return;
                int novelty=0,n=0;
                std::uint64_t mix_seed=bound+std::uint64_t(pass)*0xd1b54a32d192ed03ULL;
                for(auto [i,_]:e) {
                    if(i<pool.previous_nodes || !check.mark[i])continue;
                    const Value x=pool.nums[i];
                    const int use=std::clamp(check.uses[i],0,5);
                    const auto r=pool.choices[i][0];
                    ++n;
                    if(mode==5) {
                        // Keep reusable medium-size factors rather than
                        // simply maximizing the number of changed values.
                        novelty+=4+3*use+5*(r.op=='*')+5*(x>=128 && x<100000000);
                    } else if(mode==6) {
                        novelty+=5+5*use+6*(r.op=='*')+4*(x>=100000000);
                    } else {
                        novelty+=(mode==3?5:3)+3*use+(x>128);
                    }
                    mix_seed^=(x^(x>>29))*0x9e3779b97f4a7c15ULL;
                }
                if(mode>=5&&n)novelty=std::max(1,novelty+(mode==7?14:7)*(roots>1)+
                                      int((mix_seed^(mix_seed>>31))%(mode==7?97:29)));
                if(!novelty||novelty+roots<=fitness)return;
                fitness=novelty+roots;
                drift=e;
            };
            struct Node {Edits edits;int price,roots;};
            for(auto& lane:lanes) {
                std::vector<std::pair<int,int>> roots_order;
                for(const auto& [v,choices]:lane.groups){
                    int lean=100000;
                    for(auto& x:choices) {
                        const int length=cost(x.edits);
                        lean=std::min(lean,length);
                        if(mode!=2 && length<best){best=length;best_edit=x.edits;roots=1;}
                        scout(x.edits,length,1);
                    }
                    if(lean<base_cost+5)roots_order.emplace_back(lean,v);
                }
                if(mode==0||width==0||roots_order.size()<2)continue;
                std::sort(roots_order.begin(),roots_order.end());
                if(roots_order.size()>18)roots_order.resize(18);
                std::vector<Node> partial;
                for(auto [_,root]:roots_order){
                    std::vector<Node> step=partial;
                    for(auto& x:lane.groups.at(root)){
                        const int length=cost(x.edits);
                        if(length>=base_cost+5)continue;
                        step.push_back({x.edits,length,1});
                        for(auto& old:partial){
                            if(old.roots>=std::min(7,2+depth))continue;
                            Edits joined=old.edits;
                            if(!mix(joined,x.edits)||joined.size()==old.edits.size())continue;
                            const int score=cost(joined);
                            if(score>=base_cost+4)continue;
                            if(score<best){best=score;best_edit=joined;roots=old.roots+1;}
                            scout(joined,score,old.roots+1);
                            step.push_back({std::move(joined),score,old.roots+1});
                        }
                    }
                    std::sort(step.begin(),step.end(),[](const auto& a,const auto& b){
                        // Preserve small non-improving sets, which can pay for
                        // a shared factor only when a later child joins.
                        return std::make_pair(4*a.price-a.roots,a.edits.size())<
                               std::make_pair(4*b.price-b.roots,b.edits.size());
                    });
                    partial.clear();
                    std::set<Edits> seen;
                    for(auto it=step.begin();it!=step.end()&&(int)partial.size()<width;++it) {
                        Edits sig=it->edits;std::sort(sig.begin(),sig.end());
                        if(seen.insert(std::move(sig)).second)partial.push_back(std::move(*it));
                    }
                }
            }
            if(best_edit.empty()||!commit(best_edit,roots))break;
            improved=true;
        }
        if(!improved) {
            if(drift.empty())break;
            State neutral=current;
            for(auto [i,j]:drift)neutral[i]=j;
            if(check.eval(neutral)!=(int)out.code.size())break;
            Edits signature;
            for(int id:check.order)if(!pool.choices[id].empty())
                signature.emplace_back(int(pool.nums[id]^(pool.nums[id]>>32)),neutral[id]);
            if(!visited_morphs.insert(signature).second)break;
            auto graph=check.freeze(neutral);
            Program alternative;
            for(int k=0;k<5&&alternative.empty();++k)
                alternative=pressure_schedule(graph,k,(std::uint32_t)(bound+911*k),int(out.code.size())+1);
            if(alternative.size()!=out.code.size())break;
            working=std::move(alternative);
            ++out.neutral;
        }
    }
    return out;
}
} // namespace lanes

#ifndef ELF_MULTIPLIER_LIBRARY
int main(int argc,char** argv){
    try{
        std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
        if(argc!=7)throw std::invalid_argument("factors roots width passes depth mode");
        const int f=std::stoi(argv[1]),g=std::stoi(argv[2]),w=std::stoi(argv[3]),
                  p=std::stoi(argv[4]),d=std::stoi(argv[5]),m=std::stoi(argv[6]);
        if(f<1||g<1||w<0||p<1||d<0||m<0||m>7)throw std::invalid_argument("Invalid profile");
        std::string key;Value limit;int ns,nt,n;
        while(std::cin>>key>>limit>>ns>>nt>>n){
            if(ns<1||ns>slots||nt<1||nt>63||n<0||n>4096||!limit||limit>=(Value(1)<<62))
                throw std::invalid_argument("Invalid puzzle");
            Registers regs{};std::vector<Value> targets(nt);Program source(n);
            for(int i=0;i<ns;++i)std::cin>>regs[i];
            for(auto& x:targets)std::cin>>x;
            for(auto& x:source)std::cin>>x.kind>>x.a>>x.b>>x.destination;
            auto out=lanes::solve(regs,limit,targets,std::move(source),f,g,w,p,d,m);
            std::unordered_set<Value> seen(regs.begin(),regs.end());
            for(auto op:out.code)seen.insert(execute(regs,op,limit));
            for(auto t:targets)if(!seen.count(t))throw std::logic_error("Lost required value");
            std::cout<<key<<' '<<out.code.size()<<' '<<out.lanes<<' '<<out.proofs<<' '<<out.single
                     <<' '<<out.joint<<' '<<out.deepest<<' '<<out.neutral<<'\n';
            for(auto op:out.code)std::cout<<op.kind<<' '<<op.a<<' '<<op.b<<' '<<op.destination<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
