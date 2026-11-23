
#include <cmath>
#include "common.inc"
#include "compact.inc"
#include "cross.inc"
#include "cache_lookup.inc"
#include "cache_perturb.inc"
    bool rotate(Graph& g,size_t cap) {
        std::vector<Value> original;
        if(!order_for(g,targets,original))return false;
        g=live_only(g,original);
        std::unordered_map<Value,int> uses;
        for(Value t:targets)++uses[t];
        for(auto [v,r]:g)if(r.op){++uses[r.a];++uses[r.b];}
        std::vector<Value> vals=original;
        vals.insert(vals.end(),starts.begin(),starts.end());
        std::shuffle(vals.begin(),vals.end(),rng);
        lookup.prepare(vals,limit);
        std::vector<Value> roots=original;
        std::shuffle(roots.begin(),roots.end(),rng);
        int scanned=0;
        for(Value v:roots){
            std::unordered_set<Value> cut{v};
            std::vector<Value> changed;
            auto clear=[&](auto&& self,Value x)->void{
                changed.push_back(x);if(--uses[x]||!g.at(x).op)return;
                cut.insert(x);auto r=g.at(x);self(self,r.a);self(self,r.b);
            };
            auto old=g.at(v);
            clear(clear,old.a);clear(clear,old.b);
            for(Value x:changed)++uses[x];
            if(cut.size()<3)continue;
            std::unordered_set<Value> banned=cut,leaves;
            for(Value x:cut)for(Value a:{g.at(x).a,g.at(x).b})if(!cut.count(a))leaves.insert(a);
            for(Value x:original){auto r=g.at(x);if(banned.count(r.a)||banned.count(r.b))banned.insert(x);}
            std::vector<Value> fixed;
            for(Value x:vals)if(!banned.count(x))fixed.push_back(x);
            std::stable_partition(fixed.begin(),fixed.end(),[&](Value x){return leaves.count(x);});
            if(fixed.empty())continue;
            auto maker=[&](Value x,Recipe& answer){
                return !g.count(x) && lookup.first(x,[&](Recipe r){return g.count(r.a)&&g.count(r.b)&&!banned.count(r.a)&&!banned.count(r.b);},answer);
            };
            auto accept=[&](Value a,Recipe ra,Value b,Recipe rb,Recipe end){
                if(a==b||a==v||b==v)return false;
                g[a]=ra;g[b]=rb;g[v]=end;
                std::vector<Value> live;
                if(order_for(g,targets,live)&&live.size()<=cap){g=live_only(g,live);return true;}
                g.erase(a);g.erase(b);g[v]=old;return false;
            };
            auto balanced=[&]() {
                int tests=0;
                int width=std::min<int>(fixed.size(),24);
                std::vector<Value> temp;
                for(int i=0;i<width;i++)for(int j=0;j<=i;j++){
                    Value aa=std::max(fixed[i],fixed[j]),bb=std::min(fixed[i],fixed[j]);
                    for(char op:{'+','-','*','/'}) {
                        Value h=calculate(op,aa,bb);
                        if(h&&!g.count(h)&&lookup.contains(h))temp.push_back(h);
                    }
                }
                std::shuffle(temp.begin(),temp.end(),rng);
                for(Value a:temp)for(auto end:inverses(v,a)){
                    Value b=end.a==a?end.b:end.a;
                    Recipe ra,rb;
                    if(a!=b&&maker(b,rb)&&maker(a,ra)){
                        if(accept(a,ra,b,rb,end))return true;
                        if(++tests>=12)return false;
                    }
                }
                return false;
            };
            auto chain=[&](){
                int width=std::min<int>(fixed.size(),32),tested=0;
                for(int i=0;i<width;i++)for(auto end:inverses(v,fixed[i])){
                    Value b=end.a==fixed[i]?end.b:end.a;
                    if(g.count(b))continue;
                    for(int j=0;j<width;j++)for(auto rb:inverses(b,fixed[j])){
                        Value a=rb.a==fixed[j]?rb.b:rb.a;
                        Recipe ra;
                        if(a!=b&&maker(a,ra)){
                            if(accept(a,ra,b,rb,end))return true;
                            if(++tested>=12)return false;
                        }
                    }
                }
                return false;
            };
            if((rng()%2)?(balanced()||chain()):(chain()||balanced()))return true;
            if(++scanned>=12)break;
        }
        return false;
    }
    Tape improve(const std::vector<Tape>& tapes){
        Tape best=tapes.front();Graph champion=graph_from(best),trial=champion;
        std::vector<Graph> pool;
        for(int turn=0;turn<rounds;++turn){
            if(turn%9==0)trial=champion;
            else if(turn%9==5&&!pool.empty())trial=pool[rng()%pool.size()];
            std::shuffle(targets.begin(),targets.end(),rng);
            size_t cap=best.size()+((turn%9==1||turn%9==2)?2:turn%9<=5?1:0);
            perturb(trial,cap,3+rng()%8);
            rotate(trial,cap);
            Tape candidate;
            if(!compile(trial,candidate)||candidate.size()>cap){trial=champion;continue;}
            if(candidate.size()<best.size()){best=candidate;champion=trial;}
            if(knockout(trial,candidate)&&candidate.size()<best.size()){best=candidate;champion=trial;}
            if(candidate.size()<=best.size()+1){
                pool.push_back(trial);
                if(pool.size()>12)pool.erase(pool.begin()+rng()%5);
            }
            if(turn%5==4&&pool.size()>=4){
                Graph better;
                if(crossover(champion,pool,better,best)){champion=better;trial=better;}
            }
        }
        return best;
    }

};
#include "main.inc"
