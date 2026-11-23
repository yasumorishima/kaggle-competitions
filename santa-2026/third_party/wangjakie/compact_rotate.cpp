
#include <cmath>
#include "common.inc"
#include "neutral.inc"
#include "compact.inc"
#include "cross.inc"
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
        std::unordered_map<Value,std::vector<Recipe>> one;
        for(size_t i=0;i<vals.size();i++)for(size_t j=0;j<=i;j++){
            Value a=std::max(vals[i],vals[j]),b=std::min(vals[i],vals[j]);
            for(char op:{'+','-','*','/'}){
                Value x=calculate(op,a,b);
                if(x&&!g.count(x))one[x].push_back({op,a,b});
            }
        }
        std::vector<Value> roots=original;
        std::shuffle(roots.begin(),roots.end(),rng);
        int scanned=0;
        for(Value v:roots){
            std::unordered_set<Value> cut{v};
            auto counts=uses;
            auto clear=[&](auto&& self,Value x)->void{
                if(--counts[x]||!g.at(x).op)return;
                cut.insert(x);auto r=g.at(x);self(self,r.a);self(self,r.b);
            };
            auto old=g.at(v);
            clear(clear,old.a);clear(clear,old.b);
            if(cut.size()<3)continue;
            std::unordered_set<Value> banned=cut,leaves;
            for(Value x:cut)for(Value a:{g.at(x).a,g.at(x).b})if(!cut.count(a))leaves.insert(a);
            for(Value x:original){auto r=g.at(x);if(banned.count(r.a)||banned.count(r.b))banned.insert(x);}
            std::vector<Value> fixed;
            for(Value x:vals)if(!banned.count(x))fixed.push_back(x);
            std::stable_partition(fixed.begin(),fixed.end(),[&](Value x){return leaves.count(x);});
            if(fixed.empty())continue;
            auto maker=[&](Value x,Recipe& answer){
                auto it=one.find(x);if(it==one.end())return false;
                for(auto r:it->second)if(!banned.count(r.a)&&!banned.count(r.b)){answer=r;return true;}
                return false;
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
                        if(h&&one.count(h))temp.push_back(h);
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
    Tape improve(const std::vector<Tape>& tapes) {
        Tape best=tapes.front(); Graph champion=graph_from(best),g=champion;
        std::vector<Graph> population;
        for(int iter=0;iter<rounds;iter++){
            if(iter%4==0)g=champion;
            std::shuffle(targets.begin(),targets.end(),rng);
            if(iter%3==0)diversify(g,best.size(),4);
            bool varied=rotate(g,best.size());
            if(!varied)diversify(g,best.size(),5);
            Tape candidate;
            if(!compile(g,candidate)){g=champion;continue;}
            if(candidate.size()<best.size()){champion=g;best=candidate;}
            knockout(g,candidate);
            if(candidate.size()<best.size()){champion=g;best=candidate;}
            population.push_back(g);
            if(population.size()==5){
                Graph winner;
                if(crossover(champion,population,winner,best)){champion=winner;g=champion;}
                population.erase(population.begin(),population.begin()+3);
            }
        }
        return best;
    }
};
#include "main.inc"
