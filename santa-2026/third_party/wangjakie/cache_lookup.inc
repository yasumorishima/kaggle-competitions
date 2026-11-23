
struct RecipeLookup {
        struct Cell { Value key=0; int head=-1, count=0; };
        struct Link { Recipe rule; int previous; };
        std::vector<Cell> slots;
        std::vector<Link> links;
        std::vector<size_t> occupied;
        void reset(size_t n) {
            size_t capacity=16;
            while(capacity<3*n*(n+1))capacity*=2;
            if(slots.size()<capacity){slots.assign(capacity,{});}
            else for(size_t i:occupied)slots[i]=Cell{};
            occupied.clear();links.clear();
        }
        size_t locate(Value x)const {
            uint64_t h=static_cast<uint64_t>(x);
            h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;
            size_t i=(h*0x94d049bb133111ebULL>>17)&(slots.size()-1);
            while(slots[i].key&&slots[i].key!=x)i=(i+1)&(slots.size()-1);
            return i;
        }
        void push(Value x,Recipe r){
            size_t i=locate(x);auto& s=slots[i];
            if(!s.key){s.key=x;occupied.push_back(i);}
            links.push_back({r,s.head});s.head=int(links.size())-1;++s.count;
        }
        bool contains(Value x)const{return slots[locate(x)].key!=0;}
        bool sample(Value x,std::mt19937& rng,Recipe& result)const{
            const auto& s=slots[locate(x)];if(!s.count)return false;
            int j=s.head,skip=rng()%s.count;
            while(skip--)j=links[j].previous;
            result=links[j].rule;return true;
        }
        template<class Predicate>
        bool first(Value x,Predicate allowed,Recipe& result)const{
            for(int j=slots[locate(x)].head;j>=0;j=links[j].previous){
                auto r=links[j].rule;
                if(allowed(r)){result=r;return true;}
            }
            return false;
        }
        std::vector<Value> universe;
        std::unordered_set<Value> indexed;
        void prepare(const std::vector<Value>& active,Value limit){
            size_t additions=0;
            for(Value x:active)if(!indexed.count(x))++additions;
            if(universe.empty()||universe.size()+additions>active.size()*5/4+15){
                reset(active.size()*5/4+17);universe.clear();indexed.clear();
            }
            for(Value x:active)if(indexed.insert(x).second){
                universe.push_back(x);
                for(Value other:universe){
                    Value a=std::max(x,other),b=std::min(x,other);
                    if(a<=limit-b)push(a+b,{'+',a,b});
                    if(a>b && a-b!=b)push(a-b,{'-',a,b});
                    if(b>1 && a<=limit/b)push(a*b,{'*',a,b});
                    if(b>1 && a%b==0 && a/b!=b)push(a/b,{'/',a,b});
                }
            }
        }
        template<class Predicate>
        bool sample_active(Value x,std::mt19937& generator,Predicate live,Recipe& answer)const{
            const Cell& cell=slots[locate(x)];
            if(!cell.count)return false;
            int offset=generator()%cell.count;
            int j=cell.head;for(int i=0;i<offset;++i)j=links[j].previous;
            int current=j;
            for(int i=0;i<cell.count;++i){
                auto r=links[current].rule;
                if(live(r.a)&&live(r.b)){answer=r;return true;}
                current=links[current].previous;
                if(current<0)current=cell.head;
            }
            return false;
        }

    } lookup;
