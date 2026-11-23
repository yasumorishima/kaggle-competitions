#pragma once

// Included after Step, Script, Graph and emit_graph are defined. Replacing
// references by numeric value lets new and outside consumers share a write.
namespace numeric_splice {

Graph rewrite(const Graph& before,const Script& patch,Value limit){
    std::unordered_map<Value,Step> definitions;
    definitions.reserve(before.nodes.size()+patch.size());
    for(auto n:before.nodes)if(n.op){
        definitions.try_emplace(n.value,Step{n.value,n.op,before.nodes[n.a].value,before.nodes[n.b].value});
    }
    for(const auto& s:patch){
        if(!s.x||!s.a||!s.b||s.a>limit||s.b>limit||s.x==s.a||s.x==s.b||calculate(s.op,s.a,s.b,limit)!=s.x)
            return {};
        definitions[s.x]=s;
    }
    Graph g;
    g.initial.fill(-1);
    Index installed;
    std::unordered_set<Value> pending;
    for(int r=0;r<slots;++r)if(before.initial[r]>=0){
        const Value v=before.nodes[before.initial[r]].value;
        const auto [it,new_value]=installed.emplace(v,(int)g.nodes.size());
        if(new_value)g.nodes.push_back({v,0,-1,-1});
        g.initial[r]=it->second;
    }
    auto emit=[&](auto& self,Value v)->int{
        if(auto it=installed.find(v);it!=installed.end())return it->second;
        if(!pending.insert(v).second)return -1;
        const auto it=definitions.find(v);
        if(it==definitions.end())return -1;
        const auto s=it->second;
        int a=self(self,s.a),b=self(self,s.b);
        if(a<0||b<0)return -1;
        int id=(int)g.nodes.size();
        g.nodes.push_back({v,s.op,a,b});installed.emplace(v,id);pending.erase(v);
        return id;
    };
    for(int r:before.targets){
        int id=emit(emit,before.nodes[r].value);
        if(id<0)return {};
        g.targets.push_back(id);
    }
    return g;
}

Program apply(const Graph& original,const Script& patch,Value limit,int cap,int trials){
    if(patch.empty())return {};
    const auto g=rewrite(original,patch,limit);
    if(g.targets.size()!=original.targets.size())return {};
    return emit_graph(g,limit,cap,trials);
}
}
