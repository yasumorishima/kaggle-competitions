
#include <cmath>
#include "common.inc"
#include "compact.inc"

    Tape improve(const std::vector<Tape>& tapes) {
        Tape best=tapes.front(); Graph graph=graph_from(best);
        for (int i=0;i<rounds;++i) if(!knockout(graph,best)) break;
        return best;
    }
};
#include "main.inc"
