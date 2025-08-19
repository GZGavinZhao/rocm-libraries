#pragma once

#include <map>
#include <set>

namespace rocRoller
{
    namespace KernelGraph
    {
        // Return value of colourByUnrollValue.  A colour-mapping is...
        struct UnrollColouring
        {
            std::map<int, std::map<int, int>>
                operationColour; //< Mapping: operation tag to colour-mapping.
            std::map<int, std::map<int, int>>
                          coordinateColour; //< Mapping: coordinate tag to colour-mapping.
            std::set<int> separators; //< Separator edges in the control graph
        };
    }
}
