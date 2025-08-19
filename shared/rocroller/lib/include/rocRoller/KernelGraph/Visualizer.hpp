#pragma once

#include <map>

#include <rocRoller/Utilities/Logging.hpp>

#include <rocRoller/KernelGraph/Colouring.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        void visualize(KernelGraph const&     graph,
                       std::string const&     windowName = "rocRoller Graph",
                       UnrollColouring const& colouring  = {});
    }
}
