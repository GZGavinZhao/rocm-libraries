
#include <iostream>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visualizer.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

void rrexplorer(rocRoller::KernelGraph::KernelGraph const& original,
                std::string const&                         windowName,
                bool                                       simplify = true)
{
    auto graph = original;
    if(simplify)
    {
        removeRedundantBodyEdges(graph);
        removeRedundantSequenceEdges(graph);
    }
    try
    {
        auto colouring = rocRoller::KernelGraph::colourByUnrollValue(graph);
        rocRoller::KernelGraph::visualize(graph, windowName, colouring);
        return;
    }
    catch(...)
    {
    }
    rocRoller::KernelGraph::visualize(graph, windowName, {});
}

void rrexplorer(std::string yamlFileName)
{
    std::string yaml;
    {
        std::stringstream ss;
        std::ifstream     file;
        file.open(yamlFileName);
        ss << file.rdbuf();
        file.close();
        yaml = ss.str();
    }

    auto graph = rocRoller::KernelGraph::fromYAML(yaml);
    rrexplorer(graph, yamlFileName);
}

void usage()
{
    std::cout << "Usage: rrexplorer <graph.yaml>" << std::endl;
}

int main(int argc, const char* argv[])
{
    if(argc != 2)
    {
        usage();
        return 1;
    }
    rrexplorer(argv[1]);
}
