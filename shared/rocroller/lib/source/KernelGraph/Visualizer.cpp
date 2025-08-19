#include <SDL.h>
#include <SDL_opengl.h>

#include <backends/imgui_impl_opengl2.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>
#include <imgui_node_editor.h>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

namespace ImNE = ax::NodeEditor;

namespace ImGui
{
    void TextUnformatted(std::string const t)
    {
        TextUnformatted(t.c_str());
    }

    ImVec2 CalcTextSize(std::string const t)
    {
        return CalcTextSize(t.c_str());
    }

    bool Combo(std::string const& label, int* selected, std::vector<std::string>& items)
    {
        return ImGui::Combo(
            label.c_str(),
            selected,
            [](void* vec, int idx, const char** out_text) -> bool {
                auto v = reinterpret_cast<std::vector<std::string>*>(vec);
                if(idx < 0 || idx >= v->size())
                    return false;
                *out_text = v->at(idx).c_str();
                return true;
            },
            reinterpret_cast<void*>(&items),
            items.size());
    }
}

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        struct Flags
        {
            bool visible   = true;
            bool collapsed = false;
        };

        using FlagMap = std::map<int, Flags>;

        namespace Layout
        {
            std::map<int, std::pair<float, float>> graphvizGraphLayout(auto const&    graph,
                                                                       FlagMap const& flags  = {},
                                                                       int            xScale = 2,
                                                                       int            yScale = -1)
            {
                std::map<int, std::pair<float, float>> rv;

                char name[32];
                std::snprintf(name, 31, "tmp");

                auto g = agopen(name, Agdirected, nullptr);

                std::map<int, Agnode_t*> nodes;
                std::map<int, Agedge_t*> edges;

                // Add rr-nodes and rr-edges as gv-nodes
                for(auto elem : graph.allElements())
                {
                    bool add = flags.empty() || flags.at(elem).visible;
                    if(add)
                    {
                        std::snprintf(name, 31, "%d", elem);
                        nodes[elem] = agnode(g, name, 1);
                    }
                }
                // Add gv-edges between rr-nodes and rr-edges
                for(auto elem : graph.allElements())
                {
                    auto downstream = graph.getNeighbours(elem, Graph::Direction::Downstream);
                    for(auto down : downstream)
                    {
                        bool add
                            = flags.empty() || (flags.at(elem).visible && flags.at(down).visible);
                        if(add)
                        {
                            std::snprintf(name, 31, "%d", elem * 1000000 + down);
                            agedge(g, nodes[elem], nodes[down], name, 1);
                        }
                    }
                }

                auto c = gvContext();
                gvLayout(c, g, "dot");
                gvRender(c, g, "dot", nullptr);
                for(auto elem : graph.allElements())
                {
                    if(nodes.contains(elem))
                    {
                        auto coord = ND_coord(nodes[elem]);
                        rv[elem]   = {xScale * coord.x, yScale * coord.y};
                    }
                }
                gvFreeLayout(c, g);
                gvFinalize(c);
                agclose(g);

                return rv;
            }

            /**
	     * Compute graph layout.
	     */
            std::map<int, std::pair<float, float>> coordinateGraphLayout(
                rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph const& graph,
                FlagMap const&                                                  flags)
            {
                return graphvizGraphLayout(graph, flags, 2, -2);
            }

            std::map<int, std::pair<float, float>>
                controlGraphLayout(rocRoller::KernelGraph::ControlGraph::ControlGraph const& graph,
                                   FlagMap const&                                            flags)
            {
                return graphvizGraphLayout(graph, flags, 2, -1);
            }
        }

        using GraphColouring = std::map<int, std::map<int, int>>;

        std::vector<ImColor> ColourBrewer2Divergent8 = {
            {140, 81, 10},
            {191, 129, 45},
            {223, 194, 125},
            {246, 232, 195},
            {199, 234, 229},
            {128, 205, 193},
            {53, 151, 143},
            {1, 102, 94},
        };

        std::vector<ImColor> ColourBrewer2Qualitative12 = {
            {166, 206, 227},
            {31, 120, 180},
            {178, 223, 138},
            {51, 160, 44},
            {251, 154, 153},
            {227, 26, 28},
            {253, 191, 111},
            {255, 127, 0},
            {202, 178, 214},
            {106, 61, 154},
            {255, 255, 153},
            {177, 89, 40},
        };

        int colourByUnrollCombo(int& selectedColour, GraphColouring const& colouring)
        {
            std::vector<int>         availableUnrollCoords       = {-1};
            std::vector<std::string> availableUnrollCoordsLabels = {"None"};
            {
                std::set<int> availableUnrollCoordsSet;
                for(auto [tag, colors] : colouring)
                {
                    for(auto [coord, value] : colors)
                        availableUnrollCoordsSet.insert(coord);
                }
                for(auto x : availableUnrollCoordsSet)
                {
                    availableUnrollCoords.push_back(x);
                    availableUnrollCoordsLabels.push_back(std::to_string(x));
                }
            }

            ImGui::TextUnformatted("Colour by: ");
            ImGui::SameLine();
            ImGui::Combo(" unroll coordinate", &selectedColour, availableUnrollCoordsLabels);

            return availableUnrollCoords[selectedColour];
        }

        template <typename T>
        std::string join(std::string const& separator, T const& xs)
        {
            if(xs.empty())
                return "";

            std::vector<int> vxs;
            for(auto x : xs)
                vxs.push_back(x);

            std::string rv;
            for(auto i = 0; i < vxs.size() - 1; ++i)
                rv = rv + std::to_string(vxs[i]) + separator;
            rv = rv + std::to_string(vxs.back());
            return rv;
        }

        struct GraphVizualizer
        {
            enum Attribute
            {
                START, //< Coordinate is tagged as a 'start' coordinate (for path finding).
                END, //< Coordinate is tagged as an 'end' coordinate (for path finding).
                PATH, //< Coordinate is in the path between the 'start' and 'end' coordinates.
                STORAGE, //< Coordinate is a "storage" node
                HARDWARE, //< Coordinate is a "hardware" node
                LOOPISH, //< Coordinate is a "loopish" node

                BODY, //< Control edge is a `Body` edge
                SEQUENCE, //< Control edge is a `Sequence` edge

                CLICKED, //< Node was clicked
            };

            using AttributeMap = std::map<int, std::unordered_set<Attribute>>;
            using ColorMap = std::map<Attribute, std::vector<std::pair<ImNE::StyleColor, ImVec4>>>;

            ColorMap colors;

            // Global
            bool m_showControlGraph    = true;
            bool m_showCoordinateGraph = false;

            // Coordinate graph
            bool m_showDataFlowEdges       = true;
            bool m_showTransformEdges      = true;
            bool m_showDanglingCoordinates = false;
            bool m_showLoadPath            = true;
            bool m_showStorePath           = true;
            int  m_showCoordinateTag       = -1;

            std::vector<int> m_selectedCoordinates, m_startCoordinates, m_endCoordinates,
                m_targetCoordinates;

            std::map<int, std::pair<float, float>> m_coordinateLayout;

            // Control graph
            int m_showOperationTag = -1;

            std::set<int> m_selectedOperations;

            bool m_showDeallocateNodes = false;
            bool m_elideControlEdges   = true;

            int m_highlightOperation  = -1;
            int m_scrolledToOperation = -1;

            FlagMap m_controlFlags;

            std::map<int, std::pair<float, float>> m_operationLayout;
            std::unordered_map<int, int>           m_bodyParents;
            std::set<int>                          m_inLoadPath, m_inStorePath;

            /**
	     * Pre-compute default layouts.
	     */
            void computeLayouts(KernelGraph const& graph)
            {
                m_coordinateLayout = Layout::coordinateGraphLayout(graph.coordinates, {});
                m_operationLayout  = Layout::controlGraphLayout(graph.control, {});

                // XXX
                //m_bodyParents = ControlFlowRWTracer(graph).getBodyParents();

                std::set<int> storageNodes;
                for(auto elem : graph.coordinates.topologicalSort())
                {
                    if(isStorageCoordinate(elem, graph))
                        storageNodes.insert(elem);
                }

                // Precompute m_inLoadPath and m_inStorePath.
                for(auto elem : storageNodes)
                {
                    auto x
                        = findRequiredCoordinates(elem, Graph::Direction::Downstream, graph).second;
                    x.erase(elem);
                    std::copy(
                        x.cbegin(), x.cend(), std::inserter(m_inLoadPath, m_inLoadPath.end()));

                    auto y
                        = findRequiredCoordinates(elem, Graph::Direction::Upstream, graph).second;
                    y.erase(elem);
                    std::copy(
                        y.cbegin(), y.cend(), std::inserter(m_inStorePath, m_inStorePath.end()));
                }
            }

            /**
	     * Render a graph (generic).
	     */
            template <typename T>
            void renderGraph(T const&              graph,
                             FlagMap const&        flags,
                             AttributeMap const&   attributes,
                             GraphColouring const& colouring,
                             int                   showColourOf = -1)
            {
                bool constexpr isControl
                    = std::is_same_v<T, rocRoller::KernelGraph::ControlGraph::ControlGraph>;

                auto getOutputAttrId = [](int elem) { return (1 << 30) + elem; };
                auto getInputAttrId  = [](int elem) { return (1 << 29) + elem; };

                ImGui::PushFont(m_font);

                ImNE::PushStyleVar(ImNE::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
                ImNE::PushStyleVar(ImNE::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));

                for(auto elem : graph.topologicalSort())
                {
                    if(!flags.contains(elem))
                    {
                        std::cout << "WARNING: missing flags " << elem << std::endl;
                        continue;
                    }

                    if(!flags.at(elem).visible)
                        continue;

                    auto type = graph.getElementType(elem);
                    if(type == Graph::ElementType::Node)
                    {
                        auto node = graph.getNode(elem);

                        int styleColourCount = 0;
                        if(attributes.contains(elem))
                        {
                            for(auto attr : attributes.at(elem))
                            {
                                for(auto [colorVar, color] : colors.at(attr))
                                {
                                    ImNE::PushStyleColor(colorVar, color);
                                    ++styleColourCount;
                                }
                            }
                        }
                        if(showColourOf != -1 && colouring.contains(elem))
                        {
                            if(colouring.at(elem).contains(showColourOf))
                            {
                                auto colourVar = ImNE::StyleColor::StyleColor_NodeBg;
                                auto colour    = ColourBrewer2Qualitative12.at(
                                    colouring.at(elem).at(showColourOf));
                                ImNE::PushStyleColor(colourVar, colour);
                                ++styleColourCount;
                            }
                        }

                        ImNE::BeginNode(elem);

                        std::string nodeTextHeader;
                        if constexpr(isControl)
                        {
                            nodeTextHeader = std::format("{}: {}", elem, name(node));
                        }
                        else
                        {
                            auto getStrideNoAssert = [](const auto x) {
                                return std::visit([](const auto a) { return a.stride; }, x);
                            };
                            nodeTextHeader = std::format("{}: {}\nS: {}\nJ: {}",
                                                         elem,
                                                         name(node),
                                                         toString(getSize(node)),
                                                         toString(getStrideNoAssert(node)));
                        }

                        auto nodeTextHeaderSize = ImGui::CalcTextSize(nodeTextHeader);

                        ImNE::BeginPin(getInputAttrId(elem), ImNE::PinKind::Input);
                        ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));
                        ImNE::EndPin();

                        ImGui::TextUnformatted(nodeTextHeader);

                        ImNE::BeginPin(getOutputAttrId(elem), ImNE::PinKind::Output);
                        ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));
                        ImNE::EndPin();

                        ImNE::EndNode();
                        ImNE::PopStyleColor(styleColourCount);
                    }
                    else if(type == Graph::ElementType::Edge)
                    {
                        auto edge = graph.getEdge(elem);
                        if constexpr(isControl)
                        {
                            auto isBody     = graph.template get<Body>(elem).has_value();
                            auto isSequence = graph.template get<Sequence>(elem).has_value();

                            if((isBody || isSequence) && m_elideControlEdges)
                                continue;
                        }

                        ImNE::PushStyleVar(ImNE::StyleVar_NodeRounding, 0.f);

                        std::string nodeTextHeader = std::format("{}: {}", elem, toString(edge));
                        if constexpr(!isControl)
                        {
                            auto inOrder = graph.getNeighbours(elem, Graph::Direction::Upstream)
                                               .template to<std::vector>();
                            auto outOrder = graph.getNeighbours(elem, Graph::Direction::Downstream)
                                                .template to<std::vector>();
                            if(!inOrder.empty())
                            {
                                nodeTextHeader += "\nInputs: ";
                                for(auto i = 0; i < inOrder.size(); ++i)
                                {
                                    nodeTextHeader += std::format(
                                        "{}{}", inOrder[i], (i == inOrder.size() - 1) ? "" : ", ");
                                }
                            }
                            if(!inOrder.empty())
                            {
                                nodeTextHeader += "\nOutputs: ";
                                for(auto i = 0; i < outOrder.size(); ++i)
                                {
                                    nodeTextHeader
                                        += std::format("{}{}",
                                                       outOrder[i],
                                                       (i == outOrder.size() - 1) ? "" : ", ");
                                }
                            }
                        }

                        auto nodeTextHeaderSize = ImGui::CalcTextSize(nodeTextHeader);

                        ImNE::BeginNode(elem);

                        ImNE::BeginPin(getInputAttrId(elem), ImNE::PinKind::Input);
                        ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));

                        ImNE::EndPin();

                        ImGui::TextUnformatted(nodeTextHeader);

                        ImNE::BeginPin(getOutputAttrId(elem), ImNE::PinKind::Output);
                        ImGui::Dummy(ImVec2(nodeTextHeaderSize.x, 1));

                        ImNE::EndPin();

                        ImNE::EndNode();

                        ImNE::PopStyleVar(1);
                    }
                }

                ImNE::PopStyleVar(2);
                ImGui::PopFont();

                int linkId = 0;
                for(auto elem : graph.getEdges())
                {
                    if(!flags.contains(elem))
                    {
                        std::cout << "WARNING: missing flags " << elem << std::endl;
                        continue;
                    }

                    if(!flags.at(elem).visible)
                        continue;

                    bool isBody     = false;
                    bool isSequence = false;
                    if constexpr(std::is_same_v<T,
                                                rocRoller::KernelGraph::ControlGraph::ControlGraph>)
                    {
                        isBody     = graph.template get<Body>(elem).has_value();
                        isSequence = graph.template get<Sequence>(elem).has_value();
                    }

                    bool elide = (isSequence || isBody) && m_elideControlEdges;
                    auto color = ImColor(255, 255, 255);

                    if(isBody)
                        color = colors.at(Attribute::BODY)[0].second;
                    if(isSequence)
                        color = colors.at(Attribute::SEQUENCE)[0].second;

                    if(!elide)
                    {
                        for(auto input : graph.getNeighbours(elem, Graph::Direction::Upstream))
                        {

                            ImNE::Link(
                                linkId++, getOutputAttrId(input), getInputAttrId(elem), color);
                        }
                        for(auto output : graph.getNeighbours(elem, Graph::Direction::Downstream))
                        {
                            ImNE::Link(
                                linkId++, getOutputAttrId(elem), getInputAttrId(output), color);
                        }
                    }
                    else
                    {
                        auto parent = *only(graph.getNeighbours(elem, Graph::Direction::Upstream));
                        auto child = *only(graph.getNeighbours(elem, Graph::Direction::Downstream));
                        ImNE::Link(linkId++, getOutputAttrId(parent), getInputAttrId(child), color);
                    }
                }
            }

            /**
	     * Render the coordinate graph.
	     */
            void renderCoordinateGraph(KernelGraph const& graph, GraphColouring const& colouring)
            {
                ImGui::Begin("CoordinateTransform graph");

                //
                // Controls
                //

                auto reflow = ImGui::Button("Reflow");
                ImGui::SameLine();
                ImGui::Checkbox("DataFlow", &m_showDataFlowEdges);
                ImGui::SameLine();
                ImGui::Checkbox("CoordinateTransform", &m_showTransformEdges);
                ImGui::SameLine();
                ImGui::Checkbox("Show danglers", &m_showDanglingCoordinates);
                ImGui::SameLine();
                ImGui::Checkbox("Show load path", &m_showLoadPath);
                ImGui::SameLine();
                ImGui::Checkbox("Show store path", &m_showStorePath);

                const int   inputLength = 32;
                static char showTagInput[inputLength];
                if(ImGui::Button("Move to tag: "))
                    m_showCoordinateTag = std::stoi(showTagInput);
                else
                    m_showCoordinateTag = -1;
                ImGui::SameLine();
                ImGui::InputText("", showTagInput, inputLength);

                ImGui::TextUnformatted("Selected tags: ");
                ImGui::SameLine();
                ImGui::TextUnformatted(join(", ", m_selectedCoordinates));

                if(ImGui::Button("Start tags: "))
                {
                    m_startCoordinates = m_selectedCoordinates;
                    m_targetCoordinates.clear();
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(join(", ", m_startCoordinates));

                if(ImGui::Button("End tags:   "))
                {
                    m_endCoordinates = m_selectedCoordinates;
                    m_targetCoordinates.clear();
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(join(", ", m_endCoordinates));

                if(ImGui::Button("Required:   "))
                {
                    m_targetCoordinates = m_selectedCoordinates;
                    m_startCoordinates.clear();
                    m_endCoordinates.clear();
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(join(", ", m_targetCoordinates));

                //
                // Compute path
                //

                std::unordered_set<int> path;
                if(!m_startCoordinates.empty() && !m_endCoordinates.empty())
                {
                    path = graph.coordinates
                               .path<Graph::Direction::Downstream>(m_startCoordinates,
                                                                   m_endCoordinates)
                               .to<std::unordered_set>();

                    if(path.empty())
                    {
                        path = graph.coordinates
                                   .path<Graph::Direction::Upstream>(m_startCoordinates,
                                                                     m_endCoordinates)
                                   .to<std::unordered_set>();
                    }
                }

                if(!m_targetCoordinates.empty())
                {
                    auto store = findRequiredCoordinates(
                        m_targetCoordinates.back(), Graph::Direction::Upstream, graph);
                    auto load = findRequiredCoordinates(
                        m_targetCoordinates.back(), Graph::Direction::Downstream, graph);

                    path = store.second;
                    std::copy(
                        load.second.cbegin(), load.second.cend(), std::inserter(path, path.end()));
                }

                //
                // Compute visible
                //
                AttributeMap attributes;

                FlagMap flags;
                for(auto elem : graph.coordinates.topologicalSort())
                {
                    auto type = graph.coordinates.getElementType(elem);
                    if(type == Graph::ElementType::Node)
                    {
                        flags[elem].visible = true;

                        if(!m_showLoadPath && m_inLoadPath.contains(elem)
                           && !m_inStorePath.contains(elem))
                            flags[elem].visible = false;

                        if(!m_showStorePath && !m_inLoadPath.contains(elem)
                           && m_inStorePath.contains(elem))
                            flags[elem].visible = false;

                        if(isStorageCoordinate(elem, graph))
                            attributes[elem].insert(Attribute::STORAGE);
                        if(isHardwareCoordinate(elem, graph))
                            attributes[elem].insert(Attribute::HARDWARE);
                        if(isLoopishCoordinate(elem, graph))
                            attributes[elem].insert(Attribute::LOOPISH);
                    }
                }
                for(auto elem : graph.coordinates.getEdges())
                {
                    auto type = graph.coordinates.getElementType(elem);
                    if(type == Graph::ElementType::Edge)
                    {
                        flags[elem].visible = false;

                        auto edge = graph.coordinates.getEdge(elem);
                        flags[elem].visible |= m_showDataFlowEdges && isEdge<DataFlowEdge>(edge);
                        flags[elem].visible
                            |= m_showTransformEdges && isEdge<CoordinateTransformEdge>(edge);

                        if(flags[elem].visible)
                        {
                            bool missingUpstream = true;
                            for(auto x :
                                graph.coordinates.getNeighbours(elem, Graph::Direction::Upstream))
                            {
                                if(flags[x].visible)
                                    missingUpstream = false;
                            }
                            bool missingDownstream = true;
                            for(auto x :
                                graph.coordinates.getNeighbours(elem, Graph::Direction::Downstream))
                            {
                                if(flags[x].visible)
                                    missingDownstream = false;
                            }
                            if(missingDownstream || missingUpstream)
                                flags[elem].visible = false;
                        }
                    }
                }

                if(!m_showDanglingCoordinates)
                {
                    for(auto elem : graph.coordinates.topologicalSort())
                    {
                        auto type = graph.coordinates.getElementType(elem);
                        if(type == Graph::ElementType::Node)
                        {
                            if(!flags[elem].visible)
                                continue;

                            bool visible = false;
                            for(auto x :
                                graph.coordinates.getNeighbours(elem, Graph::Direction::Upstream))
                            {
                                if(flags[x].visible)
                                    visible = true;
                            }
                            for(auto x :
                                graph.coordinates.getNeighbours(elem, Graph::Direction::Downstream))
                            {
                                if(flags[x].visible)
                                    visible = true;
                            }
                            flags[elem].visible = visible;
                        }
                    }
                }

                for(auto elem : path)
                    attributes[elem].insert(Attribute::PATH);

                ImNE::SetCurrentEditor(m_coordinateContext);
                ImNE::Begin("Coordinate Graph");
                renderGraph(graph.coordinates, flags, attributes, colouring);

                if(reflow)
                {
                    m_coordinateLayout = Layout::coordinateGraphLayout(graph.coordinates, flags);
                    for(auto [elem, xy] : m_coordinateLayout)
                    {
                        if(flags[elem].visible)
                            ImNE::SetNodePosition(elem, {xy.first, xy.second});
                    }
                }

                if(m_showCoordinateTag > 0)
                {
                    ImNE::SelectNode(m_showCoordinateTag);
                    ImNE::NavigateToSelection();
                }

                int selectedObjectCount = ImNE::GetSelectedObjectCount();
                if(selectedObjectCount > 0)
                {
                    std::vector<ImNE::NodeId> selectedNodes(selectedObjectCount);

                    auto numSelectedNodes
                        = ImNE::GetSelectedNodes(selectedNodes.data(), selectedObjectCount);
                    m_selectedCoordinates.resize(numSelectedNodes);
                    for(auto i = 0; i < numSelectedNodes; ++i)
                    {
                        m_selectedCoordinates[i] = selectedNodes[i].Get();
                    }
                }
                else
                {
                    m_selectedCoordinates.clear();
                }

                ImNE::End();
                ImNE::SetCurrentEditor(nullptr);

                ImGui::End();
            }

            /**
	     * Render control graph.
	     */
            void renderControlGraph(KernelGraph const& graph, GraphColouring const& colouring)
            {
                ImGui::Begin("ControlFlow graph");

                //
                // Controls
                //

                auto reflow = ImGui::Button("Reflow");

                ImGui::SameLine();
                if(ImGui::Button("Uncollapse all"))
                {
                    for(auto& [tag, flags] : m_controlFlags)
                        flags.collapsed = false;
                }

                ImGui::SameLine();
                ImGui::Checkbox("Show Deallocate nodes", &m_showDeallocateNodes);

                ImGui::SameLine();
                ImGui::Checkbox("Elide edges", &m_elideControlEdges);

                ImGui::SameLine();
                ImGui::TextUnformatted("Selected tags: ");
                ImGui::SameLine();
                ImGui::TextUnformatted(join(", ", m_selectedOperations));

                static int selectedColour = 0;
                auto       showColourOf   = colourByUnrollCombo(selectedColour, colouring);

                // move this to speedbar
                const int   inputLength = 32;
                static char showTagInput[inputLength];
                if(ImGui::Button("Move to tag: "))
                    m_showOperationTag = std::stoi(showTagInput);
                ImGui::SameLine();
                ImGui::InputText("", showTagInput, inputLength);

                //
                // Visibility
                //

                // First pass: default visible, maybe hide Deallocate (nodes).
                for(auto elem : graph.control.topologicalSort())
                {
                    m_controlFlags[elem].visible = true;

                    auto isDeallocate = graph.control.get<Deallocate>(elem).has_value();
                    if(!m_showDeallocateNodes && isDeallocate)
                        m_controlFlags[elem].visible = false;
                }

                // Second pass: collapse bodies (nodes).
                for(auto elem : graph.control.topologicalSort())
                {
                    auto type = graph.control.getElementType(elem);
                    if(type != Graph::ElementType::Node)
                        continue;

                    auto target = elem;
                    while(true)
                    {
                        if(m_controlFlags[m_bodyParents[target]].collapsed)
                        {
                            m_controlFlags[elem].visible = false;
                            break;
                        }
                        auto parent = m_bodyParents[target];
                        if(m_bodyParents.contains(parent))
                        {
                            target = parent;
                            continue;
                        }
                        break;
                    }
                }

                // Third pass: hide dangling edges (edges).
                for(auto elem : graph.control.topologicalSort())
                {
                    auto type = graph.control.getElementType(elem);
                    if(type != Graph::ElementType::Edge)
                        continue;

                    bool show = false;
                    for(auto x : graph.control.getNeighbours(elem, Graph::Direction::Upstream))
                    {
                        if(m_controlFlags[x].visible)
                            show = true;
                    }
                    bool hasDownstream = false;
                    for(auto x : graph.control.getNeighbours(elem, Graph::Direction::Downstream))

                    {
                        if(m_controlFlags[x].visible)
                        {
                            hasDownstream = true;
                            show          = true;
                        }
                    }

                    m_controlFlags[elem].visible = show && hasDownstream;
                }

                //
                // Draw
                //

                AttributeMap attributes;
                ImNE::SetCurrentEditor(m_controlContext);
                ImNE::Begin("Control Flow Graph");
                renderGraph(graph.control, m_controlFlags, attributes, colouring, showColourOf);

                if(reflow)
                {
                    m_operationLayout = Layout::controlGraphLayout(graph.control, m_controlFlags);
                    for(auto [elem, xy] : m_operationLayout)
                    {
                        if(m_controlFlags[elem].visible)
                            ImNE::SetNodePosition(elem, {xy.first, xy.second});
                    }
                }

                if(m_showOperationTag > 0)
                {
                    ImNE::SelectNode(m_showOperationTag);
                    ImNE::NavigateToSelection();
                }
                m_showOperationTag = -1;

                int selectedObjectCount = ImNE::GetSelectedObjectCount();
                if(selectedObjectCount > 0)
                {
                    std::vector<ImNE::NodeId> selectedNodes(selectedObjectCount);

                    auto numSelectedNodes
                        = ImNE::GetSelectedNodes(selectedNodes.data(), selectedObjectCount);
                    m_selectedOperations.clear();
                    for(auto i = 0; i < numSelectedNodes; ++i)
                    {
                        m_selectedOperations.insert(selectedNodes[i].Get());
                    }
                }
                else
                {
                    m_selectedOperations.clear();
                }

                int doubleClickedOperation = ImNE::GetDoubleClickedNode().Get();
                if(doubleClickedOperation > 0)
                {
                    m_highlightOperation = doubleClickedOperation;
                }

                ImNE::End();
                ImNE::SetCurrentEditor(nullptr);

                ImGui::End();

                //
                // Speedbar
                //

                ImGui::Begin("Operations");
                auto operationTags = graph.control.getNodes().to<std::set>();
                for(auto elem : operationTags)
                {
                    auto node = graph.control.getNode(elem);

                    std::string label;
                    {
                        std::stringstream ss;
                        ss << std::setw(6) << elem << ": " << name(node);
                        label = ss.str();
                    }

                    ImGui::PushID(elem);
                    ImGui::Checkbox("", &m_controlFlags[elem].collapsed);

                    bool highlight = m_selectedOperations.contains(elem);
                    if(highlight)
                        ImGui::PushStyleColor(ImGuiCol_Text, colors[CLICKED][0].second);

                    ImGui::SameLine();
                    ImGui::TextUnformatted(label);

                    if(ImGui::IsItemHovered())
                    {
                        if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_G)))
                            m_showOperationTag = elem;

                        if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_U)))
                            if(m_bodyParents.contains(elem))
                            {
                                m_showOperationTag   = m_bodyParents[elem];
                                m_highlightOperation = m_bodyParents[elem];
                            }

                        if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_C)))
                            m_controlFlags[elem].collapsed = !m_controlFlags[elem].collapsed;
                    }

                    if(highlight)
                        ImGui::PopStyleColor();

                    if(elem == m_highlightOperation && elem != m_scrolledToOperation)
                    {
                        m_scrolledToOperation = elem;
                        ImGui::SetScrollHereY();
                    }

                    ImGui::PopID();
                }
                ImGui::End();
            }

            /**
	     * Render the full GUI.
	     *
	     * Backend agnostic.
	     */
            void renderGUI(KernelGraph const& graph, UnrollColouring const& colouring)
            {
                //
                // Global controls
                //
                ImGui::Begin("Controls");
                ImGui::Checkbox("Show ControlFlow graph", &m_showControlGraph);
                ImGui::Checkbox("Show CoordinateTransform graph", &m_showCoordinateGraph);
                ImGui::End();

                //
                // Graphs
                //
                if(m_showCoordinateGraph)
                    renderCoordinateGraph(graph, colouring.coordinateColour);

                if(m_showControlGraph)
                    renderControlGraph(graph, colouring.operationColour);
            }

            /**
	     * Setup colours.
	     *
	     * Backend agnostic.
	     */
            void setupTheme()
            {
                colors[GraphVizualizer::Attribute::STORAGE]
                    = {{ImNE::StyleColor_NodeBg, ImColor(200, 200, 20)}};

                colors[GraphVizualizer::Attribute::HARDWARE]
                    = {{ImNE::StyleColor_NodeBg, ImColor(20, 200, 20)}};

                colors[GraphVizualizer::Attribute::LOOPISH]
                    = {{ImNE::StyleColor_NodeBg, ImColor(20, 20, 200)}};

                colors[GraphVizualizer::Attribute::PATH]
                    = {{ImNE::StyleColor_NodeBg, ImColor(255, 0, 0)}};

                colors[GraphVizualizer::Attribute::BODY]
                    = {{ImNE::StyleColor_Count, ImColor(255, 0, 0)}};

                colors[GraphVizualizer::Attribute::SEQUENCE]
                    = {{ImNE::StyleColor_Count, ImColor(128, 255, 128)}};

                colors[GraphVizualizer::Attribute::CLICKED]
                    = {{ImNE::StyleColor_Count, ImColor(255, 176, 50, 255)}};
            }

            /**
	     * Setup the GUI.
	     *
	     * Specific to SDL2 + OpenGL2.
	     */
            void setupGUI(std::string const& windowName)
            {
                if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
                    return;

                SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
                SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
                SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
                SDL_WindowFlags window_flags
                    = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
                                        | SDL_WINDOW_ALLOW_HIGHDPI);

                m_window = SDL_CreateWindow(windowName.c_str(),
                                            SDL_WINDOWPOS_CENTERED,
                                            SDL_WINDOWPOS_CENTERED,
                                            1280,
                                            720,
                                            window_flags);
                if(m_window == nullptr)
                {
                    Log::warn("Error: SDL_CreateWindow(): {}", SDL_GetError());
                    return;
                }

                m_gl_context = SDL_GL_CreateContext(m_window);
                SDL_GL_MakeCurrent(m_window, m_gl_context);
                SDL_GL_SetSwapInterval(1);

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                ImGuiIO& io    = ImGui::GetIO();
                io.IniFilename = "rr_graph.ini";
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

                ImGui_ImplSDL2_InitForOpenGL(m_window, m_gl_context);
                ImGui_ImplOpenGL2_Init();

                // Generic
                m_coordinateConfig.SelectButtonIndex   = 0;
                m_coordinateConfig.DragButtonIndex     = 0;
                m_coordinateConfig.NavigateButtonIndex = 1;
                m_coordinateConfig.EnableSmoothZoom    = true;
                m_coordinateConfig.SettingsFile        = "rr_graph_coordinates.json";

                m_controlConfig.SelectButtonIndex   = 0;
                m_controlConfig.DragButtonIndex     = 0;
                m_controlConfig.NavigateButtonIndex = 1;
                m_controlConfig.EnableSmoothZoom    = true;
                m_controlConfig.SettingsFile        = "rr_graph_control.json";

                m_coordinateContext = ImNE::CreateEditor(&m_coordinateConfig);
                m_controlContext    = ImNE::CreateEditor(&m_controlConfig);

                io.Fonts->AddFontDefault();
                //                m_font = io.Fonts->AddFontFromFileTTF("inconsolata.ttf", 96);
                m_font = io.Fonts->AddFontFromFileTTF(
                    "/usr/share/texlive/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans.ttf",
                    18);
            }

            /**
	     * GUI main loop.
	     *
	     * Specific to SDL2 + OpenGL2.
	     */
            void renderLoop(KernelGraph const& graph, UnrollColouring const& colouring)
            {
                ImGuiIO& io = ImGui::GetIO();

                bool done = false;
                while(!done)
                {
                    SDL_Event event;
                    while(SDL_PollEvent(&event))
                    {
                        ImGui_ImplSDL2_ProcessEvent(&event);
                        if(event.type == SDL_QUIT)
                            done = true;
                        if(event.type == SDL_WINDOWEVENT
                           && event.window.event == SDL_WINDOWEVENT_CLOSE
                           && event.window.windowID == SDL_GetWindowID(m_window))
                            done = true;
                    }

                    ImGui_ImplOpenGL2_NewFrame();
                    ImGui_ImplSDL2_NewFrame();
                    ImGui::NewFrame();

                    renderGUI(graph, colouring);

                    ImGui::Render();
                    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
                    glClear(GL_COLOR_BUFFER_BIT);
                    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
                    SDL_GL_SwapWindow(m_window);
                }
            }

            /**
	     * Shutdown GUI.
	     *
	     * Specific to SDL2 + OpenGL2.
	     */
            void shutdownGUI()
            {
                // Generic
                ImNE::DestroyEditor(m_coordinateContext);
                ImNE::DestroyEditor(m_controlContext);

                // Cleanup; backend specific
                ImGui_ImplOpenGL2_Shutdown();
                ImGui_ImplSDL2_Shutdown();
                ImGui::DestroyContext();

                SDL_GL_DeleteContext(m_gl_context);
                SDL_DestroyWindow(m_window);
                SDL_Quit();
            }

            ImNE::Config         m_coordinateConfig;
            ImNE::EditorContext* m_coordinateContext;

            ImNE::Config         m_controlConfig;
            ImNE::EditorContext* m_controlContext;

            SDL_Window*   m_window;
            SDL_GLContext m_gl_context;

            ImFont* m_font;
        };

        // Might want to allow this to accept arguments so that
        // transforms can highlight nodes etc
#ifdef ROCROLLER_ENABLE_IMGUI
        void visualize(KernelGraph const&     graph,
                       std::string const&     windowName,
                       UnrollColouring const& colouring)
        {
            AssertFatal(sizeof(ImDrawIdx) >= 4,
                        "ImGUI configuration mismatch: ImDrawIdx should be at least 32bit for "
                        "rocRoller graphs.");
            GraphVizualizer viz;
            viz.setupGUI(windowName);
            viz.setupTheme();
            viz.computeLayouts(graph);
            viz.renderLoop(graph, colouring);
            viz.shutdownGUI();
        }
#else
        void visualize(KernelGraph const&     graph,
                       std::string const&     windowName,
                       UnrollColouring const& colouring)
        {
            Log::warn("rocRoller::KernelGraph::visualize called but ImGUI disabled.");
        }
#endif
    }
}
