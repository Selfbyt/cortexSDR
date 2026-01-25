/**
 * @file execution_graph.hpp
 * @brief Computational graph for proper layer execution ordering
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stdexcept>

namespace CortexAICompression {
namespace Utils {

/**
 * @brief Graph node representing a single operation
 */
struct GraphNode {
    std::string name;                           // Node identifier
    std::string op_type;                        // Operation type
    std::vector<std::string> inputs;            // Input node names
    std::vector<std::string> outputs;           // Output node names (usually one)
    std::unordered_map<std::string, std::string> attributes;  // Node attributes
    
    GraphNode(const std::string& n, const std::string& op)
        : name(n), op_type(op) {}
};

/**
 * @brief Computational graph with topological sorting
 */
class ExecutionGraph {
public:
    ExecutionGraph() = default;
    
    /**
     * @brief Add a node to the graph
     */
    void add_node(const GraphNode& node) {
        if (nodes_.find(node.name) != nodes_.end()) {
            throw std::runtime_error("Node already exists: " + node.name);
        }
        nodes_[node.name] = std::make_shared<GraphNode>(node);
        invalidate_order();
    }
    
    /**
     * @brief Add an edge from source to destination
     */
    void add_edge(const std::string& from, const std::string& to) {
        if (nodes_.find(from) == nodes_.end()) {
            throw std::runtime_error("Source node not found: " + from);
        }
        if (nodes_.find(to) == nodes_.end()) {
            throw std::runtime_error("Destination node not found: " + to);
        }
        
        adjacency_list_[from].push_back(to);
        reverse_adjacency_[to].push_back(from);
        invalidate_order();
    }
    
    /**
     * @brief Get topologically sorted execution order
     */
    std::vector<std::string> get_execution_order() {
        if (execution_order_valid_) {
            return execution_order_;
        }
        
        execution_order_ = topological_sort();
        execution_order_valid_ = true;
        return execution_order_;
    }
    
    /**
     * @brief Get node by name
     */
    std::shared_ptr<GraphNode> get_node(const std::string& name) const {
        auto it = nodes_.find(name);
        if (it == nodes_.end()) {
            return nullptr;
        }
        return it->second;
    }
    
    /**
     * @brief Get all input nodes (no predecessors)
     */
    std::vector<std::string> get_input_nodes() const {
        std::vector<std::string> inputs;
        for (const auto& pair : nodes_) {
            if (reverse_adjacency_.find(pair.first) == reverse_adjacency_.end() ||
                reverse_adjacency_.at(pair.first).empty()) {
                inputs.push_back(pair.first);
            }
        }
        return inputs;
    }
    
    /**
     * @brief Get all output nodes (no successors)
     */
    std::vector<std::string> get_output_nodes() const {
        std::vector<std::string> outputs;
        for (const auto& pair : nodes_) {
            if (adjacency_list_.find(pair.first) == adjacency_list_.end() ||
                adjacency_list_.at(pair.first).empty()) {
                outputs.push_back(pair.first);
            }
        }
        return outputs;
    }
    
    /**
     * @brief Get predecessors of a node
     */
    std::vector<std::string> get_predecessors(const std::string& node) const {
        auto it = reverse_adjacency_.find(node);
        if (it == reverse_adjacency_.end()) {
            return {};
        }
        return it->second;
    }
    
    /**
     * @brief Get successors of a node
     */
    std::vector<std::string> get_successors(const std::string& node) const {
        auto it = adjacency_list_.find(node);
        if (it == adjacency_list_.end()) {
            return {};
        }
        return it->second;
    }
    
    /**
     * @brief Check if graph has cycles
     */
    bool has_cycle() const {
        std::unordered_set<std::string> visited;
        std::unordered_set<std::string> rec_stack;
        
        for (const auto& pair : nodes_) {
            if (has_cycle_util(pair.first, visited, rec_stack)) {
                return true;
            }
        }
        return false;
    }
    
private:
    std::unordered_map<std::string, std::shared_ptr<GraphNode>> nodes_;
    std::unordered_map<std::string, std::vector<std::string>> adjacency_list_;
    std::unordered_map<std::string, std::vector<std::string>> reverse_adjacency_;
    
    std::vector<std::string> execution_order_;
    bool execution_order_valid_ = false;
    
    void invalidate_order() {
        execution_order_valid_ = false;
    }
    
    /**
     * @brief Kahn's algorithm for topological sorting
     */
    std::vector<std::string> topological_sort() const {
        // Compute in-degrees
        std::unordered_map<std::string, size_t> in_degree;
        for (const auto& pair : nodes_) {
            in_degree[pair.first] = 0;
        }
        for (const auto& pair : adjacency_list_) {
            for (const auto& neighbor : pair.second) {
                in_degree[neighbor]++;
            }
        }
        
        // Queue of nodes with no incoming edges
        std::vector<std::string> queue;
        for (const auto& pair : in_degree) {
            if (pair.second == 0) {
                queue.push_back(pair.first);
            }
        }
        
        std::vector<std::string> result;
        
        while (!queue.empty()) {
            std::string node = queue.back();
            queue.pop_back();
            result.push_back(node);
            
            // Reduce in-degree for neighbors
            auto it = adjacency_list_.find(node);
            if (it != adjacency_list_.end()) {
                for (const auto& neighbor : it->second) {
                    in_degree[neighbor]--;
                    if (in_degree[neighbor] == 0) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        if (result.size() != nodes_.size()) {
            throw std::runtime_error("Graph has cycles - cannot compute execution order");
        }
        
        return result;
    }
    
    bool has_cycle_util(
        const std::string& node,
        std::unordered_set<std::string>& visited,
        std::unordered_set<std::string>& rec_stack
    ) const {
        if (rec_stack.find(node) != rec_stack.end()) {
            return true;  // Back edge found
        }
        if (visited.find(node) != visited.end()) {
            return false;  // Already processed
        }
        
        visited.insert(node);
        rec_stack.insert(node);
        
        auto it = adjacency_list_.find(node);
        if (it != adjacency_list_.end()) {
            for (const auto& neighbor : it->second) {
                if (has_cycle_util(neighbor, visited, rec_stack)) {
                    return true;
                }
            }
        }
        
        rec_stack.erase(node);
        return false;
    }
};

/**
 * @brief Helper to build execution graph from layer metadata
 */
class GraphBuilder {
public:
    static ExecutionGraph build_from_layers(
        const std::vector<std::string>& layer_names,
        const std::unordered_map<std::string, std::vector<std::string>>& dependencies
    ) {
        ExecutionGraph graph;
        
        // Add all nodes
        for (const auto& name : layer_names) {
            GraphNode node(name, "layer");
            graph.add_node(node);
        }
        
        // Add edges based on dependencies
        for (const auto& pair : dependencies) {
            const std::string& to = pair.first;
            for (const std::string& from : pair.second) {
                graph.add_edge(from, to);
            }
        }
        
        return graph;
    }
};

} // namespace Utils
} // namespace CortexAICompression
