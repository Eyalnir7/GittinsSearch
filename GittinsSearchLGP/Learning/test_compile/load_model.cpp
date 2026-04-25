#include <torch/script.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <vector>

struct InputVariation {
    int num_ssBox;
    int num_object;
    int num_place_frame;
    int num_ssCylinder;
    int num_pick;
    int num_place;
    int num_edges_pick;
    int num_edges_place;
    int num_edges_close;
    int num_edges_time;
};

InputVariation generateRandomVariation(std::mt19937& gen) {
    std::uniform_int_distribution<> node_dist(2, 20);  // 2-20 nodes
    std::uniform_int_distribution<> action_dist(1, 10);  // 1-10 actions
    std::uniform_int_distribution<> edge_dist(0, 10);  // 0-10 edges
    
    InputVariation var;
    var.num_ssBox = node_dist(gen);
    var.num_object = node_dist(gen);
    var.num_place_frame = node_dist(gen);
    var.num_ssCylinder = node_dist(gen);
    var.num_pick = action_dist(gen);
    var.num_place = action_dist(gen);
    var.num_edges_pick = edge_dist(gen);
    var.num_edges_place = edge_dist(gen);
    var.num_edges_close = edge_dist(gen);
    var.num_edges_time = edge_dist(gen);
    
    return var;
}

void testModelForwardWithVariation(torch::jit::script::Module& module, torch::Device device, 
                                   const InputVariation& var, int run_number, const std::string& model_name) {
    torch::NoGradGuard no_grad;
    
    // Create heterogeneous graph input dictionaries
    torch::Dict<std::string, torch::Tensor> x_dict, times_dict, batch_dict, edge_index_dict;
    
    std::cout << "\n[" << model_name << "] Run " << run_number << " - ";
    std::cout << "Nodes: " << (var.num_ssBox + var.num_object + var.num_place_frame + 
                                var.num_ssCylinder + var.num_pick + var.num_place)
              << ", Edges: " << (var.num_edges_pick + var.num_edges_place + 
                                 var.num_edges_close + var.num_edges_time) << std::endl;
    
    // Create node features (using random values to simulate real data)
    if (var.num_ssBox > 0) {
        x_dict.insert("ssBox", torch::rand({var.num_ssBox, 4}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        times_dict.insert("ssBox", torch::randint(0, 10, {var.num_ssBox}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("ssBox", torch::zeros({var.num_ssBox}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    if (var.num_object > 0) {
        x_dict.insert("object", torch::rand({var.num_object, 4}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        times_dict.insert("object", torch::randint(0, 10, {var.num_object}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("object", torch::zeros({var.num_object}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    if (var.num_place_frame > 0) {
        x_dict.insert("place_frame", torch::rand({var.num_place_frame, 4}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        times_dict.insert("place_frame", torch::randint(0, 10, {var.num_place_frame}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("place_frame", torch::zeros({var.num_place_frame}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    if (var.num_ssCylinder > 0) {
        x_dict.insert("ssCylinder", torch::rand({var.num_ssCylinder, 3}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(device)));
        times_dict.insert("ssCylinder", torch::randint(0, 10, {var.num_ssCylinder}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("ssCylinder", torch::zeros({var.num_ssCylinder}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    // Create action nodes (pick and place)
    if (var.num_pick > 0) {
        times_dict.insert("pick", torch::randint(0, 10, {var.num_pick}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("pick", torch::zeros({var.num_pick}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    if (var.num_place > 0) {
        times_dict.insert("place", torch::randint(0, 10, {var.num_place}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
        batch_dict.insert("place", torch::zeros({var.num_place}, 
            torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }
    
    // Create edges with random connectivity
    if (var.num_edges_pick > 0 && var.num_ssBox > 0 && var.num_pick > 0) {
        auto src = torch::randint(0, var.num_ssBox, {var.num_edges_pick}, torch::kInt64);
        auto dst = torch::randint(0, var.num_pick, {var.num_edges_pick}, torch::kInt64);
        edge_index_dict.insert("ssBox___pick_edge___pick", 
            torch::stack({src, dst}).to(device));
    }
    
    if (var.num_edges_place > 0 && var.num_object > 0 && var.num_place > 0) {
        auto src = torch::randint(0, var.num_object, {var.num_edges_place}, torch::kInt64);
        auto dst = torch::randint(0, var.num_place, {var.num_edges_place}, torch::kInt64);
        edge_index_dict.insert("object___place_edge___place", 
            torch::stack({src, dst}).to(device));
    }
    
    if (var.num_edges_close > 0 && var.num_ssBox > 1) {
        auto src = torch::randint(0, var.num_ssBox, {var.num_edges_close}, torch::kInt64);
        auto dst = torch::randint(0, var.num_ssBox, {var.num_edges_close}, torch::kInt64);
        edge_index_dict.insert("ssBox___close_edge___ssBox", 
            torch::stack({src, dst}).to(device));
    }
    
    if (var.num_edges_time > 0 && var.num_object > 1) {
        auto src = torch::randint(0, var.num_object, {var.num_edges_time}, torch::kInt64);
        auto dst = torch::randint(0, var.num_object, {var.num_edges_time}, torch::kInt64);
        edge_index_dict.insert("object___time_edge___object", 
            torch::stack({src, dst}).to(device));
    }
    
    // Prepare inputs vector: (x_dict, times_dict, edge_index_dict, batch_dict)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_dict);
    inputs.push_back(times_dict);
    inputs.push_back(edge_index_dict);
    inputs.push_back(batch_dict);
    
    // Run forward pass with timing
    auto start = std::chrono::high_resolution_clock::now();
    
    torch::IValue output = module.forward(inputs);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    torch::Tensor result = output.toTensor();
    
    std::cout << "  Forward pass: " << elapsed.count() << "s, Output shape: " << result.sizes() << std::endl;
}

void testModelsRoundRobin(std::vector<torch::jit::script::Module>& modules, 
                         const std::vector<std::string>& model_names,
                         torch::Device device, int num_runs) {
    std::cout << "\n=== Testing " << modules.size() << " Models in Round-Robin Fashion ===" << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < num_runs; i++) {
        InputVariation var = generateRandomVariation(gen);
        
        // Test all models with the same input variation
        for (size_t model_idx = 0; model_idx < modules.size(); model_idx++) {
            testModelForwardWithVariation(modules[model_idx], device, var, i + 1, model_names[model_idx]);
        }
        
        // Small delay between runs
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "\n=== Round-Robin Test Complete ===" << std::endl;
}

void testModelForward(torch::jit::script::Module& module, torch::Device device) {
    std::cout << "\n=== Testing Model Forward Pass with Random Variations ===" << std::endl;
    
    // Initialize random number generator with current time
    std::random_device rd;
    std::mt19937 gen(rd());
    
    
    const int num_runs = 1000;
    
    for (int i = 0; i < num_runs; i++) {
        InputVariation var = generateRandomVariation(gen);
        testModelForwardWithVariation(module, device, var, i + 1, "SingleModel");
        
        // Small delay to allow any cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n=== Forward Pass Test Complete ===" << std::endl;
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: load_model <path-to-model1.pt> [path-to-model2.pt] [...]" << std::endl;
        std::cerr << "Example: load_model model1.pt model2.pt model3.pt" << std::endl;
        return -1;
    }

    // Determine device (CPU or CUDA)
    torch::Device device(torch::kCPU);
    std::cout << "Using device: CPU" << std::endl;
    
    // Vectors to store all loaded models and their names
    std::vector<torch::jit::script::Module> modules;
    std::vector<std::string> model_names;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Loading " << (argc - 1) << " models..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Load all models first
    for (int model_idx = 1; model_idx < argc; model_idx++) {
        std::string model_path = argv[model_idx];
        
        // Extract model name from path
        size_t last_slash = model_path.find_last_of("/\\");
        std::string model_name = (last_slash != std::string::npos) 
                                  ? model_path.substr(last_slash + 1) 
                                  : model_path;
        
        std::cout << "\n[" << model_idx << "/" << (argc - 1) << "] Loading: " << model_name << std::endl;

        try {
            // Load the model
            torch::jit::script::Module module = torch::jit::load(model_path);
            module.to(device);
            module.eval();
            
            modules.push_back(module);
            model_names.push_back(model_name);
            
            std::cout << "  ✓ Loaded successfully" << std::endl;
            
        } catch (const c10::Error& e) {
            std::cerr << "  ✗ Error loading: " << e.what() << std::endl;
            std::cerr << "  Skipping this model..." << std::endl;
            continue;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Standard exception: " << e.what() << std::endl;
            std::cerr << "  Skipping this model..." << std::endl;
            continue;
        }
    }

    if (modules.empty()) {
        std::cerr << "\nNo models were successfully loaded. Exiting." << std::endl;
        return -1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Successfully loaded " << modules.size() << " model(s)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test all models in round-robin fashion
    const int num_runs = 10;
    testModelsRoundRobin(modules, model_names, device, num_runs);

    std::cout << "\n========================================" << std::endl;
    std::cout << "All models tested!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
