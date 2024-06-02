#include <pybind11/embed.h> // everything needed for embedding
#include <iostream>
#include <string>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    try {
        py::module model_wrapper = py::module::import("model_wrapper");

        std::string user_input;
        while (true) {
            std::cout << "Enter text (type 'exit' to quit): ";
            std::getline(std::cin, user_input);

            if (user_input == "exit") {
                break;
            }

            auto result = model_wrapper.attr("classify_text")(user_input);
            std::string prediction = result.cast<std::string>();

            std::cout << "Prediction: " << prediction << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}