#include "parallel/cli.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        parallel::CliOptions options = parallel::parseArguments(argc, argv);
        return parallel::runCli(options);
    } catch (const std::exception& ex) {
        std::cerr << "[Parallel Fatal Error]: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[Parallel Fatal Error]: An unexpected exception occurred.\n";
        return 2;
    }
}
