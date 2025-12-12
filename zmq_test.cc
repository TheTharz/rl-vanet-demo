#include <zmq.hpp>
#include <string>
#include <iostream>

int main() {
    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::req);
    std::cout << "ZMQ Compiled!" << std::endl;
    return 0;
}
