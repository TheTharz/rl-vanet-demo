#include "rl-interface.h"
#include "ns3/log.h"
#include <iostream>
#include <nlohmann/json.hpp>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RLInterface");

TypeId RLInterface::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::RLInterface")
    .SetParent<Object>()
    .SetGroupName("Core");
  return tid;
}

using json = nlohmann::json;

RLInterface::RLInterface()
    : m_ctx(1), m_socket(m_ctx, zmq::socket_type::req) {}

RLInterface::RLInterface(const std::string &addr)
    : m_ctx(1), m_socket(m_ctx, zmq::socket_type::req) {
    Init(addr);
}

RLInterface::~RLInterface() {
    m_socket.close();
    m_ctx.close();
}

void RLInterface::Init(const std::string &addr) {
    m_addr = addr;
    m_socket.connect(addr);
    NS_LOG_UNCOND("[RLInterface] Connected to " << addr);
}

json RLInterface::SendState(const std::map<std::string, double> &state) {
    json jmsg;
    jmsg["type"] = "state";
    jmsg["data"] = state;

    std::string data = jmsg.dump();
    zmq::message_t msg(data.begin(), data.end());
    // std::cout << "[RLInterface] Sending State REQ..." << std::endl;
    m_socket.send(msg, zmq::send_flags::none);

    // Receive Action immediately (Synchronous REQ-REP)
    zmq::message_t reply;
    // std::cout << "[RLInterface] Waiting for Action REP..." << std::endl;
    m_socket.recv(reply, zmq::recv_flags::none);
    std::string replyStr(static_cast<char *>(reply.data()), reply.size());
    // std::cout << "[RLInterface] Got Action REP." << std::endl;

    json jreply;
    try {
        jreply = json::parse(replyStr);
    } catch (const std::exception &e) {
        NS_LOG_UNCOND("[RLInterface] JSON parse error in SendState: " << e.what());
    }
    return jreply;
}

json RLInterface::ReceiveAction() {
    zmq::message_t reply;
    m_socket.recv(reply, zmq::recv_flags::none);
    std::string replyStr(static_cast<char *>(reply.data()), reply.size());

    json jreply;
    try {
        jreply = json::parse(replyStr);
        // NS_LOG_UNCOND("[RLInterface] Received action JSON: " << jreply.dump(2));
    } catch (const std::exception &e) {
        NS_LOG_UNCOND("[RLInterface] JSON parse error: " << e.what());
    }

    return jreply;
}

void RLInterface::SendReward(double reward, bool done) {
    json jmsg;
    jmsg["type"] = "reward";
    jmsg["reward"] = reward;
    jmsg["done"] = done;

    std::string data = jmsg.dump();
    zmq::message_t msg(data.begin(), data.end());
    m_socket.send(msg, zmq::send_flags::none);

    // Wait for acknowledgment
    zmq::message_t ack;
    m_socket.recv(ack, zmq::recv_flags::none);
    std::string ackStr(static_cast<char *>(ack.data()), ack.size());
    // NS_LOG_UNCOND("[RLInterface] Agent ACK: " << ackStr);
}

// NEW: Receive positions from external source
std::map<uint32_t, std::pair<double, double>> RLInterface::ReceivePositions() {
    std::map<uint32_t, std::pair<double, double>> positions;
    
    // Send request for positions
    json jmsg;
    jmsg["type"] = "get_positions";
    std::string data = jmsg.dump();
    zmq::message_t msg(data.begin(), data.end());
    m_socket.send(msg, zmq::send_flags::none);
    
    // Receive positions
    zmq::message_t reply;
    m_socket.recv(reply, zmq::recv_flags::none);
    std::string replyStr(static_cast<char *>(reply.data()), reply.size());
    
    try {
        json jreply = json::parse(replyStr);
        if (jreply.contains("positions")) {
            for (auto& [id_str, pos] : jreply["positions"].items()) {
                uint32_t id = std::stoul(id_str);
                positions[id] = {pos[0], pos[1]};
            }
        }
    } catch (const std::exception &e) {
        NS_LOG_UNCOND("[RLInterface] Position parse error: " << e.what());
    }
    
    return positions;
}

} // namespace ns3
