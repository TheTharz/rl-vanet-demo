#ifndef RL_INTERFACE_H
#define RL_INTERFACE_H

#include "ns3/object.h"
#include <string>
#include <map>
#include <vector>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

namespace ns3 {

class RLInterface : public Object
{
public:
  static TypeId GetTypeId(void);
  RLInterface();
  RLInterface(const std::string &addr);
  ~RLInterface() override;

  void Init(const std::string &addr);
  
  // Send state to agent and get action
  nlohmann::json SendState(const std::map<std::string, double> &state);
  
  // Receive action from agent (Deprecated/Unused if SendState returns action)
  nlohmann::json ReceiveAction();
  
  // Send reward to agent
  void SendReward(double reward, bool done);
  
  // NEW: Receive positions from external source (SUMO via Python)
  // Returns a map of NodeID -> (x, y)
  std::map<uint32_t, std::pair<double, double>> ReceivePositions();

private:
  std::string m_addr;
  zmq::context_t m_ctx;
  zmq::socket_t m_socket;
};

} // namespace ns3

#endif
