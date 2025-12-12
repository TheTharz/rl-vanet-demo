/*
 * VANET Scenario with SUMO Integration
 * Compatible with NS-3 3.44
 * 
 * Integrated with Python Orchestrator via ZMQ
 * Receives positions from SUMO, sends metrics, receives actions
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "rl-interface.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VanetSumo");

// Global variables
// Global variables
double g_beaconInterval = 1.0;
double g_txPower = 23.0;
uint32_t g_numVehicles = 0; // Dynamic based on SUMO
double g_simulationTime = 1000.0; // Controlled by Python
double g_loggingInterval = 1.0; // Step size
bool g_enableRL = true; // Always true for this script
std::string g_rlAddress = "tcp://localhost:5555";

std::map<uint32_t, uint64_t> g_packetsSent;
std::map<uint32_t, uint64_t> g_packetsReceived;
std::map<uint32_t, uint64_t> g_expectedReceptions;
std::map<uint32_t, Time> g_channelBusyTime;
std::map<uint32_t, Time> g_lastChannelSampleTime;

struct MetricsWindow {
    uint64_t packetsSent = 0;
    uint64_t packetsReceived = 0;
    uint64_t expectedReceptions = 0;
    Time windowStart = Seconds(0);
};

std::map<uint32_t, MetricsWindow> g_currentWindow;
std::map<uint32_t, MetricsWindow> g_previousWindow;

NodeContainer g_nodes;
NetDeviceContainer g_devices;
ApplicationContainer g_onoffApps;
ApplicationContainer g_sinkApps;
Ptr<RLInterface> g_rlInterface;

// Track active nodes (SUMO might add/remove vehicles, but for simplicity we assume fixed pool for now)
// In a full implementation, we would handle dynamic node creation/deletion.
// Here we assume a max pool of nodes and only move active ones.
const uint32_t MAX_NODES = 100;

// Forward declarations
void UpdateBeaconInterval(double newInterval);
void UpdateTxPower(double newPower);

// Packet callbacks (same as before)
// Helper: Update expected receptions based on neighbors in range
void UpdateExpectedReceptions(uint32_t senderNodeId) {
    Ptr<Node> senderNode = g_nodes.Get(senderNodeId);
    Ptr<MobilityModel> senderMobility = senderNode->GetObject<MobilityModel>();
    double commRange = 300.0; 
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        if (i == senderNodeId) continue;
        
        Ptr<MobilityModel> receiverMobility = g_nodes.Get(i)->GetObject<MobilityModel>();
        double distance = senderMobility->GetDistanceFrom(receiverMobility);
        
        // Only count if distance > 0 (avoids counting inactive nodes at 0,0)
        if (distance <= commRange && distance > 0.1) {
            g_expectedReceptions[i]++;
            g_currentWindow[i].expectedReceptions++;
        }
    }
}

// Packet callbacks
void TxCallback(std::string context, Ptr<const Packet> packet) {
    size_t pos = context.find("/NodeList/");
    if (pos != std::string::npos) {
        size_t start = pos + 10;
        size_t end = context.find("/", start);
        if (end != std::string::npos) {
            uint32_t nodeId = std::stoul(context.substr(start, end - start));
            g_packetsSent[nodeId]++;
            g_currentWindow[nodeId].packetsSent++;
            UpdateExpectedReceptions(nodeId);
        }
    }
}

void RxCallback(std::string context, Ptr<const Packet> packet, const Address &address) {
    size_t pos = context.find("/NodeList/");
    if (pos != std::string::npos) {
        size_t start = pos + 10;
        size_t end = context.find("/", start);
        if (end != std::string::npos) {
            uint32_t nodeId = std::stoul(context.substr(start, end - start));
            g_packetsReceived[nodeId]++;
            g_currentWindow[nodeId].packetsReceived++;
        }
    }
}

// Helper: Count neighbors
uint32_t CountNeighbors(Ptr<Node> node, double range) {
    uint32_t count = 0;
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        if (g_nodes.Get(i)->GetId() == node->GetId()) continue;
        Ptr<MobilityModel> otherMobility = g_nodes.Get(i)->GetObject<MobilityModel>();
        double dist = mobility->GetDistanceFrom(otherMobility);
        if (dist <= range && dist > 0.1) count++;
    }
    return count;
}

// Helper: Calculate CBR
double CalculateCBR(Ptr<Node> node) {
    uint32_t nodeId = node->GetId();
    Time currentTime = Simulator::Now();
    
    // Initialize if first time
    if (g_lastChannelSampleTime.find(nodeId) == g_lastChannelSampleTime.end()) {
        g_lastChannelSampleTime[nodeId] = currentTime;
        return 0.0;
    }

    Time timeSinceLastSample = currentTime - g_lastChannelSampleTime[nodeId];
    if (timeSinceLastSample.GetSeconds() <= 0.0) return 0.0;
    
    double beaconHz = 1.0 / g_beaconInterval;
    double packetDuration = 0.001; // ~1ms
    double numNeighbors = CountNeighbors(node, 300.0);
    
    double cbr = std::min(1.0, numNeighbors * beaconHz * packetDuration);
    g_lastChannelSampleTime[nodeId] = currentTime;
    return cbr;
}

double GetAverageCBR() {
    double totalCBR = 0.0;
    uint32_t activeNodes = 0;
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        // Only count nodes that have moved from 0,0
        Ptr<Node> node = g_nodes.Get(i);
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
        if (mob->GetPosition().GetLength() > 1.0) {
             totalCBR += CalculateCBR(node);
             activeNodes++;
        }
    }
    return (activeNodes > 0) ? totalCBR / activeNodes : 0.0;
}

// Helper: Calculate Reward
double CalculateReward(double pdr, double throughput, double avgNeighbors, double cbr) {
    double beaconHz = 1.0 / g_beaconInterval;
    
    // 1. PDR Reward
    double pdrReward = 0.0;
    if (pdr < 0.2) pdrReward = -20.0 + (pdr / 0.2) * 15.0; 
    else if (pdr < 0.4) pdrReward = -5.0 + ((pdr - 0.2) / 0.2) * 15.0; 
    else if (pdr < 0.6) pdrReward = 10.0 + ((pdr - 0.4) / 0.2) * 20.0; 
    else if (pdr < 0.8) pdrReward = 30.0 + ((pdr - 0.6) / 0.2) * 20.0; 
    else pdrReward = 50.0 + ((pdr - 0.8) / 0.2) * 20.0; 
    
    // 2. Throughput Reward
    double throughputReward = 0.0;
    if (throughput < 500.0) throughputReward = (throughput / 500.0) * 10.0;
    else if (throughput <= 3000.0) throughputReward = 10.0 + ((throughput - 500.0) / 2500.0) * 15.0; 
    else throughputReward = 25.0; 
    
    // 3. BeaconHz Cost
    double beaconCost = (beaconHz / 20.0) * 10.0;
    
    // 4. Connectivity Reward
    double connectivityReward = 0.0;
    if (avgNeighbors < 3.0) connectivityReward = -5.0 + (avgNeighbors / 3.0) * 5.0; 
    else if (avgNeighbors <= 12.0) connectivityReward = (avgNeighbors / 12.0) * 15.0; 
    else connectivityReward = 15.0;
    
    // 5. Congestion Penalty
    double congestionPenalty = 0.0;
    if (cbr > 0.4) congestionPenalty = (cbr - 0.4) / 0.6 * 10.0; 

    // 6. Power Penalty
    double powerPenalty = 0.0;
    if (g_txPower > 10.0) powerPenalty = ((g_txPower - 10.0) / 20.0) * 15.0;
    
    // 7. Stability Bonus
    double stabilityBonus = (pdr > 0.5 && cbr < 0.6) ? 10.0 : 0.0;
    
    double totalReward = pdrReward + throughputReward + connectivityReward + stabilityBonus 
                         - congestionPenalty - beaconCost - powerPenalty + 10.0;
                         
    return totalReward;
}

void ResetMetricsWindow() {
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        g_previousWindow[i] = g_currentWindow[i];
        g_currentWindow[i].packetsSent = 0;
        g_currentWindow[i].packetsReceived = 0;
        g_currentWindow[i].expectedReceptions = 0;
        g_currentWindow[i].windowStart = Simulator::Now();
    }
}

// Simulation Loop Step
void SimulationStep() {
    if (Simulator::Now().GetSeconds() >= g_simulationTime) return;

    // 1. Get Positions from Python
    std::map<uint32_t, std::pair<double, double>> positions = g_rlInterface->ReceivePositions();
    
    // 2. Update Mobility
    for (auto const& [id, pos] : positions) {
        if (id < g_nodes.GetN()) {
            Ptr<Node> node = g_nodes.Get(id);
            Ptr<ConstantPositionMobilityModel> mobility = node->GetObject<ConstantPositionMobilityModel>();
            mobility->SetPosition(Vector(pos.first, pos.second, 0.0));
        }
    }
    
    // 3. Calculate Windowed Metrics
    uint64_t windowSent = 0, windowRecv = 0, windowExpected = 0;
    double avgThroughput = 0.0, avgNeighbors = 0.0;
    uint32_t activeNodes = 0;

    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        // Filter for active nodes (non-zero position)
        Ptr<Node> node = g_nodes.Get(i);
        if (node->GetObject<MobilityModel>()->GetPosition().GetLength() < 1.0) continue;

        activeNodes++;
        windowSent += g_currentWindow[i].packetsSent;
        windowRecv += g_currentWindow[i].packetsReceived;
        windowExpected += g_currentWindow[i].expectedReceptions;
        
        double tput = (g_currentWindow[i].packetsReceived * 200 * 8) / g_loggingInterval;
        avgThroughput += tput;
        avgNeighbors += CountNeighbors(node, 300.0);
    }
    
    if (activeNodes > 0) {
        avgThroughput /= activeNodes;
        avgNeighbors /= activeNodes;
    }

    double windowPDR = (windowExpected > 0) ? (double)windowRecv / windowExpected : 0.0;
    double avgCBR = GetAverageCBR();
    
    // 4. Send State
    std::map<std::string, double> state;
    state["time"] = Simulator::Now().GetSeconds();
    state["PDR"] = windowPDR;
    state["throughput"] = avgThroughput;
    state["packetsSent"] = windowSent;
    state["packetsReceived"] = windowRecv;
    state["avgNeighbors"] = avgNeighbors;
    state["beaconHz"] = 1.0 / g_beaconInterval;
    state["txPower"] = g_txPower;
    state["CBR"] = avgCBR;
    
    g_rlInterface->SendState(state);

    NS_LOG_INFO("Time: " << Simulator::Now().GetSeconds() << "s | PDR: " << windowPDR 
                << " | Tput: " << avgThroughput << " | Neigh: " << avgNeighbors 
                << " | CBR: " << avgCBR);
    
    // 5. Receive Action
    nlohmann::json actionJson = g_rlInterface->ReceiveAction();
    if (actionJson.contains("action")) {
        auto action = actionJson["action"];
        if (action.contains("beaconHz")) {
            double hz = action["beaconHz"];
            if (hz > 0) UpdateBeaconInterval(1.0 / hz);
        }
        if (action.contains("txPower")) UpdateTxPower((double)action["txPower"]);
    }
    
    // 6. Calculate & Send Reward
    double reward = CalculateReward(windowPDR, avgThroughput, avgNeighbors, avgCBR);
    bool done = (Simulator::Now().GetSeconds() >= g_simulationTime - g_loggingInterval);
    g_rlInterface->SendReward(reward, done);
    
    // 7. Reset Window & Schedule
    ResetMetricsWindow();
    Simulator::Schedule(Seconds(g_loggingInterval), &SimulationStep);
}

void UpdateBeaconInterval(double newInterval) {
    g_beaconInterval = newInterval;
    // Update logic for OnOff apps...
    // (Simplified for brevity, same as original script)
}

void UpdateTxPower(double newPower) {
    g_txPower = newPower;
    for (uint32_t i = 0; i < g_devices.GetN(); i++) {
        Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(g_devices.Get(i));
        if (wifiDev) {
            wifiDev->GetPhy()->SetTxPowerStart(newPower);
            wifiDev->GetPhy()->SetTxPowerEnd(newPower);
        }
    }
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.AddValue("rlAddress", "RL agent ZMQ address", g_rlAddress);
    cmd.Parse(argc, argv);
    
    g_rlInterface = CreateObject<RLInterface>();
    g_rlInterface->Init(g_rlAddress);
    
    // Create fixed pool of nodes
    g_nodes.Create(MAX_NODES);
    
    // WiFi Setup (Same as before)
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(300.0));
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(g_txPower));
    wifiPhy.Set("TxPowerEnd", DoubleValue(g_txPower));
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue("OfdmRate6MbpsBW10MHz"));
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    g_devices = wifi.Install(wifiPhy, wifiMac, g_nodes);
    
    // Mobility: Constant Position (Updated externally)
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_nodes);
    
    // Internet & Apps
    InternetStackHelper internet;
    internet.Install(g_nodes);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.0.0");
    ipv4.Assign(g_devices);
    
    // Install Apps (Simplified)
    uint16_t port = 9;
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        g_sinkApps.Add(sinkHelper.Install(g_nodes.Get(i)));
        
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address("10.1.255.255"), port));
        onoff.SetAttribute("PacketSize", UintegerValue(200));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate("160000bps")));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.002]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]")); // Default 10Hz
        g_onoffApps.Add(onoff.Install(g_nodes.Get(i)));
    }
    g_sinkApps.Start(Seconds(0.0));
    g_onoffApps.Start(Seconds(0.0));
    
    // Initialize metrics
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        g_packetsSent[i] = 0;
        g_packetsReceived[i] = 0;
        g_expectedReceptions[i] = 0;
        g_channelBusyTime[i] = Seconds(0.0);
        g_lastChannelSampleTime[i] = Seconds(0.0);
        g_currentWindow[i] = MetricsWindow();
        g_previousWindow[i] = MetricsWindow();
        g_currentWindow[i].windowStart = Seconds(0.0);
    }

    // Connect Traces
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::OnOffApplication/Tx", MakeCallback(&TxCallback));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&RxCallback));
    
    // Start Loop
    Simulator::Schedule(Seconds(0.1), &SimulationStep);
    
    Simulator::Stop(Seconds(g_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    return 0;
}
