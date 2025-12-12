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
#include "ns3/phy-entity.h" // Required for RxPowerWattPerChannelBand
#include "ns3/ipv4-global-routing-helper.h"
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
const uint32_t MAX_NODES = 150;

// Forward declarations
void UpdateBeaconInterval(double newInterval);
void UpdateTxPower(double newPower);

// Global counters for debugging
uint32_t g_debugPhyTx = 0;
uint32_t g_debugPhyRx = 0;
uint32_t g_debugPhyRxBegin = 0;
uint32_t g_debugPhyDrop = 0;

// Trace Callbacks
void PhyTxBeginCallback(std::string context, Ptr<const Packet> packet, double txPowerW) {
    g_debugPhyTx++;
}


void PhyRxBeginCallback(std::string context, Ptr<const Packet> packet, RxPowerWattPerChannelBand rxPowers) {
    g_debugPhyRxBegin++;
}

void PhyRxEndCallback(std::string context, Ptr<const Packet> packet) {
    g_debugPhyRx++;
}

void PhyRxDropCallback(std::string context, Ptr<const Packet> packet, WifiPhyRxfailureReason reason) {
    g_debugPhyDrop++;
    /*if (g_debugPhyDrop % 100 == 0) { // Limit output frequency
        std::cout << "PHY DROP Reason: " << reason << std::endl;
    }*/
}

void MacRxDropCallback(std::string context, Ptr<const Packet> packet) {
    std::cout << "MAC DROP" << std::endl;
}

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

// Helper: Populate ARP Cache
void PopulateArpCache() {
    Ptr<Ipv4GlobalRouting> gr = CreateObject<Ipv4GlobalRouting>();
    // This is a workaround to populate ARP cache if we don't use GlobalRoutingHelper
    // But simpler is to just use NeighborCacheHelper if available, or just ignore for broadcast.
    // Since we use broadcast, ARP shouldn't be the main issue, but let's be safe.
    // Actually, simpler way:
    // Ipv4GlobalRoutingHelper::PopulateNeighborCache();
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

// Helper: Update Beacon Interval (Frequency)
void UpdateBeaconInterval(double newInterval) {
    if (std::abs(g_beaconInterval - newInterval) < 0.001) return;
    g_beaconInterval = newInterval;
    
    // Update OnOff Applications
    // OnTime is fixed (packet duration), OffTime determines frequency
    // Frequency = 1 / (OnTime + OffTime)
    // OffTime = (1 / Frequency) - OnTime
    
    // Assuming packet size 200 bytes @ 6Mbps -> ~266us OnTime
    // We can just set OffTime to (1/Hz) - small_delta
    
    std::string offTimeStr = "ns3::ConstantRandomVariable[Constant=" + std::to_string(newInterval) + "]";
    
    for (uint32_t i = 0; i < g_onoffApps.GetN(); i++) {
        Ptr<OnOffApplication> app = DynamicCast<OnOffApplication>(g_onoffApps.Get(i));
        if (app) {
            app->SetAttribute("OffTime", StringValue(offTimeStr));
        }
    }
}

// Helper: Update Tx Power
void UpdateTxPower(double newPower) {
    if (std::abs(g_txPower - newPower) < 0.1) return;
    g_txPower = newPower;
    
    for (uint32_t i = 0; i < g_devices.GetN(); i++) {
        Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice>(g_devices.Get(i));
        if (device) {
            Ptr<WifiPhy> phy = device->GetPhy();
            phy->SetAttribute("TxPowerStart", DoubleValue(g_txPower));
            phy->SetAttribute("TxPowerEnd", DoubleValue(g_txPower));
        }
    }
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
    // std::cout << "[NS3] SimulationStep Start at " << Simulator::Now().GetSeconds() << "s" << std::endl;
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
        avgNeighbors /= activeNodes;
    }

    double windowPDR = (windowExpected > 0) ? (double)windowRecv / windowExpected : 0.0;
    


// ... inside SimulationStep ...
    // DEBUG LOGGING
    // std::cout << "DEBUG: AppSent=" << windowSent << " AppRecv=" << windowRecv 
    //           << " PhyTx=" << g_debugPhyTx << " PhyRx=" << g_debugPhyRx 
    //           << " RxBegin=" << g_debugPhyRxBegin << " Drop=" << g_debugPhyDrop
    //           << " Exp=" << windowExpected << std::endl;
    
    // Reset debug counters for next step
    g_debugPhyTx = 0;
    g_debugPhyRx = 0;
    g_debugPhyRxBegin = 0;
    g_debugPhyDrop = 0;
    
    // 4. Calculate Throughput (bps)
    if (activeNodes > 0) {
        avgThroughput = (double)windowRecv * 200.0 * 8.0 / g_loggingInterval / activeNodes; 
    }
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
    
    // 5. Calculate Reward
    double reward = CalculateReward(windowPDR, avgThroughput, avgNeighbors, avgCBR);
    bool done = (Simulator::Now().GetSeconds() >= g_simulationTime - g_loggingInterval);

    // 6. Send State and Get Action
    // std::cout << "[NS3] Sending State..." << std::endl;
    nlohmann::json actionJson = g_rlInterface->SendState(state);
    // std::cout << "[NS3] Received Action." << std::endl;
    
    // 7. Send Reward (for previous step's action)
    // std::cout << "[NS3] Sending Reward..." << std::endl;
    g_rlInterface->SendReward(reward, done);
    // std::cout << "[NS3] Reward Sent." << std::endl;
    
    // 8. Process Action
    if (actionJson.contains("action")) {
        auto action = actionJson["action"];
        if (action.contains("beaconHz")) {
            double hz = action["beaconHz"];
            if (hz > 0) UpdateBeaconInterval(1.0 / hz);
        }
        if (action.contains("txPower")) UpdateTxPower((double)action["txPower"]);
    }
    
    // 9. Reset Window & Schedule
    ResetMetricsWindow();
    // std::cout << "[NS3] Scheduling next step..." << std::endl;
    Simulator::Schedule(Seconds(g_loggingInterval), &SimulationStep);
    // std::cout << "[NS3] SimulationStep End." << std::endl;
    // std::cout.flush();
}



int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.AddValue("rlAddress", "RL agent ZMQ address", g_rlAddress);
    cmd.Parse(argc, argv);
    
    g_rlInterface = CreateObject<RLInterface>();
    g_rlInterface->Init(g_rlAddress);
    
    // Create fixed pool of nodes
    g_nodes.Create(MAX_NODES);
    
    // WiFi Setup
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(300.0));
    
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(g_txPower));
    wifiPhy.Set("TxPowerEnd", DoubleValue(g_txPower));
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac"); // Back to standard Adhoc
    g_devices = wifi.Install(wifiPhy, wifiMac, g_nodes);
    
    // Mobility: Constant Position (Updated externally)
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_nodes);

    // Initialize to far away positions to avoid interference
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        Ptr<ConstantPositionMobilityModel> mob = g_nodes.Get(i)->GetObject<ConstantPositionMobilityModel>();
        mob->SetPosition(Vector(-10000.0 - (i * 100.0), 0.0, 0.0));
    }
    
    // Internet & Apps

    InternetStackHelper internet;
    internet.SetIpv4StackInstall(true);
    internet.SetIpv6StackInstall(false); // Disable IPv6 to reduce overhead
    internet.Install(g_nodes);
    
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.0.0");
    ipv4.Assign(g_devices);
    

    // Populate ARP cache to avoid broadcast storms
    PopulateArpCache();
    
    // Install Apps (Simplified)
    uint16_t port = 9;
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        g_sinkApps.Add(sinkHelper.Install(g_nodes.Get(i)));
        
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address("10.1.255.255"), port));
        onoff.SetAttribute("PacketSize", UintegerValue(200));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate("6Mbps"))); // Restore high data rate
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.002]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]")); // Default 10Hz
        g_onoffApps.Add(onoff.Install(g_nodes.Get(i)));
    }
    g_sinkApps.Start(Seconds(0.0));
    g_onoffApps.Start(Seconds(0.0));
    
    // Initialize metrics
    for (uint32_t i = 0; i < g_nodes.GetN(); i++) {
        g_previousWindow[i] = g_currentWindow[i];
        g_currentWindow[i].packetsSent = 0;
        g_currentWindow[i].packetsReceived = 0;
        g_currentWindow[i].expectedReceptions = 0;
        g_currentWindow[i].windowStart = Simulator::Now();
    }

    // Connect Traces
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::OnOffApplication/Tx", MakeCallback(&TxCallback));
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&RxCallback));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback(&PhyTxBeginCallback));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxBegin", MakeCallback(&PhyRxBeginCallback));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxEnd", MakeCallback(&PhyRxEndCallback));
    // Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop", MakeCallback(&PhyRxDropCallback));
    // Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRxDrop", MakeCallback(&MacRxDropCallback));
    
    // Start Loop
    Simulator::Schedule(Seconds(0.1), &SimulationStep);
    
    Simulator::Stop(Seconds(g_simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    return 0;
}
