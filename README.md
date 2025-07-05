# Wireless-Sensor-Network-Localization

**Problem Description (Observables, Unknowns, Knowns)**
**Wireless Sensor Network Localization**

The Internet of Things (IoT) is an important part of new-generation information technology. It is a vast network formed by integrating the Internet with various information-sensing devices, such as sensors, Radio Frequency Identification (RFID) technology, Global Positioning Systems, and other devices and technologies that can collect real-time information on sound, light, heat, electricity, mechanics, chemistry, biology, and location. As an important component of the IoT, Wireless Sensor Networks (WSN) are composed of a large number of low-cost micro-sensor nodes deployed in the monitoring area, forming a multi-hop self-organizing network through wireless communication.
In many application scenarios of wireless sensor networks, the locations of the nodes must be known. Therefore, node localization technology is a key technology and a research hotspot in WSNs. However, equipping all nodes with GPS or other localization devices is very costly. Therefore, usually only some nodes obtain their precise locations through GPS positioning devices, and these nodes are called beacon nodes. The other unknown nodes estimate their location coordinates through network connectivity information and internal ranging among nodes using geometric calculations. It is assumed that the random ranging errors follow a normal distribution, and due to terrain or equipment reasons, there may be systematic biases in the ranging. Please solve the following problems under this assumption:

1. Localize the unknown nodes in the simulation example (the results should be stored in Table 3 in the attachment).

2. Evaluate the accuracy of the localization results.

3. Conduct an overall model test on the localization results.

问题描述（观测量、未知量、已知量）
无线传感器网络的定位问题
物联网是新一代信息技术的重要组成部分，它是通过各种信息传感设备，如传感器、射频识别（RFID）技术、全球定位系统等各种装置与技术，实时采集声、光、热、 电、力学、化学、生物、位置等各种信息，与互联网结合形成的一个巨大网络。作为物联网的重要组成部分，无线传感器网络（WSN, Wireless Sensor networks）就是由部署在监测区域内大量的廉价微型传感器节点组成，通过无线通信方式形成的一个多跳自组织网络。 
无线传感器网络的很多应用场合必须知道节点的位置，因此节点定位技术是 WSN 的关键技术和研究热点。然而，在所有节点上都配备GPS等定位设施成本很高。因此，一般只在部分节点通过 GPS 定位设备获得自身的精确位置，这些节点称为信标节点（beacon node）；而其它未知节点（unknown node）则通过网络连接信息和节点内部相互测距通过几何计算来估计其位置坐标。假设测距的随机误差服从正态分布，并且由于地形或设备原因，测距可能存在系统偏差。请在此假设条件下解决以下问题：
1. 对仿真算例中的未知节点进行定位（结果存储在附件表3中） 
2. 对定位结果精度进行评定 
3. 对定位结果进行总体模型检验
