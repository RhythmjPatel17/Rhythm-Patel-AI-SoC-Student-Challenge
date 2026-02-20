## 🚀 Real-Time Silicon Wafer Defect Image Classification Using a Hardware–Software Co-Designed MobileNetV2 Accelerator on a Xilinx Zynq SoC

## 📌 Project Overview

In semiconductor wafer manufacturing, defect detection and classification are critical processes that directly influence production yield, cost efficiency, and time to market. The increasing resolution of wafer maps and the demand for low-latency, on-site inspection challenge conventional CPU-based inference systems, which suffer from limited throughput and higher energy consumption on embedded platforms.

## ⚙️ Proposed System

This work presents a real-time silicon wafer defect classification system implemented using a hardware–software co-designed MobileNetV2 accelerator on a Xilinx Zynq SoC. The heterogeneous architecture leverages the Arm processing system (PS) for image acquisition, preprocessing, control logic, and post-processing, while compute-intensive convolutional operations are offloaded to the FPGA programmable logic (PL). The CNN accelerator is implemented using RTL with optimized pipelining, parallel multiply–accumulate (MAC) units, and on-chip BRAM buffering to minimize memory bottlenecks and maximize data reuse.

## 📊 Experimental Results

Experimental evaluation demonstrates significant performance improvement compared to a CPU-only implementation on the Arm processor, achieving real-time inference with enhanced throughput and reduced latency while efficiently utilizing FPGA resources (LUTs, DSPs, and BRAM). The proposed system validates the effectiveness of hardware-software co-design for deploying energy-efficient edge AI solutions in semiconductor inspection environments.

## 📘 Introduction and Research Background

Semiconductor wafer inspection is essential for maintaining yield and reliability in IC fabrication, but as defect patterns become more complex with technology scaling, traditional inspection methods become computationally expensive and unsuitable for real-time deployment. While convolutional neural networks (CNNs) improve defect classification through automated feature extraction, they require significant computational and memory resources. Running them on embedded CPUs often fails to meet real-time and energy constraints, and cloud-based solutions introduce latency, cost, and privacy concerns, limiting their practicality for inline fab-floor applications.

## 🏗 Proposed Architecture

To address these challenges, this work proposes a hardware–software co-designed inference system implemented on a Zynq-7000 platform from Xilinx. The Zynq architecture integrates a dual-core Arm Cortex-A processing system (PS) with FPGA programmable logic (PL) on a single chip, enabling heterogeneous acceleration through efficient workload partitioning. In the proposed architecture, image acquisition, preprocessing, system control, and post-processing are executed on the Arm processing system, while computationally intensive CNN layers are offloaded to the programmable logic for parallel hardware execution.

## 🧠 Backbone Network Selection

The selected backbone network is MobileNetV2, due to its lightweight structure and efficiency-oriented design. MobileNetV2 employs depthwise separable convolutions and inverted residual blocks to significantly reduce multiply–accumulate (MAC) operations and parameter count compared to conventional convolutional networks. This architectural efficiency makes it well-suited for FPGA-based implementation under constrained logic, DSP, and on-chip memory resources.

## 📊 Baseline Software Analysis

A baseline software-only implementation of MobileNetV2 was first executed on the Arm Cortex-A processor to establish reference performance metrics. Profiling analysis confirmed that convolutional layers dominate execution time and memory bandwidth utilization, accounting for the majority of inference latency. Based on this observation, a custom FPGA accelerator was developed to target the primary computational kernels of the network:
Standard Convolution
Depthwise Convolution
Pointwise Convolution

## ⚙️ Accelerator Design

The accelerator architecture incorporates parallel MAC arrays, pipelined datapaths, and on-chip BRAM buffers to maximize data reuse and minimize off-chip memory accesses. Communication between the PS and PL is implemented using AXI-based interfaces: AXI4-Lite for configuration and control, and AXI4-Full or AXI DMA for high-throughput data transfer. A centralized finite state machine (FSM) coordinates execution sequencing and synchronization between software and hardware components.

## 🔍 Comparison with Traditional Methods

Compared to traditional machine learning approaches such as Support Vector Machines and Random Forests—which rely on handcrafted features and limited scalability—the proposed CNN-based accelerator enables automated feature extraction with improved classification robustness. Additionally, unlike computationally intensive architectures such as ResNet or Vision Transformers, MobileNetV2 provides a favorable trade-off between accuracy and computational complexity, making it practical for real-time deployment on heterogeneous embedded SoCs.

## 📑 Functional Specifications and Methodology

## 1️⃣ Input

High-resolution wafer images acquired from inspection equipment. Images are preprocessed and quantized to INT8 format to match the fixed-point FPGA accelerator architecture.

## 2️⃣ Model Architecture

MobileNetV2 CNN quantized to INT8 using Quantization-Aware Training (QAT) to reduce computational and memory overhead while maintaining accuracy. Supported layers include Standard Convolution (SC), Depthwise Convolution (DW), and Pointwise Convolution (PW), along with Batch Normalization, ReLU, Max Pooling, and Fully Connected layers. Convolution and pooling operations are mapped onto parallel MAC/MAX-based Processing Elements (PEs) implemented in the programmable logic.

## 3️⃣ Output

Predicted wafer defect class is transmitted to the Arm Cortex-A Processing System (PS) via AXI interface for further decision-making and system-level control.

## 4️⃣ Performance Metrics

Accuracy: INT8 quantized model achieves ~95.5% classification accuracy compared to ~97.8% FP32 baseline.

Latency: ~3.92 ms per image using fully pipelined hardware execution.

Throughput: ~255 images per second, enabling real-time inspection.

Energy Efficiency: Reduced power consumption through fixed-point arithmetic and pipeline-based architecture.

## 5️⃣ Memory Architecture

On-chip BRAM buffers store feature maps, weights, biases, scales, and shifts for each CNN layer. Data movement (inputs, intermediate activations, outputs) is managed through AXI4 interfaces with burst transfer support.

DDR memory can be used for storing larger model parameters and input datasets when required.

## 6️⃣ Interfaces and Integration

AXI4-Lite for control and configuration (start/stop, status, layer parameters).

AXI4/AXI-DMA for high-speed data transfer between PS and PL.

Fully memory-mapped accelerator tightly coupled with the Arm processor within the Zynq SoC environment.

## 7️⃣ Quantization and Pipeline Optimization

MobileNetV2 converted from FP32 to INT8 using QAT, reducing computational complexity by ~75% while preserving accuracy.

Intra-layer and inter-layer pipelining maximize PE utilization and minimize inference latency.

## 8️⃣ Tools and Methodology

Design & Synthesis: Xilinx Vivado for RTL synthesis, implementation, and bitstream generation.

Hardware Description: Verilog/SystemVerilog for accelerator design.

Simulation & Verification: Vivado Simulator for functional and timing validation.

Model Training & Quantization: Python with TensorFlow (QAT) for training and INT8 conversion.

<img width="1196" height="1474" alt="image" src="https://github.com/user-attachments/assets/10f9e698-0103-469e-9d43-b65f93c15568" />

## Input

•	Image size: 224 × 224 × 3 (RGB wafer image).

•	This serves as the raw input vector for the CNN accelerator.

## Standard Convolution (3×3, same padding)

•	Operation: Applies 32 filters of size 3×3 across the input image with padding to preserve dimensions.

•	Output size: 224 × 224 × 32.

•	Processing elements: Each filter is implemented using multiple MAC PEs (Multiply-Accumulate Processing Elements) to perform convolution in parallel.

•	Purpose: Extracts low-level spatial features such as edges, scratches, and texture anomalies from wafer images.

•	Trained parameters: Weights + biases.

## Batch Normalization

•	Operation: Normalizes feature maps across each channel.

•	Parameters: Scale (γ) and shift (β), both learned during training.

•	Purpose: Stabilizes training, improves convergence, and prepares data for quantization.

•	Output: Same shape, normalized values.

## ReLU6 Activation

•	Operation: Activation function with clipping at 6 → maps values to [0, 6].

•	Significance: When quantized to INT8, values fit into 4 bits, improving power efficiency.

•	Purpose: Introduces non-linearity while controlling dynamic range.

## Max Pooling (2×2 stride)

•	Operation: Picks maximum value from every 2×2 region.

•	Processing element: Max PE compares 4 inputs and outputs the largest.

•	Output: Reduced spatial size (downsampling) while retaining strongest features.

•	Purpose: Reduces computation, increases robustness to translation.

## Depthwise Convolution (3×3, zero padding)

•	Operation: Each input channel has its own filter (channel-wise convolution).

•	MAC usage: Much fewer MACs than standard conv, since no mixing across channels.

•	Purpose: Lightweight feature extraction, preserves channel independence.

•	Trained parameters: Depthwise filter weights + biases.

## Pointwise Convolution (1×1)

•	Operation: Combines depthwise outputs across channels using 1×1 convolutions.

•	MAC usage: Intensive, but highly parallelizable via many PEs.

•	Purpose: Channel mixing and dimensionality adjustment.

•	Trained parameters: Weights + biases.

## Iterative Blocks

The pipeline alternates [Depthwise → Pointwise → BN → ReLU6 → Pooling] multiple times:

•	Each block extracts higher-level features : From scratches and edges to larger defect structures and Reduces spatial resolution while increasing depth (channels).

•	Max pooling in early stages = 2×2 stride.

•	Final pooling = 7×7 stride → produces a compact global representation.

## Flattening Layer

•	Converts the final 7×7×1280 tensor into a 1D vector.

•	This vector is the condensed “signature” of the wafer image.

## Dense (Fully Connected) Layers

•	Dense 1: Connects flattened vector to hidden representation (all MAC operations).

•	Dense 2 (final): Maps to 9 output neurons (classes).

•	Parameters: Large number of MAC operations since every neuron connects to all inputs. Weights and biases are learned.

## ArgMax

•	Operation: Selects the index of the maximum logit among the 9 classes.
0 : Center, 
1 : Donut, 
2 : Edge-Loc, 
3 : Edge-Ring, 
4 : Loc, 
5 : Near-Full, 
6 : Random, 
7 : Scratch and 
8 : None
•	Output: Final predicted wafer defect category.

<img width="1134" height="653" alt="image" src="https://github.com/user-attachments/assets/e29ebdcf-89d2-443a-85e8-cf0a3fda3385" />

## 📷 Integration of CNN Accelerator with PL, PS, and Camera (End-to-End Architecture)

The proposed system is implemented on a Zynq-7000 SoC from Xilinx, which integrates an Arm Cortex-A Processing System (PS) with FPGA Programmable Logic (PL). The system performs real-time wafer defect classification through coordinated interaction between the camera interface, FPGA accelerator, and Arm processor.

## 1️⃣ Camera to Programmable Logic (PL)
Camera Interface

The wafer image sensor (industrial camera) connects to the PL through:

MIPI CSI-2 RX (for high-speed cameras), or

Parallel CMOS interface, depending on hardware availability.

A Camera Receiver IP in the PL converts incoming pixel stream into AXI4-Stream format.

Frame Buffering

The AXI4-Stream video data is passed to:

AXI VDMA (Video DMA)

VDMA writes image frames into DDR memory through:

AXI High-Performance (HP) port of the PS.

This enables full-frame buffering before inference.

## 2️⃣ Processing System (PS) Role

The Arm Cortex-A processor performs:

Camera configuration (via I2C/SPI)

Frame acquisition control

Image preprocessing:

Resizing (if required)

Normalization

INT8 quantization

Memory allocation in DDR

Accelerator configuration via AXI4-Lite

Post-processing of classification output

The PS accesses DDR and communicates with PL using:

AXI GP (General Purpose) port → control signals

AXI HP port → high-speed data transfer

## 3️⃣ PS to CNN Accelerator (Control Path)

The CNN accelerator implemented in PL is exposed as a memory-mapped AXI slave peripheral.

Control Interface (AXI4-Lite)

Used for:

Start signal

Reset

Status monitoring

Layer parameter configuration

Interrupt enable/acknowledge

Flow:

PS writes input image DDR address to accelerator register.

PS writes model parameter base address.

PS asserts “start”.

Accelerator raises interrupt when inference completes.

## 4️⃣ Data Path (High-Speed Path)

PS programs AXI DMA with:

Source address (input image in DDR)

Destination (accelerator input stream)

AXI DMA transfers INT8 image to accelerator using AXI4-Stream.

Accelerator processes data using:

Parallel MAC arrays

BRAM-based feature map buffers

Output feature/class result is written:

Back to DDR via AXI Master interface

Streamed to DMA for DDR write-back.

This architecture:

Minimizes CPU load

Maximizes throughput

Enables pipelined execution

## 5️⃣ End-to-End Data Flow

Camera captures wafer image.

Pixel stream enters PL through MIPI/CMOS interface.

AXI VDMA writes frame into DDR.

PS preprocesses image and converts to INT8.

PS configures accelerator via AXI4-Lite.

AXI DMA streams image to accelerator.

CNN accelerator performs inference in PL.

Output class written back to DDR.

PS reads result and performs:

Defect logging

Display update

Fab-floor decision trigger

| PEs | Total Latency (cycles) | Speedup | Latency (ms) |
| --- | ---------------------- | ------- | ------------ |
| 1   | 11,48,90,005           | 1×      | 1148.9       |
| 9   | 28,79,500              | 39.88×  | 28.8         |
| 27  | 11,70,405              | 98.13×  | 11.7         |
| 32  | 7,12,039               | 161.43× | 7.12         |
| 64  | 4,94,909               | 232.12× | 4.95         |
| 128 | 3,92,314               | 292.79× | 3.92         |

## Hardware Implementation 
<img width="1048" height="694" alt="image" src="https://github.com/user-attachments/assets/8febbe3b-8732-43cd-92da-6223a9948f70" />
Fig. a : The left-side image serves as the input image displayed to camera, which captures the visual data for processing. The processing board and camera module are positioned to enable real-time image acquisition for ML inference.
Fig. b : The right-top panel shows the image captured by the camera.
Fig. c : The right-bottom panel illustrates the output prediction produced by the ML accelerator.

<img width="1197" height="188" alt="image" src="https://github.com/user-attachments/assets/4ff1b3fc-96d3-4d4d-bec6-7a152a50b7b6" />
<img width="1208" height="204" alt="image" src="https://github.com/user-attachments/assets/89f3dfbe-8343-44d6-aab4-3fd7f7e8637b" />

| SR. NO. | METRIC                   | VALUE           |
| ------- | ------------------------ | --------------- |
| 1       | THROUGHPUT               | 255 FPS         |
| 2       | ACCURACY                 | 95.52% (INT8)   |
| 3       | LATENCY PER FRAME (CPU)  | 1148.9 ms/frame |
| 4       | LATENCY PER FRAME (FPGA) | 3.92 ms/frame   |
| 5       | SPEED UP (w.r.t. cycles) | 293.09          |
| 6       | LOOK-UP TABLES (LUTs)    | 39947 (75.08%)  |
| 7       | FLIP FLOP (FF)           | 60532 (56.89%)  |
| 8       | BRAM : DSP               | 75 : 87         |
| 9       | ON CHIP POWER            | 0.147 Watt      |
| 10      | ENERGY PER FRAME         | 368 µJ/frame    |

## Conclusion

## 📌 Why Deep Learning for Wafer Defect Detection?

Deep learning, particularly Convolutional Neural Networks (CNNs), enables automatic feature extraction from wafer images without relying on handcrafted rules. It effectively handles complex and diverse defect patterns present in industrial datasets such as WM811K, maintaining robust classification performance even under high variability. Compared to traditional machine learning approaches, CNN-based methods improve generalization, scalability, and adaptability for real-time semiconductor inspection systems.

## ⚙️ Why MobileNetV2 Specifically?

MobileNetV2 is selected due to its lightweight architecture and computational efficiency. Its use of depthwise separable convolutions and inverted residual blocks significantly reduces multiply–accumulate (MAC) operations while preserving high classification accuracy. The architecture is highly suitable for FPGA acceleration because convolution operations can be parallelized efficiently in programmable logic. Its compact design makes it ideal for deployment on resource-constrained edge platforms such as the Zynq SoC, where power, memory, and logic utilization must be carefully managed.

## 🧠 Model Design for WM811K Dataset

The model processes high-resolution wafer images obtained from inspection equipment. Using Quantization-Aware Training (QAT), the FP32 baseline achieved 97% classification accuracy. After INT8 quantization, the model achieved 95.5% accuracy while reducing memory footprint and computational cost by approximately 75%. The quantized model is deployed on the FPGA accelerator, and predicted defect classes are communicated to the Arm Processing System (PS) for further decision-making and logging.

## 🚀 Custom AI Accelerator with Pipeline-Mapped MobileNetV2

The FPGA accelerator, implemented in the Programmable Logic (PL) of the Zynq SoC, maps standard convolution, depthwise convolution, pointwise convolution, batch normalization, max pooling, and fully connected layers onto parallel MAC/MAX-based processing elements. Both intra-layer and inter-layer pipelining techniques are employed to minimize latency and maximize hardware utilization. The accelerator is integrated with the Arm Cortex-A Processing System through AXI4-Lite (control) and AXI DMA (high-speed data transfer), ensuring efficient communication and synchronized execution within the SoC.

## 📊 Performance and Deployment

The complete inference pipeline—including feature extraction, dense computation, argmax, and classification—executes in approximately 3,92,314 cycles at 100 MHz, corresponding to 3.92 ms latency per frame. The accelerator achieves approximately 255 FPS throughput at 0.147 Watt Power on a ~40% utilization of board resources, meeting real-time inline inspection requirements. A significant speedup over software-only INT8 inference is achieved due to efficient depthwise–pointwise convolution mapping and pipelined execution in hardware. Computation is performed primarily using on-chip BRAM buffers to reduce external memory dependency and avoid bandwidth bottlenecks. The Arm processor manages accelerator control, result retrieval, logging, display updates, and fab-floor decision triggering without reliance on external PCs, GPUs, or cloud services, ensuring deterministic execution and data security. The modular accelerator design supports scalability to higher-resolution wafer images or expanded channel depths through parallel tiling strategies while maintaining architectural flexibility.

## References

1.	A. Su Pan, B. Xingyang Nie, C. Xiaoyu Zhai; Enhancing wafer defect detection via ensemble learning. AIP Advances 1 August 2024; 14 (8): 085301. https://doi.org/10.1063/5.0222140

2.	Mayank Jariya, Parveen Kumar, Rekha Devi, Balwinder Singh, Silicon wafer defect pattern detection using machine learning,Materials Today: Proceedings,2023,ISSN 2214-7853, https://doi.org/ 10.1016/j.matpr.2023.04.233.3
   
3.	R. K.P. and S. V., "Machine Learning Approach for Mixed type Wafer Defect Pattern Recognition by ResNet Architecture," 2023 International Conference on Control, Communication and Computing (ICCC), Thiruvananthapuram, India, 2023, pp. 1-6, doi: 10.1109/ICCC57789.2023.10165078.

4.	Wu, Di & Zhang, Yu & Jia, Xijie & Tian, Lu & Li, Tianping & Sui, Lingzhi & Xie, Dongliang & Shan, Yi. (2019). A High-Performance CNN Processor Based on FPGA for MobileNets. 136-143. 10.1109/ FPL.2019.00030.
   
5.	W. Jiang, H. Yu and Y. Ha, "A High-Throughput Full-Dataflow MobileNetv2 Accelerator on Edge FPGA," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 5, pp. 1532-1545, May 2023, doi: 10.1109/TCAD.2022.3198246.

6.	Huang, J.; Liu, X.; Guo, T.; Zhao, Z. A High-Performance FPGA-Based Depthwise Separable Convolution Accelerator. Electronics 2023, 12, 1571. https://doi.org/10.3390/electronics 12071571

7.	Sandler, Mark & Howard, Andrew & Zhu, Menglong & Zhmoginov, Andrey & Chen, Liang-Chieh. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. 4510-4520. 10.1109/CVPR .2018.00474. 

8.	A. Su Pan, B. Xingyang Nie, C. Xiaoyu Zhai, Title: Enhancing wafer defect detection via ensemble learning, Journal: AIP Advances, August 2024 
