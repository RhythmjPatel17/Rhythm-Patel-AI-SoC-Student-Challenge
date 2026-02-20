## 🚀 Real-Time Silicon Wafer Defect Image Classification Using a Hardware–Software Co-Designed MobileNetV2 Accelerator on a Xilinx Zynq SoC

📌 Project Overview

In semiconductor wafer manufacturing, defect detection and classification are critical processes that directly influence production yield, cost efficiency, and time to market. The increasing resolution of wafer maps and the demand for low-latency, on-site inspection challenge conventional CPU-based inference systems, which suffer from limited throughput and higher energy consumption on embedded platforms.

⚙️ Proposed System

This work presents a real-time silicon wafer defect classification system implemented using a hardware–software co-designed MobileNetV2 accelerator on a Xilinx Zynq SoC. The heterogeneous architecture leverages the Arm processing system (PS) for image acquisition, preprocessing, control logic, and post-processing, while compute-intensive convolutional operations are offloaded to the FPGA programmable logic (PL). The CNN accelerator is implemented using RTL with optimized pipelining, parallel multiply–accumulate (MAC) units, and on-chip BRAM buffering to minimize memory bottlenecks and maximize data reuse.

📊 Experimental Results

Experimental evaluation demonstrates significant performance improvement compared to a CPU-only implementation on the Arm processor, achieving real-time inference with enhanced throughput and reduced latency while efficiently utilizing FPGA resources (LUTs, DSPs, and BRAM). The proposed system validates the effectiveness of hardware-software co-design for deploying energy-efficient edge AI solutions in semiconductor inspection environments.

📘 Introduction and Research Background

Semiconductor wafer inspection is essential for maintaining yield and reliability in IC fabrication, but as defect patterns become more complex with technology scaling, traditional inspection methods become computationally expensive and unsuitable for real-time deployment. While convolutional neural networks (CNNs) improve defect classification through automated feature extraction, they require significant computational and memory resources. Running them on embedded CPUs often fails to meet real-time and energy constraints, and cloud-based solutions introduce latency, cost, and privacy concerns, limiting their practicality for inline fab-floor applications.

🏗 Proposed Architecture

To address these challenges, this work proposes a hardware–software co-designed inference system implemented on a Zynq-7000 platform from Xilinx. The Zynq architecture integrates a dual-core Arm Cortex-A processing system (PS) with FPGA programmable logic (PL) on a single chip, enabling heterogeneous acceleration through efficient workload partitioning. In the proposed architecture, image acquisition, preprocessing, system control, and post-processing are executed on the Arm processing system, while computationally intensive CNN layers are offloaded to the programmable logic for parallel hardware execution.

🧠 Backbone Network Selection

The selected backbone network is MobileNetV2, due to its lightweight structure and efficiency-oriented design. MobileNetV2 employs depthwise separable convolutions and inverted residual blocks to significantly reduce multiply–accumulate (MAC) operations and parameter count compared to conventional convolutional networks. This architectural efficiency makes it well-suited for FPGA-based implementation under constrained logic, DSP, and on-chip memory resources.

📊 Baseline Software Analysis

A baseline software-only implementation of MobileNetV2 was first executed on the Arm Cortex-A processor to establish reference performance metrics. Profiling analysis confirmed that convolutional layers dominate execution time and memory bandwidth utilization, accounting for the majority of inference latency. Based on this observation, a custom FPGA accelerator was developed to target the primary computational kernels of the network:
Standard Convolution
Depthwise Convolution
Pointwise Convolution

⚙️ Accelerator Design

The accelerator architecture incorporates parallel MAC arrays, pipelined datapaths, and on-chip BRAM buffers to maximize data reuse and minimize off-chip memory accesses. Communication between the PS and PL is implemented using AXI-based interfaces: AXI4-Lite for configuration and control, and AXI4-Full or AXI DMA for high-throughput data transfer. A centralized finite state machine (FSM) coordinates execution sequencing and synchronization between software and hardware components.

🔍 Comparison with Traditional Methods

Compared to traditional machine learning approaches such as Support Vector Machines and Random Forests—which rely on handcrafted features and limited scalability—the proposed CNN-based accelerator enables automated feature extraction with improved classification robustness. Additionally, unlike computationally intensive architectures such as ResNet or Vision Transformers, MobileNetV2 provides a favorable trade-off between accuracy and computational complexity, making it practical for real-time deployment on heterogeneous embedded SoCs.

📑 Functional Specifications and Methodology
1️⃣ Input

High-resolution wafer images acquired from inspection equipment.

Images are preprocessed and quantized to INT8 format to match the fixed-point FPGA accelerator architecture.

2️⃣ Model Architecture

MobileNetV2 CNN quantized to INT8 using Quantization-Aware Training (QAT) to reduce computational and memory overhead while maintaining accuracy.

Supported layers include Standard Convolution (SC), Depthwise Convolution (DW), and Pointwise Convolution (PW), along with Batch Normalization, ReLU, Max Pooling, and Fully Connected layers.

Convolution and pooling operations are mapped onto parallel MAC/MAX-based Processing Elements (PEs) implemented in the programmable logic.

3️⃣ Output

Predicted wafer defect class is transmitted to the Arm Cortex-A Processing System (PS) via AXI interface for further decision-making and system-level control.

4️⃣ Performance Metrics

Accuracy: INT8 quantized model achieves ~95.5% classification accuracy compared to ~97.8% FP32 baseline.

Latency: ~3.92 ms per image using fully pipelined hardware execution.

Throughput: ~255 images per second, enabling real-time inspection.

Energy Efficiency: Reduced power consumption through fixed-point arithmetic and pipeline-based architecture.

5️⃣ Memory Architecture

On-chip BRAM buffers store feature maps, weights, biases, scales, and shifts for each CNN layer. Data movement (inputs, intermediate activations, outputs) is managed through AXI4 interfaces with burst transfer support.

DDR memory can be used for storing larger model parameters and input datasets when required.

6️⃣ Interfaces and Integration

AXI4-Lite for control and configuration (start/stop, status, layer parameters).

AXI4/AXI-DMA for high-speed data transfer between PS and PL.

Fully memory-mapped accelerator tightly coupled with the Arm processor within the Zynq SoC environment.

7️⃣ Quantization and Pipeline Optimization

MobileNetV2 converted from FP32 to INT8 using QAT, reducing computational complexity by ~75% while preserving accuracy.

Intra-layer and inter-layer pipelining maximize PE utilization and minimize inference latency.

8️⃣ Tools and Methodology

Design & Synthesis: Xilinx Vivado for RTL synthesis, implementation, and bitstream generation.

Hardware Description: Verilog/SystemVerilog for accelerator design.

Simulation & Verification: Vivado Simulator for functional and timing validation.

Model Training & Quantization: Python with TensorFlow (QAT) for training and INT8 conversion.

<img width="1196" height="1474" alt="image" src="https://github.com/user-attachments/assets/10f9e698-0103-469e-9d43-b65f93c15568" />

Input
•	Image size: 224 × 224 × 3 (RGB wafer image).
•	This serves as the raw input vector for the CNN accelerator.

Standard Convolution (3×3, same padding)
•	Operation: Applies 32 filters of size 3×3 across the input image with padding to preserve dimensions.
•	Output size: 224 × 224 × 32.
•	Processing elements: Each filter is implemented using multiple MAC PEs (Multiply-Accumulate Processing Elements) to perform convolution in parallel.
•	Purpose: Extracts low-level spatial features such as edges, scratches, and texture anomalies from wafer images.
•	Trained parameters: Weights + biases.

Batch Normalization
•	Operation: Normalizes feature maps across each channel.
•	Parameters: Scale (γ) and shift (β), both learned during training.
•	Purpose: Stabilizes training, improves convergence, and prepares data for quantization.
•	Output: Same shape, normalized values.

ReLU6 Activation
•	Operation: Activation function with clipping at 6 → maps values to [0, 6].
•	Significance: When quantized to INT8, values fit into 4 bits, improving power efficiency.
•	Purpose: Introduces non-linearity while controlling dynamic range.

Max Pooling (2×2 stride)
•	Operation: Picks maximum value from every 2×2 region.
•	Processing element: Max PE compares 4 inputs and outputs the largest.
•	Output: Reduced spatial size (downsampling) while retaining strongest features.
•	Purpose: Reduces computation, increases robustness to translation.

Depthwise Convolution (3×3, zero padding)
•	Operation: Each input channel has its own filter (channel-wise convolution).
•	MAC usage: Much fewer MACs than standard conv, since no mixing across channels.
•	Purpose: Lightweight feature extraction, preserves channel independence.
•	Trained parameters: Depthwise filter weights + biases.

Pointwise Convolution (1×1)
•	Operation: Combines depthwise outputs across channels using 1×1 convolutions.
•	MAC usage: Intensive, but highly parallelizable via many PEs.
•	Purpose: Channel mixing and dimensionality adjustment.
•	Trained parameters: Weights + biases.

Iterative Blocks
The pipeline alternates [Depthwise → Pointwise → BN → ReLU6 → Pooling] multiple times:
•	Each block extracts higher-level features : From scratches and edges to larger defect structures and Reduces spatial resolution while increasing depth (channels).
•	Max pooling in early stages = 2×2 stride.
•	Final pooling = 7×7 stride → produces a compact global representation.

Flattening Layer
•	Converts the final 7×7×1280 tensor into a 1D vector.
•	This vector is the condensed “signature” of the wafer image.

Dense (Fully Connected) Layers
•	Dense 1: Connects flattened vector to hidden representation (all MAC operations).
•	Dense 2 (final): Maps to 9 output neurons (classes).
•	Parameters: Large number of MAC operations since every neuron connects to all inputs. Weights and biases are learned.
ArgMax
•	Operation: Selects the index of the maximum logit among the 9 classes.
Classes: 
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

📷 Integration of CNN Accelerator with PL, PS, and Camera (End-to-End Architecture)

The proposed system is implemented on a Zynq-7000 SoC from Xilinx, which integrates an Arm Cortex-A Processing System (PS) with FPGA Programmable Logic (PL). The system performs real-time wafer defect classification through coordinated interaction between the camera interface, FPGA accelerator, and Arm processor.

1️⃣ Camera to Programmable Logic (PL)
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

2️⃣ Processing System (PS) Role

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

3️⃣ PS to CNN Accelerator (Control Path)

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

4️⃣ Data Path (High-Speed Path)

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

5️⃣ End-to-End Data Flow

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
