# gf180mcu Project: SlugTPU

SlugTPU is an open source tensor processing unit that is designed to accelerate quantized neural network inference. We feature a parameterizable N x N systolic array with a full scalar post processing pipleline, on-chip SRAM, SPI host communication, and off-chip DRAM support via LiteDRAM. The design runs INT8 matrix multiplications with 32 bit accumulation, with hardware quantization to convert outputs back into INT8 for layer chaining.

This ASIC currently targets the GF180MCU process node.

> Part of the UC Santa Cruz CSE 127A/B Capstone Course

## Architecture

Our datapath can be organized into three major sections: the **compute core**, the **memory hierarchy**, and the **host interface**.

### Compute Core

The compute core performs tiled matrix multiplication that are followed by per element post-processing.

**Systolic Array**: A parameterizable N x N grid of processing elements (our current default is 8 x 8, which provides 64 MACs per cycle). Activations flow from left to right and partial sums accumulate from top to bottom. Weights are loaded top-down through a chain of shift registers Each PE performs a signed 8-bit multiply-accumulate into a 32 bit accumulator. 

The weight registers are designed to be double buffered, which allows the next layer's weights to be loaded while the current inference is still running, eliminating dead time between layers.

**Scalar Post Processing Pipeline**: A elastic pipeline that processes the systolic array's 32 bit output column by column in 4 stages:

1. **Bias Add**: Adds a 32 bit bias term per output channel
2. **ReLU**: Clamps negative values to zero
3. **Subtract Zero-Point**: Adjusts for quantization offset
4. **Fixed Point Scale + Quantize**: Multiplies by a 32 bit fixed point scale factor, rounds, and saturates to INT8

All stages use valid/ready elastic handshaking for backpressure safe pipelining.

### Memory Hierarchy

**On-Chip SRAM**: Eight SRAM blocks that store activations and intermediate results. The SRAM controller supports simultaneous read and write through an AXI-Lite interface with separate read/write channels. Address decoding uses the bottom 2 bits for bank selection and the upper bits for the intra bank address.

**Off-Chip DRAM**: Full model weights and potentially activation tensors will live in external DRAM. The design will interface with DRAM through a LiteDRAM controller exposing a Wishbone B4 port.

### Host Interface

Our host interfaces with the TPU via SPI. The host loads model data and instructions into DRAM using SPIBone as a bridge, and then sends a issues a flag to `wb_mux_2to1.sv` to give access to the TPU to begin execution.


---

## ISA

SlugTPU uses a CISC-style instruction set where each instruction maps to a high-level data movement or compute operation. Instructions are fetched from DRAM and decoded by the control unit.

| Instruction | Description |
|---|---|
| `Gmem2Smem` | DRAM to SRAM transfer |
| `Smem2Gmem` | SRAM to DRAM transfer |
| `Load_bias/zp/scale` | Load scalar parameters |
| `Load_weights` | Shift weights into systolic array |
| `Matmul` | Read activations, performs tiled matmul |
| `do_relu` | Activation function |
| `to_host_spi` | Send results to host |
| `exit` | Stop execution, return to IDLE |

---

## Verification

All RTL modules are verified with cocotb testbenches driven by pytest. The verification framework follows a producer–consumer model with Python reference models.

**The test framework currently covers:**
- Processing element (PE): MAC correctness, double buffer bank switching
- Systolic array (2 x 2 and N x N): full matrix multiply against NumPy reference
- Scalar pipeline: bias, ReLU, zero-point subtraction, fixed point quantization
- SRAM controller: read/write transactions, bank addressing
- SPI slave: host communication protocol
- FIFO: fill/drain, backpressure, boundary conditions
- Data loader: streaming activation/weight data into compute units
- Triangle shifter: input staggering for systolic array feeding

---


## Prerequisites

We use a custom fork of the [gf180mcuD PDK variant](https://github.com/wafer-space/gf180mcu) until all changes have been upstreamed.

To clone the latest PDK version, simply run `make clone-pdk`.

In the next step, install LibreLane by following the Nix-based installation instructions: https://librelane.readthedocs.io/en/latest/installation/nix_installation/index.html

## Implement the Design

This repository contains a Nix flake that provides a shell with the [`leo/gf180mcu`](https://github.com/librelane/librelane/tree/leo/gf180mcu) branch of LibreLane.

Simply run `nix-shell` in the root of this repository.

> [!NOTE]
> Since we are working on a branch of LibreLane, OpenROAD needs to be compiled locally. This will be done automatically by Nix, and the binary will be cached locally. 

With this shell enabled, run the implementation:

```
make librelane
```

## View the Design

After completion, you can view the design using the OpenROAD GUI:

```
make librelane-openroad
```

Or using KLayout:

```
make librelane-klayout
```

## Copying the Design to the Final Folder

To copy your latest run to the `final/` folder in the root directory of the repository, run the following command:

```
make copy-final
```

This will only work if the last run was completed without errors.

## Verification and Simulation

We use [cocotb](https://www.cocotb.org/), a Python-based testbench environment, for the verification of the chip.
The underlying simulator is Icarus Verilog (https://github.com/steveicarus/iverilog).

The testbenches are located in `cocotb`.  The chip top-level testbench is located inside as `chip_top_tb.py`. To run the chip-level RTL simulation, run the following command:

```
make sim
```

To run the GL (gate-level) simulation, run the following command:

```
make sim-gl
```

In addition, module specific tests are located inside `cocotb` as well as `test_*.py`. To run the module specific RTL simulation, such as the sysray_nxn module for instance, run

```
make sim-sysray-nxn
```

All RTL modules are verified with cocotb testbenches driven by pytest. The verification framework follows a producer–consumer model with Python reference models.

**The test framework currently covers:**
- Processing element (PE): MAC correctness, double buffer bank switching
- Systolic array (2 x 2 and N x N): full matrix multiply against NumPy reference
- Scalar pipeline: bias, ReLU, zero-point subtraction, fixed point quantization
- SRAM controller: read/write transactions, bank addressing
- SPI slave: host communication protocol
- FIFO: fill/drain, backpressure, boundary conditions
- Data loader: streaming activation/weight data into compute units
- Triangle shifter: input staggering for systolic array feeding

### Available Test Targets
| Target | Module Under Test |
|---|---|
| `sim-sysray-nxn` | N x N systolic array |
| `sim-scalar-pipe` | Test scalar units |
| `sim-scalar-stage` | Test scalar units, including loading |
| `sim-add-n` | Vectorized bias adder |
| `sim-relu-n` | Vectorized ReLU |
| `sim-scale-n` | Vectorized fixed point scale |
| `sim-fifo` | FIFO |
| `sim-spi` | SPI slave |
| `sim-sram` | SRAM controller |
| `sim-activation-sram` | Activation SRAM |
| `sim-tri` | Triangle shifter |
| `sim-load` | Data loader |
---


> [!NOTE]
> You need to have the latest implementation of your design in the `final/` folder. After implementing the design, execute 'make copy-final' to copy all necessary files.

In both cases, a waveform file will be generated under `cocotb/sim_build/chip_top.fst`.
You can view it using a waveform viewer, for example, [GTKWave](https://gtkwave.github.io/gtkwave/).

```
make sim-view
```

You can now update the testbench according to your design.

## Implementing Your Own Design

The source files for this template can be found in the `src/` directory. `chip_top.sv` defines the top-level ports and instantiates `chip_core`, chip ID (QR code) and the wafer.space logo. To allow for the default bonding setup, do not change the number of pads in order to keep the original bondpad positions. To be compatible with the default breakout PCB, do not change any of the power or ground pads. However, you can change the type of the signal pads, e.g. to bidirectional, input-only or e.g. analog pads. The template provides the `NUM_INPUT` and `NUM_BIDIR` parameters for this purpose.

The actual pad positions are defined in the LibreLane configuration file under `librelane/config.yaml`. The variables `PAD_SOUTH`/`PAD_EAST`/`PAD_NORTH`/`PAD_WEST` determine the respective pad placement. The LibreLane configuration also allows you to customize the flow (enable or disable steps), specify the source files, set various variables for the steps, and instantiate macros. For more information about the configuration, please refer to the LibreLane documentation: https://librelane.readthedocs.io/en/latest/

To implement your own design, simply edit `chip_core.sv`. The `chip_core` module receives the clock and reset, as well as the signals from the pads defined in `chip_top`. As an example, a 42-bit wide counter is implemented.

> [!NOTE]
> For more comprehensive SystemVerilog support, enable the `USE_SLANG` variable in the LibreLane configuration.

## Choosing a Different Slot Size

The template supports the following slot sizes: `1x1`, `0p5x1`, `1x0p5`, `0p5x0p5`.
By default, the design is implemented using the `1x1` slot definition.

To select a different slot size, simply set the `SLOT` environment variable.
This can be done when invoking a make target:

```
SLOT=0p5x0p5 make librelane
```

Alternatively, you can export the slot size:

```
export SLOT=0p5x0p5
```

You can change the slot that is selected by default in the Makefile by editing the value of `DEFAULT_SLOT`.

## Building a Standalone Padring for Analog Design

To build just the padring without any standard cell rows, digital routing or filler cells, run the following command:

```
make librelane-padring
```

It is also possible to build the padring for other slot sizes:

```
SLOT=0p5x0p5 make librelane-padring
```

## Precheck

To check whether your design is suitable for manufacturing, run the [gf180mcu-precheck](https://github.com/wafer-space/gf180mcu-precheck) with your layout.
