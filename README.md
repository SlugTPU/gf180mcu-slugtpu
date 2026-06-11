# gf180mcu Project: SlugTPU

Project template for wafer.space MPW runs using the gf180mcu PDK.

SlugTPU is an open source tensor processing unit (TPU) that is designed to accelerate quantized neural network inference. Specifically, our ASIC implements w8a8 quantization (8 bit weights, 8 bit activations), and optimizes the operation: $q*[ReLu(A @ W + b)-z_p]$. 

SlugTPU features a 8x8 Matrix Multiply Unit (MXU) with inbuilt tiling support, followed by a full scalar post processing pipleline, that add the bias term and performs requantization. Our chip has a 2KiB on-chip SRAM data cache and 256 byte instruction cache, which access an off chip DRAM via a custom DRAM controller. To communicate with a host microcontroller, SlugTPU exposes a Memory Mapped SPI interface.

This ASIC currently targets the GF180MCU process node.

> Part of the 2026 UC Santa Cruz CSE 127A/B Capstone Course

<img width="2568" height="1966" alt="image" src="https://github.com/user-attachments/assets/2e951d8f-97ee-4541-8570-14210098a10a" />


## Physical Specifications
<br>

| | |
|---|---|
| Process | GF180MCU, wafer.space 1x1 slot |
| Pads | 12 input, 40 bidirectional, 2 analog (+ power) |
| Host interface | SPI mode 0 |
| Status pin | `TPU_ACTIVE` on bidirectional pad 38 |
| Compute | 8x8 INT8 systolic array (64 MACs/cycle), 32-bit accumulation |
| On-chip memory | 2 KiB activation SRAM + 2 KiB weight SRAM + 256 B instruction FIFO (`gf180mcu_ocd_ip_sram` 256x8 macros) |
| Off-chip memory | 32 MiB SDR SDRAM (Micron MT48LC16M16A2) |
| Standard cells | `gf180mcu_as_sc_mcu7t3v3` |

## Architecture

Our ASIC can be organized into two major sections: the **compute core** and the **memory hierarchy**

### Compute Core

The compute core performs tiled matrix multiplication that are followed by per channel post-processing.
<br>
<p align="center">
  <img width="1205" alt="ComputECore" src="https://github.com/user-attachments/assets/db1ad5d9-96c4-40ec-a1d7-f299cb030d30" />
</p>
<br>
**Matrix Multiply Unit**: A parameterizable N x N systolic array of processing elements (our current default is 8 x 8, which provides 64 MACs per cycle). Activations flow from left to right and partial sums accumulate from top to bottom. Weights are loaded top-down through a chain of shift registers. Each PE performs a signed 8-bit multiply-accumulate into a 32 bit accumulator. 

The weight registers are designed to be double buffered, which allows the next layer's weights to be loaded while the current inference is still running, eliminating dead time between layers. For inner dimensions larger than 8, partial sums recirculate from the bottom of the array back into the top, summing successive 8-deep K-tiles without wider accumulators.

**Scalar Post Processing Pipeline**: A elastic pipeline that processes the systolic array's 32 bit output column by column in 4 stages:

1. **Bias Add**: Adds a 32 bit bias term per output channel
2. **ReLU**: Clamps negative values to zero. Can be toggled.
3. **Subtract Zero-Point**: Adjusts for quantization offset
4. **Fixed Point Scale + Quantize**: Multiplies by a 32 bit fixed point scale factor, rounds, and saturates to INT8

### Memory Hierarchy
<br>
<p align="center">
  <img width="473" alt="Memory" src="https://github.com/user-attachments/assets/fa02d3e5-36a3-4c32-a581-7fe9f9df3434" />
</p>
<br>

**On-Chip SRAM**: 2 Banks of eight SRAM blocks each. One stores activations, scalar data, and intermediate results, and the other stores weights. We interface with these banks via an atomic memory interface unit. 

**Off-Chip DRAM**: Model weights and activation tensors live in a 32 MiB external SDR SDRAM (Micron MT48LC16M16A2). The compute core interfaces with a SDRAM controler via a 64 bit Wishbone bus.

**SPI Interface**: Our host interfaces with the TPU via SPI. The host uses SPI to load model data and instructions into DRAM, and then writes a program counter to a specified address to tell the TPU to begin execution at that memory location. Additionally, debug options can be enabled via memory mapped SPI commands.


---

## ISA

SlugTPU uses a CISC-style instruction set where each instruction maps to a high-level data movement or compute operation. Instructions are fetched from DRAM and decoded by the control unit. Additionally, we implement a full instruction level parallelism between a limited amount of instructions.

| Instruction | Description | Length (bits) |
|---|---|---|
| `Dram2Sram` | DRAM to SRAM transfer | 32 |
| `Sram2Dram` | SRAM to DRAM transfer | 32 |
| `Load_bias/zp/scale` | Load scalar parameters | 16 |
| `Load_weights` | Shift weights into systolic array from SRAM | 16 | 
| `Matmul` | Read activations, performs tiled matmul | 24 |
| `Pipeline_setup` | Toggles Relu on/off, sets K dimension size for tiling | 16 |
| `exit` | Stop execution, return to IDLE | 8 |

---

## Dependencies

To manage all dependencies, the project template includes a Nix shell with all the required tools.
Install Nix and LibreLane by following the Nix-based installation instructions: https://librelane.readthedocs.io/en/latest/installation/nix_installation/index.html
To activate the shell, simply run `nix-shell` in the root directory of this repository. The subsequent steps assume that you are in the Nix shell of the project template.

## Prerequisites

The project template uses the open_pdks gf180mcuD variant of the PDK.
To clone the latest PDK version via [Ciel](https://github.com/fossi-foundation/ciel), run `make clone-pdk`.

## Implement the Design

With the Nix shell enabled, run the implementation:

```
make librelane
```

You can find all output artifacts in the `librelane/runs/<timestamp>/` directory.

## View the Design

After completion, you can view the design using the OpenROAD GUI:

```
make librelane-openroad
```

Or using KLayout:

```
make librelane-klayout
```

## Verification and Simulation

For the verification of the chip we use [cocotb](https://www.cocotb.org/). Cocotb is a Python-based testbench environment. The simulator that is used by the project template is [Icarus Verilog](https://github.com/steveicarus/iverilog).

The testbench is located in `cocotb/chip_top_tb.py`. To run the RTL simulation, run the following command:

Note: The top-level testbench tests with the official Micron MT48LC16M16A2 SDRAM Verilog module. Due to copyright concerns, this file is not included inside the repository. You must source the model somewhere yourself. 

If you (legally) obtained this model, you should place this model inside `src/dram/sdram_model_mt48lc16m16a2.v`.

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
| `sim-scalar-load` | Data loader |
| `sim-compute-core` | Full compute core (MXU + SRAM + scalar pipeline) |
| `sim-control-decoder` | Instruction decoder (includes a full single layer) |
| `sim-control-top` | Control unit FSM |
| `sim-wb-sdr-mt48lc16m16a2` | SDRAM controller (requires Micron model) |
| `sim-tpu-soc-with-dram` | SoC + SDRAM model (requires Micron model) |
| `sim-tpu-instructions` | SoC running assembled programs (requires Micron model) |
| `sim-general-test-example` / `sim-general-8x16` / `sim-general-8x24` | e2e workloads at chip level (requires Micron model) |
---


> [!NOTE]
> You need to have the latest implementation of your design in the `final/` folder. After a run has completed without errors, the final views will be copied to `final/`.

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

## Select Different IP Libraries

The project template has support for selecting libraries with the below environment variables:

| Env  | Available Values                                                          | Description                |
|------|---------------------------------------------------------------------------|----------------------------|
| SCL  | gf180mcu_fd_sc_mcu7t5v0, gf180mcu_fd_sc_mcu9t5v0, gf180mcu_as_sc_mcu7t3v3 | The standard cell library. |
| PAD  | gf180mcu_fd_io, gf180mcu_ocd_io                                           | The I/O pad library.       |
| SRAM | gf180mcu_fd_ip_sram, gf180mcu_ocd_ip_sram                                 | The SRAM library.          |

For example, to build the 0p5x0p5 chip with 3v3 libraries:

```
SLOT=0p5x0p5 SCL=gf180mcu_as_sc_mcu7t3v3 PAD=gf180mcu_ocd_io SRAM=gf180mcu_ocd_ip_sram make librelane
```

The default values can be changed in the Makefile.

> [!NOTE]
> Not all of the community-created IPs have been tested yet, so support for them is experimental!

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
