// SPDX-FileCopyrightText: © 2025 XXX Authors
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    input  wire VDD,
    input  wire VSS,
    `endif

    input  wire clk,       // clock
    input  wire rst_n,     // reset (active low)

    input  wire [NUM_INPUT_PADS-1:0] input_in,   // Input value
    output wire [NUM_INPUT_PADS-1:0] input_pu,   // Pull-up
    output wire [NUM_INPUT_PADS-1:0] input_pd,   // Pull-down

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,   // Input value
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,  // Output value
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,   // Output enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,   // Input type (0=CMOS Buffer, 1=Schmitt Trigger)
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,   // Slew rate (0=fast, 1=slow)
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,   // Input enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,   // Pull-up
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,   // Pull-down

    input  wire [NUM_ANALOG_PADS-1:0] analog  // Analog
);

    // input_in[0] = spi_clk, [1] = spi_cs_n, [2] = spi_mosi; [11:3] unused
    assign input_pu = '0;
    assign input_pd = '0;

    logic _unused_inputs;
    assign _unused_inputs = &input_in[NUM_INPUT_PADS-1:3];

    logic _unused_analog;
    assign _unused_analog = &analog;

    // Intermediate wires from tpu_soc
    logic        spi_miso;
    logic [15:0] sdr_dq_o;
    logic        sdr_dq_oe;
    logic [12:0] sdr_addr;
    logic [1:0]  sdr_ba;
    logic        sdr_cke, sdr_cs_n, sdr_ras_n, sdr_cas_n, sdr_we_n;
    logic [1:0]  sdr_dqm;

    // Bidir pad assignment:
    //   [15:0]  = sdr_dq[15:0]     (bidir, OE from SDRAM controller)
    //   [16]    = spi_miso          (output)
    //   [17]    = sdr_cke           (output)
    //   [18]    = sdr_cs_n          (output)
    //   [19]    = sdr_ras_n         (output)
    //   [20]    = sdr_cas_n         (output)
    //   [21]    = sdr_we_n          (output)
    //   [22]    = sdr_dqm[0]        (output)
    //   [23]    = sdr_dqm[1]        (output)
    //   [36:24] = sdr_addr[12:0]    (output)
    //   [37]    = sdr_ba[0]         (output)
    //   [38]    = sdr_ba[1]         (output)
    //   [39]    = clk → SDRAM CLK  (output)
    assign bidir_out[15:0]  = sdr_dq_o;
    assign bidir_out[16]    = spi_miso;
    assign bidir_out[17]    = sdr_cke;
    assign bidir_out[18]    = sdr_cs_n;
    assign bidir_out[19]    = sdr_ras_n;
    assign bidir_out[20]    = sdr_cas_n;
    assign bidir_out[21]    = sdr_we_n;
    assign bidir_out[22]    = sdr_dqm[0];
    assign bidir_out[23]    = sdr_dqm[1];
    assign bidir_out[36:24] = sdr_addr;
    assign bidir_out[37]    = sdr_ba[0];
    assign bidir_out[38]    = sdr_ba[1];
    assign bidir_out[39]    = clk;

    assign bidir_oe[15:0]  = {16{sdr_dq_oe}};
    assign bidir_oe[39:16] = '1;
    assign bidir_ie        = ~bidir_oe;
    assign bidir_cs        = '0;
    assign bidir_sl        = '0;
    assign bidir_pu        = '0;
    assign bidir_pd        = '0;

    tpu_soc #(
        .sys_clk_mhz_p(25)
    ) i_tpu_soc (
`ifdef USE_POWER_PINS
        .VDD(VDD),
        .VSS(VSS),
`endif
        .clk_i      (clk),
        .rst_i      (~rst_n),
        .spi_clk_i  (input_in[0]),
        .spi_cs_ni  (input_in[1]),
        .spi_mosi_i (input_in[2]),
        .spi_miso_o (spi_miso),
        .sdr_dq_i   (bidir_in[15:0]),
        .sdr_dq_o   (sdr_dq_o),
        .sdr_dq_oe_o(sdr_dq_oe),
        .sdr_addr_o (sdr_addr),
        .sdr_ba_o   (sdr_ba),
        .sdr_cke_o  (sdr_cke),
        .sdr_cs_no  (sdr_cs_n),
        .sdr_ras_no (sdr_ras_n),
        .sdr_cas_no (sdr_cas_n),
        .sdr_we_no  (sdr_we_n),
        .sdr_dqm_o  (sdr_dqm)
    );

endmodule

`default_nettype wire
