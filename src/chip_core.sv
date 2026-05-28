// SPDX-FileCopyrightText: © 2025 XXX Authors
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
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

   logic [1:0] rst_sync_d, rst_sync_q;

    // Intermediate wires from tpu_soc
    logic        spi_miso;
    logic [15:0] sdr_dq_o;
    logic        sdr_dq_oe;
    logic [12:0] sdr_addr;
    logic [1:0]  sdr_ba;
    logic        sdr_cke, sdr_cs_n, sdr_ras_n, sdr_cas_n, sdr_we_n;
    logic [1:0]  sdr_dqm;
    logic test_mode;

    // When test_mode is asserted, hold the SDRAM pads inactive. cke=0 freezes part, cs_n=1 deselects (everything else is don't-care under deselect). dq_oe=0 makes SDRAM act as if disconnected from wire
    wire sdr_cke_pad   = sdr_cke   & ~test_mode;
    wire sdr_cs_n_pad  = sdr_cs_n  |  test_mode;
    wire sdr_dq_oe_pad = sdr_dq_oe & ~test_mode;

    // Pad ring layout (1x1 slot).  [P] = power pad.  Bidir indices in [brackets].
    //
    //                                TOP (left → right)
    //  NC   NC  MISO CLK  BA1 BA0 A12 A11 [P] A10  A9  A8  A7  A6  A5  A4  A3
    //  |    |    |    |    |   |   |   |       |    |   |   |   |   |   |   |
    // ┌┴────┴────┴────┴────┴───┴───┴───┴───────┴────┴───┴───┴───┴───┴───┴───┴───┐
    // │                                                                            │
    // ├─ SPI_CLK  [i0]                                      DQ[14]   [14] ───── ─┤
    // ├─ SPI_CS_N [i1]                                      DQ[15]   [15] ───── ─┤
    // ├─ [PWR]                                              DQM[0]   [16] ───── ─┤
    // ├─ [PWR]                                              DQM[1]   [17] ───── ─┤
    // ├─ SPI_MOSI [i2]                                      SDR_CKE  [18] ───── ─┤
    // ├─ NC       [i3]                                      SDR_CS_N [19] ───── ─┤
    // ├─ NC       [i4]                                [PWR] ──────────────────── ─┤
    // ├─ NC       [i5]                                [PWR] ──────────────────── ─┤
    // ├─ [PWR]                                    SDR_RAS_N [20] ─────────────── ─┤
    // ├─ [PWR]                                    SDR_CAS_N [21] ─────────────── ─┤
    // ├─ NC       [i6]                             SDR_WE_N [22] ─────────────── ─┤
    // ├─ NC       [i7]                             SDR_A[0] [23] ─────────────── ─┤
    // ├─ NC       [i8]                             SDR_A[1] [24] ─────────────── ─┤
    // ├─ NC       [i9]                             SDR_A[2] [25] ─────────────── ─┤
    // ├─ NC      [i10]                                [PWR] ──────────────────── ─┤
    // ├─ NC      [i11]                                [PWR] ──────────────────── ─┤
    // ├─ [PWR]                                        [PWR] ──────────────────── ─┤
    // ├─ [PWR]                                        [PWR] ──────────────────── ─┤
    // └───┬──┬───┬───┬───┬───┬───┬───┬─[P]─┬───┬───┬───┬───┬───┬───┬───┬───────┘
    //     |  |   |   |   |   |   |   |     |   |   |   |   |   |   |   |
    //    CLK RST DQ0 DQ1 DQ2 DQ3 DQ4 DQ5  DQ6 DQ7 DQ8 DQ9 D10 D11 D12 D13
    //                                BOTTOM (left → right)
    assign bidir_out[15:0]  = sdr_dq_o;
    assign bidir_out[16]    = sdr_dqm[0];
    assign bidir_out[17]    = sdr_dqm[1];
    assign bidir_out[18]    = sdr_cke_pad;
    assign bidir_out[19]    = sdr_cs_n_pad;
    assign bidir_out[20]    = sdr_ras_n;
    assign bidir_out[21]    = sdr_cas_n;
    assign bidir_out[22]    = sdr_we_n;
    assign bidir_out[35:23] = sdr_addr;
    assign bidir_out[36]    = sdr_ba[0];
    assign bidir_out[37]    = sdr_ba[1];
    // assign bidir_out[38]    = clk;
    assign bidir_out[39]    = spi_miso;

    assign bidir_oe[15:0]  = {16{sdr_dq_oe_pad}};
    assign bidir_oe[39:16] = '1;

   assign rst_sync_d = {rst_sync_q[0], ~rst_n};

    generate
        if (NUM_BIDIR_PADS > 40) begin : g_tieoff_unused
            assign bidir_out[NUM_BIDIR_PADS-1:40] = '0;
            assign bidir_oe [NUM_BIDIR_PADS-1:40] = '0;
        end
    endgenerate

    assign bidir_ie        = ~bidir_oe;
    assign bidir_cs        = '0;
    assign bidir_sl        = '0;
    assign bidir_pu        = '0;
    assign bidir_pd        = '0;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         rst_sync_q <= '1;
      end else begin
         rst_sync_q <= rst_sync_d;
      end
   end

    tpu_soc #(
        .sys_clk_mhz_p(25)
    ) i_tpu_soc (
`ifdef USE_POWER_PINS
        .VDD(VDD),
        .VSS(VSS),
`endif
        .clk_i      (clk),
        .rst_i      (rst_sync_q[1]),
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
        .sdr_dqm_o  (sdr_dqm),
        .test_mode_o(test_mode)
    );

endmodule

`default_nettype wire
