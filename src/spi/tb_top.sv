// tb_top.sv
// cocotb top-level wrapper.
// Instantiates the full system and exposes all internal signals
// that the cocotb testbench needs to drive or observe.
//
// cocotb drives: clk_i, rst_i, spi_*, tpu_busy_i, tpu_done_i, axi_*ready/valid inputs
// cocotb observes: spi_miso_o, tpu_start_o, tpu_reset_o, tpu_*_o, axi_* outputs
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none
`timescale 1ns/1ps

module tb_top;

    // ------------------------------------------------------------------
    // Clocks and reset
    // ------------------------------------------------------------------
    logic clk_i;
    logic rst_i;

    // ------------------------------------------------------------------
    // SPI
    // ------------------------------------------------------------------
    logic spi_sck_i;
    logic spi_mosi_i;
    logic spi_miso_o;
    logic spi_cs_n_i;

    // ------------------------------------------------------------------
    // TPU control / status (driven by cocotb to simulate TPU behaviour)
    // ------------------------------------------------------------------
    logic        tpu_start_o;
    logic        tpu_reset_o;
    logic [31:0] tpu_input_addr_o;
    logic [31:0] tpu_output_addr_o;
    logic [31:0] tpu_length_o;
    logic        tpu_busy_i;
    logic        tpu_done_i;

    // ------------------------------------------------------------------
    // AXI4 signals (cocotb acts as the AXI slave / fake SDRAM)
    // ------------------------------------------------------------------
    logic        axi_awvalid;
    logic        axi_awready;
    logic [31:0] axi_awaddr;
    logic [3:0]  axi_awid;
    logic [7:0]  axi_awlen;
    logic [1:0]  axi_awburst;

    logic        axi_wvalid;
    logic        axi_wready;
    logic [31:0] axi_wdata;
    logic [3:0]  axi_wstrb;
    logic        axi_wlast;

    logic        axi_bvalid;
    logic        axi_bready;
    logic [1:0]  axi_bresp;

    logic        axi_arvalid;
    logic        axi_arready;
    logic [31:0] axi_araddr;
    logic [3:0]  axi_arid;
    logic [7:0]  axi_arlen;
    logic [1:0]  axi_arburst;

    logic        axi_rvalid;
    logic        axi_rready;
    logic [31:0] axi_rdata;
    logic        axi_rlast;
    logic [1:0]  axi_rresp;

    // ------------------------------------------------------------------
    // Internal Wishbone wires (visible for monitoring)
    // ------------------------------------------------------------------
    logic [31:0] spiwb_adr, spiwb_dat_w, spiwb_dat_r;
    logic        spiwb_we, spiwb_stb, spiwb_cyc, spiwb_ack;

    logic [31:0] wbm0_adr, wbm0_dat_w, wbm0_dat_r;
    logic        wbm0_we, wbm0_stb, wbm0_cyc, wbm0_ack;

    logic [31:0] wbm1_adr, wbm1_dat_w, wbm1_dat_r;
    logic        wbm1_we, wbm1_stb, wbm1_cyc, wbm1_ack;

    // ------------------------------------------------------------------
    // DUT instantiations
    // ------------------------------------------------------------------

    spibone_wb u_spibone (
        .clk_i      (clk_i),
        .rst_i      (rst_i),
        .spi_sck_i  (spi_sck_i),
        .spi_mosi_i (spi_mosi_i),
        .spi_miso_o (spi_miso_o),
        .spi_cs_n_i (spi_cs_n_i),
        .wb_adr_o   (spiwb_adr),
        .wb_dat_o   (spiwb_dat_w),
        .wb_dat_i   (spiwb_dat_r),
        .wb_we_o    (spiwb_we),
        .wb_stb_o   (spiwb_stb),
        .wb_cyc_o   (spiwb_cyc),
        .wb_ack_i   (spiwb_ack)
    );

    wb_decoder u_decoder (
        .clk_i      (clk_i),
        .rst_i      (rst_i),
        .wbs_adr_i  (spiwb_adr),
        .wbs_dat_i  (spiwb_dat_w),
        .wbs_dat_o  (spiwb_dat_r),
        .wbs_we_i   (spiwb_we),
        .wbs_stb_i  (spiwb_stb),
        .wbs_cyc_i  (spiwb_cyc),
        .wbs_ack_o  (spiwb_ack),
        .wbm0_adr_o (wbm0_adr),
        .wbm0_dat_o (wbm0_dat_w),
        .wbm0_dat_i (wbm0_dat_r),
        .wbm0_we_o  (wbm0_we),
        .wbm0_stb_o (wbm0_stb),
        .wbm0_cyc_o (wbm0_cyc),
        .wbm0_ack_i (wbm0_ack),
        .wbm1_adr_o (wbm1_adr),
        .wbm1_dat_o (wbm1_dat_w),
        .wbm1_dat_i (wbm1_dat_r),
        .wbm1_we_o  (wbm1_we),
        .wbm1_stb_o (wbm1_stb),
        .wbm1_cyc_o (wbm1_cyc),
        .wbm1_ack_i (wbm1_ack)
    );

    tpu_regs u_tpu_regs (
        .clk_i             (clk_i),
        .rst_i             (rst_i),
        .wb_adr_i          (wbm1_adr),
        .wb_dat_i          (wbm1_dat_w),
        .wb_dat_o          (wbm1_dat_r),
        .wb_we_i           (wbm1_we),
        .wb_stb_i          (wbm1_stb),
        .wb_cyc_i          (wbm1_cyc),
        .wb_ack_o          (wbm1_ack),
        .tpu_start_o       (tpu_start_o),
        .tpu_reset_o       (tpu_reset_o),
        .tpu_input_addr_o  (tpu_input_addr_o),
        .tpu_output_addr_o (tpu_output_addr_o),
        .tpu_length_o      (tpu_length_o),
        .tpu_busy_i        (tpu_busy_i),
        .tpu_done_i        (tpu_done_i)
    );

    wb_to_axi4 u_wb_axi (
        .clk          (clk_i),
        .rst          (rst_i),
        .wb_adr_i     (wbm0_adr),
        .wb_dat_i     (wbm0_dat_w),
        .wb_dat_o     (wbm0_dat_r),
        .wb_we_i      (wbm0_we),
        .wb_stb_i     (wbm0_stb),
        .wb_cyc_i     (wbm0_cyc),
        .wb_ack_o     (wbm0_ack),
        .axi_awvalid  (axi_awvalid), .axi_awready  (axi_awready),
        .axi_awaddr   (axi_awaddr),  .axi_awid     (axi_awid),
        .axi_awlen    (axi_awlen),   .axi_awburst  (axi_awburst),
        .axi_wvalid   (axi_wvalid),  .axi_wready   (axi_wready),
        .axi_wdata    (axi_wdata),   .axi_wstrb    (axi_wstrb),
        .axi_wlast    (axi_wlast),
        .axi_bvalid   (axi_bvalid),  .axi_bready   (axi_bready),
        .axi_bresp    (axi_bresp),
        .axi_arvalid  (axi_arvalid), .axi_arready  (axi_arready),
        .axi_araddr   (axi_araddr),  .axi_arid     (axi_arid),
        .axi_arlen    (axi_arlen),   .axi_arburst  (axi_arburst),
        .axi_rvalid   (axi_rvalid),  .axi_rready   (axi_rready),
        .axi_rdata    (axi_rdata),   .axi_rlast    (axi_rlast),
        .axi_rresp    (axi_rresp)
    );

    // Waveform dump for GTKWave
    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
    end

endmodule

`default_nettype wire