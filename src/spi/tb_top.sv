// tb_top.sv
// cocotb top-level wrapper for new tpu_regs interface.
//
// SPDX-License-Identifier: Apache-2.0
`default_nettype none
`timescale 1ns/1ps

module tb_top;

    logic clk_i;
    logic rst_i;

    logic spi_sck_i;
    logic spi_mosi_i;
    logic spi_miso_o;
    logic spi_cs_n_i;

    logic [1:0]  tpu_state_i;
    logic [31:0] tpu_pc_addr_o;
    logic        tpu_pc_stb_o;
    logic        tpu_enable_o;
    logic        tpu_done_o;

    logic [31:0] spiwb_adr, spiwb_dat_w, spiwb_dat_r;
    logic        spiwb_we, spiwb_stb, spiwb_cyc, spiwb_ack;

    logic [31:0] wbm0_adr, wbm0_dat_w, wbm0_dat_r;
    logic        wbm0_we, wbm0_stb, wbm0_cyc, wbm0_ack;

    logic [31:0] wbm1_adr, wbm1_dat_w, wbm1_dat_r;
    logic        wbm1_we, wbm1_stb, wbm1_cyc, wbm1_ack;

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
        .clk_i         (clk_i),
        .rst_i         (rst_i),
        .wb_adr_i      (wbm1_adr),
        .wb_dat_i      (wbm1_dat_w),
        .wb_dat_o      (wbm1_dat_r),
        .wb_we_i       (wbm1_we),
        .wb_stb_i      (wbm1_stb),
        .wb_cyc_i      (wbm1_cyc),
        .wb_ack_o      (wbm1_ack),
        .tpu_pc_addr_o (tpu_pc_addr_o),
        .tpu_pc_stb_o  (tpu_pc_stb_o),
        .tpu_enable_o  (tpu_enable_o),
        .tpu_state_i   (tpu_state_i),
        .tpu_done_o    (tpu_done_o)
    );

    // wbm0 stub — not needed for tpu_regs tests
    assign wbm0_dat_r = 32'hDEAD_BEEF;
    assign wbm0_ack   = wbm0_stb & wbm0_cyc;

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
    end

endmodule

`default_nettype wire