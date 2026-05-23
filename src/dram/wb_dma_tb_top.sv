// DMA master -> wb_mux_2to1 -> wb_mem_model.
// exposes the host-side (m0) port so the test can preload DRAM the same way SPIBone (or equivalent) eventually will.

module wb_dma_tb_top #(
    parameter DATA_W = 32
)(
    input  logic             clk_i,
    input  logic             rst_i,
    input  logic             tpu_active,

    // hostside master
    input  logic [31:0]      m0_adr,
    input  logic [31:0]      m0_dat_w,
    input  logic             m0_we,
    input  logic             m0_stb,
    input  logic             m0_cyc,
    input  logic [3:0]       m0_sel,
    output logic [31:0]      m0_dat_r,
    output logic             m0_ack,

    // DMA control
    input  logic [31:0]      dma_start_addr,
    input  logic [15:0]      dma_word_count,
    input  logic             dma_we,
    input  logic             dma_start,
    output logic             dma_busy,
    output logic             dma_done,

    // DMA stream
    output logic [DATA_W-1:0] dma_rd_data,
    output logic              dma_rd_valid,
    input  logic              dma_rd_ready,
    input  logic [DATA_W-1:0] dma_wr_data,
    input  logic              dma_wr_valid,
    output logic              dma_wr_ready
);

    // DMA -> m1
    logic [31:0] m1_adr, m1_dat_w, m1_dat_r;
    logic        m1_we, m1_stb, m1_cyc, m1_ack;
    logic [3:0]  m1_sel;

    // mux -> slave
    logic [31:0] s_adr, s_dat_w, s_dat_r;
    logic        s_we, s_stb, s_cyc, s_ack;
    logic [3:0]  s_sel;

    wb_dma_master #(
        .AddrW (32),
        .DataW (DATA_W)
    ) u_dma (
        .clk_i        (clk_i),
        .rst_i        (rst_i),
        .start_addr_i (dma_start_addr),
        .word_count_i (dma_word_count),
        .we_i         (dma_we),
        .start_i      (dma_start),
        .busy_o       (dma_busy),
        .done_o       (dma_done),
        .rd_data_o    (dma_rd_data),
        .rd_valid_o   (dma_rd_valid),
        .rd_ready_i   (dma_rd_ready),
        .wr_data_i    (dma_wr_data),
        .wr_valid_i   (dma_wr_valid),
        .wr_ready_o   (dma_wr_ready),
        .wb_adr_o     (m1_adr),
        .wb_dat_o     (m1_dat_w),
        .wb_dat_i     (m1_dat_r),
        .wb_we_o      (m1_we),
        .wb_stb_o     (m1_stb),
        .wb_cyc_o     (m1_cyc),
        .wb_sel_o     (m1_sel),
        .wb_ack_i     (m1_ack)
    );

    wb_mux_2to1 u_mux (
        .tpu_active_i (tpu_active),
        .m0_adr_i (m0_adr), .m0_dat_w_i (m0_dat_w),
        .m0_we_i  (m0_we),  .m0_stb_i   (m0_stb),
        .m0_cyc_i (m0_cyc), .m0_sel_i   (m0_sel),
        .m0_dat_r_o (m0_dat_r), .m0_ack_o (m0_ack),
        .m1_adr_i (m1_adr), .m1_dat_w_i (m1_dat_w),
        .m1_we_i  (m1_we),  .m1_stb_i   (m1_stb),
        .m1_cyc_i (m1_cyc), .m1_sel_i   (m1_sel),
        .m1_dat_r_o (m1_dat_r), .m1_ack_o (m1_ack),
        .s_adr_o (s_adr),   .s_dat_w_o (s_dat_w),
        .s_we_o  (s_we),    .s_stb_o   (s_stb),
        .s_cyc_o (s_cyc),   .s_sel_o   (s_sel),
        .s_dat_r_i (s_dat_r), .s_ack_i (s_ack)
    );

    wb_mem_model #(
        .DEPTH_LOG2 (10),
        .DATA_W     (DATA_W)
    ) u_mem (
        .clk_i (clk_i), .rst_i (rst_i),
        .adr_i (s_adr), .dat_i (s_dat_w), .dat_o (s_dat_r),
        .we_i  (s_we),  .stb_i (s_stb),   .cyc_i (s_cyc),
        .sel_i (s_sel), .ack_o (s_ack)
    );

endmodule