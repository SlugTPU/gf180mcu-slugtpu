// same as wb_dma_tb_top.sv but with the real SDRAM vs wb_mem_model
//
// Host port (m0) is exposed for the test to preload/verify DRAM the same way SPIBone will in silicon. DMA control + stream are exposed

`default_nettype none

module wb_dma_sdram_tb_top #(
    parameter sys_clk_mhz_p   = 100,
    parameter dram_burst_p    = 4,
    parameter data_bits_p     = 16,
    parameter rows_p          = 8192,
    parameter cols_p          = 512,
    parameter banks_p         = 4,
    parameter sdr_addr_bits_p = 13,
    parameter usr_addr_bits_p = $clog2(rows_p) + $clog2(banks_p)
                              + $clog2(cols_p) - $clog2(dram_burst_p),
    parameter DATA_W          = data_bits_p * dram_burst_p
)(
    input  logic                       clk_i,
    input  logic                       rst_i,
    input  logic                       tpu_active,

    // host-side WB master (m0)
    input  logic [usr_addr_bits_p-1:0] m0_adr,
    input  logic [DATA_W-1:0]          m0_dat_w,
    input  logic                       m0_we,
    input  logic                       m0_stb,
    input  logic                       m0_cyc,
    input  logic [DATA_W/8-1:0]        m0_sel,
    output logic [DATA_W-1:0]          m0_dat_r,
    output logic                       m0_ack,

    // control
    input  logic [usr_addr_bits_p-1:0] dma_start_addr,
    input  logic [15:0]                dma_word_count,
    input  logic                       dma_we,
    input  logic                       dma_start,
    output logic                       dma_busy,
    output logic                       dma_done,

    // stream
    output logic [DATA_W-1:0]          dma_rd_data,
    output logic                       dma_rd_valid,
    input  logic                       dma_rd_ready,
    input  logic [DATA_W-1:0]          dma_wr_data,
    input  logic                       dma_wr_valid,
    output logic                       dma_wr_ready
);

    // DMA -> mux m1
    logic [usr_addr_bits_p-1:0] m1_adr;
    logic [DATA_W-1:0]          m1_dat_w, m1_dat_r;
    logic                       m1_we, m1_stb, m1_cyc, m1_ack;
    logic [DATA_W/8-1:0]        m1_sel;

    // mux -> SDRAM controller
    logic [usr_addr_bits_p-1:0] s_adr;
    logic [DATA_W-1:0]          s_dat_w, s_dat_r;
    logic                       s_we, s_stb, s_cyc, s_ack;
    logic [DATA_W/8-1:0]        s_sel;

    // SDRAM controller pins
    logic [data_bits_p-1:0]     sdr_dq_o;
    logic [sdr_addr_bits_p-1:0] sdr_addr;
    logic [1:0]                 sdr_ba;
    logic                       sdr_cke, sdr_cs_n, sdr_ras_n, sdr_cas_n, sdr_we_n;
    logic [1:0]                 sdr_dqm;
    logic                       sdr_oe;

    // bidirectional DQ between controller and model
    wire  [data_bits_p-1:0]     sdram_dq;
    assign sdram_dq = sdr_oe ? sdr_dq_o : 'z;

    wb_dma_master #(
        .AddrW (usr_addr_bits_p),
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

    wb_mux_2to1 #(
        .AdrW  (usr_addr_bits_p),
        .DataW (DATA_W)
    ) u_mux (
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

    wb_sdr_mt48lc16m16a_7e #(
        .sys_clk_mhz_p    (sys_clk_mhz_p),
        .dram_burst_p     (dram_burst_p),
        ._data_bits_p     (data_bits_p),
        ._rows_p          (rows_p),
        ._cols_p          (cols_p),
        ._banks_p         (banks_p),
        ._sdr_addr_bits_p (sdr_addr_bits_p)
    ) u_sdr_ctrl (
        .clk_i    (clk_i),
        .rst_i    (rst_i),
        .m_adr_i  (s_adr),
        .m_dat_i  (s_dat_w),
        .m_dat_o  (s_dat_r),
        .m_we_i   (s_we),
        .m_stb_i  (s_stb),
        .m_cyc_i  (s_cyc),
        .m_sel_i  (s_sel),
        .m_ack_o  (s_ack),
        .s_dq_i   (sdram_dq),
        .s_dq_o   (sdr_dq_o),
        .s_addr_o (sdr_addr),
        .s_ba_o   (sdr_ba),
        .s_cke_o  (sdr_cke),
        .s_cs_no  (sdr_cs_n),
        .s_ras_no (sdr_ras_n),
        .s_cas_no (sdr_cas_n),
        .s_we_no  (sdr_we_n),
        .s_dqm_o  (sdr_dqm),
        .oe_o     (sdr_oe)
    );

    mt48lc16m16a2 #(
        .addr_bits (sdr_addr_bits_p),
        .data_bits (data_bits_p),
        .col_bits  ($clog2(cols_p)),
        .mem_sizes (rows_p * cols_p - 1)
    ) u_sdr_model (
        .Dq    (sdram_dq),
        .Addr  (sdr_addr),
        .Ba    (sdr_ba),
        .Clk   (clk_i),
        .Cke   (sdr_cke),
        .Cs_n  (sdr_cs_n),
        .Ras_n (sdr_ras_n),
        .Cas_n (sdr_cas_n),
        .We_n  (sdr_we_n),
        .Dqm   (sdr_dqm)
    );

endmodule

`default_nettype wire