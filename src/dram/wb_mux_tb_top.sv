// testing wb_mux_2to1 + our mem model wired together. (we expose both master ports and tpu_active for cocotb to drive)

module wb_mux_tb_top (
    input  logic        clk_i,
    input  logic        rst_i,
    input  logic        tpu_active,

    //SPI bridge side
    input  logic [31:0] m0_adr,
    input  logic [31:0] m0_dat_w,
    input  logic        m0_we,
    input  logic        m0_stb,
    input  logic        m0_cyc,
    input  logic [3:0]  m0_sel,
    output logic [31:0] m0_dat_r,
    output logic        m0_ack,

    //TPU side
    input  logic [31:0] m1_adr,
    input  logic [31:0] m1_dat_w,
    input  logic        m1_we,
    input  logic        m1_stb,
    input  logic        m1_cyc,
    input  logic [3:0]  m1_sel,
    output logic [31:0] m1_dat_r,
    output logic        m1_ack
);

    logic [31:0] s_adr, s_dat_w, s_dat_r;
    logic        s_we, s_stb, s_cyc, s_ack;
    logic [3:0]  s_sel;

    wb_mux_2to1 u_mux (
        .clk_i        (clk_i),
        .rst_i        (rst_i),
        .tpu_active_i (tpu_active),
        .m0_adr_i     (m0_adr),
        .m0_dat_w_i   (m0_dat_w),
        .m0_we_i      (m0_we),
        .m0_stb_i     (m0_stb),
        .m0_cyc_i     (m0_cyc),
        .m0_sel_i     (m0_sel),
        .m0_dat_r_o   (m0_dat_r),
        .m0_ack_o     (m0_ack),
        .m1_adr_i     (m1_adr),
        .m1_dat_w_i   (m1_dat_w),
        .m1_we_i      (m1_we),
        .m1_stb_i     (m1_stb),
        .m1_cyc_i     (m1_cyc),
        .m1_sel_i     (m1_sel),
        .m1_dat_r_o   (m1_dat_r),
        .m1_ack_o     (m1_ack),
        .s_adr_o      (s_adr),
        .s_dat_w_o    (s_dat_w),
        .s_we_o       (s_we),
        .s_stb_o      (s_stb),
        .s_cyc_o      (s_cyc),
        .s_sel_o      (s_sel),
        .s_dat_r_i    (s_dat_r),
        .s_ack_i      (s_ack)
    );

    wb_mem_model #(
        .DEPTH_LOG2 (10)
    ) u_slave (
        .clk_i  (clk_i),
        .rst_i  (rst_i),
        .adr_i  (s_adr),
        .dat_i  (s_dat_w),
        .dat_o  (s_dat_r),
        .we_i   (s_we),
        .stb_i  (s_stb),
        .cyc_i  (s_cyc),
        .sel_i  (s_sel),
        .ack_o  (s_ack)
    );

endmodule
