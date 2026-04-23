// testing wb_mux_2to1_pipe + our pipelined mem model wired together. (we expose both master ports and tpu_active for cocotb to drive)

module wb_mux_pipe_tb_top (
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
    output logic        m0_stall,

    //TPU side
    input  logic [31:0] m1_adr,
    input  logic [31:0] m1_dat_w,
    input  logic        m1_we,
    input  logic        m1_stb,
    input  logic        m1_cyc,
    input  logic [3:0]  m1_sel,
    output logic [31:0] m1_dat_r,
    output logic        m1_ack,
    output logic        m1_stall
);

    logic [31:0] s_adr, s_dat_w, s_dat_r;
    logic        s_we, s_stb, s_cyc, s_ack, s_stall;
    logic [3:0]  s_sel;

    wb_mux_2to1_pipe u_mux (
        .tpu_active (tpu_active),
        .m0_adr     (m0_adr),
        .m0_dat_w   (m0_dat_w),
        .m0_we      (m0_we),
        .m0_stb     (m0_stb),
        .m0_cyc     (m0_cyc),
        .m0_sel     (m0_sel),
        .m0_dat_r   (m0_dat_r),
        .m0_ack     (m0_ack),
        .m0_stall   (m0_stall),
        .m1_adr     (m1_adr),
        .m1_dat_w   (m1_dat_w),
        .m1_we      (m1_we),
        .m1_stb     (m1_stb),
        .m1_cyc     (m1_cyc),
        .m1_sel     (m1_sel),
        .m1_dat_r   (m1_dat_r),
        .m1_ack     (m1_ack),
        .m1_stall   (m1_stall),
        .s_adr      (s_adr),
        .s_dat_w    (s_dat_w),
        .s_we       (s_we),
        .s_stb      (s_stb),
        .s_cyc      (s_cyc),
        .s_sel      (s_sel),
        .s_dat_r    (s_dat_r),
        .s_ack      (s_ack),
        .s_stall    (s_stall)
    );

    wb_mem_model_pipe #(
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
        .ack_o  (s_ack),
        .stall_o(stall_o)
    );

endmodule
