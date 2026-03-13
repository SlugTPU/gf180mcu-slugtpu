module wb_mux_2to1 (
    input  logic tpu_active,  // 0 for SPI bridge to use bus, 1 = TPU

    //SPI Bridge (basically SPIBone)
    input  logic [31:0] m0_adr, m0_dat_w,
    input  logic  m0_we, m0_stb, m0_cyc,
    input  logic [3:0] m0_sel,
    output logic [31:0] m0_dat_r,
    output logic m0_ack,

    //TPU
    input  logic [31:0] m1_adr, m1_dat_w,
    input  logic m1_we, m1_stb, m1_cyc,
    input  logic [3:0] m1_sel,
    output logic [31:0] m1_dat_r,
    output logic m1_ack,

    //slave-LiteDRAM
    output logic [31:0] s_adr, s_dat_w,
    output logic s_we, s_stb, s_cyc,
    output logic [3:0] s_sel,
    input  logic [31:0] s_dat_r,
    input  logic s_ack
);
    always_comb begin
        if (tpu_active) begin
            s_adr   = m1_adr;   s_dat_w = m1_dat_w;
            s_we    = m1_we;    s_stb   = m1_stb;
            s_cyc   = m1_cyc;   s_sel   = m1_sel;
            m1_dat_r = s_dat_r; m1_ack  = s_ack;
            m0_dat_r = '0;      m0_ack  = 1'b0;
        end else begin
            s_adr   = m0_adr;   s_dat_w = m0_dat_w;
            s_we    = m0_we;    s_stb   = m0_stb;
            s_cyc   = m0_cyc;   s_sel   = m0_sel;
            m0_dat_r = s_dat_r; m0_ack  = s_ack;
            m1_dat_r = '0;      m1_ack  = 1'b0;
        end
    end
endmodule
