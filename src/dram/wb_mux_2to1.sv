module wb_mux_2to1 #(
    parameter AdrW  = 32,
    parameter DataW = 32
) (
    input  logic clk_i,
    input  logic rst_i,
    input  logic tpu_active_i,  // 0 for SPI bridge to use bus, 1 = TPU

    //SPI Bridge (basically SPIBone)
    input  logic [AdrW-1:0]    m0_adr_i,
    input  logic [DataW-1:0]   m0_dat_w_i,
    input  logic               m0_we_i, m0_stb_i, m0_cyc_i,
    input  logic [DataW/8-1:0] m0_sel_i,
    output logic [DataW-1:0]   m0_dat_r_o,
    output logic               m0_ack_o,

    //TPU
    input  logic [AdrW-1:0]    m1_adr_i,
    input  logic [DataW-1:0]   m1_dat_w_i,
    input  logic               m1_we_i, m1_stb_i, m1_cyc_i,
    input  logic [DataW/8-1:0] m1_sel_i,
    output logic [DataW-1:0]   m1_dat_r_o,
    output logic               m1_ack_o,

    //slave-LiteDRAM
    output logic [AdrW-1:0]    s_adr_o,
    output logic [DataW-1:0]   s_dat_w_o,
    output logic               s_we_o, s_stb_o, s_cyc_o,
    output logic [DataW/8-1:0] s_sel_o,
    input  logic [DataW-1:0]   s_dat_r_i,
    input  logic               s_ack_i
);
    // Combinational mux selects the active master's signals
    logic [AdrW-1:0]    s_adr_d;
    logic [DataW-1:0]   s_dat_w_d;
    logic               s_we_d, s_stb_d, s_cyc_d;
    logic [DataW/8-1:0] s_sel_d;

    always_comb begin
        if (tpu_active_i) begin
            s_adr_d   = m1_adr_i;            s_dat_w_d = m1_dat_w_i;
            s_we_d    = m1_we_i;              s_stb_d   = m1_stb_i & ~s_ack_i;
            s_cyc_d   = m1_cyc_i;            s_sel_d   = m1_sel_i;
            m1_dat_r_o = s_dat_r_i;          m1_ack_o  = s_ack_i;
            m0_dat_r_o = '0;                 m0_ack_o  = 1'b0;
        end else begin
            s_adr_d   = m0_adr_i;            s_dat_w_d = m0_dat_w_i;
            s_we_d    = m0_we_i;              s_stb_d   = m0_stb_i & ~s_ack_i;
            s_cyc_d   = m0_cyc_i;            s_sel_d   = m0_sel_i;
            m0_dat_r_o = s_dat_r_i;          m0_ack_o  = s_ack_i;
            m1_dat_r_o = '0;                 m1_ack_o  = 1'b0;
        end
    end

    // Pipeline register: break the combinational path from masters to SDRAM controller inputs
    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            s_adr_o   <= '0;
            s_dat_w_o <= '0;
            s_we_o    <= '0;
            s_stb_o   <= '0;
            s_cyc_o   <= '0;
            s_sel_o   <= '0;
        end else begin
            s_adr_o   <= s_adr_d;
            s_dat_w_o <= s_dat_w_d;
            s_we_o    <= s_we_d;
            s_stb_o   <= s_stb_d;
            s_cyc_o   <= s_cyc_d;
            s_sel_o   <= s_sel_d;
        end
    end
endmodule
