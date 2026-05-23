// slave-side demux for Wishbone B4 classic for debug

module wb_demux_1to2 #(
    parameter AdrW  = 32,
    parameter DataW = 32
)(
    input  logic sel_i,

    // master
    input  logic [AdrW-1:0]    m_adr_i,
    input  logic [DataW-1:0]   m_dat_w_i,
    output logic [DataW-1:0]   m_dat_r_o,
    input  logic [DataW/8-1:0] m_sel_i,
    input  logic               m_we_i, m_stb_i, m_cyc_i,
    output logic               m_ack_o,

    // slave 0 (like a SDRAM controller)
    output logic [AdrW-1:0]    s0_adr_o,
    output logic [DataW-1:0]   s0_dat_w_o,
    input  logic [DataW-1:0]   s0_dat_r_i,
    output logic [DataW/8-1:0] s0_sel_o,
    output logic               s0_we_o, s0_stb_o, s0_cyc_o,
    input  logic               s0_ack_i,

    // slave 1 (ex wb_test_ram)
    output logic [AdrW-1:0]    s1_adr_o,
    output logic [DataW-1:0]   s1_dat_w_o,
    input  logic [DataW-1:0]   s1_dat_r_i,
    output logic [DataW/8-1:0] s1_sel_o,
    output logic               s1_we_o, s1_stb_o, s1_cyc_o,
    input  logic               s1_ack_i
);

    assign s0_adr_o   = m_adr_i;
    assign s0_dat_w_o = m_dat_w_i;
    assign s0_sel_o   = m_sel_i;
    assign s0_we_o    = m_we_i;
    assign s0_stb_o   = m_stb_i & ~sel_i;
    assign s0_cyc_o   = m_cyc_i & ~sel_i;

    assign s1_adr_o   = m_adr_i;
    assign s1_dat_w_o = m_dat_w_i;
    assign s1_sel_o   = m_sel_i;
    assign s1_we_o    = m_we_i;
    assign s1_stb_o   = m_stb_i &  sel_i;
    assign s1_cyc_o   = m_cyc_i &  sel_i;

    assign m_dat_r_o = sel_i ? s1_dat_r_i : s0_dat_r_i;
    assign m_ack_o   = sel_i ? s1_ack_i   : s0_ack_i;

endmodule
