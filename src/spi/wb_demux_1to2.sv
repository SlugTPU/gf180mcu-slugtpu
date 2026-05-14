// slave-side demux for Wishbone B4 classic for debug

module wb_demux_1to2 #(
    parameter AdrW  = 32,
    parameter DataW = 32
)(
    input  logic sel_i,

    // master
    input  logic [AdrW-1:0]    m_adr,
    input  logic [DataW-1:0]   m_dat_w,
    output logic [DataW-1:0]   m_dat_r,
    input  logic [DataW/8-1:0] m_sel,
    input  logic               m_we, m_stb, m_cyc,
    output logic               m_ack,

    // slave 0 (like a SDRAM controller)
    output logic [AdrW-1:0]    s0_adr,
    output logic [DataW-1:0]   s0_dat_w,
    input  logic [DataW-1:0]   s0_dat_r,
    output logic [DataW/8-1:0] s0_sel,
    output logic               s0_we, s0_stb, s0_cyc,
    input  logic               s0_ack,

    // slave 1 (ex wb_test_ram)
    output logic [AdrW-1:0]    s1_adr,
    output logic [DataW-1:0]   s1_dat_w,
    input  logic [DataW-1:0]   s1_dat_r,
    output logic [DataW/8-1:0] s1_sel,
    output logic               s1_we, s1_stb, s1_cyc,
    input  logic               s1_ack
);

    assign s0_adr = m_adr;
    assign s0_dat_w = m_dat_w;
    assign s0_sel = m_sel;
    assign s0_we  = m_we;
    assign s0_stb = m_stb & ~sel_i;
    assign s0_cyc = m_cyc & ~sel_i;

    assign s1_adr = m_adr;
    assign s1_dat_w = m_dat_w;
    assign s1_sel = m_sel;
    assign s1_we  = m_we;
    assign s1_stb = m_stb &  sel_i;
    assign s1_cyc = m_cyc &  sel_i;

    assign m_dat_r = sel_i ? s1_dat_r : s0_dat_r;
    assign m_ack   = sel_i ? s1_ack   : s0_ack;

endmodule