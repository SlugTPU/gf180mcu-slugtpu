module tb_wb_sdr_mt48lc16m16a_7e #(
    parameter sys_clk_mhz_p  = 100
   ,parameter dram_burst_p    = 4
   // Note: SDRAM Pin buffering not implemented on this testbench
   ,parameter with_sdram_pin_buffering_p  = 0
   ,parameter data_bits_p     = 16
   ,parameter rows_p          = 8192
   ,parameter cols_p          = 512
   ,parameter banks_p         = 4
   ,parameter sdr_addr_bits_p = 13
   ,parameter usr_addr_bits_p = $clog2(rows_p) + $clog2(banks_p) + $clog2(cols_p) - $clog2(dram_burst_p)
) (
    input  logic clk_i
   ,input  logic rst_i

   ,input  logic [usr_addr_bits_p - 1:0]                           m_adr_i
   ,input  logic [data_bits_p * dram_burst_p - 1:0]  m_dat_i
   ,input  logic                                                    m_we_i, m_stb_i, m_cyc_i
   ,input  logic [(data_bits_p * dram_burst_p) / 8 - 1:0]         m_sel_i
   ,output logic                                                    m_ack_o
   ,output logic [data_bits_p * dram_burst_p - 1:0]  m_dat_o
);

    wire  [data_bits_p - 1:0] sdram_dq;
    logic [data_bits_p - 1:0] s_dq_o;
    logic [sdr_addr_bits_p - 1:0]          s_addr;
    logic [1:0]                             s_ba;
    logic                                   s_cke, s_cs_n, s_ras_n, s_cas_n, s_we_n;
    logic [1:0]                             s_dqm;
    logic                                   oe;

    // Tristate buffer for bidirectional DQ bus
    assign sdram_dq = oe ? s_dq_o : 'z;

    // Input pipeline to model the pad register in chip_core
    wire [data_bits_p-1:0] s_dq_i_buf;
    generate
        if (with_sdram_pin_buffering_p == 0) begin : g_no_dq_buf
            assign s_dq_i_buf = sdram_dq;
        end else begin : g_dq_buf
            logic [data_bits_p-1:0] dq_pipe [0:with_sdram_pin_buffering_p-1];
            always_ff @(posedge clk_i) begin
                dq_pipe[0] <= sdram_dq;
                for (int i = 1; i < with_sdram_pin_buffering_p; i++)
                    dq_pipe[i] <= dq_pipe[i-1];
            end
            assign s_dq_i_buf = dq_pipe[with_sdram_pin_buffering_p-1];
        end
    endgenerate

    wb_sdr_mt48lc16m16a_7e #(
        .sys_clk_mhz_p   (sys_clk_mhz_p),
        .dram_burst_p     (dram_burst_p),
        .with_sdram_pin_buffering_p   (with_sdram_pin_buffering_p),
        ._data_bits_p     (data_bits_p),
        ._rows_p          (rows_p),
        ._cols_p          (cols_p),
        ._banks_p         (banks_p),
        ._sdr_addr_bits_p (sdr_addr_bits_p)
    ) controller (
        .clk_i    (clk_i),
        .rst_i    (rst_i),

        .m_adr_i  (m_adr_i),
        .m_dat_i  (m_dat_i),
        .m_we_i   (m_we_i),
        .m_stb_i  (m_stb_i),
        .m_cyc_i  (m_cyc_i),
        .m_sel_i  (m_sel_i),
        .m_ack_o  (m_ack_o),
        .m_dat_o  (m_dat_o),

        .s_dq_i   (s_dq_i_buf),
        .s_dq_o   (s_dq_o),
        .s_addr_o (s_addr),
        .s_ba_o   (s_ba),
        .s_cke_o  (s_cke),
        .s_cs_no  (s_cs_n),
        .s_ras_no (s_ras_n),
        .s_cas_no (s_cas_n),
        .s_we_no  (s_we_n),
        .s_dqm_o  (s_dqm),
        .oe_o     (oe)
    );

    mt48lc16m16a2 #(
        .addr_bits (sdr_addr_bits_p),
        .data_bits (data_bits_p),
        .col_bits  ($clog2(cols_p)),
        .mem_sizes (rows_p * cols_p - 1) // max word index, not total bits
    ) model (
        .Dq    (sdram_dq),
        .Addr  (s_addr),
        .Ba    (s_ba),
        .Clk   (clk_i),
        .Cke   (s_cke),
        .Cs_n  (s_cs_n),
        .Ras_n (s_ras_n),
        .Cas_n (s_cas_n),
        .We_n  (s_we_n),
        .Dqm   (s_dqm)
    );

endmodule
