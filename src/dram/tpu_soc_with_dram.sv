// Testbench wrapper: tpu_soc + mt48lc16m16a2 SDRAM model.
// The bidirectional SDRAM DQ bus is resolved internally via a tristate.
// Exposes the same SPI + clock/reset interface as tpu_soc.
`default_nettype none

module tpu_soc_sdram_tb #(
    parameter sys_clk_mhz_p   = 100,
    parameter sdr_data_bits_p  = 16,
    parameter sdr_addr_bits_p  = 13,
    parameter dram_burst_p     = 4,
    parameter with_sdram_pin_buffering_p = 0,
    parameter rows_p           = 8192,
    parameter cols_p           = 512,
    parameter banks_p          = 4
) (
    input  logic clk_i,
    input  logic rst_i,

    input  logic spi_clk_i,
    input  logic spi_cs_ni,
    input  logic spi_mosi_i,
    output logic spi_miso_o
);
    wire [sdr_data_bits_p-1:0]  sdram_dq;
    logic [sdr_data_bits_p-1:0] sdr_dq_o;
    logic [sdr_data_bits_p-1:0] sdr_dq_i;
    logic                       sdr_dq_oe;
    logic [sdr_addr_bits_p-1:0] sdr_addr;
    logic [1:0]                 sdr_ba;
    logic                       sdr_cke;
    logic                       sdr_cs_n, sdr_ras_n, sdr_cas_n, sdr_we_n;
    logic [1:0]                 sdr_dqm;

    wire [sdr_data_bits_p-1:0]  sdram_dq_pad;
    // logic [sdr_data_bits_p-1:0] sdr_dq_o_pad;
    logic [sdr_addr_bits_p-1:0] sdr_addr_pad;
    logic [1:0]                 sdr_ba_pad;
    logic                       sdr_cke_pad;
    logic                       sdr_cs_n_pad, sdr_ras_n_pad, sdr_cas_n_pad, sdr_we_n_pad;
    logic [1:0]                 sdr_dqm_pad;

    initial begin
       assert (with_sdram_pin_buffering_p == 0 || with_sdram_pin_buffering_p == 1)
         else $fatal(1, "with_sdram_pin_buffering_p must be 0 or 1");
    end
    generate
       if (with_sdram_pin_buffering_p) begin : gen_with_pin_buffering
          wire [sdr_data_bits_p-1:0]  sdram_dq_buf_d, sdram_dq_buf_q;
          logic [sdr_data_bits_p-1:0] sdr_dq_o_buf_d, sdr_dq_o_buf_q;
          logic [sdr_data_bits_p-1:0] sdr_dq_i_buf_d, sdr_dq_i_buf_q;
          logic                       sdr_dq_oe_buf_d, sdr_dq_oe_buf_q;
          logic [sdr_addr_bits_p-1:0] sdr_addr_buf_d, sdr_addr_buf_q;
          logic [1:0]                 sdr_ba_buf_d, sdr_ba_buf_q;
          logic                       sdr_cke_buf_d, sdr_cke_buf_q;
          logic                       sdr_cs_n_buf_d, sdr_cs_n_buf_q;
          logic                       sdr_ras_n_buf_d, sdr_ras_n_buf_q;
          logic                       sdr_cas_n_buf_d, sdr_cas_n_buf_q;
          logic                       sdr_we_n_buf_d, sdr_we_n_buf_q;
          logic [1:0]                 sdr_dqm_buf_d, sdr_dqm_buf_q;

          // assign sdr_dq_o_pad     = sdr_dq_o_buf_q;
          // assign sdr_dq_oe_pad    = sdr_dq_oe_buf_q;
          assign sdram_dq_pad     = sdram_dq;
          assign sdr_addr_pad     = sdr_addr_buf_q;
          assign sdr_ba_pad       = sdr_ba_buf_q;
          assign sdr_cke_pad      = sdr_cke_buf_q;
          assign sdr_cs_n_pad     = sdr_cs_n_buf_q;
          assign sdr_ras_n_pad    = sdr_ras_n_buf_q;
          assign sdr_cas_n_pad    = sdr_cas_n_buf_q;
          assign sdr_we_n_pad     = sdr_we_n_buf_q;
          assign sdr_dqm_pad      = sdr_dqm_buf_q;

          assign sdr_dq_i         = sdr_dq_i_buf_q;
          assign sdram_dq         = sdr_dq_oe_buf_q ? sdr_dq_o_buf_q : 'z;

          always_ff @(posedge clk_i) begin
             if (rst_i) begin
                // sdram_dq_buf_q   <= '0;
                sdr_dq_o_buf_q   <= '0;
                sdr_dq_oe_buf_q  <= '0;
                sdr_addr_buf_q   <= '0;
                sdr_ba_buf_q     <= '0;
                sdr_cke_buf_q    <= '0;
                sdr_cs_n_buf_q   <= '1;
                sdr_ras_n_buf_q  <= '1;
                sdr_cas_n_buf_q  <= '1;
                sdr_we_n_buf_q   <= '1;
                sdr_dqm_buf_q    <= '0;
                sdr_dq_i_buf_q   <= '0;
             end else begin
                // sdram_dq_buf_q   <=  sdram_dq_buf_d;
                sdr_dq_o_buf_q   <=  sdr_dq_o_buf_d;
                sdr_dq_oe_buf_q  <=  sdr_dq_oe_buf_d;
                sdr_addr_buf_q   <=  sdr_addr_buf_d;
                sdr_ba_buf_q     <=  sdr_ba_buf_d;
                sdr_cke_buf_q    <=  sdr_cke_buf_d;
                sdr_cs_n_buf_q   <=  sdr_cs_n_buf_d;
                sdr_ras_n_buf_q  <=  sdr_ras_n_buf_d;
                sdr_cas_n_buf_q  <=  sdr_cas_n_buf_d;
                sdr_we_n_buf_q   <=  sdr_we_n_buf_d;
                sdr_dqm_buf_q    <=  sdr_dqm_buf_d;
                sdr_dq_i_buf_q   <=  sdr_dq_i_buf_d;
             end
          end

          assign sdr_dq_o_buf_d   = sdr_dq_o;
          assign sdr_dq_oe_buf_d  = sdr_dq_oe;
          assign sdr_addr_buf_d   = sdr_addr;
          assign sdr_ba_buf_d     = sdr_ba;
          assign sdr_cke_buf_d    = sdr_cke;
          assign sdr_cs_n_buf_d   = sdr_cs_n;
          assign sdr_ras_n_buf_d  = sdr_ras_n;
          assign sdr_cas_n_buf_d  = sdr_cas_n;
          assign sdr_we_n_buf_d   = sdr_we_n;
          assign sdr_dqm_buf_d    = sdr_dqm;
          assign sdr_dq_i_buf_d   = sdram_dq_pad;
       end else begin : gen_no_pin_buffering
          // assign sdr_dq_o_pad     = sdr_dq_o;
          assign sdram_dq_pad     = sdram_dq;
          assign sdr_addr_pad     = sdr_addr;
          assign sdr_ba_pad       = sdr_ba;
          assign sdr_cke_pad      = sdr_cke;
          assign sdr_cs_n_pad     = sdr_cs_n;
          assign sdr_ras_n_pad    = sdr_ras_n;
          assign sdr_cas_n_pad    = sdr_cas_n;
          assign sdr_we_n_pad     = sdr_we_n;
          assign sdr_dqm_pad      = sdr_dqm;

          assign sdr_dq_i         = sdram_dq_pad;
          assign sdram_dq         = sdr_dq_oe ? sdr_dq_o : 'z;
       end
    endgenerate

    tpu_soc #(
        .sys_clk_mhz_p   (sys_clk_mhz_p),
        .sdr_data_bits_p (sdr_data_bits_p),
        .sdr_addr_bits_p (sdr_addr_bits_p),
        .dram_burst_p    (dram_burst_p),
        .with_sdram_pin_buffering_p  (with_sdram_pin_buffering_p)
    ) soc (
        .clk_i       (clk_i),
        .rst_i       (rst_i),
        .spi_clk_i   (spi_clk_i),
        .spi_cs_ni   (spi_cs_ni),
        .spi_mosi_i  (spi_mosi_i),
        .spi_miso_o  (spi_miso_o),
        .sdr_dq_i    (sdr_dq_i),
        .sdr_dq_o    (sdr_dq_o),
        .sdr_dq_oe_o (sdr_dq_oe),
        .sdr_addr_o  (sdr_addr),
        .sdr_ba_o    (sdr_ba),
        .sdr_cke_o   (sdr_cke),
        .sdr_cs_no   (sdr_cs_n),
        .sdr_ras_no  (sdr_ras_n),
        .sdr_cas_no  (sdr_cas_n),
        .sdr_we_no   (sdr_we_n),
        .sdr_dqm_o   (sdr_dqm)
    );

    mt48lc16m16a2 #(
        .addr_bits (sdr_addr_bits_p),
        .data_bits (sdr_data_bits_p),
        .col_bits  ($clog2(cols_p)),
        .mem_sizes (rows_p * cols_p - 1)
    ) sdram_model (
        .Dq    (sdram_dq_pad),
        .Addr  (sdr_addr_pad),
        .Ba    (sdr_ba_pad),
        .Clk   (clk_i),
        .Cke   (sdr_cke_pad),
        .Cs_n  (sdr_cs_n_pad),
        .Ras_n (sdr_ras_n_pad),
        .Cas_n (sdr_cas_n_pad),
        .We_n  (sdr_we_n_pad),
        .Dqm   (sdr_dqm_pad)
    );

endmodule

`default_nettype wire
