// Testbench wrapper: tpu_soc + mt48lc16m16a2 SDRAM model.
// The bidirectional SDRAM DQ bus is resolved internally via a tristate.
// Exposes the same SPI + clock/reset interface as tpu_soc.
`default_nettype none

module tpu_soc_sdram_tb #(
    parameter sys_clk_mhz_p   = 100,
    parameter sdr_data_bits_p  = 16,
    parameter sdr_addr_bits_p  = 13,
    parameter dram_burst_p     = 4,
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

    wire [sdr_data_bits_p-1:0] sdram_dq;

    logic [sdr_data_bits_p-1:0] sdr_dq_o;
    logic                       sdr_dq_oe;
    logic [sdr_addr_bits_p-1:0] sdr_addr;
    logic [1:0]                 sdr_ba;
    logic                       sdr_cke;
    logic                       sdr_cs_n, sdr_ras_n, sdr_cas_n, sdr_we_n;
    logic [1:0]                 sdr_dqm;

    assign sdram_dq = sdr_dq_oe ? sdr_dq_o : 'z;

    tpu_soc #(
        .sys_clk_mhz_p   (sys_clk_mhz_p),
        .sdr_data_bits_p (sdr_data_bits_p),
        .sdr_addr_bits_p (sdr_addr_bits_p),
        .dram_burst_p    (dram_burst_p)
    ) soc (
        .clk_i       (clk_i),
        .rst_i       (rst_i),
        .spi_clk_i   (spi_clk_i),
        .spi_cs_ni   (spi_cs_ni),
        .spi_mosi_i  (spi_mosi_i),
        .spi_miso_o  (spi_miso_o),
        .sdr_dq_i    (sdram_dq),
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
