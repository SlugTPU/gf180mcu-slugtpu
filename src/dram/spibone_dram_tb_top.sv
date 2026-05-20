// spibone_wb -> wb_mem_model. 
// spi master hits bus, spibone_wb becomes a Wishbone master,
// swap mem model with actual controller later

`default_nettype none

module spibone_dram_tb_top #(
    parameter sys_clk_mhz_p = 100
) (
    input  logic        clk_i,
    input  logic        rst_i,

    input  logic        spi_sck_i,
    input  logic        spi_mosi_i,
    output logic        spi_miso_o,
    input  logic        spi_cs_n_i
);

    // SPIBone -> WB slave
    logic [31:0] wb_adr;
    logic [63:0]  wb_dat_w, wb_dat_r;
    logic        wb_we, wb_stb, wb_cyc, wb_ack;

    spibone_wb #(
       .addr_w_p(32),
       .data_w_p(64)
    ) u_spibone (
        .clk_i      (clk_i),
        .rst_i      (rst_i),

        .spi_sck_i  (spi_sck_i),
        .spi_mosi_i (spi_mosi_i),
        .spi_miso_o (spi_miso_o),
        .spi_cs_n_i (spi_cs_n_i),

        .wb_adr_o   (wb_adr),
        .wb_dat_o   (wb_dat_w),
        .wb_dat_i   (wb_dat_r),
        .wb_we_o    (wb_we),
        .wb_stb_o   (wb_stb),
        .wb_cyc_o   (wb_cyc),
        .wb_ack_i   (wb_ack)
    );

    wb_mem_model #(
        .DEPTH_LOG2 (10),// enough for unit tests
        .DATA_W     (64)
    ) u_mem (
        .clk_i (clk_i),
        .rst_i (rst_i),
        .adr_i (wb_adr),
        .dat_i (wb_dat_w),
        .dat_o (wb_dat_r),
        .we_i  (wb_we),
        .stb_i (wb_stb),
        .cyc_i (wb_cyc),
        .sel_i (8'hFF), // SPIBone always issues full-word transfers
        .ack_o (wb_ack)
    );

endmodule

`default_nettype wire

