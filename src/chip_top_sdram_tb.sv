// Chip-top testbench wrapper: chip_top (pad ring) + mt48lc16m16a2 SDRAM model.
`default_nettype none
`include "slot_defines.svh"

module chip_top_sdram_tb #(
    parameter sys_clk_mhz_p = 100
) (
    input  wire  clk_i,
    input  wire  rst_i,
    input  wire  spi_clk_i,
    input  wire  spi_cs_ni,
    input  wire  spi_mosi_i,
    output wire  spi_miso_o
);

    initial begin
        if (`NUM_BIDIR_PADS < 40)
            $fatal(1, "chip_top_sdram_tb: NUM_BIDIR_PADS=%0d < 40; use SLOT_1X1", `NUM_BIDIR_PADS);
    end

    wire [`NUM_BIDIR_PADS-1:0] bidir_PAD;
    wire [`NUM_INPUT_PADS-1:0] input_PAD;
    wire [`NUM_ANALOG_PADS-1:0] analog_PAD;

    wire clk_w;
    wire rst_n_w;
    assign clk_w   = clk_i;
    assign rst_n_w = ~rst_i;

    assign input_PAD[0] = spi_clk_i;
    assign input_PAD[1] = spi_cs_ni;
    assign input_PAD[2] = spi_mosi_i;
    assign input_PAD[`NUM_INPUT_PADS-1:3] = '0;

    assign spi_miso_o = bidir_PAD[39];

    chip_top #(
        .NUM_INPUT_PADS  (`NUM_INPUT_PADS),
        .NUM_BIDIR_PADS  (`NUM_BIDIR_PADS),
        .NUM_ANALOG_PADS (`NUM_ANALOG_PADS)
    ) i_chip_top (
`ifdef USE_POWER_PINS
        .VDD       (1'b1),
        .VSS       (1'b0),
`endif
        .clk_PAD   (clk_w),
        .rst_n_PAD (rst_n_w),
        .input_PAD (input_PAD),
        .bidir_PAD (bidir_PAD),
        .analog_PAD(analog_PAD)
    );

    wire sdram_clk_w = clk_i;

    // SDRAM model uses default parameters (addr_bits=13, data_bits=16, col_bits=9).
    // bidir_PAD[15:0]  = DQ  (tristate: chip drives when OE=1, SDRAM drives on reads)
    // bidir_PAD[17:16] = DQM[1:0]
    // bidir_PAD[18]    = CKE
    // bidir_PAD[19]    = CS_N
    // bidir_PAD[20]    = RAS_N
    // bidir_PAD[21]    = CAS_N
    // bidir_PAD[22]    = WE_N
    // bidir_PAD[35:23] = ADDR[12:0]
    // bidir_PAD[37:36] = BA[1:0]
    mt48lc16m16a2 sdram_model (
        .Dq   (bidir_PAD[15:0]),
        .Addr (bidir_PAD[35:23]),
        .Ba   (bidir_PAD[37:36]),
        .Clk  (sdram_clk_w),
        .Cke  (bidir_PAD[18]),
        .Cs_n (bidir_PAD[19]),
        .Ras_n(bidir_PAD[20]),
        .Cas_n(bidir_PAD[21]),
        .We_n (bidir_PAD[22]),
        .Dqm  (bidir_PAD[17:16])
    );

endmodule
`default_nettype wire
