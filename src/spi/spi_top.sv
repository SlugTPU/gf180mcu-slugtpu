// SPDX-License-Identifier: Apache-2.0
// spi_top.sv

`default_nettype none

module spi_top (
    // SPI pins
    input  logic        reset,
    input  logic        SCK,
    input  logic        SDI,
    input  logic        CSB,
    output logic        SDO,
    output logic        sdoenb,

    // Wishbone clock and reset
    input  logic        wb_clk_i,
    input  logic        wb_rst_i,

    // Wishbone B3 master
    output logic [15:0] wb_adr_o,
    output logic [7:0]  wb_dat_o,
    input  logic [7:0]  wb_dat_i,
    output logic        wb_we_o,
    output logic        wb_stb_o,
    output logic        wb_cyc_o,
    input  logic        wb_ack_i,

    // Pass-through signals
    output logic        pass_thru_mgmt,
    output logic        pass_thru_mgmt_delay,
    output logic        pass_thru_user,
    output logic        pass_thru_user_delay,
    output logic        pass_thru_mgmt_reset,
    output logic        pass_thru_user_reset
);

    wire [15:0] oaddr;
    wire [7:0]  odata;
    wire [7:0]  idata;
    wire        wrstb;
    wire        rdstb;

    housekeeping_spi u_spi (
        .reset               (reset),
        .SCK                 (SCK),
        .SDI                 (SDI),
        .CSB                 (CSB),
        .SDO                 (SDO),
        .sdoenb              (sdoenb),
        .idata               (idata),
        .odata               (odata),
        .oaddr               (oaddr),
        .rdstb               (rdstb),
        .wrstb               (wrstb),
        .pass_thru_mgmt      (pass_thru_mgmt),
        .pass_thru_mgmt_delay(pass_thru_mgmt_delay),
        .pass_thru_user      (pass_thru_user),
        .pass_thru_user_delay(pass_thru_user_delay),
        .pass_thru_mgmt_reset(pass_thru_mgmt_reset),
        .pass_thru_user_reset(pass_thru_user_reset)
    );

    decoder u_decoder (
        .wb_clk_i  (wb_clk_i),
        .wb_rst_i  (wb_rst_i),
        .oaddr     (oaddr),
        .odata     (odata),
        .wrstb     (wrstb),
        .rdstb     (rdstb),
        .wb_idata  (idata),
        .wb_adr_o  (wb_adr_o),
        .wb_dat_o  (wb_dat_o),
        .wb_dat_i  (wb_dat_i),
        .wb_we_o   (wb_we_o),
        .wb_stb_o  (wb_stb_o),
        .wb_cyc_o  (wb_cyc_o),
        .wb_ack_i  (wb_ack_i)
    );

endmodule : spi_top

`default_nettype wire