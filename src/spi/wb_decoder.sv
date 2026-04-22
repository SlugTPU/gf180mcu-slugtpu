// wb_decoder.sv
// Wishbone address decoder
//
// Address map:
//   0x0000_0000 - 0x0FFF_FFFF : DRAM  -> port 0 (wb_to_axi4)
//   0x1000_0000 - 0x1000_00FF : TPU   -> port 1 (tpu_regs)
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module wb_decoder (
    input  logic        clk_i,
    input  logic        rst_i,

    // Wishbone slave port (from spibone_wb)
    input  logic [31:0] wbs_adr_i,
    input  logic [31:0] wbs_dat_i,
    output logic [31:0] wbs_dat_o,
    input  logic        wbs_we_i,
    input  logic        wbs_stb_i,
    input  logic        wbs_cyc_i,
    output logic        wbs_ack_o,

    // Wishbone master port 0: DRAM (-> wb_to_axi4)
    output logic [31:0] wbm0_adr_o,
    output logic [31:0] wbm0_dat_o,
    input  logic [31:0] wbm0_dat_i,
    output logic        wbm0_we_o,
    output logic        wbm0_stb_o,
    output logic        wbm0_cyc_o,
    input  logic        wbm0_ack_i,

    // Wishbone master port 1: TPU regs (-> tpu_regs)
    output logic [31:0] wbm1_adr_o,
    output logic [31:0] wbm1_dat_o,
    input  logic [31:0] wbm1_dat_i,
    output logic        wbm1_we_o,
    output logic        wbm1_stb_o,
    output logic        wbm1_cyc_o,
    input  logic        wbm1_ack_i
);

    logic sel_dram, sel_tpu;

    assign sel_dram = (wbs_adr_i[31:28] == 4'h0);
    assign sel_tpu  = (wbs_adr_i[31:8]  == 24'h100000);

    always_comb begin
        // Default: pass address/data through to both, deassert selects
        wbm0_adr_o = wbs_adr_i;
        wbm0_dat_o = wbs_dat_i;
        wbm0_we_o  = wbs_we_i;
        wbm0_stb_o = '0;
        wbm0_cyc_o = '0;

        wbm1_adr_o = wbs_adr_i;
        wbm1_dat_o = wbs_dat_i;
        wbm1_we_o  = wbs_we_i;
        wbm1_stb_o = '0;
        wbm1_cyc_o = '0;

        wbs_dat_o  = '0;
        wbs_ack_o  = '0;

        if (wbs_cyc_i && wbs_stb_i) begin
            casez (1'b1)
                sel_dram: begin
                    wbm0_stb_o = '1;
                    wbm0_cyc_o = '1;
                    wbs_dat_o  = wbm0_dat_i;
                    wbs_ack_o  = wbm0_ack_i;
                end
                sel_tpu: begin
                    wbm1_stb_o = '1;
                    wbm1_cyc_o = '1;
                    wbs_dat_o  = wbm1_dat_i;
                    wbs_ack_o  = wbm1_ack_i;
                end
                default: begin
                    // Unmapped — ack immediately
                    wbs_ack_o = '1;
                    wbs_dat_o = 32'hDEAD_BEEF;
                end
            endcase
        end
    end

endmodule

`default_nettype wire