// tpu_regs.sv
// Wishbone slave: TPU control registers
//
// Register map (base = 0x1000_0000):
//   0x00  RW  CTRL          [0]=tpu_enable (1=DMA owns bus), [1]=reset
//   0x04  RO  STATUS        [0]=busy, [1]=done
//   0x08  RW  INPUT_ADDR    — DRAM address of TPU input data
//   0x0C  RW  OUTPUT_ADDR   — DRAM address for TPU to write results
//   0x10  RW  LENGTH        — number of 32-bit words to process
//   0x14  RW  SPIBONE_CTRL  [0]=burst_en (1=enable SPI read bursts)
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module tpu_regs (
    input  logic        clk_i,
    input  logic        rst_i,

    // Wishbone slave (from wb_decoder port 1)
    input  logic [31:0] wb_adr_i,
    input  logic [31:0] wb_dat_i,
    output logic [31:0] wb_dat_o,
    input  logic        wb_we_i,
    input  logic        wb_stb_i,
    input  logic        wb_cyc_i,
    output logic        wb_ack_o,

    // TPU control outputs
    output logic        tpu_enable_o,
    output logic        tpu_reset_o,
    output logic [31:0] tpu_input_addr_o,
    output logic [31:0] tpu_output_addr_o,
    output logic [31:0] tpu_length_o,
    output logic        burst_en_o,

    // TPU status inputs
    input  logic        tpu_busy_i,
    input  logic        tpu_done_i
);

    // Register offsets (byte addressed, 32-bit aligned)
    localparam logic [7:0]
        REG_CTRL         = 8'h00,
        REG_STATUS       = 8'h04,
        REG_INPUT_ADDR   = 8'h08,
        REG_OUTPUT_ADDR  = 8'h0C,
        REG_LENGTH       = 8'h10,
        REG_SPIBONE_CTRL = 8'h14;

    logic [7:0] reg_offset;
    assign reg_offset = wb_adr_i[7:0];

    // Single-cycle ack
    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) wb_ack_o <= '0;
        else        wb_ack_o <= wb_cyc_i && wb_stb_i && !wb_ack_o;
    end

    // Registers + TPU control
    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            tpu_enable_o      <= '0;
            tpu_reset_o       <= '0;
            tpu_input_addr_o  <= '0;
            tpu_output_addr_o <= '0;
            tpu_length_o      <= '0;
            burst_en_o        <= '0;
            wb_dat_o          <= '0;
        end else begin
            if (wb_cyc_i && wb_stb_i) begin
                if (wb_we_i) begin
                    casez (reg_offset)
                        REG_CTRL: begin
                            tpu_enable_o <= wb_dat_i[0];
                            tpu_reset_o  <= wb_dat_i[1];
                        end
                        REG_INPUT_ADDR:   tpu_input_addr_o  <= wb_dat_i;
                        REG_OUTPUT_ADDR:  tpu_output_addr_o <= wb_dat_i;
                        REG_LENGTH:       tpu_length_o      <= wb_dat_i;
                        REG_SPIBONE_CTRL: burst_en_o        <= wb_dat_i[0];
                        default: ;
                    endcase
                end else begin
                    casez (reg_offset)
                        REG_CTRL:         wb_dat_o <= {30'h0, tpu_reset_o, tpu_enable_o};
                        REG_STATUS:       wb_dat_o <= {30'h0, tpu_done_i, tpu_busy_i};
                        REG_INPUT_ADDR:   wb_dat_o <= tpu_input_addr_o;
                        REG_OUTPUT_ADDR:  wb_dat_o <= tpu_output_addr_o;
                        REG_LENGTH:       wb_dat_o <= tpu_length_o;
                        REG_SPIBONE_CTRL: wb_dat_o <= {31'h0, burst_en_o};
                        default:          wb_dat_o <= 32'hDEAD_BEEF;
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire