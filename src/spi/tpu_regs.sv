// tpu_regs.sv
// Specialized 32-bit registers for reading status or controlling parts of TPU
//
// Register map (base = 0x1000_0000):
//   0x00  RW  PC_ADDR  — program counter address
//   0x04  RO  STATUS   — [1:0] tpu_state_i (2'b01 = IDLE)
//   0x08  RW  CTRL     — [0]=test_mode (1=DMA routed to on-chip test RAM, SDRAM pads inactive)

//   0x0C  RW  DBG_ADDR  [7:0] byte index into the SoC debug observation map
//   0x10  RO  DBG_DATA  [7:0] byte at DBG_ADDR from the debug map
//
`default_nettype none

module tpu_regs #(
    parameter int DBG_N_WORDS = 32
)(
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

    // PC outputs
    output logic [31:0] tpu_pc_addr_o,
    output logic        tpu_pc_stb_o,   // pulses high one cycle on PC write

    // Bus mux control
    output logic        test_mode_o,    // CTRL[1]: route DMA to on chip test RAM

    // TPU state + done detection
    input  logic [1:0]  tpu_state_i,
    input  logic [DBG_N_WORDS*8 - 1:0] dbg_word_i //soc internal state
);

    localparam logic [1:0] TPU_IDLE = 2'b01;

    localparam logic [7:0]
        REG_PC     = 8'h00,
        REG_STATUS = 8'h04,
        REG_CTRL   = 8'h08,
        REG_DBG_ADDR = 8'h0C,
        REG_DBG_DATA = 8'h10;

    logic [7:0] reg_offset;
    assign reg_offset = wb_adr_i[7:0];

    // Single-cycle ack
    always_ff @(posedge clk_i) begin
        if (rst_i) wb_ack_o <= '0;
        else        wb_ack_o <= wb_cyc_i && wb_stb_i && !wb_ack_o;
    end

    // dbg window addr reg and byte sel mux. addr is host writeable, mux is combinaitonal
    logic [7:0] dbg_addr_q;
    logic [7:0] dbg_data_w;
 
    debug_mux #(.N_WORDS(DBG_N_WORDS)) i_dbg_mux (
        .addr_i (dbg_addr_q),
        .data_i (dbg_word_i),
        .data_o (dbg_data_w)
    );
 
    // PC register + wishbone read/write
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            tpu_pc_addr_o <= '0;
            tpu_pc_stb_o  <= 1'b0;
            test_mode_o   <= 1'b0;
            dbg_addr_q <= '0;
            wb_dat_o      <= '0;
        end else begin
            tpu_pc_stb_o <= 1'b0;  // default: clear strobe

            if (wb_cyc_i && wb_stb_i) begin
                if (wb_we_i) begin
                    case (reg_offset)
                        REG_PC: begin
                            tpu_pc_addr_o <= wb_dat_i;
                            tpu_pc_stb_o  <= 1'b1;
                        end
                        REG_CTRL: begin
                            test_mode_o  <= wb_dat_i[0];
                        end
                        REG_DBG_ADDR: dbg_addr_q <= wb_dat_i[7:0];
                        default: ;
                    endcase
                end else begin
                    case (reg_offset)
                        REG_PC:     wb_dat_o <= tpu_pc_addr_o;
                        REG_STATUS: wb_dat_o <= {30'h0, tpu_state_i};
                        REG_CTRL:   wb_dat_o <= {31'h0, test_mode_o};
                        REG_DBG_ADDR: wb_dat_o <= {24'h0, dbg_addr_q};
                        REG_DBG_DATA: wb_dat_o <= {24'h0, dbg_data_w};
                        default:    wb_dat_o <= 32'hDEAD_BEEF;
                    endcase
                end
            end
        end
    end

endmodule

`default_nettype wire
