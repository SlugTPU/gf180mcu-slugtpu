// NOTE: make sure to remember that this is sim only!! This file is a standin for LiteDRAM, NOT for synthesis
// Behavioral Wishbone B4 pipelined ram (technically SRAM slave but you get the idea..)

module wb_mem_model #(
    parameter DEPTH_LOG2 = 10 // 1K words is 4 KB
)(
    input  logic        clk_i,
    input  logic        rst_i,

    // actual WB4 slave
    input  logic [31:0] adr_i,
    input  logic [31:0] dat_i,
    output logic [31:0] dat_o,
    input  logic        we_i,
    input  logic        stb_i,
    input  logic        cyc_i,
    input  logic [3:0]  sel_i,
    output logic        ack_o
);

    localparam DEPTH = 1 << DEPTH_LOG2;

    logic [31:0] mem [DEPTH];
    logic [DEPTH_LOG2-1:0] addr_w;
    logic valid_cycle_w;

    assign addr_w        = adr_i[DEPTH_LOG2+1:2];  // word aligned
    assign valid_cycle_w = cyc_i & stb_i;

    // single-cyc ack
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ack_o <= 1'b0;
        end else begin
            ack_o <= valid_cycle_w & ~ack_o;
        end
    end

    assign dat_o = mem[addr_w]; //read

    //write
    always_ff @(posedge clk_i) begin 
        if (valid_cycle_w & we_i & ~ack_o) begin
            if (sel_i[0]) mem[addr_w][ 7: 0] <= dat_i[ 7: 0];
            if (sel_i[1]) mem[addr_w][15: 8] <= dat_i[15: 8];
            if (sel_i[2]) mem[addr_w][23:16] <= dat_i[23:16];
            if (sel_i[3]) mem[addr_w][31:24] <= dat_i[31:24];
        end
    end

endmodule
