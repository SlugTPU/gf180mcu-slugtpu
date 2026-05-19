// NOTE: make sure to remember that this is sim only!! This file is a standin for LiteDRAM, NOT for synthesis
// Behavioral Wishbone B4 classic ram (technically SRAM slave but you get the idea..)

module wb_mem_model #(
    parameter DEPTH_LOG2 = 10, // 1K words is 4 KB
    parameter DATA_W = 32
)(
    input  logic        clk_i,
    input  logic        rst_i,

    // actual WB4 slave
    input  logic [DATA_W-1:0] adr_i,
    input  logic [DATA_W-1:0] dat_i,
    output logic [DATA_W-1:0] dat_o,
    input  logic        we_i,
    input  logic        stb_i,
    input  logic        cyc_i,
    input  logic [DATA_W/8-1:0]  sel_i,
    output logic        ack_o
);

    localparam DEPTH = 1 << DEPTH_LOG2;
    logic [DATA_W-1:0] mem [DEPTH];
    logic [DEPTH_LOG2-1:0] addr_w;
    logic valid_cycle_w;

    assign addr_w        = adr_i[DEPTH_LOG2-1:0];
    assign valid_cycle_w = cyc_i & stb_i;

    // single-cyc ack
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ack_o <= 1'b0;
        end else begin
            ack_o <= valid_cycle_w & ~ack_o;
        end
    end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         dat_o <= '0;
      end else begin
         dat_o <= mem[addr_w]; //read
      end
   end

    //write
    always_ff @(posedge clk_i) begin 
        if (valid_cycle_w & we_i & ~ack_o) begin
            for (int b = 0; b < DATA_W/8; b++) begin
                if (sel_i[b]) mem[addr_w][b*8 +: 8] <= dat_i[b*8 +: 8];
            end
        end
    end

endmodule
