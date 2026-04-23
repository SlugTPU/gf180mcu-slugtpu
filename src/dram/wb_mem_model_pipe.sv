// NOTE: make sure to remember that this is sim only!! This file is a standin for LiteDRAM, NOT for synthesis
// Behavioral Wishbone B4 pipelined (actually this itme) ram (technically SRAM slave but you get the idea..)

// The original mem model and accompanying files follows wishbone B4 classic scheme. For wb files that end in _pipe, these are the variants and modifications made to allow for wishbone B4 pipelined ver

module wb_mem_model_pipe #(
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
    output logic        ack_o,
    output logic        stall_o
);

    localparam DEPTH = 1 << DEPTH_LOG2;
    localparam BYTE_BITS = $clog2(DATA_W/8);

    logic [DATA_W-1:0] mem [DEPTH];
    logic [DEPTH_LOG2-1:0] addr_w;
    logic accepted;

    assign addr_w = adr_i[DEPTH_LOG2+BYTE_BITS-1:BYTE_BITS];  // word aligned
    assign stall_o = 1'b0;
    assign accepted = cyc_i & stb_i;

    // ack one cyc after accept
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ack_o <= 1'b0;
        end else begin
            ack_o <= accepted;
        end
    end

    //read: pipeline addr, data valid on ack cyc
    logic [DEPTH_LOG2-1:0] addr_pipe;
    always_ff @(posedge clk_i) begin
        addr_pipe <= addr_w;
    end
    assign dat_o = mem[addr_pipe];

    //write: commit on accept cyc
    always_ff @(posedge clk_i) begin 
        if (accepted) begin
            for (int b = 0; b < DATA_W/8; b++) begin
                if (sel_i[b]) mem[addr_w][b*8 +: 8] <= dat_i[b*8 +: 8];
            end
        end
    end

endmodule
