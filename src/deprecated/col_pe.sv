module pe_col #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                  clk_i,
    input  logic                  rst_i,

    input  logic [DATA_WIDTH-1:0] act0_in,
    input  logic [DATA_WIDTH-1:0] act1_in,

    // Separate weight ports per PE — controller drives each directly
    // weight[0]   = valid
    // weight[1]   = sel
    // weight[9:2] = data
    input  logic [DATA_WIDTH+1:0] weight0_in,  // for PE0
    input  logic [DATA_WIDTH+1:0] weight1_in,  // for PE1 (data/valid)
                                                // sel is overridden from PE0's weight_out

    input  logic [ACC_WIDTH-1:0]  psum_in,
    output logic [ACC_WIDTH-1:0]  psum_out
);

    // PE0's weight_out carries sel only (valid=0, data=0)
    // We merge it with weight1_in's valid+data so PE1 gets:
    //   - sel from the chain (keeps both PEs in sync)
    //   - valid+data from the controller (unique per-PE weights)
    logic [DATA_WIDTH+1:0] weight_mid_raw;   // sel-only from PE0
    logic [DATA_WIDTH+1:0] weight1_merged;   // sel from chain + valid/data from controller
    logic [ACC_WIDTH-1:0]  psum_mid;

    // Merge: take sel from chain, valid+data from controller input
    assign weight1_merged = {weight1_in[DATA_WIDTH+1:2],  // data from controller
                             weight_mid_raw[1],             // sel from PE0 chain
                             weight1_in[0]};                // valid from controller

    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_pe0 (
        .clk_i      (clk_i),
        .rst_i      (rst_i),
        .act_in     (act0_in),
        .act_out    (),
        .weight_in  (weight0_in),
        .weight_out (weight_mid_raw),
        .psum_in    (psum_in),
        .psum_out   (psum_mid)
    );

    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_pe1 (
        .clk_i      (clk_i),
        .rst_i      (rst_i),
        .act_in     (act1_in),
        .act_out    (),
        .weight_in  (weight1_merged),
        .weight_out (),
        .psum_in    (psum_mid),
        .psum_out   (psum_out)
    );

endmodule
