module sysray_2x2 #(
  parameter DATA_WIDTH = 8,
  parameter ACC_WIDTH  = 32
)(
  input logic clk_i,
  input logic rst_i,

  input logic [DATA_WIDTH:0] act0,
  input logic [DATA_WIDTH:0] act1,
  input logic act_valid0,
  input logic act_valid1,

  input logic [DATA_WIDTH:0] weight0,
  input logic [DATA_WIDTH:0] weight1,
  input logic weight_valid0,
  input logic weight_valid1,

  output logic [ACC_WIDTH-1:0] psum_out1,
  output logic [ACC_WIDTH-1:0] psum_out2,
  output logic psum_out1_valid_o,
  output logic psum_out2_valid_o
);

  logic [DATA_WIDTH:0]  act00_out;
  logic                 act_valid00_out;
  logic [DATA_WIDTH:0]  act_10_out;
  logic                 act_valid10_out;

  logic [DATA_WIDTH:0]  weight00_out;
  logic                 weight_valid00_out;
  logic [ACC_WIDTH-1:0] psum_00_out;              
  logic [DATA_WIDTH:0]  weight01_out;
  logic                 weight_valid01_out;
  logic [ACC_WIDTH-1:0] psum_01_out;

  logic psum_valid_00_out, psum_valid_01_out;


  pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe00 (
    .clk_i(clk_i),
    .rst_i(rst_i),
    .act_i(act0),
    .act_o(act00_out),
    .weight_i(weight0),
    .weight_o(weight00_out),
    .weight_valid_i(weight_valid0),
    .weight_valid_o(weight_valid00_out),
    .act_valid_i(act_valid0),
    .act_valid_o(act_valid00_out),
    .psum_i({ACC_WIDTH{1'b0}}), // start with zero psum
    .psum_valid_i(1'b1),
    .psum_o(psum_00_out),
    .psum_valid_o(psum_valid_00_out)
  );

  pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe01 (
    .clk_i(clk_i),
    .rst_i(rst_i),
    .act_i(act00_out),
    .act_o(),
    .weight_i(weight1),
    .weight_o(weight01_out),
    .weight_valid_i(weight_valid1),
    .weight_valid_o(weight_valid01_out),
    .act_valid_i(act_valid00_out),
    .act_valid_o(),
    .psum_i({ACC_WIDTH{1'b0}}), // start with zero psum
    .psum_valid_i(1'b1),
    .psum_o(psum_01_out),
    .psum_valid_o(psum_valid_01_out)
  );

  // logic [ACC_WIDTH-1:0] psum_10_out;

  pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe10 (
    .clk_i(clk_i),
    .rst_i(rst_i),
    .act_i(act1),
    .act_o(act_10_out),
    .weight_i(weight00_out),
    .weight_o(),
    .weight_valid_i(weight_valid00_out),
    .weight_valid_o(),
    .act_valid_i(act_valid1),
    .act_valid_o(act_valid10_out),
    .psum_i(psum_00_out),
    .psum_valid_i(psum_valid_00_out),
    .psum_o(psum_out1),
    .psum_valid_o(psum_out1_valid_o)
  );

  // logic [ACC_WIDTH-1:0] psum_11_out;

  pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe11 (
    .clk_i(clk_i),
    .rst_i(rst_i)
    .weight_i(weight,
    .act_i(act_10_out),
    .act_o(),01_out),
    .weight_o(),
    .weight_valid_i(weight_valid01_out),
    .weight_valid_o(),
    .act_valid_i(act_valid10_out),
    .act_valid_o(),
    .psum_i(psum_01_out),
    .psum_valid_i(psum_valid_01_out),
    .psum_o(psum_out2),
    .psum_valid_o(psum_out2_valid_o)
  );

  // assign psum_out1 = psum_10_out[ACC_WIDTH-1:0];
  // assign psum_out2 = psum_11_out[ACC_WIDTH-1:0];

endmodule
