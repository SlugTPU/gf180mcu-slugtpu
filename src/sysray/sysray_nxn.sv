// Parameterized N×N systolic array. Weights flow top-down and activations
// flow left-right through a 2D grid of PEs; partial sums accumulate
// top-down and the final row drives the output.
//
// Double-buffering: to keep the pipeline running across layers, weights and
// activations use their MSB as a buffer select bit. This allows loading the
// next layer's weights into the shadow buffer while the current inference is
// running. This module simply wires PEs together and passes the select bit
// through — no shadow buffer logic lives here; that is implemented entirely
// within each PE.

module sysray_nxn #(
  parameter DATA_WIDTH = 8,
  parameter ACC_WIDTH  = 32,
  parameter N = 8
)(
  input logic clk_i,
  input logic rst_i,

  input logic                    act_valid_n_i         [N-1:0],
  input logic  signed [DATA_WIDTH-1:0]  act_n_i               [N-1:0],
  input logic                    act_sel_n_i           [N-1:0],  // one select bit per row

  input logic                    weight_valid_n_i      [N-1:0],
  input logic  signed [DATA_WIDTH-1:0]  weight_n_i            [N-1:0],
  input logic                    weight_sel_n_i        [N-1:0],  // one select bit per column

  // input logic                    psum_valid_n_i        [N-1:0],
  // input logic  signed [ACC_WIDTH-1:0]   psum_n_i              [N-1:0],

  output logic signed [ACC_WIDTH-1:0]   psum_out_n_o          [N-1:0],
  output logic                   psum_out_valid_n_o    [N-1:0]
);

logic [DATA_WIDTH:0]   w_conn          [N:0][N:0];
/* verilator lint_off UNOPTFLAT */  // acyclic: index strictly increases through generate loop
logic                  w_valid_conn    [N:0][N:0];
/* verilator lint_on UNOPTFLAT */
logic [DATA_WIDTH:0]   a_conn          [N:0][N:0];
logic                  a_valid_conn    [N:0][N:0];
logic [ACC_WIDTH-1:0]  psum_conn       [N:0][N:0];
// logic                  psum_valid_conn [N:0][N:0];

genvar i, j;
generate
  for (i = 0 ; i < N; i++) begin  : row_block
    for (j = 0; j < N; j++) begin : col_block
      if (i == 0) begin
        assign w_conn[i][j] = {weight_sel_n_i[j], weight_n_i[j]};
        assign w_valid_conn[i][j] = weight_valid_n_i[j];
        assign psum_conn[i][j] = '0;
        // assign psum_valid_conn[i][j] = 1'b1;
      end else if (i == N-1) begin
        assign psum_out_n_o[j]       = psum_conn[i+1][j];
        assign psum_out_valid_n_o[j] = a_valid_conn[i][j+1];
      end
      if (j == 0) begin
        assign a_conn[i][j] = {act_sel_n_i[i], act_n_i[i]};
        assign a_valid_conn[i][j] = act_valid_n_i[i];
      end

      pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_ij (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .act_i(a_conn[i][j]),
        .act_o(a_conn[i][j+1]),
        .weight_i(w_conn[i][j]),
        .weight_o(w_conn[i+1][j]),
        .weight_valid_i(w_valid_conn[i][j]),
        .weight_valid_o(w_valid_conn[i+1][j]),
        .act_valid_i(a_valid_conn[i][j]),
        .act_valid_o(a_valid_conn[i][j+1]),
        .psum_i(psum_conn[i][j]),
        // .psum_valid_i(psum_valid_conn[i][j]),
        .psum_o(psum_conn[i+1][j])
        // .psum_valid_o(psum_valid_conn[i+1][j])
      );
    end  
  end
endgenerate

endmodule
