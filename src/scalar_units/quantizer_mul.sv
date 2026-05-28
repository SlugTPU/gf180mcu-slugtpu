/* Implementation of quantizer multiplication step (scaled by m0) for 8-bit quantization output
 *  Note: Currently is hardcoded for 8-bit output, but can be parameterized if needed
*/
module quantizer_mul #(
    parameter int ACC_WIDTH = 32,
    parameter int FIXED_SHIFT = 16,
    parameter int M0_WIDTH =32 
)(
    input                              clk_i,
    input                              rst_i,

    input logic signed [ACC_WIDTH-1:0] psum,
    input logic [M0_WIDTH-1:0]         m0,
    input                              valid_i,
    input                              ready_i,
    output                             valid_o,
    output                             ready_o,
    output logic signed [7:0]          q_out
);

    logic signed [ACC_WIDTH+M0_WIDTH-1:0] product_d, product_q;
    logic signed [ACC_WIDTH+M0_WIDTH-1:0] rounded_d, rounded_q;
    logic signed [ACC_WIDTH+M0_WIDTH-1:0] shifted_d, shifted_q;

   // shift registers acting as valid_o signals for each step in quantization pipeline
   logic [2:0]                            valids_d, valids_q;

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         product_q <= '0;
         rounded_q <= '0;
         shifted_q <= '0;
         valids_q  <= '0;
      end else if (ready_o) begin
         product_q <= product_d;
         rounded_q <= rounded_d;
         shifted_q <= shifted_d;
         valids_q  <= valids_d;
      end
   end

   assign ready_o = (!valid_i || ready_i);
   assign valids_d = { valids_q[1], valids_q[0], valid_i };
   assign valid_o = valids_q[2];

    // Multiply
    // m0 is unsigned, but we want a signed product, so we need to cast m0 to signed with an extra leading 0 bit to preserve signness of the product
    assign product_d = (valid_i) ? (psum * $signed({ 1'b0, m0 })) : product_q;

    // Fixed Rounding (effective adds +0.5 to shifted result)
    assign rounded_d = (valids_q[0]) ? (product_q + (1 << (FIXED_SHIFT - 1))) : rounded_q;

    // Shift to convert from integer representation to fixed point representation
    assign shifted_d = (valids_q[1]) ? (rounded_q >>> FIXED_SHIFT) : shifted_q;

    // Output truncated with saturation
    assign q_out  = (shifted_q > 127) ?   8'sd127 :
                    (shifted_q < -128) ? -8'sd128 :
                    shifted_q[7:0];

endmodule
