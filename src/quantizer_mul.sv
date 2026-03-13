/* Implementation of quantizer multiplication step (scaled by m0) for 8-bit quantization output
 *  Note: Currently is hardcoded for 8-bit output, but can be parameterized if needed
*/
module quantizer_mul #(
    parameter int ACC_WIDTH = 32,
    parameter int FIXED_SHIFT = 16,
    parameter int M0_WIDTH =32 
)(
    input  logic signed [ACC_WIDTH-1:0] psum,
    input  logic [M0_WIDTH-1:0] m0,
    output logic signed [7:0]  q_out
);

    logic signed [ACC_WIDTH+M0_WIDTH-1:0] product;
    logic signed [ACC_WIDTH+M0_WIDTH-1:0] rounded;
    logic signed [ACC_WIDTH+M0_WIDTH-1:0] shifted;

    // Multiply
    // m0 is unsigned, but we want a signed product, so we need to cast m0 to signed with an extra leading 0 bit to preserve signness of the product
    assign product = psum * $signed({ 1'b0, m0 });

    // Fixed Rounding (effective adds +0.5 to shifted result)
    assign rounded = product + (1 << (FIXED_SHIFT - 1));

    // Shift to convert from integer representation to fixed point representation
    assign shifted = rounded >>> FIXED_SHIFT;

    // Output truncated with saturation
    assign q_out  = (shifted > 127) ?   8'sd127 :
                    (shifted < -128) ? -8'sd128 :
                    shifted[7:0];
endmodule
