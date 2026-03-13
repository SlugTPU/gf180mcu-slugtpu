// DEPRECATED: This file is no longer maintained as the scale module is no longer used in the design. However, it may still be useful as a reference for how to write cocotb tests and models.

// Implements fixed point scale factor stage to scale down the psum back to 8 bits

module scale #(
    parameter int ACC_WIDTH = 32,
    parameter int FIXED_SHIFT = 16,
    parameter int M0_WIDTH = 32
)(
    input logic clk_i,
    input logic rst_i,

    input logic signed [M0_WIDTH-1:0] scale_i,
    input logic scale_valid_i,
    input logic data_valid_i,
    input logic signed [ACC_WIDTH-1:0] data_i,
    output logic data_valid_o,
    output logic signed [7:0] data_o
);

logic signed [M0_WIDTH-1:0] scale_r;
logic scale_ready_r;
logic signed [M0_WIDTH-1:0] m0_sel;
logic signed [7:0] q_out;

assign m0_sel = (scale_valid_i) ? scale_i : scale_r;

quantizer_mul #(
    .ACC_WIDTH(ACC_WIDTH),
    .FIXED_SHIFT(FIXED_SHIFT),
    .M0_WIDTH(M0_WIDTH)
) u_quantizer_mul (
    .psum(data_i),
    .m0(m0_sel),
    .q_out(q_out)
);

always_ff @(posedge clk_i) begin
    if (rst_i) begin
        scale_r <= '0;
        scale_ready_r <= '0;
    end else if (scale_valid_i) begin
        scale_r <= scale_i;
        scale_ready_r <= '0;
    end
end

always_ff @(posedge clk_i) begin
    if (rst_i) begin
        data_o <= 8'b0;
        data_valid_o <= 1'b0;
    end
    else if (data_valid_i &&
            (scale_ready_r || scale_valid_i)) begin
        data_valid_o <= data_valid_i;

        data_o <= q_out;
    end else begin
        data_valid_o <= '0;
    end
end

endmodule
