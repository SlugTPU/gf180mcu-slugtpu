module scalar_pipe #(
    parameter int N = 8,
    parameter int PSUM_W = 32,
    parameter int M0_W = 32,
    parameter int FIXED_SHIFT  = 16
)(
    input logic clk_i,
    input logic rst_i,

    // psum input from systolic array
    input logic signed [PSUM_W-1:0] data_i [N-1:0],
    input logic data_valid_i,
    output logic data_ready_o,

    // pre-loaded scalar parameters (active during compute)
    input logic signed [PSUM_W-1:0] bias_i [N-1:0],
    input logic signed [PSUM_W-1:0] zero_point_i [N-1:0],
    input logic signed [M0_W-1:0] scale_i [N-1:0],

    // quantized 8-bit output
    output logic signed [7:0] data_o [N-1:0],
    output logic data_valid_o,
    input logic data_ready_i
);

    // bias -> relu -> sub_zp -> scale

    // all stages use elastic ready/valid handshaking. sub-zp reuses add_n
    
    // bias -> relu
    logic signed [PSUM_W-1:0] bias_data [N-1:0];
    logic bias_valid;
    logic bias_ready;

    // relu -> sub_zp
    logic signed [PSUM_W-1:0] relu_data [N-1:0];
    logic relu_valid;
    logic relu_ready;

    // sub_zp -> scale
    logic signed [PSUM_W-1:0] zp_data [N-1:0];
    logic zp_valid;
    logic zp_ready;

    // // negate zero points for subtraction via add_n
    // logic signed [PSUM_W-1:0] neg_zp [N-1:0];

    // for (genvar i = 0; i < N; i++) begin : gen_neg_zp
    //     assign neg_zp[i] = -zero_point_i[i];
    // end

    //s1: bias add

    add_n #(
        .N       (N),
        .width_p (PSUM_W)
    ) u_add_bias (
        .clk_i        (clk_i),
        .rst_i        (rst_i),
        .bias_i       (bias_i),
        .data_i       (data_i),
        .data_valid_i (data_valid_i),
        .data_ready_i (bias_ready),
        .data_o       (bias_data),
        .data_valid_o (bias_valid),
        .data_ready_o (data_ready_o)
    );

    // s2: relu

    relu_n #(
        .N       (N),
        .width_p (PSUM_W)
    ) u_relu (
        .clk_i        (clk_i),
        .rst_i        (rst_i),
        .data_i       (bias_data),
        .data_valid_i (bias_valid),
        .data_ready_i (relu_ready),
        .data_o       (relu_data),
        .data_valid_o (relu_valid),
        .data_ready_o (bias_ready)
    );

    // s3: sub zp

    add_n #(
        .N       (N),
        .width_p (PSUM_W)
    ) u_sub_zp (
        .clk_i        (clk_i),
        .rst_i        (rst_i),
        .bias_i       (zero_point_i), // reuse add_n with zero_point as "bias" for subtraction
        .data_i       (relu_data),
        .data_valid_i (relu_valid),
        .data_ready_i (zp_ready),
        .data_o       (zp_data),
        .data_valid_o (zp_valid),
        .data_ready_o (relu_ready)
    );

    // s4: fixed-point scale + quantize to int8

    scale_n #(
        .N          (N),
        .ACC_WIDTH_P (PSUM_W),
        .M0_WIDTH_P  (M0_W),
        .FIXED_SHIFT_P(FIXED_SHIFT)
    ) u_scale_n (
        .clk_i       (clk_i),
        .rst_i       (rst_i),
        .m0_i        (scale_i),
        .data_valid_i(zp_valid),
        .data_ready_i(data_ready_i),
        .data_i      (zp_data),
        .data_valid_o(data_valid_o),
        .data_ready_o(zp_ready),
        .data_o      (data_o)
    );

    // logic signed [7:0] scale_comb [N-1:0];

    // for (genvar i = 0; i < N; i++) begin : gen_qmul
    //     localparam int PW = PSUM_W + M0_W;

    //     logic signed [PW-1:0] product;
    //     logic signed [PW-1:0] rounded;
    //     logic signed [PW-1:0] shifted;

    //     assign product = zp_data[i] * scale_i[i];
    //     assign rounded = product + (1 <<< (FIXED_SHIFT - 1));
    //     assign shifted = rounded >>> FIXED_SHIFT;

    //     assign scale_comb[i] = (shifted > 127)  ? 8'sd127  :
    //                             (shifted < -128) ? -8'sd128 :
    //                             shifted[7:0];
    // end


    // elastic output reg (like relu_n)
    // assign zp_ready = ~data_valid_o | data_ready_i;

    // for (genvar i = 0; i < N; i++) begin : gen_out_reg
    //     always_ff @(posedge clk_i) begin
    //         if (rst_i) begin
    //             data_o[i] <= '0;
    //         end else if (zp_ready) begin
    //             data_o[i] <= scale_comb[i];
    //         end
    //     end
    // end

    // always_ff @(posedge clk_i) begin
    //     if (rst_i) begin
    //         data_valid_o <= 1'b0;
    //     end else if (zp_ready) begin
    //         data_valid_o <= zp_valid;
    //     end
    // end

endmodule
