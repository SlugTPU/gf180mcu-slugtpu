module scalar_stage #(
    parameter int N = 8,
    parameter int PSUM_W = 32,
    parameter int M0_W = 32,
    parameter int FIXED_SHIFT = 16,
    parameter int BUS_W = 64
)(
    input logic clk_i,
    input logic rst_i,

    // param loading (from memory/AXI)
    input logic[BUS_W-1:0] read_bus_i,
    input logic load_valid_i,
    input logic load_bias_en_i,
    input logic load_zp_en_i,
    input logic load_scale_en_i,

    // dat pipeline input (from systolic array)
    input logic signed [PSUM_W-1:0] data_i[N-1:0],
    input logic data_valid_i,
    output logic data_ready_o,

    // dat pipeline output (quantized INT8)
    output logic signed [7:0] data_o [N-1:0],
    output logic data_valid_o,
    input  logic data_ready_i
);

    localparam int LANES = BUS_W / PSUM_W;
    localparam int DEPTH = N / LANES;

    // connect loaders to the scalar pipeline
    logic signed [PSUM_W-1:0] bias_w[N-1:0];
    logic signed [PSUM_W-1:0] zp_w[N-1:0];
    logic signed [M0_W-1:0] scale_w[N-1:0];

    load_scalar_data #(
        .scalar_data_width_p(PSUM_W),
        .lane_depth_p       (DEPTH),
        .read_bus_width     (BUS_W)
    ) u_load_bias (
        .clk_i          (clk_i),
        .reset_i        (rst_i),
        .read_bus       (read_bus_i),
        .load_valid_i   (load_valid_i),
        .load_enable_i  (load_bias_en_i),
        .scalar_values_o(bias_w)
    );

    load_scalar_data #(
        .scalar_data_width_p(PSUM_W),
        .lane_depth_p       (DEPTH),
        .read_bus_width     (BUS_W)
    ) u_load_zp (
        .clk_i          (clk_i),
        .reset_i        (rst_i),
        .read_bus       (read_bus_i),
        .load_valid_i   (load_valid_i),
        .load_enable_i  (load_zp_en_i),
        .scalar_values_o(zp_w)
    );

    load_scalar_data #(
        .scalar_data_width_p(M0_W),
        .lane_depth_p       (DEPTH),
        .read_bus_width     (BUS_W)
    ) u_load_scale (
        .clk_i          (clk_i),
        .reset_i        (rst_i),
        .read_bus       (read_bus_i),
        .load_valid_i   (load_valid_i),
        .load_enable_i  (load_scale_en_i),
        .scalar_values_o(scale_w)
    );

    scalar_pipe #(
        .N          (N),
        .PSUM_W     (PSUM_W),
        .M0_W       (M0_W),
        .FIXED_SHIFT(FIXED_SHIFT)
    ) u_scalar_pipe (
        .clk_i       (clk_i),
        .rst_i       (rst_i),
        .data_i      (data_i),
        .data_valid_i(data_valid_i),
        .data_ready_o(data_ready_o),
        
        .bias_i      (bias_w),
        .zero_point_i(zp_w),
        .scale_i     (scale_w),
        
        .data_o      (data_o),
        .data_valid_o(data_valid_o),
        .data_ready_i(data_ready_i)
    );

endmodule
