`timescale 1ns/1ps

module tri_shift #(
    parameter int N      = 8,
    parameter int DATA_W = 16
)(
    input  logic                 clk,
    input  logic                 rst,
    input  logic [DATA_W-1:0]    data_i   [N],
    input  logic [N-1:0]         enable_i,
    output logic [DATA_W-1:0]    data_o   [N]
);

    logic [DATA_W-1:0] taps [N][N];

    genvar lane;
    generate
        for (lane = 0; lane < N; lane++) begin : TRI
            localparam int DEPTH_L = lane + 1;

            logic [DATA_W-1:0] lane_taps [DEPTH_L-1:0];

            shift #(
                .width_p (DATA_W),
                .depth_p (DEPTH_L)
            ) u_shift_lane (
                .clk_i   (clk),
                .reset_i (rst),
                .enable_i(enable_i[lane]),
                .data_i  (data_i[lane]),
                .data_o  (lane_taps)
            );

            assign data_o[lane] = lane_taps[DEPTH_L-1];

            for (genvar st = 0; st < N; st++) begin : PACK
                if (st < DEPTH_L) begin
                    assign taps[lane][st] = lane_taps[st];
                end else begin
                    assign taps[lane][st] = '0;
                end
            end
        end
    endgenerate

endmodule