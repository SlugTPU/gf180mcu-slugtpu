module scale_n
    #(parameter N = 8
    , parameter ACC_WIDTH_P = 32
    , parameter M0_WIDTH_P = 32
    // hardcoded for 8-bit output, but can be parameterized if needed
    , parameter Q_WIDTH_P = 8
    , parameter FIXED_SHIFT_P = 16)
    (
    input logic clk_i,
    input logic rst_i,

    input logic signed [M0_WIDTH_P-1:0] m0_i [N-1:0],
    input logic data_valid_i,
    input logic data_ready_i,
    input logic signed [ACC_WIDTH_P-1:0] data_i [N-1:0],
    output logic data_valid_o,
    output logic data_ready_o,
    output logic signed [Q_WIDTH_P-1:0] data_o [N-1:0]
);

    logic signed [Q_WIDTH_P-1:0] data_r [N-1:0];
    genvar i;
    generate
        for (i = 0; i < N; i++) begin
            quantizer_mul
                #(.ACC_WIDTH(ACC_WIDTH_P),
                .M0_WIDTH(M0_WIDTH_P),
                .FIXED_SHIFT(FIXED_SHIFT_P))
            quantizer_mul_inst
                (.psum(data_i[i]),
                .m0(m0_i[i]),
                .q_out(data_r[i])
            );
        end
    endgenerate

    elastic
        #(.width_p(Q_WIDTH_P),
        .depth_p(N))
    add_elastic
        (.clk_i(clk_i)
        ,.rst_i(rst_i)
        ,.data_i(data_r)
        ,.valid_i(data_valid_i)
        ,.ready_o(data_ready_o)
        ,.valid_o(data_valid_o)
        ,.ready_i(data_ready_i)
        ,.data_o(data_o)
    );

endmodule
