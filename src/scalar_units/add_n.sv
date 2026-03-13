module add_n
    #(parameter N = 8
    , parameter width_p = 32)
    (
    input logic clk_i,
    input logic rst_i,

    input logic signed [width_p-1:0] bias_i [N-1:0],
    input logic data_valid_i,
    input logic data_ready_i,
    input logic signed [width_p-1:0] data_i [N-1:0],
    output logic data_valid_o,
    output logic data_ready_o,
    output logic signed [width_p-1:0] data_o [N-1:0]
);

    logic signed [width_p-1:0] data_r [N-1:0];
    genvar i;
    generate
        for (i = 0; i < N; i++) begin
            assign data_r[i] = data_i[i] + bias_i[i];
        end
    endgenerate

    elastic
        #(.width_p(width_p),
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
