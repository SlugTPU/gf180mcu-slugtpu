module relu_n
    #(parameter N = 8,
     parameter width_p = 32
    )(
    input logic clk_i,
    input logic rst_i,

    input logic signed [width_p-1:0] data_i [N-1:0],
    input logic data_valid_i,
    input logic data_ready_i,

    output logic signed [width_p-1:0] data_o [N-1:0],
    output logic data_valid_o,
    output logic data_ready_o
    );

    logic signed [width_p-1:0] data_d [N-1:0];

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_relu
            assign data_d[i] = data_i[i][width_p-1] ? '0 : data_i[i];
        end
    endgenerate

    // elastic
    assign data_ready_o = ~data_valid_o | data_ready_i;

    generate
        for (i = 0; i < N; i++) begin : gen_reg
            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    data_o[i] <= '0;
                end else if (data_ready_o) begin
                    data_o[i] <= data_d[i];
                end
            end
        end
    endgenerate

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            data_valid_o <= 1'b0;
        end else if (data_ready_o) begin
            data_valid_o <= data_valid_i;
        end
    end

endmodule
