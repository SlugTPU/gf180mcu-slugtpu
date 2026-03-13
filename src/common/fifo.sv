module fifo #(
    parameter int DEPTH_LOG2_P = 3,
    parameter int WIDTH_P = 8
) (
    input clk_i,
    input rst_i,

    // data
    input logic [WIDTH_P - 1:0] data_i,
    output logic [WIDTH_P - 1:0] data_o,
    // handshake
    input logic [0:0] valid_i,
    input logic [0:0] ready_i,
    output logic [0:0] valid_o,
    output logic [0:0] ready_o
);
    logic [WIDTH_P - 1:0] mem [1 << DEPTH_LOG2_P];
    logic [DEPTH_LOG2_P:0]    wr_ptr_n, wr_ptr_q, rd_ptr_n, rd_ptr_q;
    logic [0:0]               write_en_w, read_en_w;
    wire  [0:0]               is_full_w, is_empty_w;

    assign read_en_w = (ready_i && valid_o);
    assign write_en_w = (valid_i && ready_o);

    assign rd_ptr_n = (read_en_w) ? rd_ptr_q + 1 : rd_ptr_q;
    assign wr_ptr_n = (write_en_w) ? wr_ptr_q + 1 : wr_ptr_q;

    assign is_full_w = (wr_ptr_q[DEPTH_LOG2_P] != rd_ptr_q[DEPTH_LOG2_P]) &&
                        (wr_ptr_q[DEPTH_LOG2_P - 1:0] == rd_ptr_q[DEPTH_LOG2_P - 1:0]);
    assign is_empty_w = wr_ptr_q == rd_ptr_q;

   assign ready_o = ~is_full_w;
   assign valid_o = ~is_empty_w;

   assign data_o = mem[rd_ptr_q[DEPTH_LOG2_P - 1:0]];

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wr_ptr_q <= '0;
            rd_ptr_q <= '0;
        end else begin
            wr_ptr_q <= wr_ptr_n;
            rd_ptr_q <= rd_ptr_n;
        end
    end

    always_ff @(posedge clk_i) begin
        if (write_en_w) begin
            mem[wr_ptr_q[DEPTH_LOG2_P - 1:0]] <= data_i;
        end
    end

endmodule
