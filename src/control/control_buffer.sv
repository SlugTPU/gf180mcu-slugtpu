/*
Program counter lives here
*/
module control_buffer #(
    parameter int DRAM_WIDTH = 64,
    parameter int CONTROL_WIDTH = 8,
    parameter int DRAM_ADDR_WIDTH = 12,
    parameter int DEPTH_LOG2_P = $clog2(CONTROL_WIDTH)
) (
    input clk_i,
    input rst_i,

    // load the program counter in here
    input [DRAM_ADDR_WIDTH-1:0] pc_in,
    input pc_valid_i,

    // DRAM data signals
    input [DRAM_WIDTH-1:0] wr_data_i,
    input wr_valid_i,
    output wr_ready_o,

    // DMA control signals. Control unit should write other signals
    output logic [DRAM_ADDR_WIDTH-1:0] pc_out,

    // control sram signals
    output logic [CONTROL_WIDTH-1:0] rd_data_o,
    output rd_valid_o,
    input rd_ready_i

);
    logic [DRAM_ADDR_WIDTH-1:0] pc_out_n;
    logic [DRAM_WIDTH-1:0] dff_array;
    logic [DEPTH_LOG2_P:0] rd_ptr_n, rd_ptr_q;
    logic wr_ptr_n, wr_ptr_q, write_en_w, read_en_w, is_empty_w;
    logic [DEPTH_LOG2_P-1:0] rd_ptr_l;

    assign pc_out = pc_out_n;

    assign read_en_w = (rd_ready_i && rd_valid_o);
    assign write_en_w = (wr_valid_i && wr_ready_o);

    assign rd_ptr_n = (read_en_w) ? rd_ptr_q + 1 : rd_ptr_q;
    assign wr_ptr_n = (write_en_w) ? wr_ptr_q + 1 : wr_ptr_q;
    assign rd_ptr_l = rd_ptr_q[DEPTH_LOG2_P-1:0];

    assign is_empty_w = wr_ptr_q == rd_ptr_q[DEPTH_LOG2_P];

    // We only write when the dff array is fully empty
    // Technically this wastes a cycle but whatever
    assign wr_ready_o = is_empty_w;
    assign rd_valid_o = ~is_empty_w;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            rd_ptr_q <= '0;
            wr_ptr_q <= '0;
        end else begin
            rd_ptr_q <= rd_ptr_n;
            wr_ptr_q <= wr_ptr_n;
        end
    end

    always_ff @( posedge clk_i ) begin : pc_dff
        if (rst_i)
            pc_out_n <= '0;
        else if (pc_valid_i)
            pc_out_n <= pc_in;
        else if (write_en_w)
            pc_out_n <= pc_out_n +1;
    end

    always_ff @( posedge clk_i ) begin : dff_array_write
        if (rst_i)
            dff_array <= '0;
        else if (write_en_w)
            dff_array <= wr_data_i;
    end

    // always_ff @( posedge clk_i ) begin : rd_data_write
    //     if (rst_i)
    //         rd_data_o <= '0;
    //     else if (read_en_w)
    //         rd_data_o <= dff_array[((rd_ptr_l) << DEPTH_LOG2_P) +: CONTROL_WIDTH];
    // end
    assign rd_data_o = dff_array[rd_ptr_l * CONTROL_WIDTH +: CONTROL_WIDTH];
    
endmodule