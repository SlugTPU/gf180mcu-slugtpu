/*
SRAM FIFO that interfaces with and serves as arbiter for both dram and decoder
*/
module control_sram #(
    parameter int sram_width_p = 10
) (
`ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
`endif
    input clk_i,
    input rst_i,

    // to/from buffer
    input [7:0] wr_data_i,
    input valid_i,
    output ready_o,

    // to/from decoder
    input ready_i,
    output [7:0] rd_data_o,
    output valid_o,
    
    output is_full_o //used for initial load after exiting IDLE state
);

    logic [sram_width_p:0]      wr_ptr_n, wr_ptr_q, rd_ptr_n, rd_ptr_q;
    logic [sram_width_p-1:0]    addr_l;
    logic                       write_en_w, read_en_w, rw_mode, last_cycle_read, rw_mode_last_cycle;
    wire                        is_full_w, is_empty_w;

    assign read_en_w = ready_i && ~is_empty_w;
    // assign read_en_w = (ready_i && ~is_empty_w && ~rw_mode_last_cycle);
    assign write_en_w = (valid_i && ready_o);

    assign rd_ptr_n = (read_en_w) ? rd_ptr_q + 1 : rd_ptr_q;
    assign wr_ptr_n = (write_en_w) ? wr_ptr_q + 1 : wr_ptr_q;

    assign is_full_w = (wr_ptr_q[sram_width_p] != rd_ptr_q[sram_width_p]) &&
                        (wr_ptr_q[sram_width_p - 1:0] == rd_ptr_q[sram_width_p - 1:0]);
    assign is_empty_w = wr_ptr_q == rd_ptr_q;
    assign is_full_o = is_full_w;

    // Make sure we do not write in when there is a valid read request
    assign ready_o = ~is_full_w & ~ready_i;

    assign valid_o = last_cycle_read; // valid if there was a read transaction last cycle

    assign rw_mode = write_en_w;
    assign addr_l = (write_en_w) ? wr_ptr_q[sram_width_p - 1:0] : rd_ptr_q[sram_width_p - 1:0];
    
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wr_ptr_q <= '0;
            rd_ptr_q <= '0;
            last_cycle_read <= '0;
            rw_mode_last_cycle <= '0;
        end else begin
            wr_ptr_q <= wr_ptr_n;
            rd_ptr_q <= rd_ptr_n;
            last_cycle_read <= read_en_w;
            rw_mode_last_cycle <= rw_mode;
        end
    end

    sram_1x1024
    control_sram_block(
`ifdef USE_POWER_PINS
        .VDD(VDD),
        .VSS(VSS),
`endif
        .clk_i(clk_i),
        .rst_i(rst_i),

        .addr_i(addr_l),
        .wr_data_i(wr_data_i),
        .rd_data_o(rd_data_o),
        .en_i(~rst_i),
        .rw_mode_i(rw_mode)
    );
    
endmodule
