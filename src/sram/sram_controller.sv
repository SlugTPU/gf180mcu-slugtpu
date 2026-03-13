module sram_mux #(
    parameter data_width = 64,
    parameter addr_width = 8
) (
    input clk_i,
    input rst_i,

    // IO to SRAM module
    output [addr_width-1:0] sram_addr_o,
    output [data_width-1:0] sram_wr_data_o,
    output sram_en_o,
    output sram_rw_mode_o,
    input [data_width-1:0] sram_rd_data_i,

    // IO to mem_transaction module in write mode
    input wr_valid_i,
    output wr_ready_o,
    input [addr_wdith-1:0] wr_addr_i,
    input [data_width-1:0] write_buffer_i,

    // IO to mem_transaction module in read mode
    input  rd_ready_i,
    output rd_valid_o,
    input  [addr_wdith-1:0] rd_addr_i,
    output [data_width-1:0] read_buffer_o
);

    logic rw_mode;
    logic in_use_q, in_use_d;

    assign unified_read_buffer_o = sram_rd_data_i;
    assign sram_wr_data_o = unified_write_buffer_i;
    assign sram_en_o = '1;
    assign sram_rw_mode_o = rw_mode;

    always_comb begin
        in_use_d = in_use_q;
        if(wr_valid_i) begin
            in_use_d = '1;
        end

        if(rw_mode == '0)
            sram_addr_o = rd_addr_i;
        else
            sram_addr_o = wr_addr_i;
    end

    always_ff @( posedge clk_i ) begin
        if(rst_i) begin
            which_reader <= '1;
            which_writer <= '1;
            rw_mode <= '0;
            in_use <= '0;
        end else begin
            in_use_q <= in_use_d;

        end
    end

    
endmodule
