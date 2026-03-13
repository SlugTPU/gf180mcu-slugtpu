// Direct SRAM interface
// TRANSACTION MODE IS 0 FOR READ, 1 FOR WRITE
module memory_transaction #(
    parameter counter_width = 8,
    parameter address_width = 8
)(
    input clk_i,
    input rst_i,

    // IO to SRAM module
    output logic [address_width-1:0] sram_addr_o,
    output logic sram_rw_mode_o,

    input  downstream_ready_i,
    output logic ready_o,
    output logic rd_valid_o,

    input  [address_width-1:0] addr_i,
    input  [counter_width-1:0] transaction_amount_i,
    input  transaction_rw_mode_i,
    input  load_valid_i,
    output load_ready_o
);

    logic [counter_width-1:0] current_count_q, transaction_amount_q, current_count_d;
    logic [address_width-1:0] addr_d, addr_q;
    logic in_use, rw_mode, rw_mode_q;

    // assign sram_addr_o = addr_q;
    
    // always_ff @( negedge clk_i ) begin
    //     sram_addr_o <= addr_q;
    //     sram_rw_mode_o <= rw_mode;
    // end

    // //sram_addr_o is the input to the SRAM macro on the A port
    // always @(addr_q) sram_addr_o = #200 addr_q;
    // always_ff @( posedge clk_i ) begin
    //     addr_q <= addr_d;
    // end

    always @(addr_q) sram_addr_o = #150 addr_q;
    always @(rw_mode_q) sram_rw_mode_o = #200 rw_mode_q;

    assign load_ready_o = ~in_use;

    assign ready_o = in_use;
    
    always_comb begin
        addr_d = addr_q;
        current_count_d = current_count_q;
        if (downstream_ready_i & in_use) begin
            current_count_d = current_count_q + 1'b1;
            addr_d = addr_q + 1'b1;
        end
    end

    always_comb begin
        rw_mode_q = '0;
        if (in_use & downstream_ready_i )
            rw_mode_q = rw_mode;
    end

    always_ff @( posedge clk_i ) begin
        if(rst_i) begin
            current_count_q <= '0;
            addr_q <= '0;
            in_use <= '0;
            rw_mode <= '0;
            rd_valid_o <= '0;
        end else if(load_valid_i & ~in_use) begin
            current_count_q <= '0;
            transaction_amount_q <= transaction_amount_i;
            in_use <= '1; 
            addr_q <= addr_i;
            rw_mode <= transaction_rw_mode_i;
        end else begin
            current_count_q <= current_count_d;           
            addr_q <= addr_d;
            rd_valid_o <= ready_o & ~rw_mode;

            if (in_use & downstream_ready_i &
                // if it is a read transaction, we add one for some reason I used to know
                (current_count_d  == transaction_amount_q)) begin
                in_use <= 1'b0;
                rw_mode <= '0;
            end
        end
    end

endmodule
