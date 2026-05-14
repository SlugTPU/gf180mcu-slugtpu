// stand in for SDRAM for debug mode
module wb_test_ram #(
    parameter int N_WORDS = 128, // number of 32-bit words; default 512 B
    parameter int AdrW    = 32,
    parameter int DataW   = 32
)(
    input  logic clk_i,
    input  logic rst_i,
 
    input  logic [AdrW-1:0]    wb_adr_i,
    input  logic [DataW-1:0]   wb_dat_i,
    output logic [DataW-1:0]   wb_dat_o,
    input  logic [DataW/8-1:0] wb_sel_i,
    input  logic               wb_we_i,
    input  logic               wb_stb_i,
    input  logic               wb_cyc_i,
    output logic               wb_ack_o
);
 
    localparam int IndexW = $clog2(N_WORDS);
 
    wire [IndexW-1:0] index = wb_adr_i[IndexW+1:2];
 
    logic [DataW-1:0] mem [N_WORDS];
 
    // single cyc ack
    always_ff @(posedge clk_i) begin
        if (rst_i) wb_ack_o <= 1'b0;
        else       wb_ack_o <= wb_cyc_i && wb_stb_i && !wb_ack_o;
    end
 
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wb_dat_o <= '0;
        end else if (wb_cyc_i && wb_stb_i && !wb_ack_o) begin
            if (wb_we_i) begin
                for (int b = 0; b < DataW/8; b++)
                    if (wb_sel_i[b]) mem[index][b*8 +: 8] <= wb_dat_i[b*8 +: 8];
            end else begin
                wb_dat_o <= mem[index];
            end
        end
    end
 
endmodule
 
