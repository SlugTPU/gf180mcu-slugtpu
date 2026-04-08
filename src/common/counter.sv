// Simple Counter Module
// Resets to 0
// rolls over

module counter #(
    parameter int width_p = 4
) (
    input clk_i,
    input rst_i,
    
    input up_i,
    output logic [width_p-1:0] count_o
);

    always_ff @( posedge clk_i ) begin
        if(rst_i) begin
            count_o <= '0;
        end else if (up_i) begin
            count_o <= count_o + 1;
        end
    end
    
endmodule
