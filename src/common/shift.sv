module shift
  #(parameter width_p = 8
   ,parameter depth_p = 3
    )
   (input [0:0] clk_i
   ,input [0:0] reset_i
   ,input [0:0] enable_i
   ,input [width_p-1:0] data_i
   ,output [width_p-1:0] data_o [depth_p-1:0]);

   logic [width_p-1:0] buffer_r [depth_p-1:0];

   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         buffer_r[0] <= '0;
      end else if(enable_i) begin
         buffer_r[0] <= data_i;
      end
   end

   for(genvar i = 1; i < depth_p; i++) begin : buffer_shift_loop
      always_ff @(posedge clk_i) begin
         if (reset_i) begin
            buffer_r[i] <= '0;
         end else if(enable_i) begin
            buffer_r[i] <= buffer_r[i-1];
         end
      end
   end

   for(genvar i = 0; i < depth_p; i++) begin : data_o_assign
      assign data_o[i] = buffer_r[i];
   end

endmodule
