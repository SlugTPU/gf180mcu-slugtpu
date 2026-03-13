module elastic
  #(parameter [31:0] width_p = 8
   ,parameter [31:0] depth_p = 8
   )
  (input clk_i
  ,input rst_i

  ,input signed [width_p - 1:0] data_i [depth_p-1:0]
  ,input valid_i
  ,output ready_o 

  ,output logic valid_o 
  ,output logic signed [width_p - 1:0] data_o [depth_p-1:0]
  ,input ready_i
  );
  genvar i;
  generate
    for (i = 0; i < depth_p ; i++) begin
      always_ff @( posedge clk_i) begin
        if (rst_i) begin
          data_o[i] <= '0;
        end else if( valid_i && ready_o) begin
          data_o[i] <= data_i[i];
        end
      end
    end
  endgenerate

  always_ff @( posedge clk_i) begin
    if (rst_i) begin
      valid_o <= '0;
    end else if(ready_o) begin
      valid_o <= ready_o & valid_i;
    end
  end

  assign ready_o = ~valid_o | ready_i;

endmodule
