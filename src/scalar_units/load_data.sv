/*
This module contains the logic for loading in data to the scalar units
    (adders, multiplier, relu)
ASSUMPTION: The control unit ensures that the load instruction does not
occur when the pipeline is in use.

Input: Tensor of form [val_1, val_2, ...val_n]
    divided across multiple clock cycles
    little endian - if a cycle contains two values, they must be
    top bits = val_i    lower bits = val_i+1

Output: Tensor of form [val_1, val_2, ...val_n]
    little endian -> val_1 is at index 0
*/
module load_scalar_data
   #(parameter scalar_data_width_p = 32
    ,parameter lane_depth_p = 4
    ,parameter read_bus_width = 64
    ,parameter num_lanes = read_bus_width/scalar_data_width_p
    )
    (input clk_i
    ,input reset_i
    ,input[read_bus_width-1:0] read_bus

    //this signal comes from the sram/axi -> read bus has valid data
    ,input[0:0] load_valid_i

    //this signal comes from the control unit -> data on bus is for this module
    ,input[0:0] load_enable_i

    //unused atm
    // ,output[0:0] load_ready_o

    // In our case, this should be [31:0] [7:0]
    ,output signed [scalar_data_width_p-1:0] scalar_values_o [(num_lanes*lane_depth_p)-1:0]
    );
    initial begin
        assert(read_bus_width % scalar_data_width_p == 0) 
            else $error("Scalar data width is not an exact multiple of the read bus width !!!");
    end
    wire [0:0] enable_shift;
    assign enable_shift = load_valid_i & load_enable_i;

    //shift registers
    genvar lane, i;
    generate
        for (lane = 0; lane < num_lanes; lane++) begin : loader_shift_reg
            wire[scalar_data_width_p-1:0] raw_output [lane_depth_p-1:0];
            shift
               #(.width_p(scalar_data_width_p)
                ,.depth_p(lane_depth_p)
                )
            shift_loader_inst
            (
                 .clk_i(clk_i)
                ,.reset_i(reset_i)
                ,.enable_i(enable_shift)
                ,.data_i(read_bus[(lane+1)*scalar_data_width_p-1: lane*scalar_data_width_p])
                ,.data_o(raw_output)
            );
            for (i = 0; i < lane_depth_p; i ++) begin
                assign scalar_values_o[i*num_lanes+lane] = raw_output[lane_depth_p - 1 - i];
                // assign scalar_value_o[num_lanes*i + lane*num_lanes + num_lanes - 1: num_lanes*i + lane*num_lanes] = raw_output[i+num_lanes-1:i];
            end
        end
    endgenerate


endmodule
