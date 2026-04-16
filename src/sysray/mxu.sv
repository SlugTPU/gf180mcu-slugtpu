// Matrix Multiply Unit
// Inputs: 2 NxN Matrices of DATA_WIDTH wide elements
// Output: NxN Matrix of ACC_WIDTH wide elements 
// All matrices are row major

//TODO: weight ready, activation ready logic and weight stall logic

module mxu #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    parameter int N = 8, // For this module, must be power of 2
    parameter int BUS_WIDTH = 64
) (
    input clk_i,
    input rst_i,

    //activations
    input act_enable_i, // from control
    input act_valid_i,  // from sram
    input [BUS_WIDTH-1:0] act_bus_i,
    // output act_ready_o, // may be uneeded, we could have perma 1 for activation sram

    //weights
    input weight_enable_i,
    input weight_valid_i,
    input [BUS_WIDTH-1:0] weight_bus_i,
    output weight_ready_o,

    output logic [ACC_WIDTH-1:0] psum_o [N-1:0],
    output logic psum_valid_o [N-1:0],

    output [(ACC_WIDTH)*8-1:0] debug_output,
    output [(DATA_WIDTH+2)*N-1:0] activation_debug
);

    logic [3:0] activation_count, weight_count;
    logic act_valid_l, weight_valid_l, act_select, weight_select, shift_en;
    assign act_valid_l = act_enable_i & act_valid_i;
    assign weight_valid_l = weight_enable_i & weight_valid_i;

    counter #(
        .width_p(4)
    ) weight_counter(
        .clk_i(clk_i),
        .rst_i(rst_i | ~weight_enable_i),

        .up_i(weight_valid_l),
        .count_o(weight_count)
    );

    counter #(
        .width_p(4)
    ) activation_counter(
        .clk_i(clk_i),
        .rst_i(rst_i | ~act_enable_i),

        .up_i(act_valid_l),
        .count_o(activation_count)
    );

    assign weight_select = weight_count[3];
    assign act_select = activation_count[3];

    /*
    INCOMPLETE
    easier to reason about w/complete compute core/mem behavior
    shift_en should be flip flopped
    ready_o should be combinatational
    */
    always_comb begin
        shift_en = '1;
        // if(weight_count == 4'b0111 && ~act_valid_l)
        //     shift_en = '0;
    end
    assign weight_ready_o = shift_en;


    logic [DATA_WIDTH+1:0] act_shift_in [N-1:0];
    logic [DATA_WIDTH+1:0] act_shift_out [N-1:0];
    logic [DATA_WIDTH+1:0] weight_shift_out [N-1:0];
    logic [DATA_WIDTH+1:0] weight_shift_in [N-1:0];
    logic signed [DATA_WIDTH-1:0] act_data_n [N-1:0];
    logic signed [DATA_WIDTH-1:0] weight_data_n [N-1:0];
    logic act_valid_n [N-1:0], weight_valid_n[N-1:0], act_select_n[N-1:0], weight_select_n [N-1:0]; 

    genvar i;
    generate
        for (i = 0; i < N; ++i) begin
            assign act_shift_in[i]      = (act_valid_l) ? { act_select, act_valid_l, act_bus_i[(i+1)*N - 1 : i*N] } : '0;
            assign weight_shift_in[i]   = (weight_valid_l) ? { weight_select, weight_valid_l, weight_bus_i[(i+1)*N - 1 : i*N] } : '0;

            assign act_select_n[i]      = act_shift_out[i][DATA_WIDTH+1];
            assign act_valid_n[i]       = act_shift_out[i][DATA_WIDTH];
            assign act_data_n[i]        = act_shift_out[i][DATA_WIDTH-1:0];

            assign weight_select_n[i]   = weight_shift_out[i][DATA_WIDTH+1];
            assign weight_valid_n[i]    = weight_shift_out[i][DATA_WIDTH];
            assign weight_data_n[i]     = weight_shift_out[i][DATA_WIDTH-1:0];
        end
    endgenerate

    tri_shift #(
        .N(N),
        .DATA_W(DATA_WIDTH+2)
    ) activation_shift (
        .clk(clk_i),
        .rst(rst_i),
        .data_i(act_shift_in),
        .enable_i(shift_en),
        .data_o(act_shift_out)
    );

    tri_shift #(
        .N(N),
        .DATA_W(DATA_WIDTH+2)
    ) weight_shift (
        .clk(clk_i),
        .rst(rst_i),
        .data_i(weight_shift_in),
        .enable_i(shift_en),
        .data_o(weight_shift_out)
    );

    logic signed [ACC_WIDTH-1:0]   sys_out [N-1:0];
    logic sys_valid_out[N-1:0];

    sysray_nxn #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .N(N)
    ) sysray (
        .clk_i(clk_i),
        .rst_i(rst_i),

        .act_valid_n_i(act_valid_n),
        .act_n_i(act_data_n),
        .act_sel_n_i(act_select_n),

        .weight_valid_n_i(weight_valid_n),
        .weight_n_i(weight_data_n),
        .weight_sel_n_i(weight_select_n),

        .psum_out_n_o(sys_out),
        .psum_out_valid_n_o(sys_valid_out)
    );

    logic [ACC_WIDTH:0] output_flipped [N-1:0];
    logic [ACC_WIDTH:0] mxu_out [N-1:0];

    generate
        for (i = 0; i < N; i = i + 1) begin
            assign output_flipped[i] = {sys_valid_out[N-1-i], sys_out[N-1-i]};
            assign psum_o[i] = mxu_out[N-1-i][ACC_WIDTH-1:0];
            assign psum_valid_o[i] = mxu_out[N-1-i][ACC_WIDTH];

            assign debug_output[(i+1)*ACC_WIDTH-1:i*ACC_WIDTH] = sys_out[i];
            assign activation_debug[(i+1)*(DATA_WIDTH+2)-1:i*(DATA_WIDTH+2)] = act_shift_out[i];
        end
    endgenerate
    
    tri_shift #(
        .N(N),
        .DATA_W(ACC_WIDTH+1)
    ) outputs (
        .clk(clk_i),
        .rst(rst_i),
        .data_i(output_flipped),
        .enable_i('1),
        .data_o(mxu_out)        
    );
    
    
endmodule

