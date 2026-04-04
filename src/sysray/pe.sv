// Processing Element (PE)
//
// Performs one 8-bit MAC per cycle and passes activations and weights to
// adjacent PEs in the systolic array.
//
// Data flow:
//   - Activations  : flow left-to-right, registered 1 cycle per PE
//   - Weights      : flow top-to-bottom via a shift-register chain
//   - Partial sums : accumulate top-to-bottom; psum_o = psum_i + act * weight
//
// Double-buffered weights:
//   The MSB of weight_i and act_i is a "select" bit (not data). Weights are
//   written into weight_buf[weight_sel], allowing the next inference's weights
//   to be pre-loaded into the inactive bank while the current inference runs.
//   The MAC reads from weight_buf[act_sel], so the active bank is determined
//   by the activation stream — flipping act_sel atomically switches banks.
//
// Bank-switch bubble (weight_valid_o):
//   When weight_sel toggles, weight_valid_o is suppressed for one cycle. This 

module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                  clk_i,
    input  logic                  rst_i,

    input  logic signed [DATA_WIDTH:0] act_i, //top bit is select
    output logic signed [DATA_WIDTH:0] act_o,

    input  logic signed [DATA_WIDTH:0] weight_i, //shift reg chain
    output logic signed [DATA_WIDTH:0] weight_o,

    input  logic weight_valid_i,
    output logic weight_valid_o,

    input  logic act_valid_i,
    output logic act_valid_o,

    input  logic signed [ACC_WIDTH-1:0]  psum_i,
    output logic signed [ACC_WIDTH-1:0]  psum_o
);

    logic weight_sel, act_sel, weight_edge, prev_weight_sel;
    assign weight_sel = weight_i[DATA_WIDTH];
    assign act_sel = act_i[DATA_WIDTH];

    // detect a bank switch: any toggle of weight_sel
    assign weight_edge = prev_weight_sel != weight_sel;

    // suppress valid for one cycle on a bank switch so downstream does not
    // latch before the new bank has been written into this PE's buffer
    assign weight_valid_o = weight_valid_i & ~weight_edge;

    // two weight banks for double-buffering
    logic [DATA_WIDTH:0] weight_buf [1:0];

    // track the previously active bank; updated on each bank switch
    always_ff @(posedge clk_i) begin
        if (rst_i)
            prev_weight_sel <= '0;
        else if (weight_edge)
            prev_weight_sel <= weight_sel;
    end

    // capture incoming weight into the bank indicated by the select bit
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            weight_buf[0] <= '0;
            weight_buf[1] <= '0;
        end else if (weight_valid_i)
            weight_buf[weight_sel] <= weight_i;
    end

    // MAC reads from the bank selected by the activation stream
    logic signed [DATA_WIDTH-1:0]   active_weight;
    logic signed [ACC_WIDTH-1:0]    s_product;
    assign active_weight = weight_buf[act_sel][DATA_WIDTH-1:0];
    assign s_product     = $signed(act_i[DATA_WIDTH-1:0]) * active_weight;  // 8b signed × 8b signed, sign-extended to 32b

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            psum_o <= '0;
        end else if (act_valid_i) begin
            psum_o <= $signed(psum_i) + ACC_WIDTH'(s_product);
        end else begin
            psum_o <= '0;
        end
    end

    // pass through activation
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            act_o <= '0;
            act_valid_o <= '0;
        end else begin
            act_o <= act_i;
            act_valid_o <= act_valid_i;
        end
    end

    // output from the previously-settled bank so downstream always sees stable data
    assign weight_o = weight_buf[prev_weight_sel];

endmodule
