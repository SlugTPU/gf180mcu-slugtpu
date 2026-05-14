// debug hell my beloved
// presents N_WORDS bytes of internal state as a flat byte window. address selects which byte is driven on the output; out-of-range reads return a fixed sentinel so unpopulated addresses are obvious on a scope.

module debug_mux #(
    parameter int N_WORDS = 32
)(
    input  logic [7:0]            addr_i,
    input  logic [N_WORDS*8-1:0]  data_i,
    output logic [7:0]            data_o
);

    localparam logic [7:0] OOR_SENTINEL = 8'hDE;

    always_comb begin
        if (addr_i < N_WORDS[7:0])
            data_o = data_i[addr_i*8 +: 8];
        else
            data_o = OOR_SENTINEL;
    end

endmodule

