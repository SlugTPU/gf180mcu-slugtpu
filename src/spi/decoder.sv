// SPDX-License-Identifier: Apache-2.0
// decoder.sv
//
// Wishbone B3 master bridging housekeeping_spi.
// Supports streaming read: after the first rdstb, issues back-to-back
// WB reads for each subsequent byte without returning to IDLE.

`default_nettype none

module decoder (
    input  logic        wb_clk_i,
    input  logic        wb_rst_i,

    // From housekeeping_spi
    input  logic [15:0] oaddr,
    input  logic [7:0]  odata,
    input  logic        wrstb,
    input  logic        rdstb,

    // To housekeeping_spi idata port
    output logic [7:0]  wb_idata,

    // Wishbone B3 master
    output logic [15:0] wb_adr_o,
    output logic [7:0]  wb_dat_o,
    input  logic [7:0]  wb_dat_i,
    output logic        wb_we_o,
    output logic        wb_stb_o,
    output logic        wb_cyc_o,
    input  logic        wb_ack_i
);

    typedef enum logic [2:0] {
        IDLE   = 3'b000,
        TRANS  = 3'b001,
        DONE   = 3'b010,
        STREAM = 3'b011
    } state_t;

    state_t state;

    logic wrstb_r, rdstb_r;
    logic wrstb_rising, rdstb_rising;

    always_ff @(posedge wb_clk_i) begin
        wrstb_r <= wrstb;
        rdstb_r <= rdstb;
    end

    assign wrstb_rising = wrstb & ~wrstb_r;
    assign rdstb_rising = rdstb & ~rdstb_r;

    logic [15:0] stream_addr;

    always_ff @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            state       <= IDLE;
            wb_adr_o    <= '0;
            wb_dat_o    <= '0;
            wb_we_o     <= 1'b0;
            wb_stb_o    <= 1'b0;
            wb_cyc_o    <= 1'b0;
            wb_idata    <= '0;
            stream_addr <= '0;
        end else begin
            unique case (state)

                IDLE: begin
                    wb_stb_o <= 1'b0;
                    wb_cyc_o <= 1'b0;
                    if (wrstb_rising) begin
                        wb_adr_o <= oaddr;
                        wb_dat_o <= odata;
                        wb_we_o  <= 1'b1;
                        wb_stb_o <= 1'b1;
                        wb_cyc_o <= 1'b1;
                        state    <= TRANS;
                    end else if (rdstb_rising) begin
                        wb_adr_o    <= oaddr;
                        stream_addr <= oaddr;
                        wb_dat_o    <= '0;
                        wb_we_o     <= 1'b0;
                        wb_stb_o    <= 1'b1;
                        wb_cyc_o    <= 1'b1;
                        state       <= TRANS;
                    end
                end

                TRANS: begin
                    if (wb_ack_i) begin
                        wb_stb_o <= 1'b0;
                        wb_idata <= wb_dat_i;
                        if (wb_we_o) begin
                            wb_cyc_o <= 1'b0;
                            state    <= DONE;
                        end else begin
                            // First read done — pre-fetch next address
                            stream_addr <= stream_addr + 1;
                            wb_adr_o    <= stream_addr + 1;
                            wb_dat_o    <= '0;
                            wb_we_o     <= 1'b0;
                            wb_stb_o    <= 1'b1;
                            wb_cyc_o    <= 1'b1;
                            state       <= STREAM;
                        end
                    end
                end

                STREAM: begin
                    if (wb_ack_i) begin
                        wb_idata <= wb_dat_i;
                        if (rdstb_rising) begin
                            // More bytes needed — pre-fetch next
                            stream_addr <= stream_addr + 1;
                            wb_adr_o    <= stream_addr + 1;
                            wb_stb_o    <= 1'b1;
                            wb_cyc_o    <= 1'b1;
                        end else begin
                            // Stream done
                            wb_stb_o <= 1'b0;
                            wb_cyc_o <= 1'b0;
                            state    <= DONE;
                        end
                    end else if (!rdstb && !rdstb_r && !wb_stb_o) begin
                        state <= IDLE;
                    end
                end

                DONE: state <= IDLE;

                default: state <= IDLE;

            endcase
        end
    end

endmodule : decoder

`default_nettype wire