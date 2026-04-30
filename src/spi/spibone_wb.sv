// spibone_wb.sv
// Wishbone master driven by SPI slave (spibone protocol).
//
// Protocol (big-endian, CS held low for entire burst):
//   Write: 0x00 | ADDR[31:0] | DATA[31:0] [DATA[31:0] ...]
//   Read:  0x01 | ADDR[31:0] | (clock 0xFF bytes) -> DATA[31:0] [...]
//
// Sampling strategy:
//   SCK is registered once (sck_r). Edge detected as sck_r & ~prev_sck_r.
//   MOSI is registered once (mosi_r). Sampled at sck_rise using mosi_r
//   which holds the value from the previous clock — this is the stable MOSI
//   value that was set up before SCK rose, satisfying SPI Mode 0 setup time.
//   CS is registered twice for clean deassert detection.
//
//   clk_i must be >= 4x SCK for reliable edge detection.
//
// SPDX-License-Identifier: Apache-2.0

`default_nettype none
`timescale 1ns/1ps

module spibone_wb (
    input  logic        clk_i,
    input  logic        rst_i,

    // SPI pins
    input  logic        spi_sck_i,
    input  logic        spi_mosi_i,
    output logic        spi_miso_o,
    input  logic        spi_cs_n_i,

    // Wishbone master
    output logic [31:0] wb_adr_o,
    output logic [31:0] wb_dat_o,
    input  logic [31:0] wb_dat_i,
    output logic        wb_we_o,
    output logic        wb_stb_o,
    output logic        wb_cyc_o,
    input  logic        wb_ack_i
);

    // ------------------------------------------------------------------
    // Input registers
    // sck_r:   SCK registered — use for edge detection
    // prev_sck_r: previous sck_r — edge = sck_r & ~prev_sck_r
    // mosi_r:  MOSI registered one cycle — sampled value at sck_rise
    // cs_n_r/rr: double-register CS for clean deassert detection
    // ------------------------------------------------------------------
    logic sck_r, prev_sck_r;
    logic mosi_r;
    logic cs_n_r, cs_n_rr;

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            sck_r      <= 1'b0;
            prev_sck_r <= 1'b0;
            mosi_r     <= 1'b0;
            cs_n_r     <= 1'b1;
            cs_n_rr    <= 1'b1;
        end else begin
            prev_sck_r <= sck_r;
            sck_r      <= spi_sck_i;
            mosi_r     <= spi_mosi_i;
            cs_n_r     <= spi_cs_n_i;
            cs_n_rr    <= cs_n_r;
        end
    end

    // Edge detection
    // sck_rise: registered SCK just went 0->1
    // sck_fall: registered SCK just went 1->0
    // cs_active: CS has been low for at least 2 cycles
    // cs_deassert: CS just went high (after 2-cycle delay)
    logic sck_rise, sck_fall, cs_active, cs_deassert;
    assign sck_rise   =  sck_r & ~prev_sck_r;
    assign sck_fall   = ~sck_r &  prev_sck_r;
    assign cs_active  = ~cs_n_rr;
    assign cs_deassert = cs_n_r & ~cs_n_rr;

    // ------------------------------------------------------------------
    // Receive shift register
    // Shift in MOSI on sck_rise.
    // Use pin-sampled MOSI at the detected SCK edge to avoid control/data
    // bit skew caused by an extra register delay.
    // ------------------------------------------------------------------
    logic [7:0] rx_shreg;
    logic [2:0] rx_bitcnt;
    logic       rx_byte_done;
    logic       rx_byte_boundary_fall;
    logic [7:0] rx_byte_data;

    assign rx_byte_done = sck_rise & (rx_bitcnt == 3'd7);
    // Byte boundary at falling edge after 8th sampled bit.
    // Use this for TX byte loads so newly loaded byte is stable for the
    // *next* rising-edge sample (avoids one-bit left shift on reads).
    assign rx_byte_boundary_fall = sck_fall & (rx_bitcnt == 3'd0);
    // On the cycle rx_byte_done is true, rx_shreg still holds the previous
    // 7 bits (NBAs update at end of timestep). Build the completed byte
    // explicitly so FSM consumers see the true 8-bit value.
    assign rx_byte_data = {rx_shreg[6:0], spi_mosi_i};

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i || !cs_active) begin
            rx_shreg  <= '0;
            rx_bitcnt <= '0;
        end else if (sck_rise) begin
            rx_shreg  <= {rx_shreg[6:0], spi_mosi_i};
            rx_bitcnt <= rx_bitcnt + 3'd1;
        end
    end

    // ------------------------------------------------------------------
    // Transmit shift register
    // Shift out MSB on sck_fall; FSM loads new byte via tx_load strobe.
    // ------------------------------------------------------------------
    logic [7:0] tx_shreg;
    logic       tx_load;
    logic [7:0] tx_load_val;

    assign spi_miso_o = tx_shreg[7];

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i || !cs_active)
            tx_shreg <= '0;
        else if (tx_load)
            tx_shreg <= tx_load_val;
        else if (sck_fall)
            tx_shreg <= {tx_shreg[6:0], 1'b0};
    end

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,
        S_CMD,
        S_ADDR0, S_ADDR1, S_ADDR2, S_ADDR3,
        S_WB_READ,
        S_TX0, S_TX1, S_TX2, S_TX3, S_TX4,
        S_DATA0, S_DATA1, S_DATA2, S_DATA3,
        S_WB_WRITE_SETUP,  // <-- new
        S_WB_WRITE
    } state_t;

    state_t      state;
    logic        is_read;
    logic [31:0] addr_reg;
    logic [31:0] rd_data;

    always_ff @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            state       <= S_IDLE;
            is_read     <= '0;
            addr_reg    <= '0;
            rd_data     <= '0;
            wb_adr_o    <= '0;
            wb_dat_o    <= '0;
            wb_we_o     <= '0;
            wb_stb_o    <= '0;
            wb_cyc_o    <= '0;
            tx_load     <= '0;
            tx_load_val <= '0;
        end else begin
            tx_load <= '0;  // default: no TX load

            // Deassert WB strobe on ack
            if (wb_ack_i) begin
                wb_stb_o <= '0;
                wb_cyc_o <= '0;
            end

            // CS deasserted — abort and return to idle
            if (cs_deassert || !cs_active) begin
                state    <= S_IDLE;
                wb_stb_o <= '0;
                wb_cyc_o <= '0;
            end else begin

                case (state)

                    S_IDLE: begin
                        if (cs_active)
                            state <= S_CMD;
                    end

                    // ---- Command byte: bit[0]=0 write, bit[0]=1 read ----
                    S_CMD: begin
                        if (rx_byte_done) begin
                            is_read <= rx_byte_data[0];
                            state   <= S_ADDR0;
                        end
                    end

                    // ---- Address bytes MSB first ----
                    S_ADDR0: if (rx_byte_done) begin
                        addr_reg[31:24] <= rx_byte_data;
                        state <= S_ADDR1;
                    end
                    S_ADDR1: if (rx_byte_done) begin
                        addr_reg[23:16] <= rx_byte_data;
                        state <= S_ADDR2;
                    end
                    S_ADDR2: if (rx_byte_done) begin
                        addr_reg[15:8] <= rx_byte_data;
                        state <= S_ADDR3;
                    end
                    S_ADDR3: if (rx_byte_done) begin
                        addr_reg[7:0] <= rx_byte_data;
                        if (is_read) begin
                            wb_adr_o <= {addr_reg[31:8], rx_byte_data};
                            wb_we_o  <= '0;
                            wb_stb_o <= '1;
                            wb_cyc_o <= '1;
                            state    <= S_WB_READ;
                        end else begin
                            state <= S_DATA0;
                        end
                    end

                    // ---- Wait for WB read ack, load TX ----
                    S_WB_READ: begin
                        if (wb_ack_i) begin
                            rd_data <= wb_dat_i;
                            // Align first transmit byte to a byte boundary.
                            // If we load mid-byte, the master sees bit shifts.
                            state   <= S_TX0;
                        end
                    end

                    // ---- Transmit 4 read-data bytes MSB first ----
                    // Each byte is loaded at rx_byte_done so the next full byte
                    // shifted on MISO is stable and aligned.
                    S_TX0: if (rx_byte_boundary_fall) begin
                        tx_load     <= '1;
                        tx_load_val <= rd_data[31:24];
                        state       <= S_TX1;
                    end
                    S_TX1: if (rx_byte_boundary_fall) begin
                        tx_load     <= '1;
                        tx_load_val <= rd_data[23:16];
                        state       <= S_TX2;
                    end
                    S_TX2: if (rx_byte_boundary_fall) begin
                        tx_load     <= '1;
                        tx_load_val <= rd_data[15:8];
                        state       <= S_TX3;
                    end
                    S_TX3: if (rx_byte_boundary_fall) begin
                        tx_load     <= '1;
                        tx_load_val <= rd_data[7:0];
                        state       <= S_TX4;
                    end
                    S_TX4: if (rx_byte_boundary_fall) begin
                        // Burst: increment address, start next WB read
                        addr_reg <= addr_reg + 32'd4;
                        wb_adr_o <= addr_reg + 32'd4;
                        wb_we_o  <= '0;
                        wb_stb_o <= '1;
                        wb_cyc_o <= '1;
                        state    <= S_WB_READ;
                    end

                    // ---- Receive write data bytes MSB first ----
                    S_DATA0: if (rx_byte_done) begin
                        wb_dat_o[31:24] <= rx_byte_data;
                        state <= S_DATA1;
                    end
                    S_DATA1: if (rx_byte_done) begin
                        wb_dat_o[23:16] <= rx_byte_data;
                        state <= S_DATA2;
                    end
                    S_DATA2: if (rx_byte_done) begin
                        wb_dat_o[15:8] <= rx_byte_data;
                        state <= S_DATA3;
                    end
                    S_DATA3: if (rx_byte_done) begin
                        wb_dat_o[7:0] <= rx_byte_data;
                        wb_adr_o      <= addr_reg;
                        wb_we_o       <= '1;
                        state         <= S_WB_WRITE_SETUP;  // wait one cycle
                    end

                    S_WB_WRITE_SETUP: begin
                        // wb_dat_o is now fully settled — safe to assert strobe
                        wb_stb_o <= '1;
                        wb_cyc_o <= '1;
                        state    <= S_WB_WRITE;
                    end

                    // ---- Wait for WB write ack, loop for burst ----
                    S_WB_WRITE: begin
                        if (wb_ack_i) begin
                            addr_reg <= addr_reg + 32'd4;
                            state    <= S_DATA0;
                        end
                    end

                    default: state <= S_IDLE;

                endcase
            end
        end
    end

endmodule

`default_nettype wire