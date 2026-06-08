// simple CPOL=0 CPHA=0 1-byte spi slave driver
// requires SPI master clock frequency to be 4x slower than system clock

module spi_slave (
    input clk_i,
    input rst_i,

    input  spi_sck_async_i,
    input  spi_cs_async_ni,
    input  spi_mosi_async_i,
    output spi_miso_o,

    output active_o,

    output [7:0] byte_rx_o,
    input  [7:0] byte_tx_i,
    output byte_stb_o
);

   logic [1:0] sclk_sync_q, sclk_sync_d;
   logic [2:0] cs_sync_q, cs_sync_d;
   logic [1:0] mosi_sync_q, mosi_sync_d;
   logic [3:0] ctr_bits_d, ctr_bits_q;
   logic       byte_valid;
   logic       byte_valid_prev_q, byte_valid_prev_d;

   logic       mosi_data;
   // logic       cs_prev_q, cs_prev_d;

   logic [7:0] rx_byte_d, rx_byte_q;
   logic       tx_q, tx_d;

   logic       sclk_rising_edge, sclk_falling_edge;

   assign sclk_sync_d = { sclk_sync_q[0], spi_sck_async_i };
   assign cs_sync_d = { cs_sync_q[1:0], spi_cs_async_ni };
   assign mosi_sync_d = { mosi_sync_q[0], spi_mosi_async_i };

   assign sclk_rising_edge  = ({sclk_sync_q[1], sclk_sync_q[0]} == 2'b01) ? 1'b1 : 1'b0;
   assign sclk_falling_edge = ({sclk_sync_q[1], sclk_sync_q[0]} == 2'b10) ? 1'b1 : 1'b0;

   assign byte_valid = (ctr_bits_q == 4'd8);
   assign byte_valid_prev_d = byte_valid;
   assign byte_stb_o = {byte_valid_prev_q,  byte_valid} == 2'b01;

   assign mosi_data = mosi_sync_q[1];

   assign byte_rx_o = rx_byte_q;
   assign spi_miso_o = tx_q;

   assign active_o = ~cs_sync_q[2];

   always_comb begin
      rx_byte_d = rx_byte_q;
      tx_d = tx_q;
      ctr_bits_d = ctr_bits_q;

      if (~active_o || byte_stb_o) begin
         rx_byte_d = '0;
         tx_d = byte_tx_i[7];
         ctr_bits_d = '0;
      end else if (sclk_rising_edge) begin
         rx_byte_d = {rx_byte_q[6:0] , mosi_data};
         ctr_bits_d = ctr_bits_q + 4'h1;
      end else if (sclk_falling_edge) begin
         tx_d = byte_tx_i[3'(4'd7 - ctr_bits_q)];
      end
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         ctr_bits_q <= '0;
      end else begin
         ctr_bits_q <= ctr_bits_d;
      end
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         sclk_sync_q <= '0;
         cs_sync_q <= '0;
         mosi_sync_q <= '0;

         byte_valid_prev_q <= '0;
         // cs_prev_q <= '0;

         rx_byte_q <= '0;
         tx_q <= '0;
      end else begin
         sclk_sync_q <= sclk_sync_d;
         cs_sync_q <= cs_sync_d;
         mosi_sync_q <= mosi_sync_d;

         byte_valid_prev_q <= byte_valid_prev_d;
         // cs_prev_q <= cs_prev_d;

         rx_byte_q <= rx_byte_d;
         tx_q <= tx_d;
      end
   end

endmodule
