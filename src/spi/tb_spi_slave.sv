module tb_spi_slave (
    input  clk_i,
    input rst_i,

    input  spi_sclk_i,
    input  spi_mosi_i,
    input  spi_cs_ni,
    output spi_miso_o
);

   logic [7:0] test_byte_0;
   logic [7:0] test_byte_1;
   logic [7:0] test_byte_2;
   logic [7:0] test_byte_3;

   logic       active, byte_stb;

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         test_byte_1 <= '0;
         test_byte_2 <= '0;
         test_byte_3 <= '0;
      end else if (byte_stb) begin
         test_byte_1 <= test_byte_0;
         test_byte_2 <= test_byte_1;
         test_byte_3 <= test_byte_2;
      end
   end

   spi_slave spi_slave_inst
     (.clk_i(clk_i), .rst_i(rst_i),
      .spi_sck_async_i(spi_sclk_i),
      .spi_cs_async_ni(spi_cs_ni),
      .spi_mosi_async_i(spi_mosi_i),
      .spi_miso_o(spi_miso_o),
      .active_o(active),
      .byte_stb_o(byte_stb),
      .byte_tx_i(test_byte_3),
      .byte_rx_o(test_byte_0));

endmodule
