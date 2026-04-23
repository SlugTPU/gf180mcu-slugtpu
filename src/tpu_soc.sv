module tpu_soc #(
    parameter sys_clk_mhz_p   = 67
   ,parameter sdr_data_bits_p = 16
   ,parameter sdr_addr_bits_p = 13
   ,parameter dram_burst_p    = 4
) (
    input clk_i
   ,input rst_i

   // SPI slave interface
   ,input  spi_clk_i
   ,input  spi_cs_ni
   ,input  spi_mosi_i
   ,output spi_miso_o
   ,output spi_oe_o

   // SDRAM
   ,input  [sdr_data_bits_p - 1 :0] sdr_dq_i
   ,output logic [sdr_data_bits_p - 1:0] sdr_dq_o
   ,output logic sdr_dq_oe_o
   ,output logic [sdr_addr_bits_p - 1:0] sdr_addr_o
   ,output logic [1:0] sdr_ba_o
   ,output logic sdr_cke_o
   ,output logic sdr_cs_no
   ,output logic sdr_ras_no
   ,output logic sdr_cas_no
   ,output logic sdr_we_no
   ,output logic [1:0] sdr_dqm_o
);
   localparam _rows_lp = 8192;
   localparam _cols_lp = 512;
   localparam _banks_lp = 4;
   localparam dma_data_w_lp  = sdr_data_bits_p * dram_burst_p;
   localparam dma_addr_w_lp  = $clog2(_rows_lp) + $clog2(_banks_lp) + $clog2(_cols_lp) - $clog2(dram_burst_p);

   // -----------------------------------------------------------------------
   // WishBone bus wires
   // -----------------------------------------------------------------------

   // DMA master → mux m1
   wire [dma_addr_w_lp - 1:0]   dma_wb_adr;
   wire [dma_data_w_lp - 1:0]   dma_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   dma_wb_dat_r;
   wire                          dma_wb_we;
   wire                          dma_wb_stb;
   wire                          dma_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] dma_wb_sel;
   wire                          dma_wb_ack;

   // SPI bridge → mux m0 (TODO: connect to SPI-to-WB bridge)
   wire [dma_addr_w_lp - 1:0]   spi_wb_adr;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_r;
   wire                          spi_wb_we;
   wire                          spi_wb_stb;
   wire                          spi_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] spi_wb_sel;
   wire                          spi_wb_ack;

   assign spi_wb_adr   = '0;
   assign spi_wb_dat_w = '0;
   assign spi_wb_we    = '0;
   assign spi_wb_stb   = '0;
   assign spi_wb_cyc   = '0;
   assign spi_wb_sel   = '0;

   // mux slave → SDRAM controller
   wire [dma_addr_w_lp - 1:0]   mux_s_adr;
   wire [dma_data_w_lp - 1:0]   mux_s_dat_w;
   wire [dma_data_w_lp - 1:0]   mux_s_dat_r;
   wire                          mux_s_we;
   wire                          mux_s_stb;
   wire                          mux_s_cyc;
   wire [dma_data_w_lp/8 - 1:0] mux_s_sel;
   wire                          mux_s_ack;

   // TODO: connect to control register
   wire tpu_active;

   // -----------------------------------------------------------------------
   // DMA control (TODO: connect to instruction decoder)
   // -----------------------------------------------------------------------
   wire [dma_addr_w_lp - 1:0] dma_start_addr;
   wire [15:0]                 dma_word_count;
   wire                        dma_we;
   wire                        dma_start;
   wire                        dma_busy;
   wire                        dma_done;


   // -----------------------------------------------------------------------
   // DMA stream (TODO: connect to systolic-array SRAM)
   // -----------------------------------------------------------------------
   wire [dma_data_w_lp - 1:0] dma_rd_data;
   wire                        dma_rd_valid;
   wire                        dma_rd_ready;
   wire [dma_data_w_lp - 1:0] dma_wr_data;
   wire                        dma_wr_valid;
   wire                        dma_wr_ready;

   // -----------------------------------------------------------------------
   // SPI register bus (TODO: connect to instruction decoder)
   // -----------------------------------------------------------------------

   // raw async outputs from housekeeping_spi (SCK domain)
   wire [15:0] spi_oaddr_async;
   wire [7:0]  spi_odata_async;
   wire [7:0]  spi_idata;
   wire        spi_rdstb_async;
   wire        spi_wrstb_async;

   // assign spi_idata = '0;

   // 2-FF synchronizers for strobes (SCK → clk_i)
   // _meta: first stage, may be metastable; _sync: second stage, resolved
   logic spi_rdstb_meta, spi_rdstb_sync;
   logic spi_wrstb_meta, spi_wrstb_sync;
   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         spi_rdstb_meta <= '0;  spi_rdstb_sync <= '0;
         spi_wrstb_meta <= '0;  spi_wrstb_sync <= '0;
      end else begin
         spi_rdstb_meta <= spi_rdstb_async;  spi_rdstb_sync <= spi_rdstb_meta;
         spi_wrstb_meta <= spi_wrstb_async;  spi_wrstb_sync <= spi_wrstb_meta;
      end
   end

   // latch addr/data into clk_i domain while either strobe is active;
   // safe because SCK << clk_i so oaddr/odata are stable before the strobe arrives
   logic [15:0] spi_oaddr_s;
   logic [7:0]  spi_odata_s;

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         spi_oaddr_s <= '0;
         spi_odata_s <= '0;
      end else if (spi_rdstb_sync | spi_wrstb_sync) begin
         spi_oaddr_s <= spi_oaddr_async;
         spi_odata_s <= spi_odata_async;
      end
   end

   housekeeping_spi spi (
      .reset               (rst_i),
      .SCK                 (spi_clk_i),
      .SDI                 (spi_mosi_i),
      .SDO                 (spi_miso_o),
      .CSB                 (spi_cs_ni),
      .sdoenb              (spi_oe_o),
      .idata               (spi_idata),
      .odata               (spi_odata_async),
      .oaddr               (spi_oaddr_async),
      .rdstb               (spi_rdstb_async),
      .wrstb               (spi_wrstb_async),
      .pass_thru_mgmt      (),
      .pass_thru_mgmt_delay(),
      .pass_thru_user      (),
      .pass_thru_user_delay(),
      .pass_thru_mgmt_reset(),
      .pass_thru_user_reset()
   );
   // TODO:
   instruction_decoder ();

   wb_dma_master #(
       .AddrW (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) dma (
       .clk_i        (clk_i),
       .rst_i        (rst_i),
       // control
       .start_addr_i (dma_start_addr),
       .word_count_i (dma_word_count),
       .we_i         (dma_we),
       .start_i      (dma_start),
       .busy_o       (dma_busy),
       .done_o       (dma_done),
       // stream
       .rd_data_o    (dma_rd_data),
       .rd_valid_o   (dma_rd_valid),
       .rd_ready_i   (dma_rd_ready),
       .wr_data_i    (dma_wr_data),
       .wr_valid_i   (dma_wr_valid),
       .wr_ready_o   (dma_wr_ready),
       // WB master
       .wb_adr_o     (dma_wb_adr),
       .wb_dat_o     (dma_wb_dat_w),
       .wb_dat_i     (dma_wb_dat_r),
       .wb_we_o      (dma_wb_we),
       .wb_stb_o     (dma_wb_stb),
       .wb_cyc_o     (dma_wb_cyc),
       .wb_sel_o     (dma_wb_sel),
       .wb_ack_i     (dma_wb_ack)
   );

   wb_mux_2to1 #(
       .AdrW  (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) wb_mux (
       .tpu_active  (tpu_active),
       // m0: SPI bridge
       .m0_adr      (spi_wb_adr),
       .m0_dat_w    (spi_wb_dat_w),
       .m0_dat_r    (spi_wb_dat_r),
       .m0_we       (spi_wb_we),
       .m0_stb      (spi_wb_stb),
       .m0_cyc      (spi_wb_cyc),
       .m0_sel      (spi_wb_sel),
       .m0_ack      (spi_wb_ack),
       // m1: DMA master
       .m1_adr      (dma_wb_adr),
       .m1_dat_w    (dma_wb_dat_w),
       .m1_dat_r    (dma_wb_dat_r),
       .m1_we       (dma_wb_we),
       .m1_stb      (dma_wb_stb),
       .m1_cyc      (dma_wb_cyc),
       .m1_sel      (dma_wb_sel),
       .m1_ack      (dma_wb_ack),
       // slave: SDRAM
       .s_adr       (mux_s_adr),
       .s_dat_w     (mux_s_dat_w),
       .s_dat_r     (mux_s_dat_r),
       .s_we        (mux_s_we),
       .s_stb       (mux_s_stb),
       .s_cyc       (mux_s_cyc),
       .s_sel       (mux_s_sel),
       .s_ack       (mux_s_ack)
   );

   wb_sdr_mt48lc16m16a_7e #(
       .sys_clk_mhz_p   (sys_clk_mhz_p),
       .dram_burst_p     (dram_burst_p)
   ) sdram_ctrl (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       // WB slave
       .m_adr_i  (mux_s_adr),
       .m_dat_i  (mux_s_dat_w),
       .m_we_i   (mux_s_we),
       .m_stb_i  (mux_s_stb),
       .m_cyc_i  (mux_s_cyc),
       .m_sel_i  (mux_s_sel),
       .m_ack_o  (mux_s_ack),
       .m_dat_o  (mux_s_dat_r),
       // SDRAM pins
       .s_dq_i   (sdr_dq_i),
       .s_dq_o   (sdr_dq_o),
       .s_addr_o (sdr_addr_o),
       .s_ba_o   (sdr_ba_o),
       .s_cke_o  (sdr_cke_o),
       .s_cs_no  (sdr_cs_no),
       .s_ras_no (sdr_ras_no),
       .s_cas_no (sdr_cas_no),
       .s_we_no  (sdr_we_no),
       .s_dqm_o  (sdr_dqm_o),
       .oe_o     (sdr_dq_oe_o)
   );

   // wire up with systolic array
endmodule
