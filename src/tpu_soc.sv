module tpu_soc #(
    parameter sys_clk_mhz_p   = 67
   ,parameter sdr_data_bits_p = 16
   ,parameter sdr_addr_bits_p = 13
   ,parameter dram_burst_p    = 4
   ,parameter dbg_n_words_p = 32
   ,parameter test_ram_words_p = 32
) (
`ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
`endif
    input clk_i
   ,input rst_i

   // SPI slave interface (spibone — direct SDRAM + TPU register access)
   ,input  spi_clk_i
   ,input  spi_cs_ni
   ,input  spi_mosi_i
   ,output spi_miso_o

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
   ,output logic test_mode_o 
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

   // SPI bridge → mux m0
   wire [dma_addr_w_lp - 1:0]   spi_wb_adr;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_w;
   wire [dma_data_w_lp - 1:0]   spi_wb_dat_r;
   wire                          spi_wb_we;
   wire                          spi_wb_stb;
   wire                          spi_wb_cyc;
   wire [dma_data_w_lp/8 - 1:0] spi_wb_sel;
   wire                          spi_wb_ack;

   // mux slave → SDRAM controller
   wire [dma_addr_w_lp - 1:0]   mux_s_adr;
   wire [dma_data_w_lp - 1:0]   mux_s_dat_w;
   wire [dma_data_w_lp - 1:0]   mux_s_dat_r;
   wire                          mux_s_we;
   wire                          mux_s_stb;
   wire                          mux_s_cyc;
   wire [dma_data_w_lp/8 - 1:0] mux_s_sel;
   wire                          mux_s_ack;

   // test-mode demux outputs.
   //   sdr_s_* is for SDRAM controller (slave 0, active when test_mode=0)
   //   tram_s_* is for wb_test_ram (slave 1, active when test_mode=1)
   wire [dma_addr_w_lp - 1:0]   sdr_s_adr,  tram_s_adr;
   wire [dma_data_w_lp - 1:0]   sdr_s_dat_w, tram_s_dat_w;
   wire [dma_data_w_lp - 1:0]   sdr_s_dat_r, tram_s_dat_r;
   wire                          sdr_s_we,    tram_s_we;
   wire                          sdr_s_stb,   tram_s_stb;
   wire                          sdr_s_cyc,   tram_s_cyc;
   wire [dma_data_w_lp/8 - 1:0] sdr_s_sel,   tram_s_sel;
   wire                          sdr_s_ack,   tram_s_ack;

   // CTRL[1] from tpu_regs
   wire test_mode;
   assign test_mode_o = test_mode;


   // -----------------------------------------------------------------------
   // DMA control (TODO: connect to instruction decoder)
   // -----------------------------------------------------------------------
   wire [dma_addr_w_lp - 1:0]  dma_start_addr;
   wire                        dma_start;
   wire [15:0]                 dma_word_count;
   wire                        dma_we;
   wire                        dma_busy;
   wire                        dma_done;

   // tpu_enable: software-controlled register bit; 1 = DMA owns the bus, 0 = SPI owns it
   wire tpu_enable;
//    wire tpu_active = tpu_enable;
   wire tpu_active;

   // Rising edge of tpu_enable generates the one-cycle DMA start pulse
   logic tpu_enable_r;
   always_ff @(posedge clk_i)
       if (rst_i) tpu_enable_r <= '0;
       else        tpu_enable_r <= tpu_enable;
   // wire dma_start = tpu_enable & ~tpu_enable_r;

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
   // spibone → wb_decoder wires (32-bit byte addr, dma_data_w_lp-bit data)
   // -----------------------------------------------------------------------
   wire [31:0]               dec_adr;
   wire [dma_data_w_lp-1:0] dec_dat_w;
   wire [dma_data_w_lp-1:0] dec_dat_r;
   wire                      dec_we, dec_stb, dec_cyc, dec_ack;

   // wb_decoder port 0 (DRAM) wires — byte addr converted to word addr below
   wire [31:0]               dec_dram_adr;
   wire [dma_data_w_lp-1:0] dec_dram_dat_w;
   wire [dma_data_w_lp-1:0] dec_dram_dat_r;
   wire                      dec_dram_we, dec_dram_stb, dec_dram_cyc, dec_dram_ack;

   // 32-bit byte address → dma_addr_w_lp word address for wb_mux m0
   assign spi_wb_adr   = dec_dram_adr[dma_addr_w_lp + 2 : 3];
   assign spi_wb_dat_w = dec_dram_dat_w;
   assign spi_wb_we    = dec_dram_we;
   assign spi_wb_stb   = dec_dram_stb;
   assign spi_wb_cyc   = dec_dram_cyc;
   assign spi_wb_sel   = '1;
   assign dec_dram_dat_r = spi_wb_dat_r;
   assign dec_dram_ack   = spi_wb_ack;

   // wb_decoder port 1 (TPU regs) wires
   wire [31:0]               dec_tpu_adr_wide;
   wire [dma_data_w_lp-1:0] dec_tpu_dat_w_wide;
   wire [dma_data_w_lp-1:0] dec_tpu_dat_r_wide;
   wire                      dec_tpu_we, dec_tpu_stb, dec_tpu_cyc, dec_tpu_ack;

   // tpu_regs bus (32-bit data)
   wire [31:0] tpureg_adr;
   wire [31:0] tpureg_dat_w;
   wire [31:0] tpureg_dat_r;
   wire [1:0]  tpureg_status;
   wire [31:0]  tpureg_pc_addr;
   wire        tpureg_we, tpureg_stb, tpureg_cyc, tpureg_ack, tpureg_pc_stb;
   assign tpureg_adr   = dec_tpu_adr_wide;
   assign tpureg_dat_w = dec_tpu_dat_w_wide[31:0];
   assign tpureg_we    = dec_tpu_we;
   assign tpureg_stb   = dec_tpu_stb;
   assign tpureg_cyc   = dec_tpu_cyc;
   assign dec_tpu_dat_r_wide = {{(dma_data_w_lp-32){1'b0}}, tpureg_dat_r};
   assign dec_tpu_ack        = tpureg_ack;
   assign tpu_active = tpureg_status[1]; 

   // tpu_regs outputs (32-bit addresses, truncated to DMA widths)
   wire [31:0] dma_start_addr_wide;
   wire [31:0] dma_word_count_wide;
   wire [31:0] tpureg_pc_addrwide;
   // assign dma_start_addr = dma_start_addr_wide[dma_addr_w_lp-1:0];
   // assign dma_word_count = dma_word_count_wide[15:0];

   // -----------------------------------------------------------------------
   // spibone_wb: SPI slave → Wishbone master
   // -----------------------------------------------------------------------
   spibone_wb #(
       .AdrW  (32),
       .DataW (dma_data_w_lp)
   ) spi_wb_inst (
       .clk_i       (clk_i),
       .rst_i       (rst_i),
       .spi_sck_i   (spi_clk_i),
       .spi_mosi_i  (spi_mosi_i),
       .spi_miso_o  (spi_miso_o),
       .spi_cs_n_i  (spi_cs_ni),
       .wb_adr_o    (dec_adr),
       .wb_dat_o    (dec_dat_w),
       .wb_dat_i    (dec_dat_r),
       .wb_we_o     (dec_we),
       .wb_stb_o    (dec_stb),
       .wb_cyc_o    (dec_cyc),
       .wb_sel_o    (/* full-word, unused here */),
       .wb_ack_i    (dec_ack)
   );

   // -----------------------------------------------------------------------
   // wb_decoder: route spibone transactions to DRAM or TPU registers
   // -----------------------------------------------------------------------
   wb_decoder #(
       .DataW (dma_data_w_lp)
   ) spi_wb_decoder (
       .clk_i      (clk_i),
       .rst_i      (rst_i),
       .wbs_adr_i  (dec_adr),
       .wbs_dat_i  (dec_dat_w),
       .wbs_dat_o  (dec_dat_r),
       .wbs_we_i   (dec_we),
       .wbs_stb_i  (dec_stb),
       .wbs_cyc_i  (dec_cyc),
       .wbs_ack_o  (dec_ack),
       // port 0: DRAM (byte addr + 64-bit data)
       .wbm0_adr_o (dec_dram_adr),
       .wbm0_dat_o (dec_dram_dat_w),
       .wbm0_dat_i (dec_dram_dat_r),
       .wbm0_we_o  (dec_dram_we),
       .wbm0_stb_o (dec_dram_stb),
       .wbm0_cyc_o (dec_dram_cyc),
       .wbm0_ack_i (dec_dram_ack),
       // port 1: TPU registers (32-bit, zero-extended to DataW)
       .wbm1_adr_o (dec_tpu_adr_wide),
       .wbm1_dat_o (dec_tpu_dat_w_wide),
       .wbm1_dat_i (dec_tpu_dat_r_wide),
       .wbm1_we_o  (dec_tpu_we),
       .wbm1_stb_o (dec_tpu_stb),
       .wbm1_cyc_o (dec_tpu_cyc),
       .wbm1_ack_i (dec_tpu_ack)
   );

   // -----------------------------------------------------------------------
   // tpu_regs: TPU control/status registers via spibone
   // -----------------------------------------------------------------------
   tpu_regs #(
        .DBG_N_WORDS (dbg_n_words_p)
   ) regs (
       .clk_i             (clk_i),
       .rst_i             (rst_i),
       .wb_adr_i          (tpureg_adr),
       .wb_dat_i          (tpureg_dat_w),
       .wb_dat_o          (tpureg_dat_r),
       .wb_we_i           (tpureg_we),
       .wb_stb_i          (tpureg_stb),
       .wb_cyc_i          (tpureg_cyc),
       .wb_ack_o          (tpureg_ack),
       .tpu_enable_o      (tpu_enable),
       .test_mode_o       (test_mode),
       .tpu_pc_addr_o     (tpureg_pc_addr),
       .tpu_pc_stb_o      (tpureg_pc_stb),
       .tpu_state_i       (tpureg_status),
       .dbg_word_i        (dbg_word)
   );

   // -----------------------------------------------------------------------
   // wb_dma_master: TPU's DMA master (mux m1)
   // -----------------------------------------------------------------------
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

   // -----------------------------------------------------------------------
   // wb_mux_2to1: arbitrate between SPI bridge (m0) and DMA master (m1)
   // -----------------------------------------------------------------------
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

   // -----------------------------------------------------------------------
   // wb_demux_1to2: route mux output to either the SDRAM controller (s0) or the on-chip test RAM (s1), selected by tpu_regs.CTRL[1] = test_mode.
   // -----------------------------------------------------------------------
   wb_demux_1to2 #(
       .AdrW  (dma_addr_w_lp),
       .DataW (dma_data_w_lp)
   ) test_demux (
       .sel_i    (test_mode),
       // master: output of wb_mux_2to1
       .m_adr    (mux_s_adr),
       .m_dat_w  (mux_s_dat_w),
       .m_dat_r  (mux_s_dat_r),
       .m_sel    (mux_s_sel),
       .m_we     (mux_s_we),
       .m_stb    (mux_s_stb),
       .m_cyc    (mux_s_cyc),
       .m_ack    (mux_s_ack),
       // s0: SDRAM controller
       .s0_adr   (sdr_s_adr),
       .s0_dat_w (sdr_s_dat_w),
       .s0_dat_r (sdr_s_dat_r),
       .s0_sel   (sdr_s_sel),
       .s0_we    (sdr_s_we),
       .s0_stb   (sdr_s_stb),
       .s0_cyc   (sdr_s_cyc),
       .s0_ack   (sdr_s_ack),
       // s1: test RAM
       .s1_adr   (tram_s_adr),
       .s1_dat_w (tram_s_dat_w),
       .s1_dat_r (tram_s_dat_r),
       .s1_sel   (tram_s_sel),
       .s1_we    (tram_s_we),
       .s1_stb   (tram_s_stb),
       .s1_cyc   (tram_s_cyc),
       .s1_ack   (tram_s_ack)
   );


   // -----------------------------------------------------------------------
   // SDRAM controller
   // -----------------------------------------------------------------------
   wb_sdr_mt48lc16m16a_7e #(
       .sys_clk_mhz_p   (sys_clk_mhz_p),
       .dram_burst_p     (dram_burst_p)
   ) sdram_ctrl (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       // WB slave
       .m_adr_i  (sdr_s_adr),
       .m_dat_i  (sdr_s_dat_w),
       .m_we_i   (sdr_s_we),
       .m_stb_i  (sdr_s_stb),
       .m_cyc_i  (sdr_s_cyc),
       .m_sel_i  (sdr_s_sel),
       .m_ack_o  (sdr_s_ack),
       .m_dat_o  (sdr_s_dat_r),
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
 
   // -----------------------------------------------------------------------
   // wb_test_ram: small on-chip stand-in for SDRAM
   // -----------------------------------------------------------------------
   wb_test_ram #(
       .N_WORDS (test_ram_words_p),
       .AdrW    (dma_addr_w_lp),
       .DataW   (dma_data_w_lp)
   ) test_ram (
       .clk_i    (clk_i),
       .rst_i    (rst_i),
       .wb_adr_i (tram_s_adr),
       .wb_dat_i (tram_s_dat_w),
       .wb_dat_o (tram_s_dat_r),
       .wb_sel_i (tram_s_sel),
       .wb_we_i  (tram_s_we),
       .wb_stb_i (tram_s_stb),
       .wb_cyc_i (tram_s_cyc),
       .wb_ack_o (tram_s_ack)
   );


   logic internal_error_l; // prev unconnected; now wired into the debug bus below

  control_top #(
     .EXTERNAL_DRAM_ADDR_WIDTH(dma_addr_w_lp),
     .DRAM_DATA_WIDTH(dma_data_w_lp)
  )
  control (
`ifdef USE_POWER_PINS
      .VDD(VDD),
      .VSS(VSS),
`endif
      .clk_i    (clk_i),
      .rst_i    (rst_i),

      .dram_start_addr_o(dma_start_addr),
      .dram_word_count_o(dma_word_count),
      .dram_we_o(dma_we),
      .dram_start_o(dma_start),
      .dma_busy_i(dma_busy),
      .dma_done_i(dma_done),

      .dram2sram_valid_i(dma_rd_valid),
      .dram2sram_data_i(dma_rd_data),
      .dram2sram_ready_o(dma_rd_ready),

      .sram2dram_valid_o(dma_wr_valid),
      .sram2dram_data_o(dma_wr_data),
      .sram2dram_ready_i(dma_wr_ready),

      .pc_in(tpureg_pc_addr[dma_addr_w_lp - 1:0]),
      .pc_valid_i(tpureg_pc_stb),
      .pc_ready_o(),

      .tpu_state_o(tpureg_status),
      .INTERNAL_ERROR_O(internal_error_l)
  );

  //debug observation byte map
  // driven into tpu_regs and read by host via REG_DBG_ADDR/DBG_DATA over SPI
  logic [dbg_n_words_p*8 - 1:0] dbg_word;

  always_comb begin
        dbg_word = '0;
        dbg_word[ 8'h00*8 +: 8] = {3'b0, internal_error_l, dma_done, dma_busy,
                                     tpureg_status};
        dbg_word[ 8'h01*8 +: 8] = {3'b0, tpu_enable_r, dma_start, dma_we,
                                     dma_rd_valid, dma_wr_valid};
        dbg_word[ 8'h02*8 +: 8] = {7'b0, tpureg_pc_stb};
        dbg_word[ 8'h03*8 +: 8] = {7'b0, test_mode};

        dbg_word[ 8'h10*8 +: 8] = tpureg_pc_addr[ 7: 0];
        dbg_word[ 8'h11*8 +: 8] = tpureg_pc_addr[15: 8];
        dbg_word[ 8'h12*8 +: 8] = tpureg_pc_addr[23:16];
        dbg_word[ 8'h13*8 +: 8] = tpureg_pc_addr[31:24];
  
        dbg_word[ 8'h14*8 +: 8] = dma_word_count[7:0];
 
        dbg_word[ 8'h1C*8 +: 8] = 8'h53; // 'S'
        dbg_word[ 8'h1D*8 +: 8] = 8'h4C; // 'L'
        dbg_word[ 8'h1E*8 +: 8] = 8'h55; // 'U'
        dbg_word[ 8'h1F*8 +: 8] = 8'h47; // 'G'
    end

endmodule
