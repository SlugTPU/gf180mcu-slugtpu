// SDRAM controller for mt48lc16m16a, with speed grade of 7e
module wb_sdr_mt48lc16m16a_7e #(
    parameter sys_clk_mhz_p = 67
   ,parameter dram_burst_p = 4
   // number of independent dram modules in parallel
   ,parameter parallel_p = 1
   // below should never be modified
   ,parameter _data_bits_p = 16
   ,parameter _rows_p = 8192
   ,parameter _cols_p = 512
   ,parameter _banks_p = 4
   ,parameter _sdr_addr_bits_p = 13
   ,parameter _usr_addr_bits_p = $clog2(_rows_p) +  $clog2(_banks_p) + $clog2(_cols_p) - $clog2(dram_burst_p)
) (
    input clk_i
   ,input rst_i

   ,input [_usr_addr_bits_p - 1:0]  m_adr_i
   ,input [_data_bits_p * parallel_p * dram_burst_p - 1:0]  m_dat_i // input to bus
   ,input m_we_i, m_stb_i, m_cyc_i
   // NOTE: not implemented!
   ,input [(_data_bits_p * dram_burst_p) / 8 - 1:0]  m_sel_i
   ,output logic m_ack_o
   ,output logic [_data_bits_p * parallel_p * dram_burst_p - 1:0]  m_dat_o // output to bus

   // for SDRAM
   ,input  [_data_bits_p * parallel_p - 1 :0] s_dq_i
   ,output logic [_data_bits_p * parallel_p - 1:0] s_dq_o
   ,output logic [_sdr_addr_bits_p - 1:0] s_addr_o
   ,output logic [1:0] s_ba_o
   ,output logic s_cke_o
   ,output logic s_cs_no
   ,output logic s_ras_no
   ,output logic s_cas_no
   ,output logic s_we_no
   ,output logic s_dqm_o

    // output enable signal to mux bidirectional pin
   ,output logic oe_o
);

   // ASSUMPTION: Assumes the worst path from IDLE nevere exceeds 64ms

   /** raw dram parameters from datasheet with unit conversions and maybe a bit of rounding */
   localparam tRP_us_lp = 0.015;
   localparam tRFC_us_lp = 0.066;
   // One distributed refresh every 7.8125us, which we round down to 7
   localparam tREF_dist_us_lp = 7;
   localparam tRCD_us_lp = 0.015;
   // localparam tWR_extra_us_lp = 0.007;
   localparam init_wait_us_lp = 100;
   // minimum clock period is 7ns, and 1/7.0ns ~= 142MHz
   localparam min_period_mhz_lp = 142;
   localparam tMRD_wait_cycles_lp = 2;

   /* parameterized parameters */
   // For 7E, the minimum clock period for CL=2 is 7.5ns, or 133MHz.
   // The minimum clock period for CL=3 is 7.0ns
   localparam CL_lp = (sys_clk_mhz_p <= 133) ? 2 : 3;
   // cycles needed to wait for 100us
   localparam init_wait_cycles_lp = sys_clk_mhz_p * init_wait_us_lp;
   localparam tRP_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tRP_us_lp));
   localparam tRFC_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tRFC_us_lp));
   localparam tREF_dist_wait_cycles = sys_clk_mhz_p * tREF_dist_us_lp;
   // localparam tWR_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tWR_us_lp));
   localparam tRCD_wait_cycles_lp = int'($ceil(sys_clk_mhz_p * tRCD_us_lp));
   localparam autorefresh_cycles_lp = sys_clk_mhz_p * tREF_dist_us_lp;

   localparam n_reg_lp = 3;

   typedef enum {
       INIT_WAIT,
       WAIT,
       INIT_NOP,
       INIT_PRECHARGE,
       INIT_REFRESH_1,
       INIT_REFRESH_2,
       INIT_LOAD_MODE,
       AUTO_REFRESH,
       ACTIVE,
       READ_BEGIN,
       READ_OUT,
       WRITE,
       PRECHARGE,
       IDLE
   } state_t;

   task automatic set_cmd_NOP ();
      s_cs_no = 1'b0;
      s_ras_no = 1'b1;
      s_cas_no = 1'b1;
      s_we_no  = 1'b1;
   endtask

   task automatic set_cmd_PRECHARGE_ALL ();
      s_cs_no = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b1;
      s_we_no  = 1'b0;
   endtask

   task automatic set_cmd_AUTO_REFRESH();
      s_cs_no  = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b0;
      s_we_no  = 1'b1;
   endtask

   task automatic set_cmd_LOAD_MODE_REGISTER();
      s_cs_no = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b0;
      s_we_no = 1'b0;
   endtask

   task automatic set_cmd_ACTIVE();
      s_cs_no = 1'b0;
      s_ras_no = 1'b0;
      s_cas_no = 1'b1;
      s_we_no = 1'b1;
      // do remember to set address for bank/row!
   endtask

   task automatic set_cmd_READ();
      s_cs_no = 1'b0;
      s_ras_no = 1'b1;
      s_cas_no = 1'b0;
      s_we_no = 1'b1;
   endtask

   task automatic set_cmd_WRITE();
      s_cs_no = 1'b0;
      s_ras_no = 1'b1;
      s_cas_no = 1'b0;
      s_we_no = 1'b0;
   endtask

   int wait_counter_d, wait_counter_q;
   logic [$clog2(autorefresh_cycles_lp) - 1:0] auto_refresh_counter_d, auto_refresh_counter_q;
   logic [$clog2(_rows_p) - 1:0]               auto_refreshes_needed_d, auto_refreshes_needed_q;
   logic dram_initialized;
   wire [_data_bits_p * parallel_p - 1:0] read_shifter_data[dram_burst_p - 1];
   // TODO: Possibly deprecate saved_state
   state_t state_d, state_q, saved_state_d, saved_state_q;
   // General purpose registers for managing sub-states in each state.
   //   (Beware of collisions in non-mutually exclusive states!)
   logic                                  r_d[n_reg_lp], r_q[n_reg_lp];
   // Single register for handling autorefresh

   // save previous bank/row address for handling precharging and activation
   logic                                        valid_prev_d, valid_prev_q;
   logic [$clog2(_banks_p) - 1:0]               prev_bank_d, prev_bank_q;
   logic [$clog2(_rows_p) - 1:0]                prev_row_addr_d, prev_row_addr_q;

   logic                                        shift_enable;

   wire [$clog2(_rows_p) - 1 : 0]  addr_row_w;
   wire [$clog2(_banks_p) - 1 : 0] addr_bank_w;
   wire [$clog2(_cols_p) - 1 : 0]  addr_col_w;

   task automatic reset_registers();
      for (int i = 0; i < n_reg_lp; i++) begin
         r_d[i] = 1'b0;
      end
   endtask

   function automatic is_valid_i();
      begin
         is_valid_i = m_cyc_i && m_stb_i;
      end
   endfunction

   function automatic is_same_bank();
      begin
         if (valid_prev_q) begin
            is_same_bank = (addr_bank_w ^ prev_bank_q) == '0;
         end else begin
            is_same_bank = '0;
         end
      end
   endfunction

   function automatic is_same_row();
      begin
         if (valid_prev_q) begin
            is_same_row = (addr_row_w ^ prev_row_addr_q) == '0;
         end else begin
            is_same_row = '0;
         end
      end
   endfunction

   function automatic state_t to_read_or_write();
      begin
         if (m_we_i) begin
            to_read_or_write = WRITE;
         end else begin
            to_read_or_write = READ_BEGIN;
         end
      end
   endfunction

   initial begin
      assert(sys_clk_mhz_p <= min_period_mhz_lp) else
          $error("sys_clk_mhz_p exceeds minimum clock period of 7ns!");

      assert(dram_burst_p == 1 || dram_burst_p == 2 || dram_burst_p == 4 || dram_burst_p == 8) else
        $error("Invalid burst length given. Valid burst lengths are 1, 2, 4, and 8");
   end

   assign oe_o = (state_q == WRITE) ? 1'b1 : 1'b0;

   assign addr_row_w  = m_adr_i[_usr_addr_bits_p - 1 -: $clog2(_rows_p)];
   assign addr_bank_w = m_adr_i[($clog2(_cols_p) - $clog2(dram_burst_p)) + $clog2(_banks_p) - 1 -: $clog2(_banks_p)];
   assign addr_col_w = m_adr_i[$clog2(_cols_p) - 1 :0];


   /** general state registers */
   generate
      for (genvar i = 0; i < n_reg_lp; i++) begin
         always_ff @(posedge clk_i) begin
            if (rst_i) begin
               r_q[i] <= '0;
            end else begin
               r_q[i] <= r_d[i];
            end
         end
      end
   endgenerate

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         valid_prev_q <= '0;
         prev_row_addr_q <= '0;
         prev_bank_q <= '0;
      end else begin
         valid_prev_q <= valid_prev_d;
         prev_row_addr_q <= prev_row_addr_d;
         prev_bank_q <= prev_bank_d;
      end
   end

   shift
     #(.width_p(_data_bits_p * parallel_p), .depth_p(dram_burst_p - 1))
   read_shifter (.clk_i(clk_i),
                  .reset_i(rst_i),
                  .data_i(s_dq_i),
                  .data_o(read_shifter_data),
                  .enable_i(shift_enable));

   assign m_dat_o[_data_bits_p * parallel_p * dram_burst_p - 1 -: _data_bits_p * parallel_p] = s_dq_i;
   generate
      for (genvar i = 0; i < dram_burst_p - 1; i++) begin
         assign m_dat_o[_data_bits_p * parallel_p * (dram_burst_p - 1 - i) - 1 -: _data_bits_p * parallel_p] = read_shifter_data[i];
      end
   endgenerate

   /** wait counter */
   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         // on reset, we go through the initialization process
         wait_counter_q <= init_wait_us_lp - 1'b1;
      end else begin
         wait_counter_q <= wait_counter_d;
      end
   end

   /** auto refresh counter after dram is initialized */
   always_comb begin
      if (!dram_initialized || state_q == AUTO_REFRESH || (auto_refresh_counter_q == 0)) begin
         // include the extra cycle needed to switch to AUTO_REFRESH state
         auto_refresh_counter_d = ($clog2(autorefresh_cycles_lp))'(autorefresh_cycles_lp - 1);
      end else begin
         auto_refresh_counter_d = auto_refresh_counter_q - 1'b1;
      end
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         auto_refresh_counter_q <= '0;
      end else begin
         auto_refresh_counter_q <= auto_refresh_counter_d;
      end
   end

   always_comb begin
      // should be at AUTO_REFRESH when counter hits 0
      if (dram_initialized && auto_refresh_counter_q == 2) begin
         auto_refreshes_needed_d = auto_refreshes_needed_q + 1;
      end else if (dram_initialized && state_q == AUTO_REFRESH && r_q[0] != 1) begin
         // at AUTO_REFRESH state, and is not just waiting
         auto_refreshes_needed_d = auto_refreshes_needed_q - 1;
      end else begin
         auto_refreshes_needed_d = auto_refreshes_needed_q;
      end
   end

   always_ff @(posedge clk_i)  begin
      if (rst_i) begin
         auto_refreshes_needed_q <= '0;
      end else begin
         auto_refreshes_needed_q <= auto_refreshes_needed_d;
      end
   end

   /** state machine */

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         saved_state_q <= INIT_WAIT;
      end else begin
         saved_state_q <= saved_state_d;
      end
   end

   always_comb begin
      wait_counter_d = '0;
      dram_initialized = 1'b0;
      saved_state_d = saved_state_q;
      shift_enable = 1'b0;
      prev_row_addr_d = prev_row_addr_q;
      prev_bank_d = prev_bank_q;
      valid_prev_d = valid_prev_q;
      m_ack_o = 1'b0;
      s_cke_o = 1;

      for (int i = 0; i < n_reg_lp; i++) begin
         r_d[i] = r_q[i];
      end

      case (state_q)
      /** The init process
       1. Apply power, assert CKE LOW, provide stable clock
       2. Wait 100μs minimum
       3. Bring CKE HIGH, issue a NOP command
       4. Issue PRECHARGE ALL (A10 HIGH)
       5. Wait tRP
       6. Issue two AUTO REFRESH commands, waiting tRFC between each
       7. Issue LOAD MODE REGISTER command with your desired configuration (burst length, CAS latency, burst type)
       8. Wait tMRD before any further commands
      **/
      INIT_WAIT: begin
         // special case of WAIT for init where cke is held low
         s_cke_o = 0;
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            state_d = state_q;
            wait_counter_d = wait_counter_q - 1'b1;
         end else begin
            state_d = INIT_NOP;
         end
      end
      WAIT: begin
         set_cmd_NOP();

         if (wait_counter_q > 0) begin
            state_d = state_q;
            wait_counter_d = wait_counter_q - 1'b1;
         end else begin
            state_d = saved_state_q;
         end
      end
      INIT_NOP: begin
         dram_initialized = 1'b0;
         set_cmd_NOP();

         state_d = INIT_PRECHARGE;
      end
      INIT_PRECHARGE: begin
         dram_initialized = 1'b0;
         set_cmd_PRECHARGE_ALL();
         // set A10 high for precharge all
         s_addr_o = 13'b0_0100_0000_0000;

         if (tRP_wait_cycles_lp - 1 > 0) begin
            state_d = WAIT;
            saved_state_d = INIT_REFRESH_1;
            wait_counter_d = tRP_wait_cycles_lp - 2;
         end else begin
            saved_state_d = INIT_REFRESH_1;
         end
      end
      INIT_REFRESH_1: begin
         dram_initialized = 1'b0;
         set_cmd_AUTO_REFRESH();

         if (tRFC_wait_cycles_lp - 1 > 0) begin
            state_d = WAIT;
            saved_state_d = INIT_REFRESH_2;
            wait_counter_d = tRFC_wait_cycles_lp - 2;
         end else begin
            state_d = INIT_REFRESH_2;
         end
      end
      INIT_REFRESH_2: begin
         dram_initialized = 1'b0;
         set_cmd_AUTO_REFRESH();

         if (tRFC_wait_cycles_lp - 1 > 0) begin
            state_d = WAIT;
            saved_state_d = INIT_LOAD_MODE;
            wait_counter_d = tRFC_wait_cycles_lp - 2;
         end else begin
            state_d = INIT_LOAD_MODE;
         end
      end
      INIT_LOAD_MODE: begin
         dram_initialized = 1'b0;
         set_cmd_LOAD_MODE_REGISTER();
         /**
          A[12:10] = Reserved
          A[9]     = Write Burst Mode {}
          A[8:7]   = Op Mode { 0 = Standard Operation }
          A[6:4]   = CAS Latency
          A[3]     = Burst Type { 0 = Sequential }
          A[2:0]   = Burst Length
          **/
         s_addr_o = {1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, CL_lp[2:0], 1'b0, dram_burst_p[2:0]};

         if (tMRD_wait_cycles_lp - 1'b1 > 0) begin
            state_d = WAIT;
            saved_state_d = IDLE;
            wait_counter_d = tMRD_wait_cycles_lp - 1'b1;
         end else begin
         end
      end
      IDLE: begin
         dram_initialized = 1'b1;

         set_cmd_NOP();

         reset_registers();

         if (auto_refreshes_needed_q > 0) begin
            // need to precharge if not already
            if (valid_prev_q) begin
               r_d[0] = 1'b1;
               state_d = PRECHARGE;
            end else begin
               state_d = AUTO_REFRESH;
            end
         end else if (!m_cyc_i || !m_stb_i) begin
            state_d = state_q;
         end else if (!valid_prev_q || is_same_bank()) begin
            state_d = ACTIVE;
         end else if (!is_same_row()) begin
            state_d = PRECHARGE;
         end else begin
            state_d = state_t'((m_we_i) ? WRITE : READ_BEGIN);
         end
      end
      ACTIVE: begin
         /** Register map:
          r[0]: Sent ACTIVE command
          **/
         dram_initialized = 1'b1;

         if (!r_q[0]) begin
            set_cmd_ACTIVE();
            // TODO:
            wait_counter_d = tRCD_wait_cycles_lp - 2;
            valid_prev_d = 1'b1;
            r_d[0] = 1'b1;
         end else begin
            set_cmd_NOP();
            if (wait_counter_q > 0) begin
               state_d = state_q;
               wait_counter_d = wait_counter_q - 1;
            end else begin
               reset_registers();
               state_d = to_read_or_write();
            end
         end
      end
      READ_BEGIN: begin // issue read command
         /** Register map:
          r[0]: Sent READ command AND should wait
          **/
         dram_initialized = 1'b1;

         set_cmd_READ();
         // disable auto-precharge; A[10] = 0
         s_addr_o = _sdr_addr_bits_p'({ '0, 1'b0, addr_col_w });
         s_ba_o = addr_bank_w;

         if (CL_lp - 1 == 0) begin
            reset_registers();
            state_d = READ_OUT;
         end else if (!r_q[0]) begin
            r_d[0] = 1'b1;
            wait_counter_d = CL_lp - 2;
         end else if (wait_counter_q - 1 > 0) begin
            state_d = state_q;
            wait_counter_d = wait_counter_q - 1;
         end else begin
            reset_registers();
            state_d = READ_OUT;
         end
      end
      READ_OUT: begin
         /** Register map:
          r[0]: Need to wait AND has began waiting
          **/
         dram_initialized = 1'b1;
         // TODO: Autorefresh counter assert needs fixing

         set_cmd_NOP();

         shift_enable = 1'b1;

         if (dram_burst_p - 1 == 0) begin
            m_ack_o = 1'b1;

            reset_registers();
            state_d = IDLE;
         end else if (!r_q[0]) begin
            r_d[0] = 1'b1;
            wait_counter_d = dram_burst_p - 1;

            state_d = state_q;
         end else if (wait_counter_q > 0) begin
            state_d = state_q;
            wait_counter_d = wait_counter_q - 1;
         end else begin
            m_ack_o = 1'b1;
            shift_enable = 1'b0;

            reset_registers();
            state_d = IDLE;
         end
      end
      WRITE: begin
         /** Register map:
          r[0]: Sent WRITE command already AND need to wait
          **/
         dram_initialized = 1'b1;

         // NOTE: If precharge is needed, tWR is implicitly met
         //       from state transition from to IDLE->PRECHARGE = 2 cycles.
         //       We never need > 2 cycles because the minimum period is 7ns

         if (!r_q[0]) begin
            set_cmd_WRITE();
            s_addr_o = _sdr_addr_bits_p'({ '0, 1'b0, addr_col_w });
            s_ba_o = addr_bank_w;

            if (dram_burst_p - 1 > 0) begin
               r_d[0] = 1'b1;
               wait_counter_d = dram_burst_p - 2;
               state_d = state_q;
            end else begin
               reset_registers();
               state_d = IDLE;
            end
         end else begin
            set_cmd_NOP();

            if (wait_counter_q > 0) begin
               wait_counter_d = wait_counter_q - 1;
               state_d = state_q;
            end else begin
               reset_registers();
               state_d = IDLE;
            end
         end
      end
      AUTO_REFRESH: begin
         /** Register map:
          r[0]: Auto refresh command sent?
          **/

         dram_initialized = 1'b1;

         if (!r_q[0]) begin
            set_cmd_AUTO_REFRESH();

            r_d[0] = 1'b1;

            if (tRFC_wait_cycles_lp - 1 > 0) begin
               wait_counter_d = tRFC_wait_cycles_lp - 2;
               state_d = state_q;
            end else begin // skip wait if period implicitly meetings tRFC
               reset_registers();
               state_d = state_t'((auto_refreshes_needed_q > 0) ? AUTO_REFRESH : IDLE);
            end
         end else begin
            set_cmd_NOP();

            if (wait_counter_q > 0) begin
               wait_counter_d = wait_counter_q - 1'b1;
               state_d = state_q;
            end else begin
               reset_registers();
               state_d = state_t'((auto_refreshes_needed_q > 0) ? AUTO_REFRESH : IDLE);
            end
         end
      end
      PRECHARGE: begin
         /** Register map:
          r[0]: ARG: Go to auto refresh next?
          r[1]: Sent PRECHARGE command
          **/

         // on precharge, we can no longer "cache" previous row and bank
         valid_prev_d = '0;

         // Always precharges ALL banks
         if (!r_q[1]) begin
            dram_initialized = 1'b0;
            set_cmd_PRECHARGE_ALL();
            // set A10 high for precharge all
            s_addr_o = 13'b0_0100_0000_0000;
            r_d[1] = 1'b1;

            if (tRP_wait_cycles_lp - 1 > 0) begin
               // TODO: Fix other one offs
               wait_counter_d = tRP_wait_cycles_lp - 2;
               state_d = state_q;
            end else begin
               reset_registers();

               if (r_q[0]) begin
                  // r_d[0] = 1'b1;
                  state_d = AUTO_REFRESH;
               end else begin
                  state_d = IDLE;
               end
            end

         end else begin
            if (wait_counter_q > 0) begin
               wait_counter_d = wait_counter_q - 1;

               state_d = state_q;
            end else begin
               reset_registers();

               if (r_q[0]) begin
                  // r_d[0] = 1'b1;
                  state_d = AUTO_REFRESH;
               end else begin
                  state_d = IDLE;
               end
            end
         end
      end
      default: begin
         dram_initialized = 1'b0;

         set_cmd_NOP();
         state_d = state_q;
      end
      endcase
   end

   always_ff @(posedge clk_i) begin
      if (rst_i) begin
         state_q <= INIT_WAIT;
      end else begin
         state_q <= state_d;
      end
   end

   // TODO: implement parallel dram modules
   // TODO: maybe move wait counter logic outside of state machine logic?

endmodule
