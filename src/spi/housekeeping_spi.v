// SPDX-FileCopyrightText: 2020 Efabless Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

`define COMMAND  3'b000
`define ADDRESS  3'b001
`define DATA     3'b010
`define ADDRESS2 3'b011
`define USERPASS 3'b100
`define MGMTPASS 3'b101

module housekeeping_spi(reset, SCK, SDI, CSB, SDO,
	sdoenb, idata, odata, oaddr, rdstb, wrstb,
	pass_thru_mgmt, pass_thru_mgmt_delay,
	pass_thru_user, pass_thru_user_delay,
	pass_thru_mgmt_reset, pass_thru_user_reset);

    input  reset;
    input  SCK;
    input  SDI;
    input  CSB;
    output SDO;
    output sdoenb;
    input  [7:0]  idata;
    output [7:0]  odata;
    output [15:0] oaddr;
    output rdstb;
    output wrstb;
    output pass_thru_mgmt;
    output pass_thru_mgmt_delay;
    output pass_thru_user;
    output pass_thru_user_delay;
    output pass_thru_mgmt_reset;
    output pass_thru_user_reset;

    reg  [15:0] addr;
    reg         wrstb;
    reg         rdstb;
    reg         sdoenb;
    reg  [2:0]  state;
    reg  [2:0]  count;
    reg         writemode;
    reg         readmode;
    reg  [2:0]  fixed;
    wire [7:0]  odata;
    reg  [6:0]  predata;
    wire [15:0] oaddr;
    reg  [7:0]  ldata;
    reg         pass_thru_mgmt;
    reg         pass_thru_mgmt_delay;
    reg         pre_pass_thru_mgmt;
    reg         pass_thru_user;
    reg         pass_thru_user_delay;
    reg         pre_pass_thru_user;
    wire        csb_reset;

    assign odata = {predata, SDI};
    assign oaddr = (state == `ADDRESS)  ? {addr[14:0], SDI} :
                   (state == `ADDRESS2) ? {addr[14:0], SDI} : addr;
    assign SDO   = ldata[7];
    assign csb_reset = CSB | reset;
    assign pass_thru_mgmt_reset = pass_thru_mgmt_delay | pre_pass_thru_mgmt;
    assign pass_thru_user_reset = pass_thru_user_delay | pre_pass_thru_user;

    // negedge SCK — drive SDO, wrstb
    always @(negedge SCK or posedge csb_reset) begin
        if (csb_reset == 1'b1) begin
            wrstb  <= 1'b0;
            ldata  <= 8'b00000000;
            sdoenb <= 1'b1;
        end else begin
            if (state == `DATA) begin
                if (readmode == 1'b1) begin
                    sdoenb <= 1'b0;
                    if (count == 3'b000) begin
                        ldata <= idata;
                    end else begin
                        ldata <= {ldata[6:0], 1'b0};
                    end
                end else begin
                    sdoenb <= 1'b1;
                end
                if (count == 3'b111) begin
                    if (writemode == 1'b1) begin
                        wrstb <= 1'b1;
                    end
                end else begin
                    wrstb <= 1'b0;
                end
            end else if (state == `MGMTPASS || state == `USERPASS) begin
                wrstb  <= 1'b0;
                sdoenb <= 1'b0;
            end else begin
                wrstb  <= 1'b0;
                sdoenb <= 1'b1;
            end
        end
    end

    // posedge SCK — state machine
    always @(posedge SCK or posedge csb_reset) begin
        if (csb_reset == 1'b1) begin
            addr              <= 16'h0000;
            rdstb             <= 1'b0;
            predata           <= 7'b0000000;
            state             <= `COMMAND;
            count             <= 3'b000;
            readmode          <= 1'b0;
            writemode         <= 1'b0;
            fixed             <= 3'b000;
            pass_thru_mgmt    <= 1'b0;
            pass_thru_mgmt_delay <= 1'b0;
            pre_pass_thru_mgmt   <= 1'b0;
            pass_thru_user    <= 1'b0;
            pass_thru_user_delay <= 1'b0;
            pre_pass_thru_user   <= 1'b0;
        end else begin

            if (state == `COMMAND) begin
                rdstb <= 1'b0;
                count <= count + 1;
                if (count == 3'b000) begin
                    writemode <= SDI;
                end else if (count == 3'b001) begin
                    readmode <= SDI;
                end else if (count < 3'b101) begin
                    fixed <= {fixed[1:0], SDI};
                end else if (count == 3'b101) begin
                    pre_pass_thru_mgmt <= SDI;
                end else if (count == 3'b110) begin
                    pre_pass_thru_user   <= SDI;
                    pass_thru_mgmt_delay <= pre_pass_thru_mgmt;
                end else if (count == 3'b111) begin
                    pass_thru_user_delay <= pre_pass_thru_user;
                    if (pre_pass_thru_mgmt == 1'b1) begin
                        state <= `MGMTPASS;
                        pre_pass_thru_mgmt <= 1'b0;
                    end else if (pre_pass_thru_user == 1'b1) begin
                        state <= `USERPASS;
                        pre_pass_thru_user <= 1'b0;
                    end else begin
                        state <= `ADDRESS;
                    end
                end

            end else if (state == `ADDRESS) begin
                // First address byte (upper 8 bits)
                count <= count + 1;
                addr  <= {addr[14:0], SDI};
                if (count == 3'b111) begin
                    state <= `ADDRESS2;
                end else begin
                    rdstb <= 1'b0;
                end

            end else if (state == `ADDRESS2) begin
                // Second address byte (lower 8 bits)
                count <= count + 1;
                addr  <= {addr[14:0], SDI};
                if (count == 3'b111) begin
                    state <= `DATA;
                    if (readmode == 1'b1) begin
                        rdstb <= 1'b1;
                    end
                end else begin
                    rdstb <= 1'b0;
                end

            end else if (state == `DATA) begin
                predata <= {predata[5:0], SDI};
                count   <= count + 1;
                if (count == 3'b111) begin
                    if (fixed == 3'b001) begin
                        state <= `COMMAND;
                    end else if (fixed != 3'b000) begin
                        fixed <= fixed - 1;
                        addr  <= addr + 1;
                    end else begin
                        addr  <= addr + 1;
                    end
                    if (readmode == 1'b1) begin
                        rdstb <= 1'b1;
                    end
                end else begin
                    rdstb <= 1'b0;
                end

            end else if (state == `MGMTPASS) begin
                pass_thru_mgmt <= 1'b1;
            end else if (state == `USERPASS) begin
                pass_thru_user <= 1'b1;
            end

        end
    end

endmodule // housekeeping_spi
`default_nettype wire