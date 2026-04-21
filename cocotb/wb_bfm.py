# reusable Wishbone B4 classic master BFM for cocotb tests

# basically just wraps the <prefix>_{adr,dat_w,dat_r,we,stb,cyc,sel,ack} signals on a DUT.

from cocotb.triggers import RisingEdge, FallingEdge


class WishboneMaster:
    def __init__(self, dut, prefix, clk):
        self.clk   = clk
        self.adr   = getattr(dut, f"{prefix}_adr")
        self.dat_w = getattr(dut, f"{prefix}_dat_w")
        self.dat_r = getattr(dut, f"{prefix}_dat_r")
        self.we    = getattr(dut, f"{prefix}_we")
        self.stb   = getattr(dut, f"{prefix}_stb")
        self.cyc   = getattr(dut, f"{prefix}_cyc")
        self.sel   = getattr(dut, f"{prefix}_sel")
        self.ack   = getattr(dut, f"{prefix}_ack")
        self._sel_all = (1 << len(self.sel)) - 1

    def idle(self):
        """call once before reset"""
        self.adr.value   = 0
        self.dat_w.value = 0
        self.we.value    = 0
        self.stb.value   = 0
        self.cyc.value   = 0
        self.sel.value   = 0

    async def write(self, addr, data, sel=None):
        await FallingEdge(self.clk)
        self.adr.value   = addr
        self.dat_w.value = data
        self.we.value    = 1
        self.stb.value   = 1
        self.cyc.value   = 1
        self.sel.value   = self._sel_all if sel is None else sel

        while True:
            await RisingEdge(self.clk)
            if self.ack.value == 1:
                break

        await FallingEdge(self.clk)
        self.stb.value = 0
        self.cyc.value = 0
        self.we.value  = 0

    async def read(self, addr, sel=None):
        await FallingEdge(self.clk)
        self.adr.value   = addr
        self.dat_w.value = 0
        self.we.value    = 0
        self.stb.value   = 1
        self.cyc.value   = 1
        self.sel.value   = self._sel_all if sel is None else sel

        while True:
            await RisingEdge(self.clk)
            if self.ack.value == 1:
                break

        val = self.dat_r.value.to_unsigned()
        await FallingEdge(self.clk)
        self.stb.value = 0
        self.cyc.value = 0
        return val