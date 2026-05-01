import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge


class SpiboneMaster:
    """SPI master with free-running SCLK implementing the spibone wait/ready protocol.

    SCLK runs continuously (CPOL=0, idle-low).  Transactions are framed by
    asserting/deasserting CS on falling-edge boundaries so the hardware sees a
    clean half-period setup window before the first SCK rise.
    """

    def __init__(self, dut, sclk_half_period_ns=25, cs_deassert_ns=30):
        self.dut     = dut
        self.half_ns = sclk_half_period_ns
        self.cs_ns   = cs_deassert_ns
        dut.spi_cs_ni.value  = 1
        dut.spi_clk_i.value  = 0
        dut.spi_mosi_i.value = 0
        cocotb.start_soon(self._clock())

    async def _clock(self):
        while True:
            await Timer(self.half_ns, 'ns')
            self.dut.spi_clk_i.value = 1
            await Timer(self.half_ns, 'ns')
            self.dut.spi_clk_i.value = 0

    async def _byte(self, mosi=0x00):
        """Transfer one byte MSB-first (CPOL=0 CPHA=0).
        Must be called at a falling-edge boundary; returns at the next falling-edge boundary.
        """
        miso = 0
        for bit in range(7, -1, -1):
            self.dut.spi_mosi_i.value = (mosi >> bit) & 1
            await RisingEdge(self.dut.spi_clk_i)
            await FallingEdge(self.dut.spi_clk_i)
            miso = (miso << 1) | int(self.dut.spi_miso_o.value)
        return miso

    async def _cs_assert(self):
        await FallingEdge(self.dut.spi_clk_i)
        self.dut.spi_cs_ni.value = 0

    async def _cs_deassert(self):
        # Called at a falling-edge boundary; CS goes high while SCK is low
        self.dut.spi_cs_ni.value = 1
        await Timer(self.cs_ns, 'ns')

    async def write(self, addr: int, data: int):
        """CMD=0x00 | ADDR(4B) | DATA(8B) | poll until 0x01 DONE on MISO."""
        await self._cs_assert()
        await self._byte(0x00)
        for b in addr.to_bytes(4, 'big'):
            await self._byte(b)
        for b in data.to_bytes(8, 'big'):
            await self._byte(b)
        while await self._byte() != 0x01:
            pass
        await self._cs_deassert()

    async def read(self, addr: int) -> int:
        """CMD=0x01 | ADDR(4B) | poll until 0x01 READY | receive 8B data."""
        await self._cs_assert()
        await self._byte(0x01)
        for b in addr.to_bytes(4, 'big'):
            await self._byte(b)
        while await self._byte() != 0x01:
            pass
        word = 0
        for _ in range(8):
            word = (word << 8) | await self._byte()
        await self._cs_deassert()
        return word

    async def burst_read(self, addr: int, n_words: int) -> list:
        """CMD=0x01 | ADDR(4B) | [poll 0x01 READY | 8B data] × n_words."""
        await self._cs_assert()
        await self._byte(0x01)
        for b in addr.to_bytes(4, 'big'):
            await self._byte(b)
        results = []
        for _ in range(n_words):
            while await self._byte() != 0x01:
                pass
            word = 0
            for _ in range(8):
                word = (word << 8) | await self._byte()
            results.append(word)
        await self._cs_deassert()
        return results
