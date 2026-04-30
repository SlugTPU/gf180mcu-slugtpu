from cocotb.triggers import Timer


class SpiboneMaster:
    """SPI master implementing the spibone wait/ready protocol.

    Drives spi_clk_i, spi_mosi_i, spi_cs_ni directly on the DUT and samples
    spi_miso_o.
    """

    def __init__(self, dut, sclk_half_period_ns=25, cs_deassert_ns=30):
        self.dut    = dut
        self.half_ns = sclk_half_period_ns
        self.cs_ns   = cs_deassert_ns
        dut.spi_cs_ni.value  = 1
        dut.spi_clk_i.value  = 0
        dut.spi_mosi_i.value = 0

    async def _byte(self, mosi=0x00):
        """Transfer one byte MSB-first (CPOL=0 CPHA=0). Returns received MISO byte."""
        miso = 0
        for bit in range(7, -1, -1):
            self.dut.spi_mosi_i.value = (mosi >> bit) & 1
            await Timer(self.half_ns, 'ns')
            self.dut.spi_clk_i.value = 1
            await Timer(self.half_ns, 'ns')
            miso = (miso << 1) | int(self.dut.spi_miso_o.value)
            self.dut.spi_clk_i.value = 0
        return miso

    async def write(self, addr: int, data: int):
        """CMD=0x00 | ADDR(4B) | DATA(8B) | poll until MISO==0x01 (DONE)."""
        self.dut.spi_cs_ni.value = 0
        await self._byte(0x00)
        for b in addr.to_bytes(4, 'big'):
            await self._byte(b)
        for b in data.to_bytes(8, 'big'):
            await self._byte(b)
        while await self._byte() != 0x01:
            pass
        await Timer(self.cs_ns, 'ns')
        self.dut.spi_cs_ni.value = 1
        await Timer(self.cs_ns, 'ns')

    async def read(self, addr: int) -> int:
        """CMD=0x01 | ADDR(4B) | poll until MISO==0x01 (READY) | receive 8B."""
        self.dut.spi_cs_ni.value = 0
        await self._byte(0x01)
        for b in addr.to_bytes(4, 'big'):
            await self._byte(b)
        while await self._byte() != 0x01:
            pass
        word = 0
        for _ in range(8):
            word = (word << 8) | await self._byte()
        await Timer(self.cs_ns, 'ns')
        self.dut.spi_cs_ni.value = 1
        await Timer(self.cs_ns, 'ns')
        return word

    async def burst_read(self, addr: int, n_words: int) -> list:
        """CMD=0x01 | ADDR(4B) | [poll READY | 8B data] × n_words.

        CS is deasserted immediately after the last data byte, so S_TX_BURST
        never fires for word n_words (no extra byte-boundaries in the frame).
        """
        self.dut.spi_cs_ni.value = 0
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
        await Timer(self.cs_ns, 'ns')
        self.dut.spi_cs_ni.value = 1
        await Timer(self.cs_ns, 'ns')
        return results
