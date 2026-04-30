
from cocotb.triggers import ClockCycles, Timer


class SpiboneBFM:
    def __init__(self, dut, clk_period_ns: int = 10, sck_half_ns: int = 250):
        self.dut = dut
        self.sck_half_ns = sck_half_ns
        # number of clk_i cycles per SCK half-period. used to align edges to clk_i to avoid setup/hold races against the registered sampler
        self._sck_half_cycles = sck_half_ns // clk_period_ns

    def idle(self) -> None:
        """park SPI lines in inactive state"""
        self.dut.spi_sck_i.value  = 0
        self.dut.spi_mosi_i.value = 0
        self.dut.spi_cs_n_i.value = 1

    async def _byte(self, tx: int) -> int:
        """shift one byte MSB first, return the byte received on MISO.

        each SCK edge is aligned to integer clk_i cycles meaning the registered MOSI sampler in spibone_wb sees a stable bit at every detected rise.
        """
        rx = 0
        for bit in range(7, -1, -1):
            self.dut.spi_mosi_i.value = (tx >> bit) & 1
            await ClockCycles(self.dut.clk_i, self._sck_half_cycles)
            await Timer(1, unit="ps")
            rx = (rx << 1) | int(self.dut.spi_miso_o.value)
            self.dut.spi_sck_i.value = 1
            await ClockCycles(self.dut.clk_i, self._sck_half_cycles)
            self.dut.spi_sck_i.value = 0
            await Timer(1, unit="ps")
        return rx

    async def _cs_assert(self) -> None:
        self.dut.spi_cs_n_i.value = 0
        await Timer(self.sck_half_ns, unit="ns")

    async def _cs_deassert(self) -> None:
        await Timer(self.sck_half_ns, unit="ns")
        self.dut.spi_cs_n_i.value = 1
        # let the WB side drain its current transaction before any next call
        await Timer(self.sck_half_ns * 4, unit="ns")

    async def write(self, addr: int, words: list[int]) -> None:
        await self._cs_assert()
        await self._byte(0x00)
        for shift in (24, 16, 8, 0):
            await self._byte((addr >> shift) & 0xFF)
        for word in words: #addr autoincrements by 4 per word
            for shift in (24, 16, 8, 0):
                await self._byte((word >> shift) & 0xFF)
        await self._cs_deassert()

    async def read(self, addr: int, num_words: int) -> list[int]:
        """addr increments same. one extra dummy byte is clocked between words to give spibone_wb time to fetch the next."""
        await self._cs_assert()
        await self._byte(0x01)
        for shift in (24, 16, 8, 0):
            await self._byte((addr >> shift) & 0xFF)

        out = []
        for i in range(num_words):
            if i > 0:
                # cover the cycle spibone_wb spends issuing the next WB read
                await self._byte(0xFF)
            b3 = await self._byte(0xFF)
            b2 = await self._byte(0xFF)
            b1 = await self._byte(0xFF)
            b0 = await self._byte(0xFF)
            out.append((b3 << 24) | (b2 << 16) | (b1 << 8) | b0)

        await self._cs_deassert()
        return out