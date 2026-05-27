from cocotb.triggers import ClockCycles, Timer
import cocotb

CMD_READ=0x10
CMD_WRITE=0x20

BYTE_STALL=0xFF
BYTE_X=0x00

class SpiboneBFM:
    def __init__(self, dut, signal_clk, signal_cs, spi_master):
        self.dut = dut
        self.spi_master = spi_master
        self.signal_cs = signal_cs
        self.clk = signal_clk

    async def read(self, starting_address, count=1, timeout=20):
        c=0
        addr_bytes = list(starting_address.to_bytes(4, byteorder='big'))

        self.signal_cs.value = 0
        await ClockCycles(self.clk, 100)  # let CS assert propagate

        results = []

        # CMD_READ, addr
        while (c < count):
            # Write command + address on first call; subsequent calls only need to wait one byte
            if (c == 0):
                # cocotb.log.info(f"READ: Writing {[CMD_READ] + addr_bytes}")
                await self.spi_master.write([CMD_READ] + addr_bytes)
            else:
                await self.spi_master.write([BYTE_X])

            self.spi_master.clear() # clear existing read buffer from writing to slave
            await self.spi_master.write([BYTE_X]) # poll for ack
            recv = (await self.spi_master.read(count=1))[0]
            while (recv == BYTE_STALL and timeout > 0):
                cocotb.log.info(f"Received {hex(recv)}; waiting...")
                await self.spi_master.write([BYTE_X]) # poll more...
                recv = (await self.spi_master.read(count=1))[0]
                timeout -= 1

            cocotb.log.info(f"Received {hex(recv)}")
            assert timeout > 0, "Timed out while waiting for ACK from slave"
            assert recv == 0xAC, f"Expected 0xAC, got {hex(recv)}"

            await self.spi_master.write([BYTE_X, BYTE_X, BYTE_X, BYTE_X, BYTE_X, BYTE_X, BYTE_X, BYTE_X])
            recv = await self.spi_master.read(count=8)
            results += [recv]
            cocotb.log.info(f"READ {recv}")
            c+=1

        self.signal_cs.value = 1
        await ClockCycles(self.clk, 100) # let CS deassert propagate

        return results

    async def write(self, starting_address, payloads, timeout=20):
        addr_bytes = list(starting_address.to_bytes(4, byteorder='big'))
        self.signal_cs.value = 0
        await ClockCycles(self.clk, 100)  # let CS assert propagate

        for i in range(len(payloads)):
            # Write command + address on first call; subsequent calls only need to wait one byte
            if (i == 0):
                cocotb.log.info(f"WRITE: Writing {[CMD_WRITE] + addr_bytes + payloads[i]}")
                await self.spi_master.write([CMD_WRITE] + addr_bytes + payloads[i])
            else:
                cocotb.log.info(f"WRITE: Writing {[BYTE_X] + payloads[i]}")
                await self.spi_master.write([BYTE_X] + payloads[i])
            self.spi_master.clear() # clear existing read buffer from writing to slave

            await self.spi_master.write([BYTE_X]) # poll for ack
            recv = (await self.spi_master.read(count=1))[0]
            while (recv == BYTE_STALL and timeout > 0):
                cocotb.log.info(f"Received {hex(recv)}; waiting...")
                await self.spi_master.write([BYTE_X]) # poll more...
                recv = (await self.spi_master.read(count=1))[0]
                timeout -= 1
            cocotb.log.info(f"Received {hex(recv)}")
            assert timeout > 0, "Timed out while waiting for ACK from slave"
            assert recv == 0xAC, f"Expected 0xAC, got {hex(recv)}"

        self.signal_cs.value = 1
        await ClockCycles(self.clk, 100)  # let CS assert propagate
