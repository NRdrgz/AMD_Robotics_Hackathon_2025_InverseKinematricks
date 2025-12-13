import select
import sys
import termios
import time
import tty

import serial


class BeltControl:
    def __init__(self, ser: serial.Serial):
        """
        Initialize belt control with a serial connection.
        Immediately disables the belt to prevent unwanted movement.
        """
        self.ser = ser
        self.disable()  # Disable immediately to prevent unwanted movement
        time.sleep(2)  # wait for Arduino reset

    def set_dir(self, sign: int):
        """Set belt direction (1 for forward, -1 for reverse)"""
        self.ser.write(f"DIR {sign}\n".encode())

    def set_speed(self, micros: int):
        """Set belt speed"""
        self.ser.write(f"SPD {micros}\n".encode())

    def enable(self):
        """Enable the belt motor"""
        self.ser.write(b"EN\n")

    def disable(self):
        """Disable the belt motor"""
        self.ser.write(b"DIS\n")

    def start_belt(self):
        """Start the belt moving"""
        self.enable()
        self.set_dir(-1)  # Set direction (1 for forward, -1 for reverse)
        self.set_speed(500)  # Set speed
        print("Belt started")

    def stop_belt(self):
        """Stop the belt"""
        self.disable()
        print("Belt stopped")

    def _read_key(self, timeout: float = 1 / 30):
        """Read a key from terminal stdin (only works when terminal is focused)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Wait for input with timeout (non-blocking)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if not rlist:
                return None  # No input available
            ch = sys.stdin.read(1)
            # Handle Ctrl+C in raw mode
            if ch == "\x03":
                raise KeyboardInterrupt
            # Handle escape sequences (arrow keys)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "C":
                        return "right"
                    elif ch3 == "D":
                        return "left"
                return "esc"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run(self):
        """Run the keyboard listener for belt control"""
        print(
            "Belt Control - Press RIGHT arrow to start, LEFT arrow to stop, ESC to exit"
        )
        print("Waiting for key presses...")

        try:
            while True:
                key = self._read_key()
                if key is None:
                    continue  # No input, check again at ~30fps
                elif key == "right":
                    self.start_belt()
                elif key == "left":
                    self.stop_belt()
                elif key == "esc":
                    self.stop_belt()
                    break
        except KeyboardInterrupt:
            self.stop_belt()

        # Cleanup
        self.disable()
        self.ser.close()
        print("Exiting...")


if __name__ == "__main__":
    ser = serial.Serial("/dev/cu.usbmodem1101", 115200, timeout=0.1)
    belt_control = BeltControl(ser)
    belt_control.run()
